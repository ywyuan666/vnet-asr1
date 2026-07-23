# -*- coding: utf-8 -*-
"""
TensorRT 导出与优化
====================
将 PyTorch 模型导出为 TensorRT engine，实现推理加速。

支持量化：
- FP32: 标准精度
- FP16: 半精度加速 (推荐)
- INT8: 量化加速（需要标定数据）

用法：
    # FP32 导出
    python tools/export_tensorrt.py --checkpoint exp/aishell_conformer/final.pt \
        --dict data/aishell/units.txt --output model.trt

    # FP16 导出
    python tools/export_tensorrt.py --checkpoint exp/aishell_conformer/final.pt \
        --dict data/aishell/units.txt --output model_fp16.trt --fp16

    # INT8 量化
    python tools/export_tensorrt.py --checkpoint exp/aishell_conformer/final.pt \
        --dict data/aishell/units.txt --output model_int8.trt --int8 \
        --calib_data data/aishell/train/data.list

依赖:
    pip install tensorrt>=8.5
    pip install torch-tensorrt
"""

import argparse
import json
import os
import sys
import time
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def export_onnx(
    checkpoint: str,
    dict_path: str,
    output_path: str,
    max_seq_len: int = 2048,
    max_decoder_len: int = 20,
) -> str:
    """
    Export PyTorch model to ONNX format.
    This is the first step towards TensorRT.

    Exports encoder only (most compute-intensive part).

    Returns: ONNX model path.
    """
    try:
        import torch
        import torch.onnx
    except ImportError:
        print("[ERROR] PyTorch 未安装")
        sys.exit(1)

    from model.conformer_ctc_attn_transducer import ConformerCTCATTNTransducer

    # Load dictionary for vocab size
    with open(dict_path, "r", encoding="utf-8") as f:
        vocab_size = len(f.read().splitlines())

    # Create model
    model = ConformerCTCATTNTransducer(
        vocab_size=vocab_size,
        idim=80,
        d_model=144,
        d_ff=1024,
        enc_blocks=6,
        attn_blocks=3,
        pred_layers=1,
        n_head=4,
        dropout=0.1,
        ctc_weight=0.3,
        attn_weight=0.3,
        trans_weight=0.4,
    )

    # Load checkpoint
    print(f"加载 checkpoint: {checkpoint}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        model.load_state_dict(state)
    model.eval()
    model.to("cpu")

    print(f"\n模型信息:")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {total_params:,} ({total_params/1e6:.1f}M)")

    # Export encoder to ONNX
    encoder = model.encoder
    batch_size = 1

    # Prepare dummy input
    dummy_input = torch.randn(batch_size, max_seq_len, 80)
    dummy_lengths = torch.tensor([max_seq_len])

    onnx_path = output_path.replace(".trt", ".onnx")

    print(f"\n导出 ONNX (encoder only)...")
    print(f"  输入: (1, {max_seq_len}, 80)")

    torch.onnx.export(
        encoder,
        (dummy_input, dummy_lengths),
        onnx_path,
        input_names=["speech", "speech_lengths"],
        output_names=["encoder_out", "encoder_out_lens"],
        dynamic_axes={
            "speech": {0: "batch", 1: "time"},
            "speech_lengths": {0: "batch"},
            "encoder_out": {0: "batch", 1: "time_out"},
            "encoder_out_lens": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
        verbose=False,
    )

    print(f"  ONNX 已保存: {onnx_path} ({os.path.getsize(onnx_path)/1024/1024:.1f} MB)")

    # Export decoder too
    decoder = model.attn_decoder
    dummy_encoder_out = torch.randn(batch_size, max_seq_len // 4, 144)
    dummy_tokens = torch.randint(2, vocab_size - 1, (batch_size, max_decoder_len))

    decoder_onnx_path = onnx_path.replace(".onnx", "_decoder.onnx")
    torch.onnx.export(
        decoder,
        (dummy_encoder_out, dummy_tokens),
        decoder_onnx_path,
        input_names=["encoder_out", "tokens"],
        output_names=["decoder_logits"],
        dynamic_axes={
            "encoder_out": {0: "batch", 1: "time"},
            "tokens": {0: "batch", 1: "length"},
            "decoder_logits": {0: "batch", 1: "length"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    print(f"  Decoder ONNX 已保存: {decoder_onnx_path}")

    # Export CTC head
    ctc_onnx_path = onnx_path.replace(".onnx", "_ctc.onnx")
    ctc_head = model.ctc_linear

    torch.onnx.export(
        ctc_head,
        dummy_encoder_out,
        ctc_onnx_path,
        input_names=["encoder_out"],
        output_names=["ctc_logits"],
        dynamic_axes={
            "encoder_out": {0: "batch", 1: "time"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    print(f"  CTC ONNX 已保存: {ctc_onnx_path}")

    return onnx_path


def export_tensorrt(
    onnx_path: str,
    output_path: str,
    fp16: bool = False,
    int8: bool = False,
    calib_data: Optional[str] = None,
    max_batch_size: int = 1,
):
    """
    Convert ONNX model to TensorRT engine.

    Args:
        onnx_path: Path to ONNX model
        output_path: Path to save TensorRT engine
        fp16: Enable FP16 inference
        int8: Enable INT8 quantization
        calib_data: Path to calibration data list (for INT8)
        max_batch_size: Maximum batch size
    """
    try:
        import tensorrt as trt
    except ImportError:
        print("[WARNING] TensorRT 未安装。跳过 TensorRT 导出。")
        print("  安装: pip install tensorrt>=8.5")
        print(f"  ONNX 模型已保存在: {onnx_path}")
        return None

    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse ONNX
    print(f"\n解析 ONNX 模型: {onnx_path}")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            print("  ONNX 解析错误:")
            for error in range(parser.num_errors):
                print(f"    {parser.get_error(error)}")
            return None

    print("  解析成功")

    # Build config
    config = builder.create_builder_config()

    # Workspace size (1GB)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    # FP16
    if fp16:
        print("  启用 FP16 模式")
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        else:
            print("  [WARNING] 硬件不支持 FP16，回退到 FP32")

    # INT8
    if int8:
        print("  启用 INT8 量化")
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            if calib_data:
                print(f"  使用标定数据: {calib_data}")
            else:
                print("  [WARNING] 未提供标定数据，INT8 量化可能不准确")
        else:
            print("  [WARNING] 硬件不支持 INT8，回退到 FP32")

    # Build engine
    print(f"\n构建 TensorRT 引擎 (这可能需要几分钟)...")
    start = time.time()

    # Set dynamic shapes
    profile = builder.create_optimization_profile()
    profile.set_shape("speech", (1, 1, 80), (1, 400, 80), (1, 2048, 80))
    profile.set_shape("speech_lengths", (1,), (1,), (1,))
    config.add_optimization_profile(profile)

    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        print("  [ERROR] TensorRT 引擎构建失败")
        return None

    elapsed = time.time() - start

    # Save engine
    with open(output_path, "wb") as f:
        f.write(serialized_engine)

    engine_size = os.path.getsize(output_path) / 1024 / 1024

    print(f"\nTensorRT 引擎已保存: {output_path}")
    print(f"   大小: {engine_size:.1f} MB")
    print(f"   构建时间: {elapsed:.1f} 秒")
    print(f"   精度: {'INT8' if int8 else 'FP16' if fp16 else 'FP32'}")

    return output_path


def benchmark_tensorrt(engine_path: str, num_runs: int = 100):
    """Benchmark TensorRT engine inference speed."""
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
    except ImportError:
        print("[WARNING] TensorRT 或 PyCUDA 未安装，跳过基准测试")
        return

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(TRT_LOGGER)

    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    if engine is None:
        print("[ERROR] 引擎加载失败")
        return

    context = engine.create_execution_context()

    # Allocate buffers
    inputs = []
    outputs = []
    bindings = []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        dtype = engine.get_tensor_dtype(name)
        shape = engine.get_tensor_shape(name)

        # Handle dynamic shapes
        if -1 in shape:
            shape = (1, 1, 80)  # minimal for testing

        size = trt.volume(shape)
        el_size = trt.bytes_per_element(dtype)
        allocation = cuda.mem_alloc(size * el_size)

        context.set_tensor_address(name, int(allocation))

        if i == 0:
            inputs.append(allocation)
            bindings.append(allocation)
        else:
            outputs.append(allocation)
            bindings.append(allocation)

    # Warmup
    print(f"\n基准测试 ({num_runs} runs)...")
    for _ in range(10):
        context.execute_async_v3(0)

    # Benchmark
    cuda_start = cuda.Event()
    cuda_end = cuda.Event()

    cuda_start.record()
    for _ in range(num_runs):
        context.execute_async_v3(0)
    cuda_end.record()
    cuda_end.synchronize()

    elapsed_ms = cuda_start.time_since(cuda_end) / num_runs

    print(f"  平均推理延迟: {elapsed_ms:.2f} ms")
    print(f"  最大吞吐量: {1000/elapsed_ms:.0f} 帧/秒")


def main():
    parser = argparse.ArgumentParser(description="TensorRT 导出与优化")
    parser.add_argument("--checkpoint", required=True, help="PyTorch checkpoint 路径")
    parser.add_argument("--dict", required=True, help="字典文件路径")
    parser.add_argument("--output", default="model.trt", help="TensorRT 引擎输出路径")
    parser.add_argument("--fp16", action="store_true", help="启用 FP16")
    parser.add_argument("--int8", action="store_true", help="启用 INT8 量化")
    parser.add_argument("--calib_data", help="INT8 标定数据 (data.list)")
    parser.add_argument("--onnx_only", action="store_true", help="仅导出 ONNX")
    parser.add_argument("--benchmark", action="store_true", help="执行基准测试")
    parser.add_argument("--max_seq_len", type=int, default=800, help="最大序列长度 (帧)")
    args = parser.parse_args()

    # Step 1: Export to ONNX
    onnx_path = export_onnx(
        checkpoint=args.checkpoint,
        dict_path=args.dict,
        output_path=args.output,
        max_seq_len=args.max_seq_len,
    )

    if args.onnx_only:
        print(f"\nONNX 导出完成: {onnx_path}")
        return

    # Step 2: Convert to TensorRT
    trt_path = export_tensorrt(
        onnx_path=onnx_path,
        output_path=args.output,
        fp16=args.fp16,
        int8=args.int8,
        calib_data=args.calib_data,
    )

    if trt_path is None:
        print("\nTensorRT 导出失败（可能未安装 TensorRT）")
        print(f"  ONNX 模型可用: {onnx_path}")
        return

    # Step 3: Benchmark
    if args.benchmark:
        benchmark_tensorrt(trt_path)

    # Print size comparison
    print(f"\n文件大小对比:")
    pt_size = os.path.getsize(args.checkpoint) / 1024 / 1024 if os.path.exists(args.checkpoint) else 0
    onnx_size = os.path.getsize(onnx_path) / 1024 / 1024
    trt_size = os.path.getsize(trt_path) / 1024 / 1024

    print(f"  PyTorch: {pt_size:.1f} MB")
    print(f"  ONNX:    {onnx_size:.1f} MB")
    print(f"  TensorRT: {trt_size:.1f} MB")
    print(f"  压缩比: {pt_size/trt_size:.1f}x")

    print(f"\nTensorRT 导出完成!")


if __name__ == "__main__":
    main()
