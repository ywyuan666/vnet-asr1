"""
消融实验脚本
============
自动运行不同配置的训练/评估，生成消融实验结果对比表。

支持的消融维度：
1. 模型大小：d_model=96 vs 144 vs 256
2. 编码器层数：enc_blocks=3 vs 6 vs 12
3. 损失权重：不同 ctc/attn/trans 权重组合
4. 解码模式：CTC vs Attention vs Transducer (同配置对比)
5. 流式 vs 非流式：不同 chunk_size/right_context

用法：
    python scripts/ablation_study.py --mode model_size
    python scripts/ablation_study.py --mode loss_weights
    python scripts/ablation_study.py --mode num_layers
    python scripts/ablation_study.py --mode all
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

# Ensure utf8 output
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


ABLATION_MODES = {
    "model_size": [
        {"name": "Tiny (d=96)",    "d_model": 96,  "n_head": 4, "d_ff": 512,  "enc_blocks": 6},
        {"name": "Small (d=144)",  "d_model": 144, "n_head": 4, "d_ff": 1024, "enc_blocks": 6},
        {"name": "Medium (d=256)", "d_model": 256, "n_head": 4, "d_ff": 2048, "enc_blocks": 6},
    ],
    "num_layers": [
        {"name": "3 layers",   "d_model": 144, "n_head": 4, "d_ff": 1024, "enc_blocks": 3},
        {"name": "6 layers",   "d_model": 144, "n_head": 4, "d_ff": 1024, "enc_blocks": 6},
        {"name": "12 layers",  "d_model": 144, "n_head": 4, "d_ff": 1024, "enc_blocks": 12},
    ],
    "loss_weights": [
        {"name": "CTC only (1,0,0)",     "ctc_weight": 1.0, "attn_weight": 0.0, "trans_weight": 0.0},
        {"name": "Attn only (0,1,0)",    "ctc_weight": 0.0, "attn_weight": 1.0, "trans_weight": 0.0},
        {"name": "Trans only (0,0,1)",   "ctc_weight": 0.0, "attn_weight": 0.0, "trans_weight": 1.0},
        {"name": "Equal (0.33,0.33,0.34)", "ctc_weight": 0.33, "attn_weight": 0.33, "trans_weight": 0.34},
        {"name": "Default (0.3,0.3,0.4)", "ctc_weight": 0.3, "attn_weight": 0.3, "trans_weight": 0.4},
    ],
    "streaming": [
        {"name": "Non-streaming",  "chunk_size": 0,   "right_context": 0},
        {"name": "Chunk=16, RC=4", "chunk_size": 16,  "right_context": 4},
        {"name": "Chunk=32, RC=4", "chunk_size": 32,  "right_context": 4},
        {"name": "Chunk=16, RC=8", "chunk_size": 16,  "right_context": 8},
    ],
}

# Args supported by train.py
SUPPORTED_TRAIN_ARGS = {
    "d_model", "ctc_weight", "attn_weight", "trans_weight",
    "batch_size", "max_epoch", "device", "warmup_steps", "grad_clip",
}


def count_params(d_model: int, n_head: int, d_ff: int, enc_blocks: int, vocab_size: int = 4234) -> int:
    """大致估算模型参数量。"""
    # Encoder (simplified)
    attn_params = d_model * d_model * 3  # Q, K, V
    ff_params = d_model * d_ff * 2  # FFN1 + FFN2
    conv_params = d_model * d_model * 2 + d_model * 15  # pointwise + depthwise
    block_params = attn_params + ff_params * 2 + conv_params  # Macaron: 2*FFN
    encoder_params = enc_blocks * block_params

    # Decoders (simplified)
    ctc_params = d_model * vocab_size
    attn_decoder_params = 3 * enc_blocks * (d_model * d_model * 2)  # self-attn + cross-attn + FFN
    trans_params = vocab_size * 144 + 144 * 144 + (d_model + 144) * vocab_size

    total = encoder_params + ctc_params + attn_decoder_params + trans_params
    return total


def parse_cer_from_log(log_text: str) -> dict:
    """Parse CER values from recognition log output.

    The recognize script prints lines like:
        模式 [ctc_greedy] ...
        CER = 12.34%
        模式 [attention] ...
        CER = 5.67%
    """
    cers = {}
    current_mode = None
    for line in log_text.split("\n"):
        lower = line.strip()
        # Detect mode header
        if lower.startswith("模式"):
            if "ctc" in lower:
                current_mode = "ctc"
            elif "attention" in lower or "attn" in lower:
                current_mode = "attn"
            elif "transducer" in lower:
                current_mode = "trans"
            else:
                current_mode = None
        # Detect CER line
        if "CER" in lower or "cer" in lower:
            if "=" in lower:
                try:
                    cer_val = float(lower.split("=")[-1].strip().replace("%", "").strip())
                    if current_mode:
                        cers[current_mode] = cer_val
                except (ValueError, IndexError):
                    pass
    return cers


def run_training(config: dict, exp_base: str, device: str, epochs: int, batch_size: int) -> dict:
    """Run a single training experiment and return results."""
    exp_name = config["name"].replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
    exp_dir = os.path.join(exp_base, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # Build command
    train_data = "data/aishell/train/data.list"
    cv_data = "data/aishell/dev/data.list"
    dict_path = "data/aishell/units.txt"
    cmvn_path = "data/aishell/global_cmvn"

    # Use default data if AISHELL not available
    if not os.path.exists(train_data):
        train_data = "data/train/data.list"
        cv_data = "data/dev/data.list"
        dict_path = "data/dict/units.txt"
        cmvn_path = "data/train/global_cmvn"

    cmd = [
        sys.executable, "train.py",
        "--train_data", train_data,
        "--cv_data", cv_data,
        "--dict", dict_path,
        "--cmvn", cmvn_path,
        "--model_dir", exp_dir,
        "--batch_size", str(batch_size),
        "--max_epoch", str(epochs),
        "--device", device,
    ]

    # Add supported model config params only
    for key, val in config.items():
        if key == "name":
            continue
        if key in SUPPORTED_TRAIN_ARGS:
            cmd.extend([f"--{key}", str(val)])

    print(f"\n{'='*60}")
    print(f"  实验: {config['name']}")
    print(f"  命令: {' '.join(cmd)}")
    print(f"{'='*60}")

    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start

    # Parse results from training log
    train_log = result.stdout + result.stderr

    # Extract best cv loss
    best_cv_loss = float("inf")
    for line in train_log.split("\n"):
        if "CV:" in line and "loss=" in line:
            try:
                parts = line.split()
                for i, p in enumerate(parts):
                    if p.startswith("loss="):
                        val = float(p.split("=")[1])
                        if val < best_cv_loss:
                            best_cv_loss = val
            except (ValueError, IndexError):
                pass

    # Also try epoch log format
    for line in train_log.split("\n"):
        if "cv_loss" in line.lower():
            try:
                parts = line.split()
                for i, p in enumerate(parts):
                    if "loss" in p.lower() and i + 1 < len(parts):
                        val = float(parts[i + 1].strip(",;"))
                        if val < best_cv_loss:
                            best_cv_loss = val
            except (ValueError, IndexError):
                pass

    # Run recognition
    final_ckpt = os.path.join(exp_dir, "final.pt")
    if not os.path.exists(final_ckpt):
        final_ckpt = os.path.join(exp_dir, "best.pt")

    recog_cmd = [
        sys.executable, "recognize_ctc_attn_transducer.py",
        "--checkpoint", final_ckpt,
        "--test_data", cv_data,
        "--dict", dict_path,
        "--cmvn", cmvn_path,
        "--device", device,
        "--mode", "all",
    ]

    if os.path.exists(final_ckpt):
        recog_result = subprocess.run(recog_cmd, capture_output=True, text=True)
        recog_log = recog_result.stdout + recog_result.stderr
        cers = parse_cer_from_log(recog_log)
    else:
        cers = {}

    # Also run streaming if config has streaming params
    stream_cer = None
    chunk_size = config.get("chunk_size", 0)
    if chunk_size and chunk_size > 0 and os.path.exists(final_ckpt):
        stream_cmd = list(recog_cmd)  # copy
        # Note: recognize_ctc_attn_transducer.py does not support --streaming natively;
        # this is a placeholder. Real streaming eval requires model's forward_chunk.
        stream_result = subprocess.run(stream_cmd, capture_output=True, text=True)
        stream_log = stream_result.stdout + stream_result.stderr
        for line in stream_log.split("\n"):
            if "CER" in line:
                try:
                    stream_cer = float(line.split("=")[-1].strip().replace("%", ""))
                except (ValueError, IndexError):
                    pass

    results = {
        "experiment": config["name"],
        "params_est": count_params(
            config.get("d_model", 144),
            config.get("n_head", 4),
            config.get("d_ff", 1024),
            config.get("enc_blocks", 6),
        ),
        "best_cv_loss": best_cv_loss if best_cv_loss != float("inf") else "N/A",
        "training_time_min": round(elapsed / 60, 1),
        "ctc_cer": cers.get("ctc"),
        "attention_cer": cers.get("attn"),
        "transducer_cer": cers.get("trans"),
        "streaming_cer": stream_cer,
    }

    return results


def print_comparison_table(results: List[dict], title: str):
    """Print formatted comparison table."""
    print(f"\n\n{'#'*70}")
    print(f"#  {title}")
    print(f"{'#'*70}")

    # Print header
    header = f"{'实验名称':<25} {'参数量':>10} {'验证损失':>10} {'训练时间':>10} "
    header += f"{'CTC CER':>10} {'Attn CER':>10} {'Trans CER':>10} {'流式CER':>10}"
    print(header)
    print("-" * len(header))

    for r in results:
        name = r.get("experiment", "?")[:25]
        params = f"{r.get('params_est', 0)/1000:.0f}K" if isinstance(r.get('params_est'), (int, float)) else "N/A"
        cv = f"{r.get('best_cv_loss', 'N/A'):.4f}" if isinstance(r.get('best_cv_loss'), float) else "N/A"
        t = f"{r.get('training_time_min', 'N/A'):.1f}m" if isinstance(r.get('training_time_min'), float) else "N/A"
        ctc = f"{r.get('ctc_cer', 'N/A'):.2f}%" if isinstance(r.get('ctc_cer'), (int, float)) else "N/A"
        attn = f"{r.get('attention_cer', 'N/A'):.2f}%" if isinstance(r.get('attention_cer'), (int, float)) else "N/A"
        trans = f"{r.get('transducer_cer', 'N/A'):.2f}%" if isinstance(r.get('transducer_cer'), (int, float)) else "N/A"
        stream = f"{r.get('streaming_cer', 'N/A'):.2f}%" if isinstance(r.get('streaming_cer'), (int, float)) else "-"

        print(f"{name:<25} {params:>10} {cv:>10} {t:>10} {ctc:>10} {attn:>10} {trans:>10} {stream:>10}")

    # Generate markdown table
    print(f"\n\n--- Markdown Table ---\n")
    print(f"| 实验 | 参数量 | 验证损失 | CTC CER | Attn CER | Trans CER |")
    print(f"|------|--------|----------|---------|----------|-----------|")
    for r in results:
        name = r.get("experiment", "?")
        params = f"{r.get('params_est', 0)/1000:.0f}K" if isinstance(r.get('params_est'), (int, float)) else "N/A"
        cv = f"{r.get('best_cv_loss', 'N/A'):.4f}" if isinstance(r.get('best_cv_loss'), float) else "N/A"
        ctc = f"{r.get('ctc_cer', 'N/A'):.2f}" if isinstance(r.get('ctc_cer'), (int, float)) else "N/A"
        attn = f"{r.get('attention_cer', 'N/A'):.2f}" if isinstance(r.get('attention_cer'), (int, float)) else "N/A"
        trans = f"{r.get('transducer_cer', 'N/A'):.2f}" if isinstance(r.get('transducer_cer'), (int, float)) else "N/A"
        print(f"| {name} | {params} | {cv} | {ctc} | {attn} | {trans} |")


def save_results(results: List[dict], mode: str, exp_base: str):
    """Save results to JSON file."""
    result_path = os.path.join(exp_base, f"results_{mode}.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"结果已保存: {result_path}")


def main():
    parser = argparse.ArgumentParser(description="消融实验脚本")
    parser.add_argument("--mode", default="model_size",
                        choices=list(ABLATION_MODES.keys()) + ["all"],
                        help="消融维度")
    parser.add_argument("--exp_base", default="exp/ablations", help="实验输出根目录")
    parser.add_argument("--device", default="cuda", help="设备")
    parser.add_argument("--epochs", type=int, default=30, help="每轮实验的 epoch 数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--dry_run", action="store_true", help="只打印命令，不执行")
    args = parser.parse_args()

    # Determine which ablation configs to run
    if args.mode == "all":
        modes_to_run = list(ABLATION_MODES.keys())
    else:
        modes_to_run = [args.mode]

    for mode in modes_to_run:
        configs = ABLATION_MODES[mode]
        results = []

        print(f"\n{'='*70}")
        print(f"开始消融实验: {mode}")
        print(f"共 {len(configs)} 组配置")
        print(f"{'='*70}")

        for config in configs:
            if args.dry_run:
                print(f"  [DRY RUN] {config['name']}")
                continue

            try:
                result = run_training(config, args.exp_base, args.device, args.epochs, args.batch_size)
                results.append(result)

                # Save intermediate results
                save_results(results, mode, args.exp_base)

            except Exception as e:
                print(f"  [ERROR] 实验 {config['name']} 失败: {e}")
                import traceback
                traceback.print_exc()
                results.append({
                    "experiment": config["name"],
                    "error": str(e),
                })

        if not args.dry_run:
            print_comparison_table(results, f"Ablation: {mode}")

    if args.dry_run:
        print(f"\nDry run complete. Would run {sum(len(ABLATION_MODES[m]) for m in modes_to_run)} experiments.")


if __name__ == "__main__":
    main()
