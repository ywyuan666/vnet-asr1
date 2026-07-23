# -*- coding: utf-8 -*-
"""
VAD + Streaming ASR 集成演示
=============================
展示 VAD 端点检测到流式 ASR 的完整链路。

注意：运行此脚本需要 VAD 模型文件。
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def main():
    parser = argparse.ArgumentParser(description="VAD + Streaming ASR 集成演示")
    parser.add_argument("--vad_model", default="../vad-system/checkpoints/best_model.pt",
                       help="VAD 模型路径")
    parser.add_argument("--asr_checkpoint", required=True, help="ASR 模型路径")
    parser.add_argument("--dict", required=True, help="字典文件")
    parser.add_argument("--cmvn", required=True, help="CMVN 文件")
    parser.add_argument("--wav", required=True, help="输入音频")
    parser.add_argument("--device", default="cpu", help="推理设备")
    args = parser.parse_args()

    # Step 1: VAD 检测
    print("=" * 60)
    print("VAD + Streaming ASR 集成演示")
    print("=" * 60)
    print(f"\n[1/4] 加载 VAD 模型: {args.vad_model}")

    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "vad-system"))
        import soundfile as sf
        import numpy as np
        import torch
        import torchaudio.compliance.kaldi as kaldi

        # Load audio
        audio, sr = sf.read(args.wav)
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000

        print(f"   音频: {os.path.basename(args.wav)}")
        print(f"   时长: {len(audio)/sr:.1f}s")
        print(f"   采样率: {sr}Hz")

        # VAD inference
        print(f"\n[2/4] VAD 端点检测...")
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vad-system"))
        from vad import dnn_vad
        vad = dnn_vad.DNNVAD(model_path=args.vad_model, device=args.device)
        segments = vad(audio, sr=sr)

        print(f"   检测到 {len(segments)} 个语音段:")
        for i, (start, end) in enumerate(segments):
            print(f"     段 {i+1}: {start:.2f}s - {end:.2f}s ({end-start:.2f}s)")

        # Step 2: ASR 流式推理
        print(f"\n[3/4] Streaming ASR 推理...")
        from model.conformer_ctc_attn_transducer import ConformerCTCATTNTransducer

        # Load model
        dict_path = args.dict
        with open(dict_path, "r", encoding="utf-8") as f:
            vocab_size = len(f.read().splitlines())

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

        state = torch.load(args.asr_checkpoint, map_location=args.device, weights_only=True)
        if "model_state" in state:
            model.load_state_dict(state["model_state"])
        else:
            model.load_state_dict(state)
        model.to(args.device)
        model.eval()

        # Load dictionary tokens
        with open(dict_path, "r", encoding="utf-8") as f:
            id2token = [line.strip() for line in f]

        sos_id = vocab_size - 1

        # Process each segment
        print(f"\n[4/4] 识别结果:")
        for i, (start, end) in enumerate(segments):
            segment_audio = audio[int(start * sr):int(end * sr)]

            if len(segment_audio) < sr * 0.1:  # < 100ms, skip
                print(f"   段 {i+1} ({start:.2f}s-{end:.2f}s): [跳过，语音过短]")
                continue

            # Extract features
            waveform = torch.from_numpy(segment_audio * (1 << 15)).float()
            feat = kaldi.fbank(
                waveform.unsqueeze(0),
                num_mel_bins=80,
                frame_length=25,
                frame_shift=10,
                dither=1.0,
            )

            # CMVN normalization
            if args.cmvn and os.path.exists(args.cmvn):
                with open(args.cmvn, encoding="utf-8") as f:
                    import json
                    cmvn = json.load(f)
                frame_num = cmvn["frame_num"]
                mean_stat = torch.tensor(cmvn["mean_stat"], dtype=torch.float32)
                var_stat = torch.tensor(cmvn["var_stat"], dtype=torch.float32)
                cmvn_mean = mean_stat / frame_num
                cmvn_var = var_stat / frame_num - cmvn_mean * cmvn_mean
                feat = (feat - cmvn_mean) / (cmvn_var.sqrt() + 1e-10)
            else:
                feat = (feat - feat.mean()) / (feat.std() + 1e-10)

            # Transducer decoding (streaming-friendly)
            feat = feat.unsqueeze(0).to(args.device)

            result = model.recognize_transducer(feat, max_len=50, sos_id=sos_id)

            # Convert IDs to text
            text = "".join(id2token[tid] for tid in result[0]
                          if 1 < tid < len(id2token) - 1)

            print(f"   段 {i+1} ({start:.2f}s-{end:.2f}s): {text}")

        print(f"\n集成演示完成")

    except ImportError as e:
        print(f"\n[ERROR] 依赖缺失: {e}")
        print("  请确保 VAD 系统和所有依赖已正确安装")
    except FileNotFoundError as e:
        print(f"\n[ERROR] 文件未找到: {e}")
    except Exception as e:
        print(f"\n[ERROR] 运行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
