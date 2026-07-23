# -*- coding: utf-8 -*-
"""
recognize_ctc_attn_transducer.py
=================================
三种解码模式评估脚本：
  1. ctc_greedy  - CTC 贪心解码
  2. attention   - Attention 自回归解码
  3. transducer  - Transducer 贪心解码
计算每种模式下的 CER（字错误率）。
"""

import argparse
import json
import os
import sys

import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.conformer_ctc_attn_transducer import ConformerCTCATTNTransducer

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def load_cmvn(cmvn_path):
    """加载 CMVN，返回 (mean, var) 均值和方差（已从累加和转换为真实统计量）"""
    if cmvn_path and os.path.exists(cmvn_path):
        with open(cmvn_path) as f:
            cmvn = json.load(f)
        frame_num = cmvn["frame_num"]
        mean_stat = torch.tensor(cmvn["mean_stat"], dtype=torch.float32)
        var_stat = torch.tensor(cmvn["var_stat"], dtype=torch.float32)
        mean = mean_stat / frame_num
        var = var_stat / frame_num - mean * mean
        return mean, var
    return None, None


def extract_fbank(wav_path, cmvn_mean=None, cmvn_var=None):
    """提取 Fbank 特征（用 soundfile 避免 ffmpeg 依赖）"""
    import soundfile as sf
    data, sr = sf.read(wav_path)
    if sr != 16000:
        import librosa
        data = librosa.resample(data, orig_sr=sr, target_sr=16000)
        sr = 16000
    waveform = torch.from_numpy(data).float().unsqueeze(0)
    waveform = waveform * (1 << 15)
    feat = kaldi.fbank(
        waveform,
        num_mel_bins=80,
        frame_length=25,
        frame_shift=10,
        dither=0.0,
        sample_frequency=16000,
    )
    if cmvn_mean is not None:
        feat = (feat - cmvn_mean) / (cmvn_var.sqrt() + 1e-10)
    return feat.unsqueeze(0)  # [1, T, 80]


def edit_distance(ref_list, hyp_list):
    """字符级编辑距离"""
    n, m = len(ref_list), len(hyp_list)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_list[i - 1] == hyp_list[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[n][m]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--dict", required=True)
    parser.add_argument("--cmvn", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mode", default="all",
                        choices=["ctc_greedy", "attention", "transducer", "all",
                                 "ctc_streaming", "transducer_streaming"])
    parser.add_argument("--streaming", action="store_true",
                        help="启用流式解码（等价于 ctc_streaming mode）")
    parser.add_argument("--chunk_size", type=int, default=16,
                        help="流式解码的 chunk size")
    parser.add_argument("--right_context", type=int, default=4,
                        help="流式解码的右侧上下文帧数")
    args = parser.parse_args()

    device = torch.device(args.device)

    # 加载字典
    vocab = {}
    idx2token = {}
    with open(args.dict, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                token, idx = parts[0], int(parts[1])
                vocab[token] = idx
                idx2token[idx] = token
    vocab_size = len(vocab)
    sos_id = vocab_size - 1
    eos_id = vocab_size - 1

    print(f"词表大小: {vocab_size}, <sos/eos> id: {sos_id}")

    # 加载 CMVN
    cmvn_mean, cmvn_var = load_cmvn(args.cmvn)

    # 加载模型（从 checkpoint 配置中读取 d_model）
    ckpt = torch.load(args.checkpoint, map_location=device)
    if "config" in ckpt:
        d_model = ckpt["config"].get("d_model", 144)
    else:
        d_model = 144
    model = ConformerCTCATTNTransducer(vocab_size=vocab_size, d_model=d_model).to(device)
    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    print(f"模型加载成功: {args.checkpoint} (d_model={d_model})")

    # 读取测试数据
    test_items = []
    with open(args.test_data, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                test_items.append(json.loads(line))

    print(f"测试集: {len(test_items)} 条\n")

    # 确定要测试的模式
    modes = []
    if args.mode == "all":
        modes = ["ctc_greedy", "attention", "transducer"]
    else:
        modes = [args.mode]

    # 如果 --streaming 标志启用，添加流式模式
    if args.streaming:
        modes = ["ctc_streaming"]

    for mode in modes:
        total_err, total_ref = 0, 0
        all_correct = 0
        print(f"{'='*60}")
        print(f"  解码模式: {mode}")
        if mode in ("ctc_streaming", "transducer_streaming"):
            print(f"  chunk_size={args.chunk_size}, right_context={args.right_context}")
        print(f"{'='*60}")

        for item in test_items:
            wav_path = item["wav"]
            ref_text = item["txt"]

            feats = extract_fbank(wav_path, cmvn_mean, cmvn_var).to(device)

            if mode == "ctc_greedy":
                hyps = model.recognize_ctc_greedy(feats, idx2token)
                hyp_text = hyps[0]
            elif mode == "attention":
                ys = model.recognize_attention(feats, max_len=20, sos_id=sos_id, eos_id=eos_id)
                tokens = []
                for t in range(1, ys.size(1)):
                    tok = ys[0, t].item()
                    if tok == eos_id:
                        break
                    tokens.append(idx2token.get(tok, ""))
                hyp_text = "".join(tokens)
            elif mode == "transducer":
                results = model.recognize_transducer(feats, max_len=20, sos_id=sos_id)
                hyp_text = "".join(idx2token.get(t, "") for t in results[0])
            elif mode == "ctc_streaming":
                hyps = model.recognize_ctc_streaming(
                    feats, idx2token,
                    chunk_size=args.chunk_size,
                    right_context=args.right_context,
                )
                hyp_text = hyps[0]
            elif mode == "transducer_streaming":
                results = model.recognize_transducer_streaming(
                    feats,
                    chunk_size=args.chunk_size,
                    right_context=args.right_context,
                    max_len=20, sos_id=sos_id,
                )
                hyp_text = "".join(idx2token.get(t, "") for t in results[0])
            else:
                continue

            ref_list = list(ref_text)
            hyp_list = list(hyp_text)
            err = edit_distance(ref_list, hyp_list)
            total_err += err
            total_ref += len(ref_list)
            if ref_text == hyp_text:
                all_correct += 1

            flag = "✓" if ref_text == hyp_text else "✗"
            print(f"{flag} 参考: {ref_text}  | 识别: {hyp_text}")

        cer = 100.0 * total_err / max(1, total_ref)
        acc = 100.0 * all_correct / max(1, len(test_items))
        print(f"\n  模式 [{mode}]  总字数: {total_ref}  错误: {total_err}")
        print(f"  CER = {cer:.2f}%   精确匹配率 = {acc:.2f}%\n")


if __name__ == "__main__":
    main()
