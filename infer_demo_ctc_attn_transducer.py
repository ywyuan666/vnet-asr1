# -*- coding: utf-8 -*-
"""
infer_demo_ctc_attn_transducer.py
===================================
单条音频识别演示（支持 CTC / Attention / Transducer 三种解码模式）
"""
import argparse
import json
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recognize_ctc_attn_transducer import extract_fbank, load_cmvn
from model.conformer_ctc_attn_transducer import ConformerCTCATTNTransducer

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dict", required=True)
    parser.add_argument("--cmvn", default=None)
    parser.add_argument("--wav", required=True)
    parser.add_argument("--mode", default="ctc_greedy",
                        choices=["ctc_greedy", "attention", "transducer",
                                 "ctc_streaming", "transducer_streaming"])
    parser.add_argument("--device", default="cpu")
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

    # 提取特征
    feats = extract_fbank(args.wav, cmvn_mean, cmvn_var).to(device)

    # 解码
    if args.mode == "ctc_greedy":
        hyps = model.recognize_ctc_greedy(feats, idx2token)
        result = hyps[0]
    elif args.mode == "attention":
        ys = model.recognize_attention(feats, max_len=20, sos_id=sos_id, eos_id=eos_id)
        tokens = []
        for t in range(1, ys.size(1)):
            tok = ys[0, t].item()
            if tok == eos_id:
                break
            tokens.append(idx2token.get(tok, ""))
        result = "".join(tokens)
    elif args.mode == "transducer":
        results = model.recognize_transducer(feats, max_len=20, sos_id=sos_id)
        result = "".join(idx2token.get(t, "") for t in results[0])
    elif args.mode == "ctc_streaming":
        hyps = model.recognize_ctc_streaming(
            feats, idx2token,
            chunk_size=args.chunk_size,
            right_context=args.right_context,
        )
        result = hyps[0]
    elif args.mode == "transducer_streaming":
        results = model.recognize_transducer_streaming(
            feats,
            chunk_size=args.chunk_size,
            right_context=args.right_context,
            max_len=20, sos_id=sos_id,
        )
        result = "".join(idx2token.get(t, "") for t in results[0])

    print("\n" + "=" * 50)
    print(f"  解码模式: {args.mode}")
    if "streaming" in args.mode:
        print(f"  chunk_size={args.chunk_size}, right_context={args.right_context}")
    print(f"  识别结果: 【{result}】")
    print("=" * 50)


if __name__ == "__main__":
    main()
