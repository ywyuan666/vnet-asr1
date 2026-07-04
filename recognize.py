# -*- coding: utf-8 -*-
"""
recognize.py
============
用训练好的 U2++ 模型对测试集解码，并计算 CER（字错误率）。

实现方式：调用 WeNet 自带、跨版本最稳定的解码入口 `wenet.bin.recognize`，
拿到识别结果后，本脚本再对照标注算 CER —— 这样不依赖各版本易变的 Python API。

用法：
  python recognize.py --config exp/u2pp_conformer/train.yaml \
      --checkpoint exp/u2pp_conformer/final.pt \
      --test_data data/test/data.list --dict data/dict/units.txt \
      --mode attention_rescoring

--mode 可选: ctc_greedy_search / ctc_prefix_beam_search / attention / attention_rescoring
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

try:  # Windows 终端正常显示中文
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def edit_distance(ref, hyp):
    """字符级编辑距离（Levenshtein）。"""
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1,
                           dp[i][j - 1] + 1,
                           dp[i - 1][j - 1] + cost)
    return dp[n][m]


def run_wenet_recognize(args, result_file):
    """调用 WeNet 解码模块，结果写入 result_file。"""
    result_dir = os.path.dirname(os.path.dirname(result_file))
    cmd = [
        sys.executable, "-m", "wenet.bin.recognize",
        "--device", "cpu",
        "--gpu", "-1",  # 使用 CPU 解码
        "--modes", args.mode,
        "--config", args.config,
        "--test_data", args.test_data,
        "--checkpoint", args.checkpoint,
        "--beam_size", str(args.beam_size),
        "--batch_size", "1",
        "--ctc_weight", str(args.ctc_weight),
        "--reverse_weight", str(args.reverse_weight),
        "--result_dir", result_dir,
        "--data_type", "raw",
    ]
    print("执行解码...")
    subprocess.run(cmd, check=True, capture_output=False)


def load_refs(test_data):
    refs = {}
    with open(test_data, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                o = json.loads(line)
                refs[o["key"]] = o["txt"].strip()
    return refs


def load_hyps(result_file):
    """wenet 结果文件每行: `key 识 别 文 字`（空格分隔的建模单元）。"""
    hyps = {}
    with open(result_file, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if not parts:
                continue
            key = parts[0]
            text = parts[1].replace(" ", "") if len(parts) > 1 else ""
            hyps[key] = text
    return hyps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--test_data", required=True)
    p.add_argument("--dict", required=True)
    p.add_argument("--cmvn", default=None, help="兼容参数，recognize 不需要")
    p.add_argument("--mode", default="attention_rescoring")
    p.add_argument("--beam_size", type=int, default=10)
    p.add_argument("--ctc_weight", type=float, default=0.3)
    p.add_argument("--reverse_weight", type=float, default=0.3)
    args = p.parse_args()

    tmp_dir = tempfile.mkdtemp()
    result_file = os.path.join(tmp_dir, args.mode, "text")
    run_wenet_recognize(args, result_file)

    refs = load_refs(args.test_data)
    hyps = load_hyps(result_file)

    total_err, total_ref = 0, 0
    print("\n================ 识别结果 ================")
    for key, ref in refs.items():
        hyp = hyps.get(key, "")
        err = edit_distance(list(ref), list(hyp))
        total_err += err
        total_ref += len(ref)
        flag = "✅" if hyp == ref else "❌"
        print(f"{flag} [{key}] 参考: {ref}  | 识别: {hyp}")

    cer = 100.0 * total_err / max(1, total_ref)
    print("==========================================")
    print(f"模式: {args.mode}   总字数: {total_ref}   错误: {total_err}")
    print(f"CER (字错误率) = {cer:.2f}%")


if __name__ == "__main__":
    main()
