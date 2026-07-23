# -*- coding: utf-8 -*-
"""
KenLM N-gram 语言模型训练
========================
使用 KenLM 训练中文 n-gram 语言模型，用于 Attention 解码的 LM rescoring。

依赖:
    pip install https://github.com/kpu/kenlm/archive/master.zip
    或从 conda: conda install -c conda-forge kenlm

用法:
    python tools/train_kenlm.py --text_path data/lm_corpus.txt --arpa_path data/lm/aishell_4gram.arpa --order 4 --prune "0 1 1 1"
"""

import argparse
import json
import os
import shutil
import subprocess
import sys


def get_kenlm_bin(name):
    """Get path to a KenLM binary tool."""
    candidates = [
        os.path.join(os.path.dirname(sys.executable), name),
        os.path.join(os.path.dirname(sys.executable), "..", "bin", name),
        name,
    ]
    for c in candidates:
        if os.path.exists(c) or shutil.which(name):
            return name
    raise FileNotFoundError(
        "KenLM binary '%s' not found. Install: conda install -c conda-forge kenlm" % name
    )


def build_lm_corpus(data_lists, output_path):
    """Build plain text corpus from data.list files for LM training."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    lines = []
    for dl_path in data_lists:
        with open(dl_path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line.strip())
                text = record["txt"]
                spaced = " ".join(text)
                lines.append(spaced)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("LM corpus: %d lines -> %s" % (len(lines), output_path))
    return output_path


def train_kenlm(text_path, arpa_path, order=4, prune=None):
    """Train KenLM n-gram model."""
    os.makedirs(os.path.dirname(arpa_path), exist_ok=True)
    lmplz = get_kenlm_bin("lmplz")
    cmd = [lmplz, "-o", str(order), "--text", text_path, "--arpa", arpa_path, "--discount_fallback"]
    if prune:
        prune_values = [int(v) for v in prune.split()]
        cmd += ["--prune"] + [str(v) for v in prune_values]
    print("训练 KenLM %d-gram 模型..." % order)
    print("  命令: %s" % " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("  STDERR: %s" % result.stderr)
        raise RuntimeError("KenLM 训练失败 (exit code %d)" % result.returncode)
    print("\nARPA 模型已保存: %s" % arpa_path)
    print("   文件大小: %.1f MB" % (os.path.getsize(arpa_path) / 1024 / 1024))
    return arpa_path


def build_binary(arpa_path, binary_path):
    """Convert ARPA to KenLM binary format."""
    os.makedirs(os.path.dirname(binary_path), exist_ok=True)
    bin_cmd = get_kenlm_bin("build_binary")
    cmd = [bin_cmd, "-q", "-s", arpa_path, binary_path]
    print("转换 ARPA 为 binary 格式...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("  STDERR: %s" % result.stderr)
        raise RuntimeError("Binary 转换失败 (exit code %d)" % result.returncode)
    print("\nBinary 模型已保存: %s" % binary_path)
    print("   文件大小: %.1f MB" % (os.path.getsize(binary_path) / 1024 / 1024))
    return binary_path


class NgramLM:
    """Python wrapper for KenLM n-gram model."""

    def __init__(self, model_path):
        try:
            import kenlm
        except ImportError:
            raise ImportError(
                "KenLM Python bindings required. "
                "pip install https://github.com/kpu/kenlm/archive/master.zip"
            )
        self.model = kenlm.Model(model_path)
        self.order = self.model.order

    def score(self, sentence):
        """Score a sentence (space-separated Chinese chars)."""
        spaced = " ".join(sentence)
        return self.model.score(spaced, bos=True, eos=True)

    def score_token_list(self, tokens):
        """Score a token list directly."""
        return self.model.score(" ".join(tokens), bos=True, eos=True)

    def perplexity(self, sentence):
        """Calculate perplexity of a sentence."""
        return 10 ** (-self.score(sentence) / len(sentence))


def main():
    parser = argparse.ArgumentParser(description="KenLM N-gram 语言模型训练")
    parser.add_argument("--text_path", help="训练语料文本路径")
    parser.add_argument("--data_list", nargs="*", help="从 data.list 提取文本作为语料")
    parser.add_argument("--arpa_path", default="data/lm/aishell_lm.arpa", help="ARPA 模型输出路径")
    parser.add_argument("--binary_path", help="Binary 模型输出路径")
    parser.add_argument("--order", type=int, default=4, help="N-gram 阶数")
    parser.add_argument("--prune", default="0 1 1 1", help="剪枝阈值")
    parser.add_argument("--build_binary", action="store_true", help="构建 binary 格式")
    parser.add_argument("--output_corpus", help="生成的语料文件路径")
    args = parser.parse_args()

    if args.data_list and args.text_path is None:
        args.text_path = args.output_corpus or os.path.join(
            os.path.dirname(args.arpa_path), "lm_corpus.txt"
        )
        build_lm_corpus(args.data_list, args.text_path)

    if args.text_path is None or not os.path.exists(args.text_path):
        parser.error("请提供 --text_path 或 --data_list")

    train_kenlm(args.text_path, args.arpa_path, args.order, args.prune)

    if args.build_binary:
        binary_path = args.binary_path or args.arpa_path.replace(".arpa", ".klm")
        build_binary(args.arpa_path, binary_path)


if __name__ == "__main__":
    main()
