# -*- coding: utf-8 -*-
"""
prepare_data.py
===============
把 generate_anker_corpus.py 生成的音频 + 标注，转换成 WeNet 训练需要的格式：

  data/train/data.list   data/dev/data.list   data/test/data.list
      每行一个 JSON: {"key": "utt_xxx", "wav": "/abs/path.wav", "txt": "下一首"}

  data/dict/units.txt
      建模单元(汉字)字典，格式: 每行 `token id`
      固定包含: <blank>=0, <unk>=1, ...汉字..., <sos/eos>=最后一个

用法：
  python local/prepare_data.py --audio_dir data/audio --out_dir data
"""
import argparse
import json
import os
import random
import sys

try:  # Windows 终端正常显示中文
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def read_metadata(audio_dir):
    meta = os.path.join(audio_dir, "metadata.tsv")
    if not os.path.exists(meta):
        raise FileNotFoundError(
            f"未找到 {meta}，请先运行 local/generate_anker_corpus.py 生成数据")
    items = []
    with open(meta, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            wav, txt = line.split("\t")
            items.append((wav, txt))
    return items


def split_data(items, seed=2024):
    """按 8:1:1 划分 train/dev/test，保证每条指令都覆盖到。"""
    random.Random(seed).shuffle(items)
    n = len(items)
    n_dev = max(1, int(n * 0.1))
    n_test = max(1, int(n * 0.1))
    dev = items[:n_dev]
    test = items[n_dev:n_dev + n_test]
    train = items[n_dev + n_test:]
    return {"train": train, "dev": dev, "test": test}


def write_data_list(items, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i, (wav, txt) in enumerate(items):
            key = os.path.splitext(os.path.basename(wav))[0]
            obj = {"key": key, "wav": os.path.abspath(wav), "txt": txt}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def build_dict(items, path):
    """汉字级建模单元字典。"""
    chars = set()
    for _, txt in items:
        for ch in txt.strip():
            if ch.strip():
                chars.add(ch)
    chars = sorted(chars)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        idx = 0
        f.write(f"<blank> {idx}\n"); idx += 1     # CTC 空白符，必须 id=0
        f.write(f"<unk> {idx}\n");   idx += 1     # 未知字
        for ch in chars:
            f.write(f"{ch} {idx}\n"); idx += 1
        f.write(f"<sos/eos> {idx}\n")             # 句子起止符
    return len(chars) + 3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audio_dir", default="data/audio")
    p.add_argument("--out_dir", default="data")
    args = p.parse_args()

    items = read_metadata(args.audio_dir)
    print(f"读取到 {len(items)} 条音频")

    splits = split_data(items)
    for name, subset in splits.items():
        out = os.path.join(args.out_dir, name, "data.list")
        write_data_list(subset, out)
        print(f"  {name:5s}: {len(subset):4d} 条 -> {out}")

    dict_path = os.path.join(args.out_dir, "dict", "units.txt")
    vocab = build_dict(items, dict_path)
    print(f"字典大小(含特殊符): {vocab}  -> {dict_path}")
    print("数据准备完成！下一步: 计算 CMVN，然后训练。")


if __name__ == "__main__":
    main()
