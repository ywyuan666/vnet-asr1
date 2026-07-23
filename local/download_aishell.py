# -*- coding: utf-8 -*-
"""
AISHELL-1 数据集下载与准备
==========================
自动下载、解压、创建 WeNet 风格 data.list。

使用方式：
    python local/download_aishell.py --out_dir data/aishell

下载源：
    - OpenSLR: https://www.openslr.org/33/
    - 国内镜像: https://openslr.magicdatatech.com/33/
"""

import argparse
import json
import os
import re
import sys
import tarfile
import urllib.request


# 防止中文乱码
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


AISHELL_URLS = {
    "openslr": "https://www.openslr.org/resources/33/data_aishell.tgz",
    "magicdata": "https://openslr.magicdatatech.com/resources/33/data_aishell.tgz",
}


def download_file(url, dest):
    """下载文件并显示进度。"""
    print("下载中: %s" % url)

    def report(block_num, block_size, total_size):
        downloaded = block_num * block_size / 1024 / 1024
        if total_size > 0:
            total = total_size / 1024 / 1024
            print(
                "\r  进度: %.1f/%.1f MB (%.1f%%)" % (downloaded, total, downloaded / total * 100),
                end="",
            )

    urllib.request.urlretrieve(url, dest, reporthook=report)
    print()


def extract_tgz(tgz_path, out_dir):
    """解压 .tgz 文件。"""
    print("解压中: %s" % tgz_path)
    extract_dir = os.path.join(out_dir, "raw")
    os.makedirs(extract_dir, exist_ok=True)
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)
    print("  解压到: %s" % extract_dir)
    return extract_dir


def parse_transcript(trans_path):
    """
    解析 AISHELL-1 的 transcript.
    格式: BAC009S0001W0122 而 今 天 的 天 气 真 好 啊
    Returns: {utt_id: text_without_spaces}
    """
    transcripts = {}
    with open(trans_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            utt_id = parts[0]
            text = "".join(parts[1:])
            transcripts[utt_id] = text
    return transcripts


def scan_wavs(data_dir):
    """扫描所有 wav 文件，返回 {utt_id: wav_path}。"""
    wavs = {}
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith(".wav"):
                utt_id = f.replace(".wav", "")
                wavs[utt_id] = os.path.abspath(os.path.join(root, f))
    return wavs


def create_data_list(transcripts, wavs, utt_ids, out_path, subset_name):
    """创建 WeNet 风格 data.list。"""
    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for utt_id in utt_ids:
            if utt_id not in wavs:
                print("  [警告] 未找到 %s 的 wav 文件" % utt_id)
                continue
            if utt_id not in transcripts:
                print("  [警告] 未找到 %s 的转录文本" % utt_id)
                continue
            record = {"key": utt_id, "wav": wavs[utt_id], "txt": transcripts[utt_id]}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    print("  %s: 写入 %d 条到 %s" % (subset_name, count, out_path))
    return count


def build_dict(data_lists, dict_path):
    """从 data.list 构建字符级字典。"""
    chars = set()
    for dl_path in data_lists:
        with open(dl_path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line.strip())
                for ch in record["txt"]:
                    chars.add(ch)
    sorted_chars = sorted(chars)
    with open(dict_path, "w", encoding="utf-8") as f:
        f.write("<blank>\n")
        f.write("<unk>\n")
        for ch in sorted_chars:
            f.write(ch + "\n")
        f.write("<sos/eos>\n")
    total = len(sorted_chars) + 3
    print("字典已保存到 %s: %d 个 token" % (dict_path, total))


def main():
    parser = argparse.ArgumentParser(description="AISHELL-1 数据集下载与准备")
    parser.add_argument("--out_dir", default="data/aishell", help="输出目录")
    parser.add_argument("--mirror", default="openslr",
                        choices=["openslr", "magicdata"], help="下载镜像")
    parser.add_argument("--skip_download", action="store_true",
                        help="跳过下载")
    parser.add_argument("--skip_extract", action="store_true",
                        help="跳过解压")
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # 下载
    tgz_path = os.path.join(out_dir, "data_aishell.tgz")
    if not args.skip_download and not os.path.exists(tgz_path):
        download_file(AISHELL_URLS[args.mirror], tgz_path)
    else:
        print("跳过下载，使用已存在的: %s" % tgz_path)

    # 解压
    raw_dir = os.path.join(out_dir, "raw")
    if not args.skip_extract and not os.path.exists(raw_dir):
        extract_tgz(tgz_path, out_dir)
    else:
        print("跳过解压，使用已存在的: %s" % raw_dir)

    # 解析转录
    transcript_path = os.path.join(
        raw_dir, "data_aishell", "transcript", "aishell_transcript_v0.8.txt"
    )
    print("解析转录文件: %s" % transcript_path)
    transcripts = parse_transcript(transcript_path)
    print("  共 %d 条转录" % len(transcripts))

    # 扫描音频
    wav_dir = os.path.join(raw_dir, "data_aishell", "wav")
    print("扫描音频文件: %s" % wav_dir)
    all_wavs = scan_wavs(wav_dir)
    print("  共 %d 个 wav 文件" % len(all_wavs))

    # 划分数据集 (按 speaker: S0001-S0040 train, S0041-S0047 dev, S0048-S0054 test)
    train_speakers = set(range(1, 41))
    dev_speakers = set(range(41, 48))
    test_speakers = set(range(48, 55))

    train_ids, dev_ids, test_ids = [], [], []
    for utt_id in transcripts:
        match = re.search(r'S(\d{4})', utt_id)
        if match is None:
            continue
        spk_num = int(match.group(1))
        if spk_num in train_speakers:
            train_ids.append(utt_id)
        elif spk_num in dev_speakers:
            dev_ids.append(utt_id)
        elif spk_num in test_speakers:
            test_ids.append(utt_id)

    print("\n数据集划分:")
    print("  Train: %d utterances" % len(train_ids))
    print("  Dev:   %d utterances" % len(dev_ids))
    print("  Test:  %d utterances" % len(test_ids))

    # 创建 data.list
    base = os.path.join(out_dir)
    os.makedirs(base, exist_ok=True)

    train_list = os.path.join(base, "train", "data.list")
    dev_list = os.path.join(base, "dev", "data.list")
    test_list = os.path.join(base, "test", "data.list")

    os.makedirs(os.path.dirname(train_list), exist_ok=True)
    os.makedirs(os.path.dirname(dev_list), exist_ok=True)
    os.makedirs(os.path.dirname(test_list), exist_ok=True)

    create_data_list(transcripts, all_wavs, train_ids, train_list, "Train")
    create_data_list(transcripts, all_wavs, dev_ids, dev_list, "Dev")
    create_data_list(transcripts, all_wavs, test_ids, test_list, "Test")

    # 构建字典
    dict_path = os.path.join(base, "units.txt")
    build_dict([train_list, dev_list, test_list], dict_path)

    print("\nAISHELL-1 准备完成!")
    print("   数据目录: %s" % os.path.abspath(base))
    print("   CMVN: python tools/make_cmvn.py --data_list %s --out %s/global_cmvn" % (train_list, base))
    print("   训练: python train.py --train_data %s --cv_data %s --dict %s ..." % (train_list, dev_list, dict_path))


if __name__ == "__main__":
    main()
