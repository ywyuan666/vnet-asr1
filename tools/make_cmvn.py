# -*- coding: utf-8 -*-
"""
make_cmvn.py
============
计算全局 CMVN（Cepstral Mean and Variance Normalization）统计量。
作用：把 Fbank 特征做「零均值、单位方差」归一化，让训练更稳更快。

输出文件 global_cmvn 是一个 JSON，WeNet 训练时通过 --cmvn 读入：
  {"mean_stat": [...80个...], "var_stat": [...80个...], "frame_num": N}

用法：
  python tools/make_cmvn.py --data_list data/train/data.list --out data/train/global_cmvn
"""
import argparse
import json
import sys

try:  # Windows 终端正常显示中文
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from tqdm import tqdm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_list", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--num_mel_bins", type=int, default=80)
    args = p.parse_args()

    mean_stat = torch.zeros(args.num_mel_bins)
    var_stat = torch.zeros(args.num_mel_bins)
    frame_num = 0

    with open(args.data_list, encoding="utf-8") as f:
        lines = [json.loads(x) for x in f if x.strip()]

    for obj in tqdm(lines, desc="计算CMVN统计量"):
        waveform, sr = torchaudio.load(obj["wav"])
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
        waveform = waveform * (1 << 15)  # 归一化到 16bit PCM 量级
        feat = kaldi.fbank(
            waveform,
            num_mel_bins=args.num_mel_bins,
            frame_length=25,
            frame_shift=10,
            dither=0.0,
            energy_floor=0.0,
            sample_frequency=16000,
        )  # [T, 80]
        mean_stat += feat.sum(0)
        var_stat += (feat * feat).sum(0)
        frame_num += feat.shape[0]

    cmvn = {
        "mean_stat": mean_stat.tolist(),
        "var_stat": var_stat.tolist(),
        "frame_num": frame_num,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(cmvn, f)
    print(f"CMVN 完成: {frame_num} 帧 -> {args.out}")


if __name__ == "__main__":
    main()
