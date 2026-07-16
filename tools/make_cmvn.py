# -*- coding: utf-8 -*-
"""
make_cmvn.py
============
计算全局 CMVN 统计量。
使用 soundfile + torchaudio.compliance.kaldi 提取 Fbank 特征，
避免对系统 ffmpeg 的依赖。

输出：JSON 格式的全局均值和方差统计量。
"""
import argparse
import json
import sys

import numpy as np
import soundfile as sf
import torch
import torchaudio.compliance.kaldi as kaldi
from tqdm import tqdm

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


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
        # 用 soundfile 读取 wav（无需 ffmpeg）
        data, sr = sf.read(obj["wav"])
        if sr != 16000:
            # 简单重采样到 16k
            import librosa
            data = librosa.resample(data, orig_sr=sr, target_sr=16000)
            sr = 16000
        waveform = torch.from_numpy(data).float().unsqueeze(0)  # [1, T]
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
