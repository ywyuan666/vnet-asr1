# -*- coding: utf-8 -*-
"""
infer_demo.py
=============
演示用：识别「单条 wav 文件」或「麦克风实时录音」。

底层同样复用 wenet.bin.recognize（最稳定），先把输入封装成一条 data.list，
解码后打印识别文字。

用法：
  # 识别一个 wav 文件
  python infer_demo.py --config exp/u2pp_conformer/train.yaml \
      --checkpoint exp/u2pp_conformer/final.pt --dict data/dict/units.txt --wav test.wav

  # 用麦克风录 3 秒并识别
  python infer_demo.py --config exp/u2pp_conformer/train.yaml \
      --checkpoint exp/u2pp_conformer/final.pt --dict data/dict/units.txt --mic 3
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


def record_from_mic(seconds, out_wav):
    import sounddevice as sd
    import soundfile as sf
    fs = 16000
    print(f"🎤 开始录音 {seconds} 秒，请说出语音指令（如「打开降噪」）...")
    audio = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype="int16")
    sd.wait()
    sf.write(out_wav, audio, fs, subtype="PCM_16")
    print(f"录音保存: {out_wav}")


def recognize_wav(args, wav_path):
    tmp_dir = tempfile.mkdtemp()
    # 构造单条 data.list
    list_path = os.path.join(tmp_dir, "data.list")
    with open(list_path, "w", encoding="utf-8") as f:
        obj = {"key": "demo", "wav": os.path.abspath(wav_path), "txt": ""}
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    result_file = os.path.join(tmp_dir, "text")
    cmd = [
        sys.executable, "-m", "wenet.bin.recognize",
        "--gpu", "-1",
        "--mode", args.mode,
        "--config", args.config,
        "--test_data", list_path,
        "--checkpoint", args.checkpoint,
        "--beam_size", "10",
        "--batch_size", "1",
        "--penalty", "0.0",
        "--dict", args.dict,
        "--ctc_weight", "0.3",
        "--reverse_weight", "0.3",
        "--result_file", result_file,
        "--data_type", "raw",
    ]
    subprocess.run(cmd, check=True)

    text = ""
    with open(result_file, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if parts and parts[0] == "demo" and len(parts) > 1:
                text = parts[1].replace(" ", "")
    return text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dict", required=True)
    p.add_argument("--cmvn", default=None, help="兼容参数")
    p.add_argument("--wav", default=None, help="要识别的 wav 文件")
    p.add_argument("--mic", type=float, default=None, help="麦克风录音秒数")
    p.add_argument("--mode", default="attention_rescoring")
    args = p.parse_args()

    if args.mic:
        tmp_wav = os.path.join(tempfile.mkdtemp(), "mic.wav")
        record_from_mic(args.mic, tmp_wav)
        wav_path = tmp_wav
    elif args.wav:
        wav_path = args.wav
    else:
        print("请指定 --wav <文件> 或 --mic <秒数>")
        sys.exit(1)

    text = recognize_wav(args, wav_path)
    print("\n========================================")
    print(f"🎧 识别结果: 【{text}】")
    print("========================================")


if __name__ == "__main__":
    main()
