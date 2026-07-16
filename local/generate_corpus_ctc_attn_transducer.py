# -*- coding: utf-8 -*-
"""
generate_corpus_ctc_attn_transducer.py
======================================
用微软 edge-tts 自动生成语音指令数据集（无需 ffmpeg）。
使用 miniaudio 纯 Python mp3 解码，避免 ffmpeg 依赖。

输出：
  <out>/<command_id>_<voice>.wav   16kHz、单声道、PCM wav
  <out>/metadata.tsv               每行: wav路径 <Tab> 文字标注
"""
import argparse
import asyncio
import io
import os
import struct
import sys
import wave

try:
    import edge_tts
except ImportError:
    print("缺少 edge-tts，请执行: pip install edge-tts")
    sys.exit(1)

try:
    import miniaudio
except ImportError:
    print("缺少 miniaudio，请执行: pip install miniaudio")
    sys.exit(1)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# 耳机语音指令清单
COMMANDS = {
    "next":  "下一首",
    "prev":  "上一首",
    "play":  "播放音乐",
    "pause": "暂停播放",
    "vol_up": "调高音量",
    "vol_down": "调低音量",
    "anc_on": "打开降噪",
    "anc_off": "关闭降噪",
    "answer": "接听电话",
    "hangup": "挂断电话",
    "battery": "查询电量",
    "assistant": "你好助手",
}

VOICES = [
    "zh-CN-XiaoxiaoNeural",
    "zh-CN-YunxiNeural",
    "zh-CN-YunyangNeural",
    "zh-CN-XiaoyiNeural",
    "zh-CN-YunjianNeural",
]

RATES = ["+0%", "-10%", "+12%"]


def mp3_bytes_to_wav16k(mp3_data, wav_path):
    """用 miniaudio 解码 mp3 字节到 16kHz 单声道 wav"""
    # 解码 mp3 → 原始 PCM
    decoded = miniaudio.decode(
        mp3_data,
        output_format=miniaudio.SampleFormat.SIGNED16,
        nchannels=1,
        sample_rate=16000,
    )
    # decoded.samples: bytes, decoded.sample_rate: int, decoded.nchannels: int
    sample_rate = decoded.sample_rate
    samples = decoded.samples

    # 如果解码后不是 16kHz，重采样
    if sample_rate != 16000:
        # 简单线性重采样
        import math
        ratio = 16000.0 / sample_rate
        old_len = len(samples) // 2  # 16bit = 2 bytes
        new_len = int(old_len * ratio)
        new_samples = bytearray(new_len * 2)
        for i in range(new_len):
            src_idx = min(int(i / ratio), old_len - 1)
            val = struct.unpack("<h", samples[src_idx * 2: src_idx * 2 + 2])[0]
            struct.pack_into("<h", new_samples, i * 2, val)
        samples = bytes(new_samples)
        sample_rate = 16000

    # 写 wav 文件
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(samples)


async def main_async(args):
    os.makedirs(args.out, exist_ok=True)
    meta_lines = []
    total = 0

    for r in range(args.repeat):
        for cmd_id, text in COMMANDS.items():
            for voice in VOICES:
                rate = RATES[(total + r) % len(RATES)]
                short_voice = voice.split("-")[-1].replace("Neural", "")
                tag = f"{cmd_id}_{short_voice}_r{r}"
                wav_path = os.path.join(args.out, tag + ".wav")

                try:
                    # 合成 mp3 到内存
                    communicate = edge_tts.Communicate(text, voice=voice, rate=rate)
                    mp3_data = b""
                    async for chunk in communicate.stream():
                        if chunk["type"] == "audio":
                            mp3_data += chunk["data"]

                    # 解码 mp3 → wav
                    mp3_bytes_to_wav16k(mp3_data, wav_path)
                    meta_lines.append(f"{os.path.abspath(wav_path)}\t{text}")
                    total += 1
                    print(f"[{total}] {tag}  ->  {text}")
                except Exception as e:
                    print(f"  跳过 {tag}: {e}")

    # 写 metadata
    meta_path = os.path.join(args.out, "metadata.tsv")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write("\n".join(meta_lines) + "\n")

    print("=" * 60)
    print(f"完成！共生成 {total} 条音频")
    print(f"音频目录 : {os.path.abspath(args.out)}")
    print(f"标注文件 : {os.path.abspath(meta_path)}")
    print("=" * 60)


def main():
    p = argparse.ArgumentParser(description="生成语音指令数据集（无需 ffmpeg）")
    p.add_argument("--out", default="data/audio", help="输出音频目录")
    p.add_argument("--repeat", type=int, default=1,
                   help="重复轮数（每轮换语速），调大=更多数据")
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
