# -*- coding: utf-8 -*-
"""
generate_corpus.py
========================
用微软 edge-tts（在线神经网络语音合成）自动生成「耳机语音指令」数据集。

为什么这么做？
  小白通常没有现成的语音数据集，也没条件大量录音。
  这里用多个「不同发音人」把每条指令念出来，再配合 WeNet 训练时的
  速度扰动 / 频谱掩蔽数据增强，就能得到一个可训练的小型中文语音识别数据集。

输出：
  <out>/<command_id>_<voice>.wav   16kHz、单声道、PCM wav
  <out>/metadata.tsv               每行: wav路径 <Tab> 文字标注

依赖：edge-tts、ffmpeg（用于把 mp3 转 16k 单声道 wav）

用法：
  python local/generate_corpus.py --out data/audio --repeat 1
  --repeat 调大可生成更多（不同语速）样本
"""
import argparse
import asyncio
import os
import shutil
import subprocess
import sys

try:  # Windows 终端正常显示中文
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

try:
    import edge_tts
except ImportError:
    print("缺少 edge-tts，请先执行: pip install edge-tts")
    sys.exit(1)


# ============ 1. 耳机语音指令清单（可自由增删）============
# key 是英文 id（做文件名用），value 是中文标注（模型要学会识别的目标文字）
COMMANDS = {
    "next":        "下一首",
    "prev":        "上一首",
    "play":        "播放音乐",
    "pause":       "暂停播放",
    "vol_up":      "调高音量",
    "vol_down":    "调低音量",
    "anc_on":      "打开降噪",
    "anc_off":     "关闭降噪",
    "transparency":"打开通透模式",
    "answer":      "接听电话",
    "hangup":      "挂断电话",
    "battery":     "查询电量",
    "pair":        "进入配对模式",
    "assistant":   "你好助手",
}

# ============ 2. 发音人列表（不同人声=数据多样性）============
# edge-tts 中文神经语音；不同发音人音色不同，相当于不同说话人。
VOICES = [
    "zh-CN-XiaoxiaoNeural",   # 女声
    "zh-CN-YunxiNeural",      # 男声
    "zh-CN-YunyangNeural",    # 男声(新闻)
    "zh-CN-XiaoyiNeural",     # 女声
    "zh-CN-YunjianNeural",    # 男声(体育)
    "zh-CN-liaoning-XiaobeiNeural",  # 东北女声(口音多样)
]

# 不同语速/音调，进一步扩充数据
RATES = ["+0%", "-10%", "+12%"]


def ensure_ffmpeg():
    if shutil.which("ffmpeg") is None:
        print("=" * 60)
        print("未检测到 ffmpeg！请先安装（见 README 2.2）：")
        print("  Windows : winget install --id=Gyan.FFmpeg -e")
        print("  Linux   : sudo apt install ffmpeg")
        print("  Mac     : brew install ffmpeg")
        print("=" * 60)
        sys.exit(1)


def mp3_to_wav16k(mp3_path, wav_path):
    """用 ffmpeg 把 mp3 转成 16kHz 单声道 16bit PCM wav（WeNet 标准输入）。"""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", mp3_path,
        "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
        wav_path,
    ]
    subprocess.run(cmd, check=True)


async def synth_one(text, voice, rate, mp3_path):
    communicate = edge_tts.Communicate(text, voice=voice, rate=rate)
    await communicate.save(mp3_path)


async def main_async(args):
    ensure_ffmpeg()
    os.makedirs(args.out, exist_ok=True)
    tmp_dir = os.path.join(args.out, "_tmp_mp3")
    os.makedirs(tmp_dir, exist_ok=True)

    meta_lines = []
    total = 0
    for r in range(args.repeat):
        for cmd_id, text in COMMANDS.items():
            for voice in VOICES:
                rate = RATES[(r + total) % len(RATES)]  # 轮换语速
                short_voice = voice.split("-")[-1].replace("Neural", "")
                tag = f"{cmd_id}_{short_voice}_r{r}"
                mp3_path = os.path.join(tmp_dir, tag + ".mp3")
                wav_path = os.path.join(args.out, tag + ".wav")
                try:
                    await synth_one(text, voice, rate, mp3_path)
                    mp3_to_wav16k(mp3_path, wav_path)
                    meta_lines.append(f"{os.path.abspath(wav_path)}\t{text}")
                    total += 1
                    print(f"[{total}] {tag}  ->  {text}")
                except Exception as e:
                    print(f"  跳过 {tag}: {e}")

    # 写 metadata
    meta_path = os.path.join(args.out, "metadata.tsv")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write("\n".join(meta_lines) + "\n")

    # 清理临时 mp3
    shutil.rmtree(tmp_dir, ignore_errors=True)

    print("=" * 60)
    print(f"完成！共生成 {total} 条音频")
    print(f"音频目录 : {os.path.abspath(args.out)}")
    print(f"标注文件 : {os.path.abspath(meta_path)}")
    print("下一步   : python local/prepare_data.py --audio_dir %s --out_dir data" % args.out)
    print("=" * 60)


def main():
    p = argparse.ArgumentParser(description="生成耳机语音指令数据集")
    p.add_argument("--out", default="data/audio", help="输出音频目录")
    p.add_argument("--repeat", type=int, default=1,
                   help="重复轮数（每轮换语速），调大=更多数据")
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
