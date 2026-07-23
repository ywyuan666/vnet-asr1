# -*- coding: utf-8 -*-
"""
server/app.py
=============
耳机语音指令识别 —— Web Demo 后端（FastAPI）。

提供两个接口：
  GET  /            返回前端页面（上传音频 / 麦克风录音 → 显示识别结果）
  POST /api/recognize   接收上传的音频文件，返回识别文字（JSON）
  GET  /api/health      健康检查，返回模型是否就绪

设计要点：
  - 真实推理：底层调用 WeNet 的 wenet.bin.recognize（跨版本最稳定的解码入口）。
  - 浏览器录音多为 webm/ogg/mp3，统一用 ffmpeg 转成 16k 单声道 wav 再喂给模型。
  - 通过环境变量配置模型路径，方便在 Docker 容器里挂载不同的模型。
"""
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ---------------- 配置（可用环境变量覆盖，便于容器化） ----------------
ROOT = Path(__file__).resolve().parent.parent          # 项目根目录
MODEL_DIR = Path(os.getenv("MODEL_DIR", ROOT / "exp" / "u2pp_conformer"))
DICT_PATH = Path(os.getenv("DICT_PATH", ROOT / "data" / "dict" / "units.txt"))
CMVN_PATH = Path(os.getenv("CMVN_PATH", ROOT / "data" / "train" / "global_cmvn"))
CONFIG_PATH = Path(os.getenv("CONFIG_PATH", MODEL_DIR / "train.yaml"))
CHECKPOINT = Path(os.getenv("CHECKPOINT", MODEL_DIR / "final.pt"))
DECODE_MODE = os.getenv("DECODE_MODE", "attention_rescoring")
FRONTEND_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="耳机语音指令识别 Demo", version="1.0")


def model_ready() -> bool:
    """检查真实模型文件是否齐全。"""
    return CHECKPOINT.exists() and CONFIG_PATH.exists() and DICT_PATH.exists() and CMVN_PATH.exists()


def _pick_checkpoint() -> Path:
    """优先用 final.pt；没有就取序号最大的一个 *.pt。"""
    if CHECKPOINT.exists():
        return CHECKPOINT
    pts = sorted(MODEL_DIR.glob("*.pt"))
    if pts:
        return pts[-1]
    raise FileNotFoundError(f"未找到模型 checkpoint：{MODEL_DIR}")


def convert_to_wav16k(src_path: str, dst_path: str) -> None:
    """用 ffmpeg 把任意音频转成 16kHz 单声道 16bit PCM wav（WeNet 标准输入）。"""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", src_path,
        "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
        dst_path,
    ]
    subprocess.run(cmd, check=True)


def recognize_wav(wav_path: str) -> str:
    """调用 WeNet 解码模块识别单条 wav，返回识别文字。"""
    tmp_dir = tempfile.mkdtemp()
    list_path = os.path.join(tmp_dir, "data.list")
    with open(list_path, "w", encoding="utf-8") as f:
        obj = {"key": "demo", "wav": os.path.abspath(wav_path), "txt": ""}
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    result_file = os.path.join(tmp_dir, DECODE_MODE, "text")
    cmd = [
        sys.executable, "-m", "wenet.bin.recognize",
        "--device", "cpu",
        "--gpu", "-1",                       # CPU 解码
        "--modes", DECODE_MODE,
        "--config", str(CONFIG_PATH),
        "--test_data", list_path,
        "--checkpoint", str(_pick_checkpoint()),
        "--beam_size", "10",
        "--batch_size", "1",
        "--ctc_weight", "0.3",
        "--reverse_weight", "0.3",
        "--result_dir", tmp_dir,
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


@app.get("/api/health")
def health():
    """健康检查：容器编排 / 技术演示前自检用。"""
    return {
        "status": "ok",
        "model_ready": model_ready(),
        "decode_mode": DECODE_MODE,
        "checkpoint": str(CHECKPOINT),
        "cmvn": str(CMVN_PATH),
    }


@app.post("/api/recognize")
async def api_recognize(audio: UploadFile = File(...)):
    """接收上传音频，返回识别结果。"""
    if not model_ready():
        raise HTTPException(
            status_code=503,
            detail="模型尚未就绪。请先训练模型（bash run.sh）或挂载已训练好的 exp/ 与 data/dict。",
        )

    # 保存上传的原始音频
    suffix = os.path.splitext(audio.filename or "")[1] or ".bin"
    tmp_dir = tempfile.mkdtemp()
    raw_path = os.path.join(tmp_dir, "input" + suffix)
    with open(raw_path, "wb") as f:
        f.write(await audio.read())

    # 统一转成 16k 单声道 wav
    wav_path = os.path.join(tmp_dir, "input_16k.wav")
    try:
        convert_to_wav16k(raw_path, wav_path)
    except subprocess.CalledProcessError:
        raise HTTPException(status_code=400, detail="音频格式无法解析（ffmpeg 转换失败）。")

    # 真实推理并计时
    t0 = time.time()
    try:
        text = recognize_wav(wav_path)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"解码失败：{e}")
    elapsed = round(time.time() - t0, 2)

    return JSONResponse({
        "text": text or "(未识别到有效指令)",
        "elapsed_sec": elapsed,
        "decode_mode": DECODE_MODE,
    })


# 前端静态资源（放在最后挂载，避免覆盖 /api 路由）
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
