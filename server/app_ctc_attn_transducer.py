# -*- coding: utf-8 -*-
"""
server/app_ctc_attn_transducer.py
=================================
Conformer + CTC/Attention/Transducer 语音识别 —— Web Demo 后端（FastAPI）。

提供三个接口：
  GET  /                返回前端页面（上传/录音 → 显示识别结果）
  POST /api/recognize   接收上传音频，返回识别文字
  GET  /api/health      健康检查

设计要点：
  - 直接加载 PyTorch 模型推理（不依赖 WeNet CLI）
  - 使用 soundfile + librosa 处理音频，无需 ffmpeg
  - 支持三种解码模式：ctc_greedy / attention / transducer
  - 模型文件通过 volume 挂载，镜像与模型分离
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.conformer_ctc_attn_transducer import ConformerCTCATTNTransducer

# ---------------- 配置（环境变量覆盖，便于容器化） ----------------
ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(os.getenv("MODEL_DIR", ROOT / "exp" / "conformer_ctc_attn_transducer"))
DICT_PATH = Path(os.getenv("DICT_PATH", ROOT / "data" / "dict" / "units.txt"))
CMVN_PATH = Path(os.getenv("CMVN_PATH", ROOT / "data" / "train" / "global_cmvn"))
CHECKPOINT = Path(os.getenv("CHECKPOINT", MODEL_DIR / "best.pt"))
DECODE_MODE = os.getenv("DECODE_MODE", "attention")
FRONTEND_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Conformer + CTC/Attention/Transducer 语音指令识别", version="1.0")

# ---- 全局：模型懒加载 ----
_model = None
_vocab = None
_idx2token = None
_sos_id = None


def load_model():
    """加载模型（首次调用时加载，之后复用）"""
    global _model, _vocab, _idx2token, _sos_id

    if _model is not None:
        return _model, _vocab, _idx2token, _sos_id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载字典
    vocab = {}
    idx2token = {}
    with open(DICT_PATH, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                token, idx = parts[0], int(parts[1])
                vocab[token] = idx
                idx2token[idx] = token
    vocab_size = len(vocab)
    sos_id = vocab_size - 1

    # 加载 checkpoint
    ckpt = torch.load(CHECKPOINT, map_location=device)
    d_model = 144
    if "config" in ckpt:
        d_model = ckpt["config"].get("d_model", 144)

    model = ConformerCTCATTNTransducer(vocab_size=vocab_size, d_model=d_model).to(device)
    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    _model = model
    _vocab = vocab
    _idx2token = idx2token
    _sos_id = sos_id
    return model, vocab, idx2token, sos_id


def extract_fbank(wav_path):
    """提取 Fbank 特征（用 soundfile 避免 ffmpeg 依赖）"""
    import soundfile as sf
    import librosa
    data, sr = sf.read(wav_path)
    if sr != 16000:
        data = librosa.resample(data, orig_sr=sr, target_sr=16000)
        sr = 16000
    waveform = torch.from_numpy(data).float().unsqueeze(0)
    waveform = waveform * (1 << 15)

    feat = kaldi.fbank(
        waveform,
        num_mel_bins=80,
        frame_length=25,
        frame_shift=10,
        dither=0.0,
        sample_frequency=16000,
    )  # [T, 80]

    # CMVN 归一化
    if CMVN_PATH.exists():
        with open(CMVN_PATH) as f:
            cmvn = json.load(f)
        frame_num = cmvn["frame_num"]
        mean = torch.tensor(cmvn["mean_stat"], dtype=torch.float32) / frame_num
        var = torch.tensor(cmvn["var_stat"], dtype=torch.float32) / frame_num - mean * mean
        feat = (feat - mean) / (var.sqrt() + 1e-10)

    return feat.unsqueeze(0)  # [1, T, 80]


def model_ready() -> bool:
    """检查模型文件是否齐全"""
    return CHECKPOINT.exists() and DICT_PATH.exists() and CMVN_PATH.exists()


def _pick_checkpoint() -> Path:
    """优先用 best.pt；没有就取序号最大的 *.pt"""
    if CHECKPOINT.exists():
        return CHECKPOINT
    pts = sorted(MODEL_DIR.glob("*.pt"))
    if pts:
        return pts[-1]
    raise FileNotFoundError(f"未找到模型 checkpoint：{MODEL_DIR}")


@app.on_event("startup")
def startup():
    """服务启动时预加载模型"""
    if model_ready():
        try:
            load_model()
            print(f"[启动] 模型加载成功！解码模式: {DECODE_MODE}")
        except Exception as e:
            print(f"[启动] 模型加载失败: {e}")
    else:
        print(f"[启动] 模型文件未就绪（挂载路径: {CHECKPOINT}），服务启动后可通过 /api/health 检查状态")


@app.get("/api/health")
def health():
    """健康检查"""
    ready = model_ready()
    return {
        "status": "ok" if ready else "degraded",
        "model_ready": ready,
        "decode_mode": DECODE_MODE,
        "checkpoint": str(CHECKPOINT),
        "cmvn": str(CMVN_PATH),
    }


@app.post("/api/recognize")
async def api_recognize(
    audio: UploadFile = File(...),
    mode: str = DECODE_MODE,
):
    """接收上传音频，返回识别结果"""
    if not model_ready():
        raise HTTPException(
            status_code=503,
            detail="模型尚未就绪。请先训练模型（bash run_ctc_attn_transducer.sh）或挂载已训练好的 exp/ 与 data/。",
        )

    # 加载模型（如果尚未加载）
    try:
        model, vocab, idx2token, sos_id = load_model()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"模型加载失败：{e}")

    # 保存上传的音频
    suffix = os.path.splitext(audio.filename or "")[1] or ".wav"
    tmp_dir = tempfile.mkdtemp()
    raw_path = os.path.join(tmp_dir, "input" + suffix)
    with open(raw_path, "wb") as f:
        f.write(await audio.read())

    # 提取 Fbank 特征（支持各种格式，soundfile 自动解码）
    try:
        feats = extract_fbank(raw_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"音频格式无法解析：{e}")

    device = next(model.parameters()).device
    feats = feats.to(device)

    # 解码
    t0 = time.time()
    device = next(model.parameters()).device
    feats = feats.to(device)

    try:
        if mode == "ctc_greedy":
            hyps = model.recognize_ctc_greedy(feats, idx2token)
            result = hyps[0]
        elif mode == "attention":
            ys = model.recognize_attention(feats, max_len=20, sos_id=sos_id, eos_id=sos_id)
            tokens = []
            for t in range(1, ys.size(1)):
                tok = ys[0, t].item()
                if tok == sos_id:
                    break
                tokens.append(idx2token.get(tok, ""))
            result = "".join(tokens)
        elif mode == "transducer":
            results = model.recognize_transducer(feats, max_len=20, sos_id=sos_id)
            result = "".join(idx2token.get(t, "") for t in results[0])
        else:
            raise HTTPException(status_code=400, detail=f"不支持的解码模式: {mode}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"解码失败：{e}")

    elapsed = round(time.time() - t0, 3)

    return JSONResponse({
        "text": result or "(未识别到有效指令)",
        "elapsed_sec": elapsed,
        "decode_mode": mode,
    })


# 前端静态资源
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
