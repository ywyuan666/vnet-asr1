# syntax=docker/dockerfile:1
# ==============================================================================
#  Dockerfile —— Conformer + CTC/Attention/Transducer 语音识别 Demo 镜像
# ------------------------------------------------------------------------------
#  设计目标：
#    1) 一条命令构建出「带 Web 界面 + 真实 PyTorch 推理」的可运行镜像
#    2) 分层清晰、缓存友好（依赖层与代码层分开）
#    3) 无 ffmpeg 依赖，纯 Python 音频处理
#  说明：模型文件（exp/、data/）不打进镜像，通过 volume 挂载。
# ==============================================================================

FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# ---- 依赖层：只拷贝依赖清单再安装 ----
COPY requirements_ctc_attn_transducer.txt ./
RUN pip install --upgrade pip && \
    pip install -r requirements_ctc_attn_transducer.txt && \
    pip install "fastapi>=0.110.0" "uvicorn[standard]>=0.27.0" "python-multipart>=0.0.9"

# ---- 代码层：拷贝项目代码 ----
COPY . .

# ---- 运行时配置 ----
ENV MODEL_DIR=/app/exp/conformer_ctc_attn_transducer \
    DICT_PATH=/app/data/dict/units.txt \
    CMVN_PATH=/app/data/train/global_cmvn \
    CHECKPOINT=/app/exp/conformer_ctc_attn_transducer/best.pt \
    DECODE_MODE=attention

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/api/health').status==200 else 1)"

CMD ["uvicorn", "server.app_ctc_attn_transducer:app", "--host", "0.0.0.0", "--port", "8000"]
