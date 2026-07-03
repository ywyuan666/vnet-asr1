# syntax=docker/dockerfile:1
# ==============================================================================
#  Dockerfile —— 耳机语音指令识别 Web Demo 镜像
# ------------------------------------------------------------------------------
#  设计目标：
#    1) 一条命令即可构建出「带 Web 界面 + 真实推理」的可运行镜像
#    2) 分层清晰、缓存友好（依赖层与代码层分开，改代码不必重装依赖）
#    3) 内置 ffmpeg（浏览器录音格式转换必需）
#  说明：模型文件（exp/、data/dict/）不打进镜像，运行时通过 volume 挂载，
#        这样镜像小、模型可替换，符合「代码与数据分离」的工程实践。
# ==============================================================================

# ---- 基础镜像：官方 Python 精简版（Debian slim，体积小） ----
FROM python:3.10-slim AS base

# 避免交互式提示、让 Python 日志实时输出（容器里看日志更顺）
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# ---- 系统依赖：ffmpeg 用于音频转码 ----
# 单独成层：系统包很少变动，可长期命中缓存
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- 依赖层：先只拷贝依赖清单再安装 ----
# 关键技巧：只要 requirements.txt 不变，这一层就复用缓存，改业务代码不会重装 torch
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install fastapi "uvicorn[standard]" python-multipart

# ---- 代码层：最后才拷贝源代码（改动最频繁，放最后减少缓存失效） ----
COPY . .

# ---- 运行时配置 ----
# 模型路径用环境变量，容器启动时可覆盖，指向挂载进来的模型
ENV MODEL_DIR=/app/exp/u2pp_conformer \
    DICT_PATH=/app/data/dict/units.txt \
    CONFIG_PATH=/app/exp/u2pp_conformer/train.yaml \
    CHECKPOINT=/app/exp/u2pp_conformer/final.pt \
    DECODE_MODE=attention_rescoring

EXPOSE 8000

# 健康检查：容器编排系统据此判断服务是否存活
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/api/health').status==200 else 1)"

# ---- 启动命令 ----
# 用 uvicorn 启动 FastAPI 应用；0.0.0.0 让容器外也能访问
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
