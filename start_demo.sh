#!/usr/bin/env bash
# ==============================================================================
#  start_demo.sh —— 一键启动 Web Demo（面试演示用）
# ------------------------------------------------------------------------------
#  两种运行方式：
#    ./start_demo.sh docker    # 用 Docker 构建并启动（推荐，学 Docker）
#    ./start_demo.sh local     # 直接用本机 Python 启动（快速调试）
#
#  前置条件：已有训练好的模型（exp/u2pp_conformer/final.pt 等）。
#  若还没有模型，先执行：  bash run.sh    （用 TTS 合成数据并训练）
# ==============================================================================
set -e

MODE="${1:-docker}"
MODEL="exp/u2pp_conformer/final.pt"
CONFIG="exp/u2pp_conformer/train.yaml"
DICT="data/dict/units.txt"
CMVN="data/train/global_cmvn"

# ---- 模型自检 ----
if [ ! -f "$MODEL" ] || [ ! -f "$CONFIG" ] || [ ! -f "$DICT" ] || [ ! -f "$CMVN" ]; then
    echo "⚠️  推理文件不完整，需要以下文件都存在："
    echo "    $MODEL"
    echo "    $CONFIG"
    echo "    $DICT"
    echo "    $CMVN"
    echo "    请先运行  bash run.sh  完成训练，或从 AutoDL 下载完整 model_bundle。"
    echo "    （CPU 训练约 1~2 小时；有 GPU 会快很多）"
    exit 1
fi

case "$MODE" in
  docker)
    echo "🐳 使用 Docker 构建并启动 Demo..."
    docker compose up -d --build
    echo "✅ 已启动！浏览器打开 http://localhost:8000"
    echo "   查看日志： docker compose logs -f"
    echo "   停止服务： docker compose down"
    ;;
  local)
    echo "🐍 使用本机 Python 启动 Demo..."
    pip install fastapi "uvicorn[standard]" python-multipart >/dev/null 2>&1 || true
    echo "✅ 启动中！浏览器打开 http://localhost:8000  （Ctrl+C 停止）"
    uvicorn server.app:app --host 0.0.0.0 --port 8000
    ;;
  *)
    echo "用法: ./start_demo.sh [docker|local]"
    exit 1
    ;;
esac
