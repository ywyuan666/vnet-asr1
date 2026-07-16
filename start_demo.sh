#!/usr/bin/env bash
# ==============================================================================
#  start_demo.sh —— 一键启动 Web Demo
# ------------------------------------------------------------------------------
#  用法：
#    ./start_demo.sh docker    # 用 Docker 构建并启动（推荐）
#    ./start_demo.sh local     # 直接用本机 Python 启动（快速调试）
#
#  前置条件：已有训练好的模型（exp/conformer_ctc_attn_transducer/best.pt 等）。
#  若还没有模型，先执行：  bash run_ctc_attn_transducer.sh
# ==============================================================================
set -e

MODE="${1:-docker}"
MODEL="exp/conformer_ctc_attn_transducer/best.pt"
DICT="data/dict/units.txt"
CMVN="data/train/global_cmvn"

# ---- 模型自检 ----
if [ ! -f "$MODEL" ] || [ ! -f "$DICT" ] || [ ! -f "$CMVN" ]; then
    echo "⚠️  推理文件不完整，需要以下文件都存在："
    echo "    $MODEL"
    echo "    $DICT"
    echo "    $CMVN"
    echo "    请先运行  bash run_ctc_attn_transducer.sh  完成训练。"
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
    uvicorn server.app_ctc_attn_transducer:app --host 0.0.0.0 --port 8000
    ;;
  *)
    echo "用法: ./start_demo.sh [docker|local]"
    exit 1
    ;;
esac
