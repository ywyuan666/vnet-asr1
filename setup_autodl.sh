#!/usr/bin/env bash
# ==============================================================================
#  setup_autodl.sh —— AutoDL (RTX 5090 / Blackwell) 一键环境配置
# ------------------------------------------------------------------------------
#  作用：把「配环境」的所有步骤打包成一条命令，配好即可直接训练。
#  用法（在项目根目录 /root/autodl-tmp/wenet 下执行）：
#       bash setup_autodl.sh
#  配完后训练：
#       config=conf/train_u2pp_conformer_gpu.yaml repeat=5 num_workers=8 bash run.sh
#
#  关键点（针对 RTX 5090）：
#    - 5090 是 Blackwell 架构(算力 sm_120)，必须 CUDA 12.8 + 新版 PyTorch，
#      否则报 "no kernel image is available"。
#    - 本脚本会「先装 GPU 版 torch → 装其余依赖 → 装 WeNet 源码 →
#      最后再确认一次 GPU torch」，防止依赖解析把 torch 换成 CPU 版。
# ==============================================================================
set -u  # 用未定义变量报错；不加 -e，便于逐步容错

# ---- 可配置项（一般不用改）----
CUDA_WHL="${CUDA_WHL:-cu128}"                       # 5090 用 cu128
WENET_SRC="${WENET_SRC:-/root/autodl-tmp/wenet_src}" # WeNet 源码克隆位置

# ---- 彩色输出小工具 ----
info()  { echo -e "\n\033[1;34m[步骤]\033[0m $*"; }
ok()    { echo -e "\033[1;32m[成功]\033[0m $*"; }
warn()  { echo -e "\033[1;33m[注意]\033[0m $*"; }
fail()  { echo -e "\033[1;31m[失败]\033[0m $*"; }

echo "=============================================="
echo "   AutoDL RTX 5090 环境配置 (CUDA ${CUDA_WHL})"
echo "=============================================="

# ---------- 1. 学术加速 ----------
info "开启学术加速（访问 GitHub / 微软TTS 更快）"
if [ -f /etc/network_turbo ]; then
    source /etc/network_turbo && ok "学术加速已开启"
else
    warn "未找到 /etc/network_turbo（非 AutoDL 环境？跳过，不影响后续）"
fi

# ---------- 2. 检查显卡 ----------
info "检查 GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    ok "检测到 GPU"
else
    fail "未检测到 nvidia-smi！请确认租的是 GPU 实例。"
    exit 1
fi

# ---------- 3. 安装 GPU 版 PyTorch（5090 关键步骤）----------
info "安装 PyTorch (${CUDA_WHL} 版，支持 5090 Blackwell)"
pip install --upgrade pip >/dev/null 2>&1
pip install torch torchaudio --index-url "https://download.pytorch.org/whl/${CUDA_WHL}"
if [ $? -eq 0 ]; then ok "PyTorch 安装完成"; else fail "PyTorch 安装失败，检查网络/CUDA_WHL"; exit 1; fi

# ---------- 4. 安装项目其余依赖（不含 torch，避免覆盖 GPU 版）----------
info "安装项目依赖（跳过 torch，防止覆盖 GPU 版）"
pip install \
    "edge-tts>=6.1.0" \
    "soundfile>=0.12.1" "numpy>=1.23" "librosa>=0.10.0" \
    "sounddevice>=0.4.6" \
    "fastapi>=0.110.0" "uvicorn[standard]>=0.27.0" "python-multipart>=0.0.9" \
    "tqdm>=4.65" "pyyaml>=6.0"
ok "项目依赖安装完成"

# ---------- 5. 安装 WeNet 训练代码（源码方式，含 wenet.bin.train）----------
info "安装 WeNet 训练代码"
if [ ! -d "${WENET_SRC}/.git" ]; then
    git clone https://github.com/wenet-e2e/wenet.git "${WENET_SRC}"
else
    warn "WeNet 源码已存在，跳过克隆：${WENET_SRC}"
fi
# --no-deps：只注册 wenet 包（拿到训练入口），不让它重装 torch 等依赖
pip install -e "${WENET_SRC}" --no-deps
ok "WeNet 训练代码安装完成"

# ---------- 6. 重新确认 GPU 版 torch（防止被上一步降级）----------
info "复核并锁定 GPU 版 PyTorch"
pip install torch torchaudio --index-url "https://download.pytorch.org/whl/${CUDA_WHL}" --upgrade >/dev/null 2>&1
ok "已复核 GPU 版 PyTorch"

# ---------- 7. 安装 ffmpeg（TTS 音频转码用）----------
info "安装 ffmpeg"
if command -v ffmpeg >/dev/null 2>&1; then
    ok "ffmpeg 已存在，跳过"
else
    apt-get update -qq && apt-get install -y -qq ffmpeg && ok "ffmpeg 安装完成" \
        || warn "ffmpeg 安装失败，请手动执行 apt-get install -y ffmpeg"
fi

# ---------- 8. 最终自检 ----------
info "最终自检"
echo "--------------------------------------------------"
python - <<'PYEOF'
import sys
ok = True
try:
    import torch
    cuda = torch.cuda.is_available()
    name = torch.cuda.get_device_name(0) if cuda else "无"
    print(f"  PyTorch 版本 : {torch.__version__}")
    print(f"  CUDA 可用    : {cuda}")
    print(f"  显卡         : {name}")
    if not cuda:
        ok = False
        print("  [警告] CUDA 不可用！torch 可能被装成了 CPU 版。")
    # 实测能否在 GPU 上分配张量（验证 5090 kernel 支持）
    if cuda:
        x = torch.randn(8, 8, device="cuda")
        y = (x @ x).sum().item()
        print(f"  GPU 计算测试 : 通过 (sum={y:.2f})")
except Exception as e:
    ok = False
    print(f"  [错误] torch 自检失败: {e}")

# 检查 WeNet 训练入口
import importlib.util
spec = importlib.util.find_spec("wenet")
print(f"  WeNet 模块   : {'已安装' if spec else '未找到'}")

sys.exit(0 if ok else 1)
PYEOF
SELFCHECK=$?
echo "--------------------------------------------------"

if [ $SELFCHECK -eq 0 ]; then
    echo ""
    ok "环境配置完成！可以开始训练了 🎉"
    echo ""
    echo "  下一步（GPU 训练）："
    echo "    config=conf/train_u2pp_conformer_gpu.yaml repeat=5 num_workers=8 bash run.sh"
    echo ""
    echo "  另开终端监控 GPU："
    echo "    watch -n 1 nvidia-smi"
else
    echo ""
    fail "自检未通过，请查看上面的警告。常见原因："
    echo "    - CUDA 不可用：确认镜像是 CUDA 12.8；或手动重装 torch cu128"
    echo "    - 'no kernel image is available'：torch 版本太旧不支持 5090"
    exit 1
fi
