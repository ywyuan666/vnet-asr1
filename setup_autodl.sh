#!/usr/bin/env bash
# ==============================================================================
#  setup_autodl.sh —— AutoDL 一键环境配置（Conformer + CTC/Attention/Transducer）
# ------------------------------------------------------------------------------
#  用法（在项目根目录下执行）：
#       bash setup_autodl.sh
#  配完后训练：
#       bash run_ctc_attn_transducer.sh --device cuda
# ==============================================================================
set -euo pipefail

# ---- 彩色输出 ----
info()  { echo -e "\n\033[1;34m[步骤]\033[0m $*"; }
ok()    { echo -e "\033[1;32m[成功]\033[0m $*"; }
warn()  { echo -e "\033[1;33m[注意]\033[0m $*"; }
fail()  { echo -e "\033[1;31m[失败]\033[0m $*"; }

echo "=============================================="
echo "   AutoDL 环境配置 (Conformer + 三解码头)"
echo "=============================================="

# ---------- 1. 学术加速 ----------
info "开启学术加速（访问 GitHub / 微软TTS 更快）"
if [ -f /etc/network_turbo ]; then
    source /etc/network_turbo && ok "学术加速已开启"
else
    warn "未找到 /etc/network_turbo（非 AutoDL 环境？跳过）"
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

# ---------- 3. 安装 PyTorch（GPU 版）----------
info "安装 PyTorch（GPU 版）"
python -m pip install --upgrade pip
# 自动检测 CUDA 版本并安装对应的 PyTorch
CUDA_VER=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "none")
if [ "$CUDA_VER" = "none" ] || [ "$CUDA_VER" = "12.8" ]; then
    # CUDA 12.8 或 torch 未安装 → 安装 cu128 版
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
elif [ "$CUDA_VER" = "12.4" ] || [ "$CUDA_VER" = "12.1" ]; then
    # 已安装 CUDA 12.x，装对应版本
    pip install torch torchaudio --index-url "https://download.pytorch.org/whl/cu${CUDA_VER//./}"
else
    pip install torch torchaudio
fi
ok "PyTorch 安装完成"

# ---------- 4. 验证 GPU 可用 ----------
info "验证 GPU 可用"
python -c "
import torch
cuda = torch.cuda.is_available()
name = torch.cuda.get_device_name(0) if cuda else '无'
print(f'CUDA 可用: {cuda}')
print(f'显卡    : {name}')
if cuda:
    x = torch.randn(8, 8, device='cuda')
    y = (x @ x).sum().item()
    print(f'GPU计算 : 通过 (sum={y:.2f})')
assert cuda, 'CUDA 不可用！'
print('[OK] GPU 验证通过')
"
ok "GPU 验证通过"

# ---------- 5. 安装项目依赖 ----------
info "安装项目依赖"
pip install -r requirements_ctc_attn_transducer.txt
pip install "fastapi>=0.110.0" "uvicorn[standard]>=0.27.0" "python-multipart>=0.0.9"
ok "项目依赖安装完成"

# ---------- 6. 最终自检 ----------
info "最终自检"
echo "--------------------------------------------------"
python - <<'PYEOF'
import sys, importlib.util
ok = True

# PyTorch
import torch
cuda = torch.cuda.is_available()
name = torch.cuda.get_device_name(0) if cuda else "无"
print(f"  PyTorch 版本 : {torch.__version__}")
print(f"  torchaudio版本: {torchaudio.__version__}" if "torchaudio" in sys.modules else "  torchaudio: 异常")
print(f"  CUDA 可用    : {cuda}")
print(f"  显卡         : {name}")
if not cuda: ok = False

# 项目依赖
for mod in ["soundfile", "librosa", "edge_tts", "tqdm", "fastapi", "uvicorn"]:
    if importlib.util.find_spec(mod) is None:
        ok = False
        print(f"  [错误] 缺少依赖: {mod}")
    else:
        print(f"  {mod:15s}: 已安装")

sys.exit(0 if ok else 1)
PYEOF
SELFCHECK=$?
set -e

echo "--------------------------------------------------"
if [ $SELFCHECK -eq 0 ]; then
    echo ""
    ok "环境配置完成！可以开始训练 🎉"
    echo ""
    echo "  下一步（GPU 训练）："
    echo "    bash run_ctc_attn_transducer.sh --device cuda --repeat 5"
    echo ""
    echo "  另开终端监控 GPU："
    echo "    watch -n 1 nvidia-smi"
else
    echo ""
    fail "自检未通过，请查看上面的警告。"
    exit 1
fi
