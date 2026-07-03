# AutoDL GPU 训练 + 本地 Docker 部署 · 完整操作清单

> 目标：在 AutoDL 云 GPU 上快速训练出模型 → 下载到本地 Mac → 用 Docker 部署成 Web Demo。
> 核心原则：**云上只训练，本地做 Docker 部署**（因为 AutoDL 实例本身是容器，不能再跑 Docker）。

---

## 第 0 部分：租机前须知

| 事项 | 建议 |
| --- | --- |
| 显卡选择 | 这个项目模型小，**4090 足够**且 torch 兼容成熟；5090 也行但要注意 CUDA 版本 |
| 镜像选择 | 选 **PyTorch 2.7+ / CUDA 12.8** 官方镜像（5090 必须；省去自己配环境的钱） |
| 计费提醒 | AutoDL 按小时计费，**配环境也烧钱**，训练完**立刻关机** |
| 数据盘 | 项目放 `/root/autodl-tmp/`（关机不丢；`/root/` 重置镜像会清空） |

> ⚠️ **RTX 5090 的坑**：Blackwell 架构(算力 sm_120)需要 CUDA 12.8+ 和新版 PyTorch。
> 旧 torch 会报 `no kernel image is available`。选对镜像就能避开。

---

## 第 1 部分：上传项目到 AutoDL

**方式 A：用 Git（推荐）**
```bash
cd /root/autodl-tmp
git clone <你的项目仓库地址> wenet
cd wenet
```

**方式 B：用 JupyterLab 拖拽上传**
在 AutoDL 控制台打开 JupyterLab，把本地 `wenet` 文件夹压缩后拖上去解压。
（`exp/`、`data/` 不用传，会在云上重新生成。）

---

## 第 2 部分：配置环境（一键脚本）

进入项目目录，运行一键配置脚本即可：

```bash
cd /root/autodl-tmp/wenet
bash setup_autodl.sh
```

脚本会自动完成：开学术加速 → 检查 GPU → 装 cu128 版 PyTorch → 装项目依赖 →
装 WeNet 训练代码 → 复核锁定 GPU 版 torch → 装 ffmpeg → 最终自检。

✅ 看到 **「环境配置完成 🎉」** 即成功，脚本还会自动打印下一步的训练命令。

> 💡 脚本专门规避了两个坑：① 分步安装并在最后复核，防止依赖解析把 GPU 版 torch
> 覆盖成 CPU 版；② WeNet 源码用 `--no-deps` 安装，只取训练入口、不动 torch。

<details>
<summary>（备用）手动分步配置——脚本失败时可逐条执行</summary>

```bash
# 1) 开学术加速
source /etc/network_turbo

# 2) 确认显卡被识别
nvidia-smi                       # 应看到 RTX 5090 和显存

# 3) 装 PyTorch（5090 必须 CUDA 12.8 版）
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# 4) 验证 GPU 可用（关键，必须打印 True）
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"

# 5) 装项目依赖（不含 torch，避免覆盖 GPU 版）
pip install edge-tts soundfile numpy librosa sounddevice \
            fastapi "uvicorn[standard]" python-multipart tqdm pyyaml

# 6) 装 WeNet 训练代码（--no-deps 只取训练入口，不动 torch）
cd /root/autodl-tmp
git clone https://github.com/wenet-e2e/wenet.git wenet_src
cd wenet_src && pip install -e . --no-deps && cd /root/autodl-tmp/wenet

# 7) 复核 GPU 版 torch（防止被降级）
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128 --upgrade

# 8) 装 ffmpeg
apt-get update && apt-get install -y ffmpeg
```

</details>

✅ 检查点：`torch.cuda.is_available()` 打印 `True`，`python -m wenet.bin.train --help` 能出帮助。

---

## 第 3 部分：GPU 训练（核心，几分钟搞定）

```bash
# 用 GPU 专用配置：更大模型 + 更多数据 + 多进程加载
config=conf/train_u2pp_conformer_gpu.yaml repeat=5 num_workers=8 bash run.sh
```

这条命令会自动依次执行：
- Stage 0：TTS 合成语音数据（repeat=5 造更多数据）
- Stage 1：准备 data.list 和字典
- Stage 2：计算 CMVN
- Stage 3：**GPU 训练** U2++ Conformer → 产出 `exp/u2pp_conformer/final.pt`
- Stage 4：测试集解码算 CER
- Stage 5：单条识别演示

**另开一个终端监控 GPU 是否真的在用：**
```bash
watch -n 1 nvidia-smi            # 看到 python 进程占显存、GPU利用率>0 即成功
```

> 💡 如果只想重跑训练那一步：
> `config=conf/train_u2pp_conformer_gpu.yaml stage=3 stop_stage=3 bash run.sh`
>
> 💡 若报 **CUDA out of memory**：编辑 `conf/train_u2pp_conformer_gpu.yaml`，把 `batch_size` 调小（如 24 或 16）。

---

## 第 4 部分：把模型下载回本地 Mac

训练产物就在 `exp/u2pp_conformer/`，把这两样下载到本地：

```
exp/u2pp_conformer/final.pt      # 训练好的模型权重（几十 MB）
exp/u2pp_conformer/train.yaml    # 训练时生成的配置（推理要用）
data/dict/units.txt              # 字典（推理要用）
```

**下载方式**：JupyterLab 文件树里右键 → Download；或用 AutoDL 的文件管理下载。

> 建议把整个 `exp/` 和 `data/dict/` 目录打包下载，保持路径结构：
> ```bash
> cd /root/autodl-tmp/wenet
> tar -czf model_bundle.tar.gz exp/u2pp_conformer/final.pt exp/u2pp_conformer/train.yaml data/dict/units.txt
> ```
> 下载 `model_bundle.tar.gz` 到本地即可。

---

## 第 5 部分：⚠️ 关机停止计费

```
在 AutoDL 控制台点【关机】，确认实例已停止计费。
（模型已下载到本地，云上实例可以关了。想省钱又保留环境可用"无卡模式开机"。）
```

---

## 第 6 部分：回到本地 Mac，用 Docker 部署

```bash
cd ~/Documents/项目/wenet          # 你的本地项目目录

# 1) 把从云上下载的模型放回原位（解压 model_bundle.tar.gz 到项目根目录）
tar -xzf ~/Downloads/model_bundle.tar.gz -C .
# 确认这三个文件就位：
ls exp/u2pp_conformer/final.pt exp/u2pp_conformer/train.yaml data/dict/units.txt

# 2) Docker 一键部署
docker compose up -d --build       # 首次构建约几分钟

# 3) 浏览器打开演示
open http://localhost:8000
```

✅ 网页上传音频或点麦克风录音，应能实时显示识别结果。

**常用运维命令：**
```bash
docker compose logs -f             # 看日志
docker compose ps                  # 看容器状态
docker compose down                # 停止并删除容器
curl http://localhost:8000/api/health   # 命令行验证模型就绪
```

> 💡 内存兜底：你的 Mac 是 8GB，若 Docker 演示时卡顿，改用本机模式（更省内存）：
> ```bash
> ./start_demo.sh local
> ```

---

## 全流程速查（一图流）

```
┌─ AutoDL (GPU) ──────────────┐        ┌─ 本地 Mac ──────────────┐
│ 1 上传项目                   │        │ 6 放回模型文件          │
│ 2 配环境(torch cu128 + wenet)│        │ 7 docker compose up     │
│ 3 GPU训练 → final.pt         │  下载  │ 8 浏览器 localhost:8000 │
│ 4 打包下载模型 ─────────────────────▶ │   演示识别 ✅           │
│ 5 关机停止计费!              │        │  (兜底: start_demo.sh)  │
└─────────────────────────────┘        └─────────────────────────┘
    云上只训练                              本地做Docker部署
```

---

## 为什么"云上训练、本地部署"（面试可讲）

> AutoDL 的 GPU 实例**本身就是一个 Docker 容器**，容器内默认不能再嵌套跑 Docker
> （Docker-in-Docker 受限）。所以最佳实践是**训练与部署分离**：
> 云 GPU 只负责算力密集的训练，把模型产物下载到本地用容器化部署。
> 这也正好体现了 **"代码与数据/模型分离"** 的工程思想——模型通过 volume 挂载进容器，
> 镜像不变、模型可随时替换。

---

## 常见问题排查

| 现象 | 原因 | 解决 |
| --- | --- | --- |
| `CUDA available: False` | 装了 CPU 版 torch | 用 `--index-url .../cu128` 重装 |
| `no kernel image is available` | torch 太旧不支持 5090 | 换 CUDA 12.8 + 新版 torch |
| `No module named wenet.bin` | WeNet 训练代码没装 | 在 wenet_src 里 `pip install -e .` |
| `CUDA out of memory` | batch 太大 | 调小 yaml 里 `batch_size` |
| edge-tts 超时 | 没开学术加速 | `source /etc/network_turbo` |
| Docker 里模型未就绪(503) | 模型文件没放对位置 | 确认 `exp/` 和 `data/dict/` 就位 |
| `docker` 命令在 AutoDL 用不了 | 实例是容器,不支持DinD | 正常现象,改为本地部署 |
