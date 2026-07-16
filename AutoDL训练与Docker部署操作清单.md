# AutoDL GPU 训练 + Docker 部署 · 完整操作清单

> 目标：在 AutoDL 云 GPU 上训练 Conformer + CTC/Attention/Transducer 模型 → 下载到本地 → 用 Docker 部署成 Web Demo。
> 核心原则：**云上只训练，本地做 Docker 部署**（AutoDL 实例本身是容器，不能嵌套 Docker）。

---

## 第 0 部分：租机前须知

| 事项 | 建议 |
| --- | --- |
| 显卡选择 | 本项目模型小（~6.6M 参数），**4090 足够** |
| 镜像选择 | 选 **PyTorch 2.7+ / CUDA 12.8** 官方镜像，省去配环境时间 |
| 计费提醒 | AutoDL 按小时计费，训练完**立刻关机** |
| 数据盘 | 项目放 `/root/autodl-tmp/`（关机不丢） |

---

## 第 1 部分：上传项目到 AutoDL

**方式 A：Git 克隆（推荐）**
```bash
cd /root/autodl-tmp
git clone https://github.com/ywyuan666/vnet-asr1.git conformer-asr
cd conformer-asr
```

**方式 B：JupyterLab 拖拽上传**
在 AutoDL 控制台打开 JupyterLab，把本地项目文件夹压缩后拖上去解压。
`exp/`、`data/` 目录不用传，会在云上重新生成。

---

## 第 2 部分：配置环境（一键脚本）

```bash
cd /root/autodl-tmp/conformer-asr
bash setup_autodl.sh
```

脚本会自动完成：开学术加速 → 检查 GPU → 安装 GPU 版 PyTorch → 安装项目依赖 → 最终自检。

✅ 看到 **「环境配置完成 🎉」** 即成功。

> ⚠️ 本项目**不依赖 WeNet 框架**，训练脚本 `train.py` 是纯 PyTorch 实现，无需安装 WeNet 源码。

---

## 第 3 部分：GPU 训练

```bash
# GPU 训练（300 条数据，200 epochs，几分钟搞定）
bash run_ctc_attn_transducer.sh --device cuda --repeat 5
```

这条命令会自动依次执行：
| Stage | 内容 |
|-------|------|
| 0 | edge-tts 合成 300 条语音指令数据 |
| 1 | 准备 data.list 和字典 |
| 2 | 计算 CMVN 特征归一化 |
| 3 | **GPU 训练** Conformer + CTC/Attention/Transducer 模型（200 epochs） |
| 4 | 测试集三种模式解码并计算 CER |
| 5 | 单条音频识别演示 |

**监控 GPU：** 另开终端 `watch -n 1 nvidia-smi` 确认 GPU 利用率 > 0。

> 💡 如果想只重跑训练：`bash run_ctc_attn_transducer.sh --stage 3 --stop 3 --device cuda`

---

## 第 4 部分：把模型下载回本地

训练产物在 `exp/conformer_ctc_attn_transducer/`，需要下载：

```
exp/conformer_ctc_attn_transducer/best.pt   # 最佳模型权重
data/dict/units.txt                          # 字典
data/train/global_cmvn                       # CMVN 统计量
```

**打包下载**（推荐）：
```bash
cd /root/autodl-tmp/conformer-asr
tar -czf model_bundle.tar.gz \
    exp/conformer_ctc_attn_transducer/best.pt \
    data/dict/units.txt \
    data/train/global_cmvn
```
然后从 JupyterLab 或 AutoDL 文件管理下载 `model_bundle.tar.gz`。

---

## 第 5 部分：⚠️ 关机停止计费

在 AutoDL 控制台点【关机】，确认实例已停止计费。

---

## 第 6 部分：回到本地，用 Docker 部署

```bash
cd /path/to/conformer-asr          # 你的本地项目目录

# 1) 把从云上下载的模型放回原位
tar -xzf ~/Downloads/model_bundle.tar.gz -C .

# 2) Docker 一键部署（首次构建约几分钟）
docker compose up -d --build

# 3) 浏览器打开演示
open http://localhost:8000
```

✅ 网页上传音频或点麦克风录音，应能实时显示识别结果。

**常用运维命令：**
```bash
docker compose logs -f             # 看日志
docker compose ps                  # 看容器状态
docker compose down                # 停止并删除容器
curl http://localhost:8000/api/health   # 健康检查
```

### 本机 Python 快速启动（无 Docker）

```bash
pip install fastapi "uvicorn[standard]" python-multipart
uvicorn server.app_ctc_attn_transducer:app --host 0.0.0.0 --port 8000
```

---

## 全流程速查

```
┌─ AutoDL (GPU) ──────────────────┐        ┌─ 本地 ────────────────────┐
│ 1 git clone 上传项目             │        │ 6 放回模型文件            │
│ 2 bash setup_autodl.sh 配环境    │        │ 7 docker compose up      │
│ 3 bash run_*.sh 训练             │  下载  │ 8 浏览器 localhost:8000   │
│ 4 打包下载模型 ────────────────────────→ │   三种解码模式可选 ✅     │
│ 5 关机停止计费!                  │        └──────────────────────────┘
└──────────────────────────────────┘
```

---

## 三种解码模式

| 解码模式 | 说明 | Web Demo 选择 |
|---------|------|-------------|
| **Attention** | 精度最高，CER=0.87% | 默认模式 |
| **Transducer** | 流式友好，CER=9.57% | 可选 |
| **CTC Greedy** | 速度最快，CER=25.22% | 可选 |

---

## 常见问题

| 现象 | 原因 | 解决 |
|------|------|------|
| `CUDA available: False` | 装了 CPU 版 torch | 用 `--index-url .../cu128` 重装 |
| edge-tts 超时 | 没开学术加速 | `source /etc/network_turbo` |
| Docker 里模型未就绪 (503) | 模型文件没放对位置 | 确认 `exp/` 和 `data/` 已挂载 |
| `docker` 命令在 AutoDL 用不了 | 实例是容器，不支持嵌套 Docker | 正常现象，改到本地部署 |
