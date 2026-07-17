# -*- coding: utf-8 -*-
"""
VNet ASR — Hugging Face Space Gradio App
=========================================
Conformer + CTC / Attention / Transducer 联合语音识别演示。
"""
import os
import json
import time
from pathlib import Path

import numpy as np
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import gradio as gr

from model import ConformerCTCATTNTransducer

# ======================================================================
# 配置
# ======================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best.pt"
CMVN_PATH = "cmvn_stats.pt"
DICT_PATH = "units.txt"
SAMPLE_RATE = 16000
NUM_MEL_BINS = 80

# ======================================================================
# 加载字典
# ======================================================================
id2token = {}
token2id = {}
with open(DICT_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 2:
            token = parts[0]
            tid = int(parts[1])
            token2id[token] = tid
            id2token[tid] = token
VOCAB_SIZE = len(id2token)
SOS_EOS_ID = VOCAB_SIZE - 1  # <sos/eos>

print(f"词汇表大小: {VOCAB_SIZE}")
print(f"设备: {DEVICE}")

# ======================================================================
# 加载 CMVN
# ======================================================================
cmvn_data = torch.load(CMVN_PATH, map_location="cpu", weights_only=False)
cmvn_mean = cmvn_data["mean"].to(DEVICE)   # [80]
cmvn_std = cmvn_data["std"].to(DEVICE)     # [80]
print(f"CMVN 加载完成: mean={cmvn_mean[:3].tolist()}..., std={cmvn_std[:3].tolist()}...")

# ======================================================================
# 加载模型
# ======================================================================
model = ConformerCTCATTNTransducer(
    vocab_size=VOCAB_SIZE, idim=NUM_MEL_BINS,
    d_model=144, n_head=4, d_ff=1024,
    enc_blocks=6, attn_blocks=3,
    pred_dim=144, pred_layers=1,
).to(DEVICE)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(checkpoint["model_state"])
model.eval()
print(f"模型加载完成: {MODEL_PATH}")


# ======================================================================
# 音频预处理
# ======================================================================
def load_and_preprocess(audio_input, sample_rate):
    """
    处理 Gradio 传来的音频数据:
    - audio_input: numpy array (audio data)
    - sample_rate: int (audio sample rate)
    返回: [1, T, 80] tensor (fbank features)
    """
    # 确保是 float32
    if audio_input.dtype != np.float32:
        audio_input = audio_input.astype(np.float32)

    # 转为 torch tensor
    waveform = torch.from_numpy(audio_input).unsqueeze(0)  # [1, T]

    # 重采样到 16kHz
    if sample_rate != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sample_rate, SAMPLE_RATE)
        waveform = resampler(waveform)

    # 提取 Fbank 特征 (与训练一致)
    fbank = kaldi.fbank(
        waveform,
        num_mel_bins=NUM_MEL_BINS,
        frame_length=25,
        frame_shift=10,
        dither=0.0,       # 推理时不用 dither
        sample_frequency=SAMPLE_RATE,
        window_type="hamming",
    )  # [T', 80]

    # CMVN 归一化
    fbank = (fbank - cmvn_mean) / cmvn_std

    # 加 batch 维度 [1, T', 80]
    return fbank.unsqueeze(0).to(DEVICE)


# ======================================================================
# 解码函数
# ======================================================================
def decode_attention(tokens, eos_id=None):
    """将 attention 解码的 token id 序列转为文本"""
    if eos_id is None:
        eos_id = SOS_EOS_ID
    texts = []
    for b in range(tokens.size(0)):
        chars = []
        for t in range(tokens.size(1)):
            tok = tokens[b, t].item()
            if tok == eos_id:
                break
            if tok == 0 or tok == 1:  # skip blank/unk
                continue
            chars.append(id2token.get(tok, ""))
        texts.append("".join(chars))
    return texts


def decode_transducer(token_ids_list, eos_id=None):
    """将 transducer 解码的 token id 列表转为文本"""
    if eos_id is None:
        eos_id = SOS_EOS_ID
    texts = []
    for ids in token_ids_list:
        chars = []
        for tok in ids:
            if tok == eos_id:
                break
            if tok in (0, 1):
                continue
            chars.append(id2token.get(tok, ""))
        texts.append("".join(chars))
    return texts


# ======================================================================
# Gradio 推理接口
# ======================================================================
MODES = [
    ("Attention", "attention"),
    ("Transducer", "transducer"),
    ("CTC Greedy", "ctc_greedy"),
]

def predict(audio, mode="attention"):
    """
    主推理函数
    audio: Gradio 音频输入 (sample_rate, numpy_array) 或 None
    mode: 解码模式
    """
    if audio is None:
        return "", "请先录制或上传音频文件"

    sample_rate, audio_data = audio

    # 检测静音
    if np.abs(audio_data).max() < 1e-6:
        return "", "未检测到声音信号"

    # 预处理
    t0 = time.time()
    try:
        feats = load_and_preprocess(audio_data, sample_rate)
    except Exception as e:
        return "", f"音频处理失败: {str(e)}"

    # 推理
    try:
        if mode == "attention":
            token_ids = model.recognize_attention(
                feats, max_len=20, sos_id=SOS_EOS_ID, eos_id=SOS_EOS_ID
            )
            text = decode_attention(token_ids)[0]
        elif mode == "transducer":
            token_lists = model.recognize_transducer(
                feats, max_len=50, sos_id=SOS_EOS_ID
            )
            text = decode_transducer(token_lists)[0]
        elif mode == "ctc_greedy":
            texts = model.recognize_ctc_greedy(feats, id2token)
            text = texts[0]
        else:
            return "", f"不支持的解码模式: {mode}"
    except Exception as e:
        return "", f"推理失败: {str(e)}"

    elapsed = time.time() - t0

    if not text:
        text = "(无识别结果)"

    detail = f"模式: {mode} | 耗时: {elapsed:.3f}s"
    return text, detail


# ======================================================================
# 构建 Gradio UI
# ======================================================================

CSS = """
.gradio-container { max-width: 800px !important; margin: auto; }
.app-title { text-align: center; font-size: 1.8em; font-weight: bold; margin-bottom: 0.2em; }
.app-subtitle { text-align: center; color: #666; margin-bottom: 1em; }
.result-text { font-size: 1.5em; font-weight: bold; text-align: center; padding: 0.5em; }
.detail-text { text-align: center; color: #888; }
.mode-btn { min-width: 120px; }
"""

with gr.Blocks(title="VNet ASR Demo") as demo:
    gr.HTML("""
    <div class="app-title">🎤 VNet ASR 语音识别</div>
    <div class="app-subtitle">Conformer + CTC / Attention / Transducer 联合模型</div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                sources=["microphone", "upload"],
                type="numpy",
                label="输入音频",
                waveform_options=gr.WaveformOptions(
                    waveform_color="#4A90D9",
                    waveform_progress_color="#2E5A8A",
                ),
            )

        with gr.Column(scale=1):
            mode_radio = gr.Radio(
                choices=[m[0] for m in MODES],
                value="Attention",
                label="解码模式",
                info="Attention: 最准确 | Transducer: 流式友好 | CTC: 最快",
            )
            submit_btn = gr.Button("🚀 开始识别", variant="primary", size="lg")
            clear_btn = gr.Button("🗑️ 清空", size="sm")

    with gr.Row():
        result_text = gr.Textbox(
            label="识别结果",
            interactive=False,
            elem_classes="result-text",
            lines=2,
        )
        detail_text = gr.Textbox(
            label="详情",
            interactive=False,
            elem_classes="detail-text",
            lines=1,
        )

    # 模式映射
    mode_map = {m[0]: m[1] for m in MODES}

    def predict_wrapper(audio, mode_name):
        return predict(audio, mode_map[mode_name])

    submit_btn.click(
        fn=predict_wrapper,
        inputs=[audio_input, mode_radio],
        outputs=[result_text, detail_text],
    )

    clear_btn.click(
        fn=lambda: (None, "", ""),
        inputs=[],
        outputs=[audio_input, result_text, detail_text],
    )

    gr.Markdown("---")
    gr.Markdown(
        "💡 **提示**: 点击麦克风按钮说话，或上传 wav/mp3 文件。"
        " Attention 模式推荐用于非流式场景， Transducer 适合流式场景。"
    )

if __name__ == "__main__":
    demo.launch(css=CSS)
