"""流式辅助函数：chunk mask 生成、KV cache 初始化等"""

from typing import List, Optional, Tuple

import torch


def make_chunk_mask(
    seq_len: int,
    chunk_size: int,
    right_context: int,
    dtype: torch.dtype = None,
    device: torch.device = None,
) -> torch.Tensor:
    """
    生成 chunk-wise 注意力掩码。

    每个位置可以关注：
    - 同一 chunk 内的所有位置
    - 当前 chunk 后面的 right_context 个帧
    - 前面的所有 chunk（用于流式中保留历史上下文）

    Args:
        seq_len: 序列长度（下采样后）
        chunk_size: 每 chunk 帧数（下采样后）
        right_context: 右侧上下文帧数
        dtype: tensor dtype
        device: tensor device

    Returns:
        mask: (seq_len, seq_len) 0 = 可关注, -inf = 被掩码
    """
    if dtype is None:
        dtype = torch.float32
    mask = torch.full((seq_len, seq_len), -float("inf"), dtype=dtype, device=device)

    for i in range(seq_len):
        # 当前帧所属 chunk 的起始位置
        chunk_start = (i // chunk_size) * chunk_size
        # 当前 chunk 的可关注终点（含 right_context）
        chunk_end = min(chunk_start + chunk_size + right_context, seq_len)
        # 当前帧可以关注从 chunk_start 到 chunk_end
        mask[i, chunk_start:chunk_end] = 0
        # 还可以关注前面所有 chunk（累积历史上下文）
        if chunk_start > 0:
            mask[i, :chunk_start] = 0

    return mask


def init_attn_cache(
    num_blocks: int,
    batch_size: int,
    max_cache_len: int,
    d_model: int,
    device: torch.device,
) -> List[Optional[Tuple[torch.Tensor, torch.Tensor]]]:
    """
    初始化 KV cache 列表，用于流式推理。

    Args:
        num_blocks: Conformer block 数量
        batch_size: batch size
        max_cache_len: cache 最大长度
        d_model: 模型维度
        device: 设备

    Returns:
        cache: 长度为 num_blocks 的列表，每项为 (k, v) 元组或 None
    """
    cache = []
    for _ in range(num_blocks):
        k = torch.zeros(batch_size, 0, d_model, device=device)
        v = torch.zeros(batch_size, 0, d_model, device=device)
        cache.append((k, v))
    return cache
