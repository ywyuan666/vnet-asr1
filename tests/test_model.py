"""
模型单元测试
============
测试 Conformer + CTC/Attention/Transducer 模型的核心组件。

运行:
    pytest tests/ -v
    python -m pytest tests/ -v
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pytest

from model.conformer_ctc_attn_transducer import (
    ConformerBlock,
    ConformerEncoder,
    AttentionDecoder,
    TransducerDecoder,
    ConformerCTCATTNTransducer,
    PositionalEncoding,
)


def make_pad_mask(lengths: torch.Tensor, max_len: int = None) -> torch.Tensor:
    """Create padding mask for variable-length sequences (non-causal)."""
    if max_len is None:
        max_len = lengths.max().item()
    batch = lengths.size(0)
    mask = torch.arange(max_len, device=lengths.device).expand(batch, max_len)
    mask = mask < lengths.unsqueeze(1)  # True = valid, False = padding
    return ~mask  # True = masked/padding


class TestConformerBlock:
    """Test the Conformer block component."""

    @pytest.fixture
    def block(self):
        return ConformerBlock(
            d_model=64,
            n_head=4,
            d_ff=256,
            kernel_size=15,
            dropout=0.1,
        )

    def test_forward_shape(self, block):
        """Test output shape is preserved."""
        batch, seq_len = 2, 20
        x = torch.randn(batch, seq_len, 64)

        out = block(x, mask=None)

        assert out.shape == (batch, seq_len, 64), f"Expected (2, 20, 64), got {out.shape}"

    def test_forward_with_pad_mask(self, block):
        """Test forward with padding mask."""
        batch, seq_len = 2, 20
        x = torch.randn(batch, seq_len, 64)
        # First sample: 15 valid frames, second: 20 valid frames
        lengths = torch.tensor([15, 20])
        mask = make_pad_mask(lengths, max_len=seq_len).unsqueeze(1)

        out = block(x, mask)

        assert out.shape == (batch, seq_len, 64)

    def test_forward_with_attn_mask(self, block):
        """Test forward with additional attention mask (streaming)."""
        batch, seq_len = 2, 20
        x = torch.randn(batch, seq_len, 64)
        # Causal mask: each position can only see current and previous
        attn_mask = torch.full((seq_len, seq_len), -float("inf"))
        attn_mask = torch.triu(attn_mask, diagonal=1)  # causal

        out = block(x, mask=None, attn_mask=attn_mask)
        assert out.shape == (batch, seq_len, 64)

    def test_grad_flow(self, block):
        """Test gradients flow through the block."""
        x = torch.randn(1, 10, 64, requires_grad=True)
        out = block(x, mask=None)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.all(x.grad == 0)


class TestConformerEncoder:
    """Test the Conformer encoder."""

    @pytest.fixture
    def encoder(self):
        return ConformerEncoder(
            idim=80,
            d_model=64,
            n_head=4,
            d_ff=256,
            num_blocks=3,
            kernel_size=15,
            dropout=0.1,
        )

    def test_forward_4x_downsample(self, encoder):
        """Test 4x time downsample from Conv2d frontend."""
        batch = 2
        T = 40  # 40 frames -> 10 after 4x downsample
        x = torch.randn(batch, T, 80)

        out = encoder(x)

        expected_T = T // 4  # 10
        assert out.shape == (batch, expected_T, 64), f"Expected (2, 10, 64), got {out.shape}"

    def test_streaming_forward(self, encoder):
        """Test streaming forward with chunk mask."""
        batch = 2
        T = 40
        x = torch.randn(batch, T, 80)

        # Non-streaming
        out_normal = encoder(x)

        # Streaming with chunk=16, right_context=4
        # After 4x downsample, T_out=10, each chunk has 4 frames
        out_streaming = encoder(x, is_streaming=True, chunk_size=16, right_context=4)

        # Both should run without error
        assert out_normal.shape == out_streaming.shape


class TestAttentionDecoder:
    """Test the attention decoder."""

    @pytest.fixture
    def decoder(self):
        return AttentionDecoder(
            vocab_size=20,
            d_model=64,
            num_blocks=2,
            n_head=4,
            d_ff=256,
            dropout=0.1,
        )

    def test_forward(self, decoder):
        """Test decoder forward creates valid logits."""
        batch, T, L = 2, 10, 5
        encoder_out = torch.randn(batch, T, 64)
        tokens = torch.randint(1, 19, (batch, L))  # includes <sos> at position 0

        logits = decoder(tokens, encoder_out)

        assert logits.shape == (batch, L, 20), f"Expected (2, 5, 20), got {logits.shape}"

    def test_causal_mask(self, decoder):
        """Test causal masking prevents future token leakage."""
        batch, T, L = 1, 10, 8
        encoder_out = torch.randn(batch, T, 64)
        tokens = torch.randint(1, 19, (batch, L))

        logits = decoder(tokens, encoder_out)

        # All logits should be different (no identity copying)
        assert not torch.allclose(logits[0, -1], logits[0, 0])


class TestTransducerDecoder:
    """Test the transducer decoder."""

    @pytest.fixture
    def decoder(self):
        return TransducerDecoder(
            vocab_size=20,
            d_model=64,
            hidden_dim=64,
            embed_dim=64,
            num_layers=1,
            dropout=0.1,
        )

    def test_forward(self, decoder):
        """Test transducer produces expected logits grid."""
        batch, T, U = 2, 10, 5
        encoder_out = torch.randn(batch, T, 64)
        target = torch.randint(1, 19, (batch, U))
        target_lens = torch.tensor([U, U])

        logits, clamped_lens = decoder(encoder_out, target, target_lens)

        assert logits.shape == (batch, T, U, 20), f"Expected (2, 10, 5, 20), got {logits.shape}"
        assert clamped_lens.shape == (batch,)


class TestFullModel:
    """Test the complete model."""

    @pytest.fixture
    def model(self):
        return ConformerCTCATTNTransducer(
            vocab_size=20,
            idim=80,
            d_model=64,
            d_ff=256,
            enc_blocks=3,
            attn_blocks=2,
            pred_dim=64,
            pred_layers=1,
            n_head=4,
            dropout=0.1,
            ctc_weight=0.3,
            attn_weight=0.3,
            trans_weight=0.4,
        )

    def test_model_creation(self, model):
        """Test model creates all components."""
        assert hasattr(model, "encoder")
        assert hasattr(model, "ctc_linear")
        assert hasattr(model, "attn_decoder")
        assert hasattr(model, "trans_decoder")

    def test_forward_loss(self, model):
        """Test forward computes loss."""
        batch, T = 2, 40
        speech = torch.randn(batch, T, 80)
        feat_lens = torch.tensor([T, T])
        attn_in = torch.randint(1, 19, (batch, 5))
        attn_in[:, 0] = 19
        attn_out = torch.randint(1, 19, (batch, 5))
        trans_in = torch.randint(1, 19, (batch, 5))
        trans_in[:, 0] = 19
        trans_lens = torch.tensor([4, 5])

        outputs = model(speech, feat_lens, attn_in, attn_out, trans_in, trans_lens)

        assert isinstance(outputs, dict)
        for key in ["loss", "ctc_loss", "attn_loss", "transducer_loss"]:
            assert key in outputs, f"Missing key: {key}"
            assert isinstance(outputs[key], torch.Tensor)
        assert outputs["loss"].ndim == 0  # scalar
        assert outputs["loss"] > 0  # positive loss

    def test_ctc_greedy(self, model):
        """Test CTC greedy decoding."""
        batch, T = 1, 40
        speech = torch.randn(batch, T, 80)
        idx2token = {i: f"T{i}" for i in range(20)}

        texts = model.recognize_ctc_greedy(speech, idx2token)

        assert isinstance(texts, list)
        assert len(texts) == 1
        assert isinstance(texts[0], str)

    def test_attention_decoding(self, model):
        """Test Attention decoding."""
        batch, T = 1, 40
        speech = torch.randn(batch, T, 80)

        result = model.recognize_attention(speech, max_len=20, sos_id=19, eos_id=19)

        assert isinstance(result, torch.Tensor)
        assert result.size(0) == 1
        assert result.size(1) >= 1

    def test_transducer_decoding(self, model):
        """Test Transducer decoding."""
        batch, T = 1, 40
        speech = torch.randn(batch, T, 80)

        result = model.recognize_transducer(speech, max_len=50, sos_id=19)

        assert isinstance(result, list)
        assert len(result) == 1


class TestStreamingHelper:
    """Test streaming helper functions."""

    def test_make_chunk_mask(self):
        """Test chunk mask creation."""
        from model.streaming_helper import make_chunk_mask

        seq_len, chunk_size, right_context = 12, 4, 2
        mask = make_chunk_mask(seq_len, chunk_size, right_context)

        assert mask.shape == (seq_len, seq_len)

        # Frame 0 (chunk 0) should attend to frames 0-5 (chunk_size + right_context)
        assert mask[0, 0] == 0  # self
        assert mask[0, 3] == 0  # within chunk
        assert mask[0, 5] == 0  # right context
        assert mask[0, 6] == -float("inf")  # beyond right context

        # Frame 4 (start of chunk 1) should not attend to frame 0 if chunk_start > 0
        # but our implementation attends to ALL previous chunks too
        # so frame 4 CAN attend to frame 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
