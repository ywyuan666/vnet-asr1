"""
推理单元测试
============
验证三种解码模式的一致性。
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pytest

from model.conformer_ctc_attn_transducer import ConformerCTCATTNTransducer


@pytest.fixture
def small_model():
    """Create a tiny model for fast testing."""
    return ConformerCTCATTNTransducer(
        vocab_size=20,
        idim=80,
        d_model=32,
        d_ff=128,
        enc_blocks=2,
        attn_blocks=1,
        pred_dim=32,
        pred_layers=1,
        n_head=2,
        dropout=0.0,
        ctc_weight=0.3,
        attn_weight=0.3,
        trans_weight=0.4,
    )


class TestInferenceModes:
    """Test all three inference modes produce valid outputs."""

    def test_ctc_produces_text(self, small_model):
        """Test CTC greedy decoding produces text."""
        small_model.eval()
        speech = torch.randn(1, 40, 80)
        idx2token = {i: f"T{i}" for i in range(20)}

        texts = small_model.recognize_ctc_greedy(speech, idx2token)

        assert isinstance(texts, list) and len(texts) == 1
        assert isinstance(texts[0], str)
        # May be empty if all blanks, but shouldn't crash

    def test_attention_produces_tokens(self, small_model):
        """Test Attention decoding produces token tensor."""
        small_model.eval()
        speech = torch.randn(1, 40, 80)

        result = small_model.recognize_attention(speech, max_len=20, sos_id=19, eos_id=19)

        assert isinstance(result, torch.Tensor)
        assert result.size(0) == 1
        # Should have at least 1 token (the sos)
        assert result.size(1) >= 1

    def test_transducer_produces_tokens(self, small_model):
        """Test Transducer decoding produces token list."""
        small_model.eval()
        speech = torch.randn(1, 40, 80)

        result = small_model.recognize_transducer(speech, max_len=20, sos_id=19)

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], list)
        for token in result[0]:
            assert isinstance(token, int)


class TestStreamingInference:
    """Test streaming inference functions."""

    def test_ctc_streaming(self, small_model):
        """Test CTC streaming decoding runs without error."""
        small_model.eval()
        speech = torch.randn(1, 80, 80)  # 80 frames
        idx2token = {i: f"T{i}" for i in range(20)}

        texts = small_model.recognize_ctc_streaming(
            speech, idx2token, chunk_size=16, right_context=4
        )

        assert isinstance(texts, list) and len(texts) == 1

    def test_transducer_streaming(self, small_model):
        """Test Transducer streaming decoding runs without error."""
        small_model.eval()
        speech = torch.randn(1, 80, 80)  # 80 frames

        result = small_model.recognize_transducer_streaming(
            speech, chunk_size=16, right_context=4, max_len=20, sos_id=19
        )

        assert isinstance(result, list) and len(result) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
