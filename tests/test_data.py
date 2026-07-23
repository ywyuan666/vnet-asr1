"""
数据加载单元测试
================
测试 AsrDataset 和数据预处理流水线。
"""

import sys
import os
import json
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import pytest


def create_dummy_wav(path: str, duration: float = 1.0, sr: int = 16000):
    """Create a dummy sine wave wav file."""
    import soundfile as sf
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440Hz sine wave
    sf.write(path, audio, sr)
    return path


def create_dummy_data_list(path: str, num_samples: int = 5, wav_dir: str = None):
    """Create a dummy data.list file."""
    if wav_dir is None:
        wav_dir = tempfile.mkdtemp()

    records = []
    for i in range(num_samples):
        wav_path = os.path.join(wav_dir, f"test_{i:03d}.wav")
        create_dummy_wav(wav_path, duration=0.5)
        records.append({
            "key": f"test_{i:03d}",
            "wav": wav_path,
            "txt": "你好世界" if i % 2 == 0 else "测试语音",
        })

    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return records


class TestAsrDataset:
    """Test the ASR dataset class."""

    @pytest.fixture
    def setup(self):
        """Setup temp dataset."""
        self.tmp = tempfile.mkdtemp()
        self.dict_path = os.path.join(self.tmp, "units.txt")
        self.data_list_path = os.path.join(self.tmp, "data.list")
        self.wav_dir = os.path.join(self.tmp, "wavs")
        os.makedirs(self.wav_dir, exist_ok=True)

        # Create dictionary
        with open(self.dict_path, "w", encoding="utf-8") as f:
            f.write("<blank>\n<unk>\n你好世界\n测试语音\n<sos/eos>\n")

        # Create data
        self.records = create_dummy_data_list(self.data_list_path, 5, self.wav_dir)

        yield

        # Cleanup
        import shutil
        shutil.rmtree(self.tmp)

    def test_dataset_len(self, setup):
        """Test dataset length."""
        from train import AsrDataset
        dataset = AsrDataset(self.data_list_path, cmvn_path=None)
        assert len(dataset) == 5

    def test_dataset_getitem(self, setup):
        """Test __getitem__ returns (feat, txt) tuple."""
        from train import AsrDataset
        dataset = AsrDataset(self.data_list_path, cmvn_path=None)
        feat, txt = dataset[0]

        assert isinstance(feat, torch.Tensor), f"Expected Tensor, got {type(feat)}"
        assert feat.dim() == 2, f"Expected 2D, got {feat.dim()}D"
        assert feat.size(-1) == 80, f"Expected 80-dim Fbank, got {feat.size(-1)}"
        assert isinstance(txt, str), f"Expected str, got {type(txt)}"

    def test_collate_fn(self, setup):
        """Test collate function produces correct shapes."""
        from train import AsrDataset, collate_fn

        # Build vocab
        vocab = {}
        with open(self.dict_path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    vocab[parts[0]] = int(parts[1])
        sos_id = len(vocab) - 1

        dataset = AsrDataset(self.data_list_path, cmvn_path=None)
        batch_data = [dataset[i] for i in range(2)]
        collated = collate_fn(batch_data, vocab, sos_id)

        # Should return 6 elements
        assert len(collated) == 6, f"Expected 6 elements, got {len(collated)}"

        feats_padded, feat_lens, attn_in, attn_out, trans_in, trans_lens = collated

        assert isinstance(feats_padded, torch.Tensor)
        assert isinstance(feat_lens, torch.Tensor)
        assert isinstance(attn_in, torch.LongTensor) or isinstance(attn_in, torch.Tensor)
        assert isinstance(attn_out, torch.LongTensor) or isinstance(attn_out, torch.Tensor)
        assert isinstance(trans_in, torch.LongTensor) or isinstance(trans_in, torch.Tensor)
        assert isinstance(trans_lens, torch.Tensor)

        # Check batch dimension
        assert feats_padded.size(0) == 2
        assert feat_lens.size(0) == 2
        assert trans_lens.size(0) == 2

        # Check feature dimension
        assert feats_padded.size(-1) == 80

        print(f"feats_padded shape: {feats_padded.shape}")
        print(f"feat_lens: {feat_lens}")
        print(f"attn_in shape: {attn_in.shape}")
        print(f"attn_out shape: {attn_out.shape}")
        print(f"trans_in shape: {trans_in.shape}")
        print(f"trans_lens: {trans_lens}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
