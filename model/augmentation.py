# -*- coding: utf-8 -*-
"""
augmentation.py
================
SpecAugment (frequency masking + time masking) and speed perturbation.
"""

import random

import librosa
import numpy as np
import torch
import torch.nn as nn


class SpecAugment(nn.Module):
    """
    SpecAugment: frequency masking + time masking.

    Args:
        freq_mask_width: frequency mask width range (0, max)
        time_mask_width: time mask width range (0, max)
        num_freq_masks: number of frequency masks
        num_time_masks: number of time masks
    """

    def __init__(
        self,
        freq_mask_width: int = 10,
        time_mask_width: int = 25,
        num_freq_masks: int = 2,
        num_time_masks: int = 2,
    ):
        super().__init__()
        self.freq_mask_width = freq_mask_width
        self.time_mask_width = time_mask_width
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks

    def forward(self, x: torch.Tensor, x_lengths: torch.Tensor = None) -> torch.Tensor:
        """
        Apply SpecAugment.
        x: (B, T, F) Fbank features
        x_lengths: (B,) original lengths before padding

        Returns:
            (B, T, F) augmented features
        """
        if not self.training:
            return x

        batch_size, max_len, num_feats = x.shape
        output = x.clone()

        # Time masking
        for _ in range(self.num_time_masks):
            t = random.randint(0, self.time_mask_width)
            if t <= 0:
                continue
            t_start = random.randint(0, max(0, max_len - t))
            output[:, t_start:t_start + t, :] = 0

        # Frequency masking
        for _ in range(self.num_freq_masks):
            f = random.randint(0, self.freq_mask_width)
            if f <= 0:
                continue
            f_start = random.randint(0, max(0, num_feats - f))
            output[:, :, f_start:f_start + f] = 0

        return output


def speed_perturb(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    Speed perturbation: randomly change speed by factor 0.9-1.1.
    Returns: perturbed audio at original sample rate, with same length as input.
    """
    speed = random.choice([0.9, 1.0, 1.1])
    if speed == 1.0:
        return audio

    # librosa time stretch
    stretched = librosa.effects.time_stretch(y=audio, rate=speed)
    # Pad or truncate to match original length
    if len(stretched) > len(audio):
        return stretched[:len(audio)]
    else:
        return np.pad(stretched, (0, len(audio) - len(stretched)), mode='constant')
