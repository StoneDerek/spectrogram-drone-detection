"""
Audio feature extraction for drone detection.
Converts raw waveforms into log-mel spectrograms for the EfficientNet classifier.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torchaudio.transforms as T


class MelSpectrogramExtractor:
    """Converts a waveform tensor into a log-mel spectrogram."""

    def __init__(
        self,
        sample_rate: int = 22050,
        n_mels: int = 128,
        n_fft: int = 1024,
        hop_length: int = 512,
        f_min: float = 50.0,
        f_max: float = 11025.0,
        power: float = 2.0,
        log_offset: float = 1e-9,
    ):
        self.sample_rate = sample_rate
        self.log_offset = log_offset
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=power,
        )

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (channels, samples) or (samples,) float tensor
        Returns:
            log_mel: (1, n_mels, time_frames) float tensor
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        # Mix down to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        mel = self.mel_transform(waveform)           # (1, n_mels, T)
        log_mel = torch.log(mel + self.log_offset)   # log-compress
        return log_mel


def peak_normalize(waveform: torch.Tensor, target_peak: float = 0.99) -> torch.Tensor:
    """Normalise waveform so its peak absolute amplitude equals target_peak."""
    peak = waveform.abs().max()
    if peak > 0:
        waveform = waveform * (target_peak / peak)
    return waveform


def load_stats(stats_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load pre-computed mean and std arrays saved during training.

    Args:
        stats_dir: directory containing mean.npy and std.npy
    Returns:
        (mean, std) as numpy arrays shaped (n_mels,)
    """
    stats_dir = Path(stats_dir)
    mean = np.load(stats_dir / "mean.npy")
    std = np.load(stats_dir / "std.npy")
    return mean, std


def standardize(
    spectrogram: torch.Tensor,
    mean: np.ndarray | torch.Tensor,
    std: np.ndarray | torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Standardise a spectrogram to zero mean and unit variance using
    training-set statistics.

    Args:
        spectrogram: (1, n_mels, T) tensor
        mean: per-mel mean from training set, shape (n_mels,)
        std:  per-mel std  from training set, shape (n_mels,)
        eps:  small constant to avoid division by zero
    Returns:
        standardised spectrogram of the same shape
    """
    if isinstance(mean, np.ndarray):
        mean = torch.from_numpy(mean).float()
    if isinstance(std, np.ndarray):
        std = torch.from_numpy(std).float()

    # Reshape to broadcast over the time dimension → (1, n_mels, 1)
    mean = mean.view(1, -1, 1)
    std = std.view(1, -1, 1)

    return (spectrogram - mean) / (std + eps)