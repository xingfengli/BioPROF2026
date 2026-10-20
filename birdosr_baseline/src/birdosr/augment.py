from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AugmentConfig:
    random_gain_db: float = 0.0
    time_shift_fraction: float = 0.0
    time_mask_param: int = 0
    freq_mask_param: int = 0
    background_noise_prob: float = 0.0
    background_noise_snr_db: tuple[float, float] | list[float] = (5.0, 20.0)
    background_noise_labels: tuple[str, ...] | list[str] = ()
    mixup_alpha: float = 0.0


class WaveformAugment:
    def __init__(self, cfg: AugmentConfig):
        self.cfg = cfg

    def __call__(self, waveform):
        import torch

        if self.cfg.random_gain_db > 0:
            gain_db = (torch.rand(1, device=waveform.device) * 2 - 1) * self.cfg.random_gain_db
            waveform = waveform * torch.pow(torch.tensor(10.0, device=waveform.device), gain_db / 20)

        if self.cfg.time_shift_fraction > 0:
            max_shift = int(waveform.shape[-1] * self.cfg.time_shift_fraction)
            if max_shift > 0:
                shift = int(torch.randint(-max_shift, max_shift + 1, (1,), device=waveform.device).item())
                waveform = torch.roll(waveform, shifts=shift, dims=-1)

        return waveform


class SpecAugment:
    def __init__(self, cfg: AugmentConfig):
        import torchaudio

        self.time_mask = torchaudio.transforms.TimeMasking(cfg.time_mask_param) if cfg.time_mask_param > 0 else None
        self.freq_mask = torchaudio.transforms.FrequencyMasking(cfg.freq_mask_param) if cfg.freq_mask_param > 0 else None

    def __call__(self, spec):
        if self.freq_mask is not None:
            spec = self.freq_mask(spec)
        if self.time_mask is not None:
            spec = self.time_mask(spec)
        return spec


def mixup_batch(inputs, targets, alpha: float):
    import torch

    if alpha <= 0:
        return inputs, targets, targets, 1.0
    distribution = torch.distributions.Beta(alpha, alpha)
    lam = float(distribution.sample().item())
    index = torch.randperm(inputs.size(0), device=inputs.device)
    mixed = lam * inputs + (1.0 - lam) * inputs[index]
    return mixed, targets, targets[index], lam


class BackgroundNoiseMixer:
    def __init__(self, cfg: AugmentConfig, noise_paths: list, audio_cfg):
        self.cfg = cfg
        self.noise_paths = list(noise_paths)
        self.audio_cfg = audio_cfg

    def __call__(self, waveform):
        import random
        import torch

        if not self.noise_paths or self.cfg.background_noise_prob <= 0:
            return waveform
        if random.random() > self.cfg.background_noise_prob:
            return waveform

        from .audio import load_waveform

        noise_path = random.choice(self.noise_paths)
        noise = load_waveform(noise_path, self.audio_cfg).to(waveform.device)
        while noise.dim() < waveform.dim():
            noise = noise.unsqueeze(0)
        if noise.shape[0] == 1 and waveform.shape[0] > 1:
            noise = noise.repeat(waveform.shape[0], 1, 1)

        signal_rms = torch.sqrt(torch.mean(waveform ** 2, dim=-1, keepdim=True) + 1e-8)
        noise_rms = torch.sqrt(torch.mean(noise ** 2, dim=-1, keepdim=True) + 1e-8)
        lo, hi = self.cfg.background_noise_snr_db
        snr_db = torch.empty((waveform.shape[0], 1, 1), device=waveform.device).uniform_(float(lo), float(hi))
        target_noise_rms = signal_rms / torch.pow(torch.tensor(10.0, device=waveform.device), snr_db / 20)
        noise = noise * (target_noise_rms / noise_rms)
        return waveform + noise
