from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16_000
    clip_seconds: float = 8.0
    n_mels: int = 128
    n_fft: int = 2048
    hop_length: int = 512
    frontend: str = "torchaudio"

    @property
    def num_samples(self) -> int:
        return int(round(self.sample_rate * self.clip_seconds))


def load_waveform(path: str | Path, cfg: AudioConfig):
    import torch
    import torchaudio

    waveform, sample_rate = torchaudio.load(str(path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != cfg.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, cfg.sample_rate)
    target = cfg.num_samples
    if waveform.shape[-1] < target:
        waveform = torch.nn.functional.pad(waveform, (0, target - waveform.shape[-1]))
    elif waveform.shape[-1] > target:
        waveform = waveform[..., :target]
    return waveform


class LogMelFrontend:
    def __init__(self, cfg: AudioConfig):
        import torch
        import torchaudio

        self.cfg = cfg
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mels,
            power=2.0,
        )
        self.amp_to_db = torchaudio.transforms.AmplitudeToDB(stype="power")
        self.eps = torch.finfo(torch.float32).eps

    def to(self, device):
        self.mel = self.mel.to(device)
        self.amp_to_db = self.amp_to_db.to(device)
        return self

    def __call__(self, waveform):
        spec = self.mel(waveform)
        spec = self.amp_to_db(spec + self.eps)
        return spec


def opensoundscape_logmel(path: str | Path, cfg: AudioConfig):
    """Create a log-Mel spectrogram via OpenSoundscape when available.

    OpenSoundscape is useful for bioacoustic preprocessing, but its API changes
    across releases. This helper keeps the dependency isolated; the default
    training frontend uses torchaudio for reproducibility.
    """
    try:
        from opensoundscape import Audio
        from opensoundscape.spectrogram import MelSpectrogram
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("OpenSoundscape is not installed or its API changed.") from exc

    audio = Audio.from_file(str(path))
    audio = audio.resample(cfg.sample_rate)
    audio = audio.trim(0, cfg.clip_seconds, extend=True)
    spec = MelSpectrogram.from_audio(
        audio,
        window_samples=cfg.n_fft,
        overlap_samples=cfg.n_fft - cfg.hop_length,
        n_mels=cfg.n_mels,
    )
    return spec.to_image(return_type="np")
