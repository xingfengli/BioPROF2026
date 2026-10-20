from __future__ import annotations

from pathlib import Path

import pandas as pd

from .audio import AudioConfig, load_waveform
from .embeddings import canonical_filepath, load_cached_embedding, resolve_embedding_cache_dir


class AudioManifestDataset:
    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        audio_cfg: AudioConfig,
        root: str | Path | None = None,
        known_only: bool = False,
        unknown_only: bool = False,
    ):
        self.root = Path.cwd() if root is None else Path(root)
        self.audio_cfg = audio_cfg
        df = pd.read_csv(manifest_path)
        df = df[df["split"] == split].copy()
        if known_only and unknown_only:
            raise ValueError("known_only and unknown_only cannot both be True.")
        if known_only:
            df = df[df["is_known"].astype(int) == 1].copy()
        if unknown_only:
            df = df[df["is_known"].astype(int) == 0].copy()
        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        import torch

        row = self.df.iloc[idx]
        path = self.root / row["filepath"]
        waveform = load_waveform(path, self.audio_cfg)
        label_id = int(row["label_id"])
        return {
            "waveform": waveform,
            "label_id": torch.tensor(label_id, dtype=torch.long),
            "label": row["label"],
            "path": str(path),
            "is_known": bool(int(row["is_known"])),
        }


class EmbeddingManifestDataset:
    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        cfg: dict,
        root: str | Path | None = None,
        known_only: bool = False,
        unknown_only: bool = False,
    ):
        self.root = Path.cwd() if root is None else Path(root)
        self.cache_dir = resolve_embedding_cache_dir(cfg, self.root)
        self.l2_normalize = bool(cfg.get("embedding", {}).get("l2_normalize", True))
        df = pd.read_csv(manifest_path)
        df = df[df["split"] == split].copy()
        if known_only and unknown_only:
            raise ValueError("known_only and unknown_only cannot both be True.")
        if known_only:
            df = df[df["is_known"].astype(int) == 1].copy()
        if unknown_only:
            df = df[df["is_known"].astype(int) == 0].copy()
        self.df = df.reset_index(drop=True)
        if self.df.empty:
            raise ValueError(f"No rows found for split '{split}'.")

        first = canonical_filepath(self.df.iloc[0]["filepath"], self.root)
        self.embedding_dim = int(load_cached_embedding(self.cache_dir, first, self.l2_normalize).shape[0])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        import torch

        row = self.df.iloc[idx]
        path = canonical_filepath(row["filepath"], self.root)
        embedding = load_cached_embedding(self.cache_dir, path, self.l2_normalize)
        if embedding.shape[0] != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch for '{path}': "
                f"expected {self.embedding_dim}, got {embedding.shape[0]}"
            )
        return {
            "embedding": torch.from_numpy(embedding),
            "label_id": torch.tensor(int(row["label_id"]), dtype=torch.long),
            "label": row["label"],
            "path": path,
            "is_known": bool(int(row["is_known"])),
        }


def make_class_balanced_sampler(dataset: AudioManifestDataset, generator=None):
    import torch
    from torch.utils.data import WeightedRandomSampler

    labels = dataset.df["label_id"].astype(int).to_numpy()
    counts = {label: int((labels == label).sum()) for label in sorted(set(labels))}
    weights = torch.tensor([1.0 / counts[int(label)] for label in labels], dtype=torch.double)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True, generator=generator)
