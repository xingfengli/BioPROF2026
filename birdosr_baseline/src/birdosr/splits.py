from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


@dataclass(frozen=True)
class ManifestRow:
    filepath: str
    label: str
    label_id: int
    site: str
    split: str
    is_known: bool


def site_known_dir(data_root: Path, site: str) -> Path:
    return data_root / site / f"known{site}"


def site_unknown_dir(data_root: Path, site: str) -> Path:
    return data_root / site / f"unknown{site}"


def find_audio_files(label_dir: Path) -> list[Path]:
    if not label_dir.exists():
        raise FileNotFoundError(f"Missing label directory: {label_dir}")
    files = [
        p for p in label_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    ]
    return sorted(files)


def split_files(files: list[Path], fractions: tuple[float, float], seed: int) -> tuple[list[Path], list[Path], list[Path]]:
    train_fraction, val_fraction = fractions
    if train_fraction <= 0 or val_fraction < 0:
        raise ValueError("train_fraction must be > 0 and val_fraction must be >= 0")
    rng = random.Random(seed)
    items = list(files)
    rng.shuffle(items)
    n = len(items)
    n_train = int(round(n * train_fraction))
    n_val = int(round(n * val_fraction))
    n_train = min(n_train, n)
    n_val = min(n_val, max(0, n - n_train))
    train = items[:n_train]
    val = items[n_train:n_train + n_val]
    test = items[n_train + n_val:]
    return sorted(train), sorted(val), sorted(test)


def _rows_for_files(
    files: Iterable[Path],
    label: str,
    label_id: int,
    site: str,
    split: str,
    is_known: bool,
    root: Path,
) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    for file in files:
        try:
            manifest_path = str(file.relative_to(root))
        except ValueError:
            manifest_path = str(file)
        rows.append(
            ManifestRow(
                filepath=manifest_path,
                label=label,
                label_id=label_id,
                site=site,
                split=split,
                is_known=is_known,
            )
        )
    return rows


def build_manifest(cfg: dict) -> list[ManifestRow]:
    root = Path.cwd()
    data_root = Path(cfg["data_root"])
    if not data_root.is_absolute():
        data_root = root / data_root

    train_site = cfg["train_site"]
    test_site = cfg["test_site"]
    known_labels = list(cfg["known_labels"])
    unknown_labels = list(cfg["unknown_labels"])
    label_to_id = {label: idx for idx, label in enumerate(known_labels)}
    seed = int(cfg.get("seed", 26))

    split_cfg = cfg.get("split", {})
    train_fraction = float(split_cfg.get("train_fraction", 0.8))
    val_fraction = float(split_cfg.get("val_fraction", 0.1))
    unknown_val_fraction = float(split_cfg.get("unknown_val_fraction", 0.0))
    unknown_oe_fraction = float(split_cfg.get("unknown_oe_fraction", 0.0))

    rows: list[ManifestRow] = []

    for label in known_labels:
        files = find_audio_files(site_known_dir(data_root, train_site) / label)
        train_files, val_files, same_site_test_files = split_files(
            files,
            (train_fraction, val_fraction),
            seed + label_to_id[label],
        )
        rows.extend(_rows_for_files(train_files, label, label_to_id[label], train_site, "train", True, root))
        rows.extend(_rows_for_files(val_files, label, label_to_id[label], train_site, "val", True, root))

        if train_site == test_site:
            rows.extend(_rows_for_files(same_site_test_files, label, label_to_id[label], test_site, "test", True, root))
        else:
            target_files = find_audio_files(site_known_dir(data_root, test_site) / label)
            rows.extend(_rows_for_files(target_files, label, label_to_id[label], test_site, "test", True, root))

    if train_site != test_site and (unknown_val_fraction > 0 or unknown_oe_fraction > 0):
        calibration_labels = cfg.get("calibration_unknown_labels")
        if calibration_labels is None:
            calibration_root = site_unknown_dir(data_root, train_site)
            calibration_labels = sorted(p.name for p in calibration_root.iterdir() if p.is_dir())
        for label_idx, label in enumerate(calibration_labels):
            files = find_audio_files(site_unknown_dir(data_root, train_site) / label)
            rng = random.Random(seed + 20_000 + label_idx)
            shuffled = list(files)
            rng.shuffle(shuffled)
            n_oe = int(round(len(shuffled) * unknown_oe_fraction))
            n_val = int(round(len(shuffled) * unknown_val_fraction))
            n_oe = min(n_oe, len(shuffled))
            n_val = min(n_val, max(0, len(shuffled) - n_oe))
            rows.extend(_rows_for_files(sorted(shuffled[:n_oe]), label, -1, train_site, "oe", False, root))
            rows.extend(_rows_for_files(sorted(shuffled[n_oe:n_oe + n_val]), label, -1, train_site, "val", False, root))

    for label_idx, label in enumerate(unknown_labels):
        files = find_audio_files(site_unknown_dir(data_root, test_site) / label)
        if train_site == test_site:
            rng = random.Random(seed + 10_000 + label_idx)
            shuffled = list(files)
            rng.shuffle(shuffled)
            n_oe = int(round(len(shuffled) * unknown_oe_fraction))
            n_val = int(round(len(shuffled) * unknown_val_fraction))
            n_oe = min(n_oe, len(shuffled))
            n_val = min(n_val, max(0, len(shuffled) - n_oe))
            oe_files = sorted(shuffled[:n_oe])
            val_files = sorted(shuffled[n_oe:n_oe + n_val])
            test_files = sorted(shuffled[n_oe + n_val:])
        else:
            oe_files = []
            val_files = []
            test_files = files
        rows.extend(_rows_for_files(oe_files, label, -1, test_site, "oe", False, root))
        rows.extend(_rows_for_files(val_files, label, -1, test_site, "val", False, root))
        rows.extend(_rows_for_files(test_files, label, -1, test_site, "test", False, root))

    return rows


def write_manifest(rows: list[ManifestRow], path: str | Path) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = Path.cwd() / path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filepath", "label", "label_id", "site", "split", "is_known"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "filepath": row.filepath,
                    "label": row.label,
                    "label_id": row.label_id,
                    "site": row.site,
                    "split": row.split,
                    "is_known": int(row.is_known),
                }
            )
    return path


def summarize_manifest(rows: list[ManifestRow]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for row in rows:
        key = f"{row.split}:{'known' if row.is_known else 'unknown'}"
        summary.setdefault(key, {})
        summary[key][row.label] = summary[key].get(row.label, 0) + 1
    return summary
