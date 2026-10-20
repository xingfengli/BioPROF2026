#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from birdosr.config import load_config
from birdosr.splits import build_manifest, write_manifest
from birdosr.train import run_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--remake-manifest", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    manifest = Path(cfg["manifest_path"])
    if not manifest.is_absolute():
        manifest = Path.cwd() / manifest
    if args.remake_manifest or not manifest.exists():
        rows = build_manifest(cfg)
        write_manifest(rows, manifest)
    checkpoint = run_training(cfg)
    print(f"Best checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()

