#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from birdosr.config import load_config
from birdosr.evaluate import evaluate_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = evaluate_checkpoint(cfg, args.checkpoint)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

