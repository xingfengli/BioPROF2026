#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from birdosr.config import load_config
from birdosr.splits import build_manifest, summarize_manifest, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    rows = build_manifest(cfg)
    manifest_path = write_manifest(rows, cfg["manifest_path"])
    print(f"Wrote {len(rows)} rows to {manifest_path}")
    print(json.dumps(summarize_manifest(rows), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

