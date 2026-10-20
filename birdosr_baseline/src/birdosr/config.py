from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["_config_path"] = str(path)
    return cfg


def resolve_path(path: str | Path, root: str | Path | None = None) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    if root is None:
        root = Path.cwd()
    return Path(root) / path


def ensure_output_dir(cfg: dict[str, Any]) -> Path:
    out = resolve_path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    return out

