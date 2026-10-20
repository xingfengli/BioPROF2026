from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


SUPPORTED_BMZ_MODELS = {
    "birdnet": "BirdNET",
    "perch": "Perch",
    "perch2": "Perch2",
}

SUPPORTED_EMBEDDING_AGGREGATIONS = {"mean", "max", "concat"}


def normalize_bmz_model_name(name: str) -> str:
    normalized = name.strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "birdnet": "birdnet",
        "perch": "perch",
        "perch2": "perch2",
    }
    if normalized not in aliases:
        valid = ", ".join(sorted(SUPPORTED_BMZ_MODELS))
        raise ValueError(f"Unsupported BMZ embedding model '{name}'. Valid models: {valid}")
    return aliases[normalized]


def normalize_embedding_aggregation(name: str | None) -> str:
    normalized = (name or "mean").strip().lower().replace("-", "_")
    aliases = {
        "mean": "mean",
        "mean_pool": "mean",
        "mean_pooling": "mean",
        "avg": "mean",
        "average": "mean",
        "max": "max",
        "max_pool": "max",
        "max_pooling": "max",
        "concat": "concat",
        "concatenate": "concat",
        "concat2": "concat",
    }
    if normalized not in aliases:
        valid = ", ".join(sorted(SUPPORTED_EMBEDDING_AGGREGATIONS))
        raise ValueError(f"Unsupported embedding aggregation '{name}'. Valid aggregations: {valid}")
    return aliases[normalized]


def resolve_embedding_cache_dir(cfg: dict, root: str | Path | None = None) -> Path:
    embedding_cfg = cfg.get("embedding")
    if not embedding_cfg:
        raise ValueError("Missing 'embedding' config for BMZ embedding backbone.")
    cache_dir = Path(embedding_cfg["cache_dir"])
    if not cache_dir.is_absolute():
        cache_dir = (Path.cwd() if root is None else Path(root)) / cache_dir
    return cache_dir


def canonical_filepath(path: str | Path, root: str | Path | None = None) -> str:
    path = Path(path)
    if not path.is_absolute():
        path = (Path.cwd() if root is None else Path(root)) / path
    return str(path.resolve())


def embedding_cache_path(cache_dir: str | Path, filepath: str | Path) -> Path:
    key = hashlib.sha256(str(filepath).encode("utf-8")).hexdigest()
    return Path(cache_dir) / key[:2] / f"{key}.npy"


def load_cached_embedding(cache_dir: str | Path, filepath: str | Path, l2_normalize: bool = False) -> np.ndarray:
    path = embedding_cache_path(cache_dir, filepath)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing cached embedding for '{filepath}': {path}. "
            "Run scripts/precompute_embeddings.py before training."
        )
    embedding = np.asarray(np.load(path), dtype=np.float32).reshape(-1)
    if l2_normalize:
        norm = float(np.linalg.norm(embedding))
        if norm > 0:
            embedding = embedding / norm
    return embedding


def create_bmz_model(name: str, device: str = "cpu"):
    model_name = normalize_bmz_model_name(name)
    if device == "cpu":
        # TensorFlow probes visible GPUs during Perch2 SavedModel loading even
        # for the CPU model. Hide them before BMZ imports TensorFlow so CPU
        # extraction remains usable on hosts with an older NVIDIA driver.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        import bioacoustics_model_zoo as bmz
    except ImportError as exc:
        raise RuntimeError(
            "bioacoustics-model-zoo is required for BirdNET, Perch, and Perch2 embeddings. "
            "Install the appropriate BMZ extras before precomputing embeddings."
        ) from exc

    class_name = SUPPORTED_BMZ_MODELS[model_name]
    factory = getattr(bmz, class_name, None)
    if factory is None:
        raise RuntimeError(
            f"Installed bioacoustics-model-zoo does not expose '{class_name}'. "
            "Install a recent BMZ release with the requested model extra."
        )
    parameters = inspect.signature(factory).parameters
    kwargs = {"device": device} if "device" in parameters else {}
    return factory(**kwargs)


def _call_supported(function, *args, **kwargs):
    parameters = inspect.signature(function).parameters
    accepts_kwargs = any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())
    accepted = kwargs if accepts_kwargs else {key: value for key, value in kwargs.items() if key in parameters}
    return function(*args, **accepted)


def _run_embed(model, filepaths: list[str], batch_size: int, num_workers: int):
    for method_name in ("embed", "generate_embeddings", "embed_files"):
        method = getattr(model, method_name, None)
        if callable(method):
            return _call_supported(
                method,
                filepaths,
                batch_size=batch_size,
                num_workers=num_workers,
            )
    raise RuntimeError(
        f"{type(model).__name__} does not expose an embedding method. "
        "Expected one of: embed, generate_embeddings, embed_files."
    )


def _embedding_columns(frame: pd.DataFrame) -> list:
    numeric = list(frame.select_dtypes(include=[np.number]).columns)
    excluded = {"start_time", "end_time", "start", "end", "duration"}
    columns = [column for column in numeric if str(column).lower() not in excluded]
    if not columns:
        raise ValueError("BMZ embedding table did not contain numeric embedding columns.")
    return columns


def _embedding_window_order(frame: pd.DataFrame) -> list[float]:
    if isinstance(frame.index, pd.MultiIndex):
        names = list(frame.index.names)
        for name in ("start_time", "start"):
            if name in names:
                return [float(value) for value in frame.index.get_level_values(name)]
        if frame.index.nlevels >= 2:
            try:
                return [float(value) for value in frame.index.get_level_values(1)]
            except (TypeError, ValueError):
                pass
    for name in ("start_time", "start"):
        if name in frame.columns:
            return [float(value) for value in frame[name].to_numpy()]
    return [float(index) for index in range(len(frame))]


def _aggregate_window_vectors(vectors, aggregation: str = "mean", max_windows: int | None = None) -> np.ndarray:
    aggregation = normalize_embedding_aggregation(aggregation)
    windows = np.asarray(vectors, dtype=np.float32)
    if windows.ndim == 1:
        windows = windows.reshape(1, -1)
    if windows.ndim != 2 or windows.shape[0] == 0:
        raise ValueError(f"Expected one or more 2D window embeddings, got shape {windows.shape}.")

    if aggregation == "mean":
        return windows.mean(axis=0).astype(np.float32)
    if aggregation == "max":
        return windows.max(axis=0).astype(np.float32)

    if max_windows is None:
        raise ValueError("embedding.max_windows is required when embedding.aggregation='concat'.")
    max_windows = int(max_windows)
    if max_windows <= 0:
        raise ValueError("embedding.max_windows must be > 0 when embedding.aggregation='concat'.")

    dim = int(windows.shape[1])
    selected = windows[:max_windows]
    if selected.shape[0] < max_windows:
        padding = np.zeros((max_windows - selected.shape[0], dim), dtype=np.float32)
        selected = np.concatenate([selected, padding], axis=0)
    return selected.reshape(max_windows * dim).astype(np.float32)


def _frame_groups(
    frame: pd.DataFrame,
    aggregation: str = "mean",
    max_windows: int | None = None,
) -> dict[str, np.ndarray]:
    columns = _embedding_columns(frame)
    if isinstance(frame.index, pd.MultiIndex):
        keys = frame.index.get_level_values(0)
    else:
        keys = frame.index

    groups: dict[str, list[tuple[float, int, np.ndarray]]] = {}
    orders = _embedding_window_order(frame)
    for row_index, (key, order, values) in enumerate(zip(keys, orders, frame[columns].to_numpy(dtype=np.float32))):
        try:
            canonical = canonical_filepath(key)
        except TypeError:
            return {}
        groups.setdefault(canonical, []).append((float(order), row_index, values))

    aggregated = {}
    for filepath, rows in groups.items():
        rows = sorted(rows, key=lambda item: (item[0], item[1]))
        vectors = np.stack([values for _, _, values in rows], axis=0)
        aggregated[filepath] = _aggregate_window_vectors(vectors, aggregation, max_windows)
    return aggregated


def aggregate_bmz_embeddings(
    result,
    requested_filepaths: list[str],
    aggregation: str = "mean",
    max_windows: int | None = None,
) -> dict[str, np.ndarray]:
    requested = [canonical_filepath(path) for path in requested_filepaths]
    aggregation = normalize_embedding_aggregation(aggregation)

    if isinstance(result, pd.Series):
        result = result.to_frame().T

    if isinstance(result, pd.DataFrame):
        groups = _frame_groups(result, aggregation, max_windows)
        if all(filepath in groups for filepath in requested):
            return {filepath: groups[filepath] for filepath in requested}

        values = result.select_dtypes(include=[np.number]).to_numpy(dtype=np.float32)
        if len(values) == len(requested):
            return {
                filepath: _aggregate_window_vectors(values[index], aggregation, max_windows)
                for index, filepath in enumerate(requested)
            }

    values = np.asarray(result, dtype=np.float32)
    if values.ndim == 1 and len(requested) == 1:
        return {requested[0]: _aggregate_window_vectors(values, aggregation, max_windows)}
    if values.ndim == 2 and values.shape[0] == len(requested):
        return {
            filepath: _aggregate_window_vectors(values[index], aggregation, max_windows)
            for index, filepath in enumerate(requested)
        }

    raise ValueError(
        "Could not align BMZ embeddings with requested audio files. "
        f"Requested {len(requested)} files, received result type {type(result).__name__}."
    )


def precompute_manifest_embeddings(
    cfg: dict,
    chunk_size: int = 32,
    batch_size: int = 16,
    num_workers: int = 1,
) -> dict[str, int | str]:
    manifest_path = Path(cfg["manifest_path"])
    if not manifest_path.is_absolute():
        manifest_path = Path.cwd() / manifest_path
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}. Run make_splits.py first.")

    embedding_cfg = cfg["embedding"]
    model_name = normalize_bmz_model_name(embedding_cfg["model"])
    aggregation = normalize_embedding_aggregation(embedding_cfg.get("aggregation", "mean"))
    max_windows = embedding_cfg.get("max_windows")
    max_windows = int(max_windows) if max_windows is not None else None
    cache_dir = resolve_embedding_cache_dir(cfg)
    cache_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = cache_dir / "metadata.json"
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = {
            "model": model_name,
            "aggregation": aggregation,
            "max_windows": max_windows,
        }
        previous_signature = {key: previous.get(key) for key in expected}
        if any(previous_signature[key] is not None and previous_signature[key] != value for key, value in expected.items()):
            raise ValueError(
                f"Embedding cache '{cache_dir}' contains signature {previous_signature}, "
                f"but config requested {expected}."
            )

    manifest = pd.read_csv(manifest_path)
    filepaths = sorted({canonical_filepath(path) for path in manifest["filepath"].tolist()})
    pending = [path for path in filepaths if not embedding_cache_path(cache_dir, path).exists()]

    print(f"BMZ model: {model_name}")
    print(f"Aggregation: {aggregation}, max_windows: {max_windows}")
    print(f"Cache directory: {cache_dir}")
    print(f"Audio files: {len(filepaths)}, cached: {len(filepaths) - len(pending)}, pending: {len(pending)}")

    device = str(embedding_cfg.get("device", "cpu"))
    model = create_bmz_model(model_name, device=device) if pending else None
    written = 0
    dimensions: set[int] = set()
    for start in range(0, len(pending), chunk_size):
        chunk = pending[start:start + chunk_size]
        result = _run_embed(model, chunk, batch_size=batch_size, num_workers=num_workers)
        embeddings = aggregate_bmz_embeddings(result, chunk, aggregation=aggregation, max_windows=max_windows)
        for filepath in chunk:
            embedding = np.asarray(embeddings[filepath], dtype=np.float32).reshape(-1)
            dimensions.add(int(embedding.shape[0]))
            cache_path = embedding_cache_path(cache_dir, filepath)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, embedding)
            written += 1
        print(f"Cached {min(start + len(chunk), len(pending))}/{len(pending)} pending files")

    if not dimensions and filepaths:
        dimensions.add(int(load_cached_embedding(cache_dir, filepaths[0]).shape[0]))

    metadata = {
        "model": model_name,
        "aggregation": aggregation,
        "max_windows": max_windows,
        "manifest_path": str(manifest_path),
        "cache_dir": str(cache_dir),
        "device": device,
        "audio_files": len(filepaths),
        "written": written,
        "dimensions": sorted(dimensions),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
