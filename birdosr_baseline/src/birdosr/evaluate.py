from __future__ import annotations

from pathlib import Path

import pandas as pd

from .audio import AudioConfig, LogMelFrontend
from .config import ensure_output_dir
from .datasets import AudioManifestDataset, EmbeddingManifestDataset
from .metrics import (
    UNKNOWN_ID,
    apply_classwise_threshold,
    apply_threshold,
    calibrate_classwise_tau,
    calibrate_subset_classwise_tau,
    calibrate_subset_tau,
    calibrate_tau,
    known_class_report,
    known_confusion_matrix,
    main_metrics,
    prediction_metrics,
    reconstruction_report,
    top_confusion_pairs,
    unknown_subset_acc_report,
    unknown_class_report,
    unknown_to_known_matrix,
)
from .models import SpectrogramClassifier, backbone_uses_cached_embeddings, model_expects_waveform
from .osr_training import attach_osr_modules
from .reproducibility import make_torch_generator, seed_everything, seed_worker


def _collate_eval(batch):
    import torch

    result = {
        "label_id": torch.stack([item["label_id"] for item in batch], dim=0),
        "path": [item["path"] for item in batch],
        "label": [item["label"] for item in batch],
        "is_known": [item["is_known"] for item in batch],
    }
    if "embedding" in batch[0]:
        result["embedding"] = torch.stack([item["embedding"] for item in batch], dim=0)
    else:
        result["waveform"] = torch.stack([item["waveform"] for item in batch], dim=0)
    return result


def score_split(
    cfg: dict,
    checkpoint_path: str | Path,
    split: str,
    return_embeddings: bool = False,
    include_embedding_scores: bool = False,
):
    import torch
    from torch.utils.data import DataLoader

    seed = int(cfg.get("seed", 26))
    deterministic = bool(cfg.get("training", {}).get("deterministic", True))
    seed_everything(seed, deterministic=deterministic)

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path.cwd() / checkpoint_path
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    label_names = ckpt.get("label_names", cfg["known_labels"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_cfg = AudioConfig(**cfg["audio"])
    manifest = Path(cfg["manifest_path"])
    if not manifest.is_absolute():
        manifest = Path.cwd() / manifest
    model_cfg = cfg["model"]
    osr_cfg = cfg.get("osr", {})
    uses_embeddings = backbone_uses_cached_embeddings(model_cfg["backbone"])
    if uses_embeddings:
        ds = EmbeddingManifestDataset(manifest, split, cfg, known_only=False)
    else:
        ds = AudioManifestDataset(manifest, split, audio_cfg, known_only=False)
    loader = DataLoader(
        ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"].get("num_workers", 4),
        collate_fn=_collate_eval,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=make_torch_generator(seed + 10),
    )

    model = SpectrogramClassifier(
        model_cfg["backbone"],
        num_classes=len(label_names),
        pretrained=False,
        dropout=float(model_cfg.get("dropout", 0.2)),
        in_channels=1,
        embedding_dim=getattr(ds, "embedding_dim", None),
    ).to(device)
    attach_osr_modules(model, cfg, len(label_names))
    model.to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    frontend = None if uses_embeddings or model_expects_waveform(model) else LogMelFrontend(audio_cfg).to(device)

    rows = []
    embedding_batches = []
    needs_embeddings = bool(return_embeddings or include_embedding_scores)
    with torch.no_grad():
        for batch in loader:
            if uses_embeddings:
                inputs = batch["embedding"].to(device)
            else:
                waveform = batch["waveform"].to(device)
                inputs = waveform if model_expects_waveform(model) else frontend(waveform).to(device)
            embeddings = None
            if needs_embeddings:
                if not hasattr(model, "extract_embedding"):
                    raise ValueError(
                        f"Model backbone '{model_cfg['backbone']}' does not expose embeddings for embedding-norm OSR scores."
                    )
                embeddings = model.extract_embedding(inputs)
                if hasattr(model, "classify_embedding"):
                    logits = model.classify_embedding(embeddings)
                else:
                    logits = model(inputs)
                if return_embeddings:
                    embedding_batches.append(embeddings.detach().cpu().float().numpy())
            else:
                logits = model(inputs)
            scores, preds = logits.max(dim=1)
            probabilities = logits.softmax(dim=1)
            msp = probabilities.max(dim=1).values
            msp_temperature = float(osr_cfg.get("msp_temperature", osr_cfg.get("temperature", 1.0)))
            if msp_temperature <= 0:
                raise ValueError("osr.msp_temperature must be > 0.")
            temperature_probabilities = (logits / msp_temperature).softmax(dim=1)
            tmsp = temperature_probabilities.max(dim=1).values
            energy = logits.logsumexp(dim=1)
            logit_norm = logits.float().norm(dim=1)
            feature_norm = None
            premax = None
            proto_selected = None
            proto_nearest = None
            rpl_selected = None
            rpl_nearest = None
            if embeddings is not None:
                norm_epsilon = float(osr_cfg.get("norm_epsilon", 1e-12))
                premax_alpha = float(osr_cfg.get("premax_alpha", 1.0))
                feature_norm = embeddings.float().norm(dim=1)
                premax = scores / feature_norm.clamp_min(norm_epsilon).pow(premax_alpha)
                osr_head = getattr(model, "osr_head", None)
                if osr_head is not None and hasattr(osr_head, "distances"):
                    distances = osr_head.distances(embeddings)
                    selected = distances[torch.arange(len(preds), device=preds.device), preds]
                    nearest = distances.min(dim=1).values
                    head_type = getattr(osr_head, "osr_head_type", "")
                    if head_type == "protosphere":
                        proto_selected = -selected
                        proto_nearest = -nearest
                    elif head_type == "rpl":
                        rpl_selected = selected
                        rpl_nearest = nearest
            topk = logits.topk(k=min(2, logits.shape[1]), dim=1).values
            if topk.shape[1] > 1:
                margin = topk[:, 0] - topk[:, 1]
            else:
                margin = topk[:, 0]
            for i in range(len(batch["path"])):
                row = {
                    "filepath": batch["path"][i],
                    "true_label": batch["label"][i],
                    "is_known": int(batch["is_known"][i]),
                    "y_true": int(batch["label_id"][i].item()),
                    "y_pred_known": int(preds[i].item()),
                    "score": float(scores[i].item()),
                    "score_mls": float(scores[i].item()),
                    "score_energy": float(energy[i].item()),
                    "score_msp": float(msp[i].item()),
                    "score_tmsp": float(tmsp[i].item()),
                    "score_margin": float(margin[i].item()),
                    "score_logit_norm": float(logit_norm[i].item()),
                }
                if feature_norm is not None and premax is not None:
                    row["score_feature_norm"] = float(feature_norm[i].item())
                    row["score_premax"] = float(premax[i].item())
                if proto_selected is not None and proto_nearest is not None:
                    row["score_proto_distance"] = float(proto_selected[i].item())
                    row["score_proto_nearest"] = float(proto_nearest[i].item())
                if rpl_selected is not None and rpl_nearest is not None:
                    row["score_rpl_distance"] = float(rpl_selected[i].item())
                    row["score_rpl_nearest"] = float(rpl_nearest[i].item())
                rows.append(row)
    df = pd.DataFrame(rows)
    if return_embeddings:
        import numpy as np

        return df, np.concatenate(embedding_batches, axis=0)
    return df


def _score_column(score_name: str) -> str:
    aliases = {
        "mls": "score_mls",
        "max_logit": "score_mls",
        "energy": "score_energy",
        "msp": "score_msp",
        "max_softmax": "score_msp",
        "tmsp": "score_tmsp",
        "t_msp": "score_tmsp",
        "temperature_msp": "score_tmsp",
        "temperature_scaled_msp": "score_tmsp",
        "margin": "score_margin",
        "feature_norm": "score_feature_norm",
        "embedding_norm": "score_feature_norm",
        "birdmae_norm": "score_feature_norm",
        "logit_norm": "score_logit_norm",
        "logits_norm": "score_logit_norm",
        "premax": "score_premax",
        "postmax_like": "score_premax",
        "premx": "score_premax",
        "premax_mls": "score_premax",
        "mls_over_feature_norm": "score_premax",
        "normalized_mls": "score_premax",
        "proto_distance": "score_proto_distance",
        "prototype_distance": "score_proto_distance",
        "protosphere": "score_proto_distance",
        "proto_nearest": "score_proto_nearest",
        "prototype_nearest": "score_proto_nearest",
        "rpl_distance": "score_rpl_distance",
        "reciprocal_distance": "score_rpl_distance",
        "rpl_nearest": "score_rpl_nearest",
    }
    normalized = score_name.strip().lower().replace("-", "_")
    if normalized not in aliases:
        valid = ", ".join(sorted(aliases))
        raise ValueError(f"Unsupported OSR score '{score_name}'. Valid scores: {valid}")
    return aliases[normalized]


def _set_primary_score(df: pd.DataFrame, score_name: str) -> pd.DataFrame:
    df = df.copy()
    column = _score_column(score_name)
    if column not in df.columns:
        if score_name == "mls" and "score" in df.columns:
            column = "score"
        else:
            raise ValueError(f"Missing score column '{column}'.")
    df["score"] = df[column].astype(float)
    return df


def _score_requires_embedding(score_name: str) -> bool:
    return _score_column(score_name) in {
        "score_feature_norm",
        "score_premax",
        "score_proto_distance",
        "score_proto_nearest",
        "score_rpl_distance",
        "score_rpl_nearest",
    }


def _osr_requires_embedding_scores(osr_method: str, osr_cfg: dict, score_name: str) -> bool:
    if osr_method in {
        "multi_score_fusion",
        "fusion_subset_tau",
        "global_fusion_classwise_tau",
        "learned_fusion_tau",
        "learned_fusion_subset_tau",
        "learned_fusion_classwise_tau",
        "learned_fusion_subset_classwise_tau",
        "evt_fusion_classwise_tau",
        "classwise_fusion_tau",
    }:
        score_names = list(osr_cfg.get("scores", ["mls", "energy", "msp", "margin"]))
    else:
        score_names = [score_name]
    return any(_score_requires_embedding(name) for name in score_names)


def _fusion_score(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    score_names: list[str],
    weights: list[float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not score_names:
        raise ValueError("multi_score_fusion requires at least one score name.")
    if weights is None:
        weights = [1.0] * len(score_names)
    if len(weights) != len(score_names):
        raise ValueError("multi_score_fusion scores and weights must have the same length.")

    val_df = val_df.copy()
    test_df = test_df.copy()
    val_score = 0.0
    test_score = 0.0
    rows = []
    weight_norm = sum(abs(float(weight)) for weight in weights) or 1.0
    for score_name, weight in zip(score_names, weights):
        column = _score_column(score_name)
        if column not in val_df.columns or column not in test_df.columns:
            raise ValueError(f"Missing score column '{column}' for multi_score_fusion.")
        mean = float(val_df[column].mean())
        std = float(val_df[column].std())
        if std == 0 or pd.isna(std):
            std = 1.0
        weight = float(weight)
        val_component = (val_df[column].astype(float) - mean) / std
        test_component = (test_df[column].astype(float) - mean) / std
        val_score = val_score + weight * val_component
        test_score = test_score + weight * test_component
        rows.append({"score": score_name, "column": column, "weight": weight, "mean": mean, "std": std})

    val_df["score"] = val_score / weight_norm
    test_df["score"] = test_score / weight_norm
    return val_df, test_df, pd.DataFrame(rows)


def _score_matrix(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    score_names: list[str],
) -> tuple:
    import numpy as np

    if not score_names:
        raise ValueError("Fusion scoring requires at least one score name.")
    columns = [_score_column(score_name) for score_name in score_names]
    missing = [column for column in columns if column not in val_df.columns or column not in test_df.columns]
    if missing:
        raise ValueError(f"Missing score columns for fusion scoring: {', '.join(missing)}")

    means = []
    stds = []
    val_components = []
    test_components = []
    rows = []
    for score_name, column in zip(score_names, columns):
        mean = float(val_df[column].mean())
        std = float(val_df[column].std())
        if std == 0 or pd.isna(std):
            std = 1.0
        means.append(mean)
        stds.append(std)
        val_components.append((val_df[column].astype(float).to_numpy() - mean) / std)
        test_components.append((test_df[column].astype(float).to_numpy() - mean) / std)
        rows.append({"score": score_name, "column": column, "mean": mean, "std": std})

    return np.stack(val_components, axis=1), np.stack(test_components, axis=1), rows


def _learned_fusion_score(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    score_names: list[str],
    seed: int,
    steps: int = 2000,
    learning_rate: float = 0.05,
    l2: float = 1e-3,
    nonnegative: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    import numpy as np

    x_val, x_test, rows = _score_matrix(val_df, test_df, score_names)
    y = (val_df["y_true"].to_numpy(dtype=int) != UNKNOWN_ID).astype(float)
    positives = y == 1.0
    negatives = ~positives
    if not positives.any() or not negatives.any():
        raise ValueError("learned_fusion_classwise_tau requires both known and unknown validation samples.")

    sample_weight = np.zeros_like(y, dtype=float)
    sample_weight[positives] = 0.5 / float(positives.sum())
    sample_weight[negatives] = 0.5 / float(negatives.sum())

    rng = np.random.default_rng(seed)
    weights = np.ones(x_val.shape[1], dtype=float) / float(max(x_val.shape[1], 1))
    weights = weights + rng.normal(0.0, 1e-4, size=weights.shape)
    bias = 0.0

    for _ in range(int(steps)):
        logits = x_val @ weights + bias
        logits = np.clip(logits, -50.0, 50.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        error = (probabilities - y) * sample_weight
        grad_w = x_val.T @ error + float(l2) * weights
        grad_b = float(error.sum())
        weights -= float(learning_rate) * grad_w
        bias -= float(learning_rate) * grad_b
        if nonnegative:
            weights = np.maximum(weights, 0.0)

    val_score = x_val @ weights + bias
    test_score = x_test @ weights + bias

    val_df = val_df.copy()
    test_df = test_df.copy()
    val_df["score"] = val_score
    test_df["score"] = test_score
    val_df["score_learned_fusion"] = val_score
    test_df["score_learned_fusion"] = test_score

    params = []
    for row, weight in zip(rows, weights):
        item = dict(row)
        item["weight"] = float(weight)
        item["bias"] = float(bias)
        item["learn_steps"] = int(steps)
        item["learn_lr"] = float(learning_rate)
        item["learn_l2"] = float(l2)
        item["learn_nonnegative"] = int(bool(nonnegative))
        params.append(item)
    return val_df, test_df, pd.DataFrame(params)


def _gpd_loss_quantile(losses, probability: float, tail_fraction: float, min_tail: int) -> tuple[float, dict]:
    import numpy as np

    losses = np.asarray(losses, dtype=float)
    losses = losses[np.isfinite(losses)]
    if len(losses) == 0:
        return float("nan"), {"tail_model": "empty"}

    probability = float(np.clip(probability, 0.0, 0.999999))
    tail_fraction = float(np.clip(tail_fraction, 1e-6, 1.0))
    q0 = max(0.0, min(0.999999, 1.0 - tail_fraction))
    empirical = float(np.quantile(losses, probability))
    if probability <= q0 or len(losses) < max(int(min_tail), 2):
        return empirical, {"tail_model": "empirical", "tail_threshold": float(np.quantile(losses, q0))}

    threshold = float(np.quantile(losses, q0))
    exceedances = losses[losses >= threshold] - threshold
    exceedances = exceedances[exceedances > 0]
    if len(exceedances) < max(int(min_tail), 2):
        return empirical, {"tail_model": "empirical", "tail_threshold": threshold}

    mean = float(exceedances.mean())
    variance = float(exceedances.var(ddof=1)) if len(exceedances) > 1 else 0.0
    if mean <= 0 or variance <= 0:
        return empirical, {"tail_model": "empirical", "tail_threshold": threshold}

    xi = 0.5 * (1.0 - (mean * mean / variance))
    xi = float(np.clip(xi, -0.5, 0.5))
    beta = float(max(mean * (1.0 - xi), 1e-12))
    conditional_probability = (probability - q0) / max(1.0 - q0, 1e-12)
    conditional_probability = float(np.clip(conditional_probability, 0.0, 0.999999))
    if abs(xi) < 1e-6:
        excess_quantile = beta * (-np.log(max(1.0 - conditional_probability, 1e-12)))
    else:
        excess_quantile = beta / xi * ((max(1.0 - conditional_probability, 1e-12)) ** (-xi) - 1.0)
    return float(threshold + excess_quantile), {
        "tail_model": "gpd",
        "tail_threshold": threshold,
        "tail_fraction": tail_fraction,
        "tail_size": int(len(exceedances)),
        "xi": xi,
        "beta": beta,
    }


def _classwise_evt_thresholds(
    y_true,
    y_pred_known,
    scores,
    num_classes: int,
    target_tpr: float,
    tail_fraction: float,
    min_tail: int,
    fallback_thresholds,
) -> tuple:
    import numpy as np

    y_true = np.asarray(y_true, dtype=int)
    y_pred_known = np.asarray(y_pred_known, dtype=int)
    scores = np.asarray(scores, dtype=float)
    fallback_thresholds = np.asarray(fallback_thresholds, dtype=float)
    thresholds = np.full(num_classes, float("nan"), dtype=float)
    rows = []

    for class_id in range(num_classes):
        positive = (y_pred_known == class_id) & (y_true == class_id)
        class_scores = scores[positive]
        if len(class_scores) == 0:
            thresholds[class_id] = fallback_thresholds[class_id]
            rows.append(
                {
                    "class_id": class_id,
                    "target_tpr": float(target_tpr),
                    "support": 0,
                    "score_tau": thresholds[class_id],
                    "tail_model": "fallback",
                }
            )
            continue
        loss_threshold, info = _gpd_loss_quantile(
            -class_scores,
            probability=float(target_tpr),
            tail_fraction=tail_fraction,
            min_tail=min_tail,
        )
        score_threshold = -float(loss_threshold) if np.isfinite(loss_threshold) else fallback_thresholds[class_id]
        thresholds[class_id] = score_threshold
        row = {
            "class_id": class_id,
            "target_tpr": float(target_tpr),
            "support": int(len(class_scores)),
            "score_tau": float(score_threshold),
        }
        row.update(info)
        rows.append(row)
    return thresholds, pd.DataFrame(rows)


def _evt_classwise_calibration(
    val_df: pd.DataFrame,
    num_classes: int,
    target_tprs: list[float],
    tail_fraction: float,
    min_tail: int,
) -> tuple:
    y_true = val_df["y_true"].to_numpy()
    y_pred_known = val_df["y_pred_known"].to_numpy()
    scores = val_df["score"].to_numpy()
    fallback = calibrate_classwise_tau(y_true, y_pred_known, scores, num_classes).thresholds

    best_thresholds = None
    best_pred = None
    best_metrics = None
    best_params = None
    best_target_tpr = None
    for target_tpr in target_tprs:
        thresholds, params = _classwise_evt_thresholds(
            y_true,
            y_pred_known,
            scores,
            num_classes,
            float(target_tpr),
            tail_fraction,
            min_tail,
            fallback,
        )
        y_pred = apply_classwise_threshold(y_pred_known, scores, thresholds)
        metrics = prediction_metrics(y_true, y_pred_known, scores, y_pred, _finite_tau_summary(thresholds))
        if best_metrics is None or metrics["ACC"] > best_metrics["ACC"]:
            best_thresholds = thresholds
            best_pred = y_pred
            best_metrics = metrics
            best_params = params
            best_target_tpr = float(target_tpr)

    if best_params is not None:
        best_params = best_params.copy()
        best_params["selected_target_tpr"] = best_target_tpr
        best_params["val_ACC_k"] = best_metrics["ACC_k"]
        best_params["val_ACC_u"] = best_metrics["ACC_u"]
        best_params["val_ACC"] = best_metrics["ACC"]
    return best_thresholds, best_pred, best_params, best_target_tpr


def _classwise_fusion_score(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    score_names: list[str],
    label_names: list[str],
    weights: list[float] | None = None,
    min_samples: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    import numpy as np

    if not score_names:
        raise ValueError("classwise_fusion_tau requires at least one score name.")
    if weights is None:
        weights = [1.0] * len(score_names)
    if len(weights) != len(score_names):
        raise ValueError("classwise_fusion_tau scores and weights must have the same length.")

    columns = [_score_column(score_name) for score_name in score_names]
    missing = [column for column in columns if column not in val_df.columns or column not in test_df.columns]
    if missing:
        raise ValueError(f"Missing score columns for classwise_fusion_tau: {', '.join(missing)}")

    val_df = val_df.copy()
    test_df = test_df.copy()
    val_score = np.zeros(len(val_df), dtype=float)
    test_score = np.zeros(len(test_df), dtype=float)
    params = []
    weight_norm = sum(abs(float(weight)) for weight in weights) or 1.0

    global_stats = {}
    for score_name, column in zip(score_names, columns):
        mean = float(val_df[column].mean())
        std = float(val_df[column].std())
        if std == 0 or pd.isna(std):
            std = 1.0
        global_stats[column] = (mean, std)

    for class_id, label in enumerate(label_names):
        val_mask = val_df["y_pred_known"].astype(int).to_numpy() == class_id
        test_mask = test_df["y_pred_known"].astype(int).to_numpy() == class_id
        use_global = int(val_mask.sum()) < int(min_samples)
        for score_name, column, weight in zip(score_names, columns, weights):
            if use_global:
                mean, std = global_stats[column]
            else:
                mean = float(val_df.loc[val_mask, column].mean())
                std = float(val_df.loc[val_mask, column].std())
                if std == 0 or pd.isna(std):
                    std = global_stats[column][1]
            weight = float(weight)
            if val_mask.any():
                val_score[val_mask] += weight * ((val_df.loc[val_mask, column].astype(float).to_numpy() - mean) / std)
            if test_mask.any():
                test_score[test_mask] += weight * ((test_df.loc[test_mask, column].astype(float).to_numpy() - mean) / std)
            params.append(
                {
                    "class_id": class_id,
                    "label": label,
                    "score": score_name,
                    "column": column,
                    "weight": weight,
                    "mean": mean,
                    "std": std,
                    "support": int(val_mask.sum()),
                    "fallback_global": int(use_global),
                }
            )

    val_df["score"] = val_score / weight_norm
    test_df["score"] = test_score / weight_norm
    val_df["score_classwise_fusion"] = val_df["score"]
    test_df["score_classwise_fusion"] = test_df["score"]
    return val_df, test_df, pd.DataFrame(params)


def _finite_tau_summary(thresholds) -> float:
    import numpy as np

    thresholds = np.asarray(thresholds, dtype=float)
    finite = thresholds[np.isfinite(thresholds)]
    return float(finite.mean()) if len(finite) else float("nan")


def _normalize_embedding_matrix(embeddings, enabled: bool):
    import numpy as np

    embeddings = np.asarray(embeddings, dtype=np.float32)
    if not enabled:
        return embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-12)


def _fit_classwise_gaussians(
    train_df: pd.DataFrame,
    train_embeddings,
    num_classes: int,
    regularization: float,
    shrinkage: float,
) -> tuple:
    import numpy as np

    embeddings = np.asarray(train_embeddings, dtype=np.float32)
    labels = train_df["y_true"].to_numpy(dtype=int)
    global_variance = embeddings.var(axis=0, ddof=1) if len(embeddings) > 1 else np.ones(embeddings.shape[1])
    global_variance = np.maximum(global_variance, regularization)

    means = []
    inv_variances = []
    rows = []
    for class_id in range(num_classes):
        class_embeddings = embeddings[labels == class_id]
        if len(class_embeddings) == 0:
            raise ValueError(f"No training embeddings found for class id {class_id}.")
        mean = class_embeddings.mean(axis=0)
        if len(class_embeddings) > 1:
            variance = class_embeddings.var(axis=0, ddof=1)
        else:
            variance = global_variance.copy()
        variance = (1.0 - shrinkage) * variance + shrinkage * global_variance
        variance = np.maximum(variance, regularization)
        means.append(mean)
        inv_variances.append(1.0 / variance)
        rows.append(
            {
                "class_id": class_id,
                "support": int(len(class_embeddings)),
                "mean_norm": float(np.linalg.norm(mean)),
                "avg_variance": float(variance.mean()),
                "min_variance": float(variance.min()),
                "max_variance": float(variance.max()),
            }
        )

    return np.stack(means), np.stack(inv_variances), pd.DataFrame(rows)


def _apply_classwise_gaussian_scores(
    df: pd.DataFrame,
    embeddings,
    means,
    inv_variances,
    label_names: list[str],
) -> pd.DataFrame:
    import numpy as np

    embeddings = np.asarray(embeddings, dtype=np.float32)
    diff = embeddings[:, None, :] - means[None, :, :]
    distances = np.sum((diff * diff) * inv_variances[None, :, :], axis=2)
    y_pred_known = df["y_pred_known"].to_numpy(dtype=int)
    predicted = np.clip(y_pred_known, 0, len(label_names) - 1)
    selected_distances = distances[np.arange(len(df)), predicted]
    nearest = distances.argmin(axis=1)
    nearest_distances = distances[np.arange(len(df)), nearest]

    df = df.copy()
    df["gaussian_predicted_class"] = [label_names[int(idx)] for idx in predicted]
    df["gaussian_distance"] = selected_distances.astype(float)
    df["score_gaussian"] = (-selected_distances).astype(float)
    df["gaussian_nearest_class"] = [label_names[int(idx)] for idx in nearest]
    df["gaussian_nearest_distance"] = nearest_distances.astype(float)
    df["score_gaussian_nearest"] = (-nearest_distances).astype(float)
    df["score"] = df["score_gaussian"]
    return df


def evaluate_checkpoint(cfg: dict, checkpoint_path: str | Path) -> dict[str, Path]:
    out = ensure_output_dir(cfg)
    osr_cfg = cfg.get("osr", {})
    osr_method = str(osr_cfg.get("method", "mls")).strip().lower().replace("-", "_")
    score_name = str(osr_cfg.get("score", "mls"))
    fusion_params = None
    classwise_thresholds = None
    evt_params = None
    gaussian_params = None
    gaussian_model = None

    if osr_method == "classwise_gaussian":
        train_df, train_embeddings = score_split(cfg, checkpoint_path, "train", return_embeddings=True)
        val_df, val_embeddings = score_split(cfg, checkpoint_path, "val", return_embeddings=True)
        test_df, test_embeddings = score_split(cfg, checkpoint_path, "test", return_embeddings=True)
    else:
        include_embedding_scores = _osr_requires_embedding_scores(osr_method, osr_cfg, score_name)
        val_df = score_split(cfg, checkpoint_path, "val", include_embedding_scores=include_embedding_scores)
        test_df = score_split(cfg, checkpoint_path, "test", include_embedding_scores=include_embedding_scores)

    if osr_method == "multi_score_fusion":
        score_names = list(osr_cfg.get("scores", ["mls", "energy", "msp", "margin"]))
        weights = osr_cfg.get("weights")
        weights = [float(weight) for weight in weights] if weights is not None else None
        val_df, test_df, fusion_params = _fusion_score(val_df, test_df, score_names, weights)
        tau_result = calibrate_tau(
            val_df["y_true"].to_numpy(),
            val_df["y_pred_known"].to_numpy(),
            val_df["score"].to_numpy(),
        )
        test_pred = apply_threshold(test_df["y_pred_known"].to_numpy(), test_df["score"].to_numpy(), tau_result.tau)
        metrics = main_metrics(
            test_df["y_true"].to_numpy(),
            test_df["y_pred_known"].to_numpy(),
            test_df["score"].to_numpy(),
            tau_result.tau,
        )
        metrics["tau"] = tau_result.tau
    elif osr_method == "fusion_subset_tau":
        score_names = list(osr_cfg.get("scores", ["mls", "energy", "msp", "margin"]))
        weights = osr_cfg.get("weights")
        weights = [float(weight) for weight in weights] if weights is not None else None
        val_df, test_df, fusion_params = _fusion_score(val_df, test_df, score_names, weights)
        val_df["score_raw_fusion"] = val_df["score"]
        test_df["score_raw_fusion"] = test_df["score"]
        tau_result = calibrate_subset_tau(
            val_df["y_true"].to_numpy(),
            val_df["y_pred_known"].to_numpy(),
            val_df["score"].to_numpy(),
            val_df["true_label"].to_numpy(),
            subset_size=int(osr_cfg.get("unknown_subset_size", cfg.get("evaluation", {}).get("unknown_subset_size", 3))),
            max_candidates=int(osr_cfg.get("subset_tau_max_candidates", 200)),
            objective=str(osr_cfg.get("subset_tau_objective", "ACC_subset_mean")),
        )
        test_pred = apply_threshold(test_df["y_pred_known"].to_numpy(), test_df["score"].to_numpy(), tau_result.tau)
        metrics = main_metrics(
            test_df["y_true"].to_numpy(),
            test_df["y_pred_known"].to_numpy(),
            test_df["score"].to_numpy(),
            tau_result.tau,
        )
        metrics["tau"] = tau_result.tau
        metrics["subset_tau_objective"] = str(osr_cfg.get("subset_tau_objective", "ACC_subset_mean"))
    elif osr_method == "classwise_tau":
        val_df = _set_primary_score(val_df, score_name)
        test_df = _set_primary_score(test_df, score_name)
        tau_result = calibrate_classwise_tau(
            val_df["y_true"].to_numpy(),
            val_df["y_pred_known"].to_numpy(),
            val_df["score"].to_numpy(),
            len(cfg["known_labels"]),
        )
        classwise_thresholds = tau_result.thresholds
        test_pred = apply_classwise_threshold(
            test_df["y_pred_known"].to_numpy(),
            test_df["score"].to_numpy(),
            tau_result.thresholds,
        )
        metrics = prediction_metrics(
            test_df["y_true"].to_numpy(),
            test_df["y_pred_known"].to_numpy(),
            test_df["score"].to_numpy(),
            test_pred,
            _finite_tau_summary(tau_result.thresholds),
        )
    elif osr_method == "subset_tau":
        val_df = _set_primary_score(val_df, score_name)
        test_df = _set_primary_score(test_df, score_name)
        tau_result = calibrate_subset_tau(
            val_df["y_true"].to_numpy(),
            val_df["y_pred_known"].to_numpy(),
            val_df["score"].to_numpy(),
            val_df["true_label"].to_numpy(),
            subset_size=int(osr_cfg.get("unknown_subset_size", cfg.get("evaluation", {}).get("unknown_subset_size", 3))),
            max_candidates=int(osr_cfg.get("subset_tau_max_candidates", 200)),
            objective=str(osr_cfg.get("subset_tau_objective", "ACC_subset_mean")),
        )
        test_pred = apply_threshold(test_df["y_pred_known"].to_numpy(), test_df["score"].to_numpy(), tau_result.tau)
        metrics = main_metrics(
            test_df["y_true"].to_numpy(),
            test_df["y_pred_known"].to_numpy(),
            test_df["score"].to_numpy(),
            tau_result.tau,
        )
        metrics["tau"] = tau_result.tau
        metrics["subset_tau_objective"] = str(osr_cfg.get("subset_tau_objective", "ACC_subset_mean"))
    elif osr_method == "subset_classwise_tau":
        val_df = _set_primary_score(val_df, score_name)
        test_df = _set_primary_score(test_df, score_name)
        tau_result = calibrate_subset_classwise_tau(
            val_df["y_true"].to_numpy(),
            val_df["y_pred_known"].to_numpy(),
            val_df["score"].to_numpy(),
            val_df["true_label"].to_numpy(),
            len(cfg["known_labels"]),
            subset_size=int(osr_cfg.get("unknown_subset_size", cfg.get("evaluation", {}).get("unknown_subset_size", 3))),
            max_candidates=int(osr_cfg.get("subset_tau_max_candidates", 200)),
            passes=int(osr_cfg.get("subset_tau_passes", 2)),
            objective=str(osr_cfg.get("subset_tau_objective", "ACC_subset_mean")),
        )
        classwise_thresholds = tau_result.thresholds
        test_pred = apply_classwise_threshold(
            test_df["y_pred_known"].to_numpy(),
            test_df["score"].to_numpy(),
            tau_result.thresholds,
        )
        metrics = prediction_metrics(
            test_df["y_true"].to_numpy(),
            test_df["y_pred_known"].to_numpy(),
            test_df["score"].to_numpy(),
            test_pred,
            _finite_tau_summary(tau_result.thresholds),
        )
        metrics["subset_tau_objective"] = str(osr_cfg.get("subset_tau_objective", "ACC_subset_mean"))
    elif osr_method == "global_fusion_classwise_tau":
        score_names = list(osr_cfg.get("scores", ["mls", "energy", "msp", "margin"]))
        weights = osr_cfg.get("weights")
        weights = [float(weight) for weight in weights] if weights is not None else None
        val_df, test_df, fusion_params = _fusion_score(val_df, test_df, score_names, weights)
        val_df["score_global_fusion"] = val_df["score"]
        test_df["score_global_fusion"] = test_df["score"]
        tau_result = calibrate_classwise_tau(
            val_df["y_true"].to_numpy(),
            val_df["y_pred_known"].to_numpy(),
            val_df["score"].to_numpy(),
            len(cfg["known_labels"]),
        )
        classwise_thresholds = tau_result.thresholds
        test_pred = apply_classwise_threshold(
            test_df["y_pred_known"].to_numpy(),
            test_df["score"].to_numpy(),
            tau_result.thresholds,
        )
        metrics = prediction_metrics(
            test_df["y_true"].to_numpy(),
            test_df["y_pred_known"].to_numpy(),
            test_df["score"].to_numpy(),
            test_pred,
            _finite_tau_summary(tau_result.thresholds),
        )
    elif osr_method == "learned_fusion_tau":
        score_names = list(osr_cfg.get("scores", ["mls", "energy", "msp", "margin"]))
        learn_cfg = osr_cfg.get("learned_fusion", {})
        val_df, test_df, fusion_params = _learned_fusion_score(
            val_df,
            test_df,
            score_names,
            seed=int(cfg.get("seed", 26)),
            steps=int(learn_cfg.get("steps", osr_cfg.get("learn_steps", 2000))),
            learning_rate=float(learn_cfg.get("learning_rate", osr_cfg.get("learn_lr", 0.05))),
            l2=float(learn_cfg.get("l2", osr_cfg.get("learn_l2", 1e-3))),
            nonnegative=bool(learn_cfg.get("nonnegative", osr_cfg.get("learn_nonnegative", False))),
        )
        tau_result = calibrate_tau(
            val_df["y_true"].to_numpy(),
            val_df["y_pred_known"].to_numpy(),
            val_df["score"].to_numpy(),
        )
        test_pred = apply_threshold(test_df["y_pred_known"].to_numpy(), test_df["score"].to_numpy(), tau_result.tau)
        metrics = main_metrics(
            test_df["y_true"].to_numpy(),
            test_df["y_pred_known"].to_numpy(),
            test_df["score"].to_numpy(),
            tau_result.tau,
        )
        metrics["tau"] = tau_result.tau
    elif osr_method == "learned_fusion_subset_tau":
        score_names = list(osr_cfg.get("scores", ["mls", "energy", "msp", "margin"]))
        learn_cfg = osr_cfg.get("learned_fusion", {})
        val_df, test_df, fusion_params = _learned_fusion_score(
            val_df,
            test_df,
            score_names,
            seed=int(cfg.get("seed", 26)),
            steps=int(learn_cfg.get("steps", osr_cfg.get("learn_steps", 2000))),
            learning_rate=float(learn_cfg.get("learning_rate", osr_cfg.get("learn_lr", 0.05))),
            l2=float(learn_cfg.get("l2", osr_cfg.get("learn_l2", 1e-3))),
            nonnegative=bool(learn_cfg.get("nonnegative", osr_cfg.get("learn_nonnegative", False))),
        )
        tau_result = calibrate_subset_tau(
            val_df["y_true"].to_numpy(),
            val_df["y_pred_known"].to_numpy(),
            val_df["score"].to_numpy(),
            val_df["true_label"].to_numpy(),
            subset_size=int(osr_cfg.get("unknown_subset_size", cfg.get("evaluation", {}).get("unknown_subset_size", 3))),
            max_candidates=int(osr_cfg.get("subset_tau_max_candidates", 200)),
            objective=str(osr_cfg.get("subset_tau_objective", "ACC_subset_mean")),
        )
        test_pred = apply_threshold(test_df["y_pred_known"].to_numpy(), test_df["score"].to_numpy(), tau_result.tau)
        metrics = main_metrics(
            test_df["y_true"].to_numpy(),
            test_df["y_pred_known"].to_numpy(),
            test_df["score"].to_numpy(),
            tau_result.tau,
        )
        metrics["tau"] = tau_result.tau
        metrics["subset_tau_objective"] = str(osr_cfg.get("subset_tau_objective", "ACC_subset_mean"))
    elif osr_method == "learned_fusion_classwise_tau":
        score_names = list(osr_cfg.get("scores", ["mls", "energy", "msp", "margin"]))
        learn_cfg = osr_cfg.get("learned_fusion", {})
        val_df, test_df, fusion_params = _learned_fusion_score(
            val_df,
            test_df,
            score_names,
            seed=int(cfg.get("seed", 26)),
            steps=int(learn_cfg.get("steps", osr_cfg.get("learn_steps", 2000))),
            learning_rate=float(learn_cfg.get("learning_rate", osr_cfg.get("learn_lr", 0.05))),
            l2=float(learn_cfg.get("l2", osr_cfg.get("learn_l2", 1e-3))),
            nonnegative=bool(learn_cfg.get("nonnegative", osr_cfg.get("learn_nonnegative", False))),
        )
        tau_result = calibrate_classwise_tau(
            val_df["y_true"].to_numpy(),
            val_df["y_pred_known"].to_numpy(),
            val_df["score"].to_numpy(),
            len(cfg["known_labels"]),
        )
        classwise_thresholds = tau_result.thresholds
        test_pred = apply_classwise_threshold(
            test_df["y_pred_known"].to_numpy(),
            test_df["score"].to_numpy(),
            tau_result.thresholds,
        )
        metrics = prediction_metrics(
            test_df["y_true"].to_numpy(),
            test_df["y_pred_known"].to_numpy(),
            test_df["score"].to_numpy(),
            test_pred,
            _finite_tau_summary(tau_result.thresholds),
        )
    elif osr_method == "learned_fusion_subset_classwise_tau":
        score_names = list(osr_cfg.get("scores", ["mls", "energy", "msp", "margin"]))
        learn_cfg = osr_cfg.get("learned_fusion", {})
        val_df, test_df, fusion_params = _learned_fusion_score(
            val_df,
            test_df,
            score_names,
            seed=int(cfg.get("seed", 26)),
            steps=int(learn_cfg.get("steps", osr_cfg.get("learn_steps", 2000))),
            learning_rate=float(learn_cfg.get("learning_rate", osr_cfg.get("learn_lr", 0.05))),
            l2=float(learn_cfg.get("l2", osr_cfg.get("learn_l2", 1e-3))),
            nonnegative=bool(learn_cfg.get("nonnegative", osr_cfg.get("learn_nonnegative", False))),
        )
        tau_result = calibrate_subset_classwise_tau(
            val_df["y_true"].to_numpy(),
            val_df["y_pred_known"].to_numpy(),
            val_df["score"].to_numpy(),
            val_df["true_label"].to_numpy(),
            len(cfg["known_labels"]),
            subset_size=int(osr_cfg.get("unknown_subset_size", cfg.get("evaluation", {}).get("unknown_subset_size", 3))),
            max_candidates=int(osr_cfg.get("subset_tau_max_candidates", 200)),
            passes=int(osr_cfg.get("subset_tau_passes", 2)),
            objective=str(osr_cfg.get("subset_tau_objective", "ACC_subset_mean")),
        )
        classwise_thresholds = tau_result.thresholds
        test_pred = apply_classwise_threshold(
            test_df["y_pred_known"].to_numpy(),
            test_df["score"].to_numpy(),
            tau_result.thresholds,
        )
        metrics = prediction_metrics(
            test_df["y_true"].to_numpy(),
            test_df["y_pred_known"].to_numpy(),
            test_df["score"].to_numpy(),
            test_pred,
            _finite_tau_summary(tau_result.thresholds),
        )
        metrics["subset_tau_objective"] = str(osr_cfg.get("subset_tau_objective", "ACC_subset_mean"))
    elif osr_method == "evt_fusion_classwise_tau":
        score_names = list(osr_cfg.get("scores", ["mls", "energy", "msp", "margin"]))
        weights = osr_cfg.get("weights")
        weights = [float(weight) for weight in weights] if weights is not None else None
        val_df, test_df, fusion_params = _fusion_score(val_df, test_df, score_names, weights)
        val_df["score_evt_fusion"] = val_df["score"]
        test_df["score_evt_fusion"] = test_df["score"]
        target_tprs = osr_cfg.get("evt_target_tprs")
        if target_tprs is None:
            target_tprs = [0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.99]
        target_tprs = [float(value) for value in target_tprs]
        tail_fraction = float(osr_cfg.get("evt_tail_fraction", 0.2))
        min_tail = int(osr_cfg.get("evt_min_tail", 5))
        classwise_thresholds, _, evt_params, selected_tpr = _evt_classwise_calibration(
            val_df,
            len(cfg["known_labels"]),
            target_tprs,
            tail_fraction,
            min_tail,
        )
        test_pred = apply_classwise_threshold(
            test_df["y_pred_known"].to_numpy(),
            test_df["score"].to_numpy(),
            classwise_thresholds,
        )
        metrics = prediction_metrics(
            test_df["y_true"].to_numpy(),
            test_df["y_pred_known"].to_numpy(),
            test_df["score"].to_numpy(),
            test_pred,
            _finite_tau_summary(classwise_thresholds),
        )
        metrics["evt_selected_target_tpr"] = float(selected_tpr)
    elif osr_method == "classwise_fusion_tau":
        label_names = list(cfg["known_labels"])
        score_names = list(osr_cfg.get("scores", ["mls", "energy", "msp", "margin"]))
        weights = osr_cfg.get("weights")
        weights = [float(weight) for weight in weights] if weights is not None else None
        min_samples = int(osr_cfg.get("classwise_min_samples", 10))
        val_df, test_df, fusion_params = _classwise_fusion_score(
            val_df,
            test_df,
            score_names,
            label_names,
            weights,
            min_samples=min_samples,
        )
        tau_result = calibrate_classwise_tau(
            val_df["y_true"].to_numpy(),
            val_df["y_pred_known"].to_numpy(),
            val_df["score"].to_numpy(),
            len(label_names),
        )
        classwise_thresholds = tau_result.thresholds
        test_pred = apply_classwise_threshold(
            test_df["y_pred_known"].to_numpy(),
            test_df["score"].to_numpy(),
            tau_result.thresholds,
        )
        metrics = prediction_metrics(
            test_df["y_true"].to_numpy(),
            test_df["y_pred_known"].to_numpy(),
            test_df["score"].to_numpy(),
            test_pred,
            _finite_tau_summary(tau_result.thresholds),
        )
    elif osr_method == "classwise_gaussian":
        label_names = list(cfg["known_labels"])
        covariance = str(osr_cfg.get("gaussian_covariance", "diag")).strip().lower()
        if covariance != "diag":
            raise ValueError("classwise_gaussian currently supports gaussian_covariance='diag'.")
        l2_normalize = bool(osr_cfg.get("l2_normalize_embeddings", True))
        regularization = float(osr_cfg.get("gaussian_regularization", 1e-4))
        shrinkage = float(osr_cfg.get("gaussian_shrinkage", 0.1))

        train_embeddings = _normalize_embedding_matrix(train_embeddings, l2_normalize)
        val_embeddings = _normalize_embedding_matrix(val_embeddings, l2_normalize)
        test_embeddings = _normalize_embedding_matrix(test_embeddings, l2_normalize)
        means, inv_variances, gaussian_params = _fit_classwise_gaussians(
            train_df,
            train_embeddings,
            len(label_names),
            regularization,
            shrinkage,
        )
        gaussian_params["label"] = label_names
        gaussian_params["covariance"] = covariance
        gaussian_params["l2_normalize_embeddings"] = l2_normalize
        gaussian_params["regularization"] = regularization
        gaussian_params["shrinkage"] = shrinkage
        gaussian_model = {"means": means, "inv_variances": inv_variances}

        val_df = _apply_classwise_gaussian_scores(val_df, val_embeddings, means, inv_variances, label_names)
        test_df = _apply_classwise_gaussian_scores(test_df, test_embeddings, means, inv_variances, label_names)
        tau_result = calibrate_classwise_tau(
            val_df["y_true"].to_numpy(),
            val_df["y_pred_known"].to_numpy(),
            val_df["score"].to_numpy(),
            len(label_names),
        )
        classwise_thresholds = tau_result.thresholds
        test_pred = apply_classwise_threshold(
            test_df["y_pred_known"].to_numpy(),
            test_df["score"].to_numpy(),
            tau_result.thresholds,
        )
        metrics = prediction_metrics(
            test_df["y_true"].to_numpy(),
            test_df["y_pred_known"].to_numpy(),
            test_df["score"].to_numpy(),
            test_pred,
            _finite_tau_summary(tau_result.thresholds),
        )
    elif osr_method == "mls":
        val_df = _set_primary_score(val_df, score_name)
        test_df = _set_primary_score(test_df, score_name)
        tau_result = calibrate_tau(
            val_df["y_true"].to_numpy(),
            val_df["y_pred_known"].to_numpy(),
            val_df["score"].to_numpy(),
        )
        test_pred = apply_threshold(test_df["y_pred_known"].to_numpy(), test_df["score"].to_numpy(), tau_result.tau)
        metrics = main_metrics(
            test_df["y_true"].to_numpy(),
            test_df["y_pred_known"].to_numpy(),
            test_df["score"].to_numpy(),
            tau_result.tau,
        )
    else:
        raise ValueError(
            "Unsupported osr.method. Valid methods: "
            "mls, classwise_tau, subset_tau, subset_classwise_tau, "
            "multi_score_fusion, fusion_subset_tau, global_fusion_classwise_tau, "
            "learned_fusion_tau, learned_fusion_subset_tau, learned_fusion_classwise_tau, "
            "learned_fusion_subset_classwise_tau, evt_fusion_classwise_tau, "
            "classwise_fusion_tau, classwise_gaussian."
        )

    test_df = test_df.copy()
    test_df["y_pred"] = test_pred

    label_names = list(cfg["known_labels"])
    subset_size = int(osr_cfg.get("unknown_subset_size", cfg.get("evaluation", {}).get("unknown_subset_size", 3)))
    metrics.update(unknown_subset_acc_report(test_df, subset_size=subset_size))
    metrics["osr_method"] = osr_method
    if osr_method == "multi_score_fusion":
        metrics["score_name"] = "fusion"
    elif osr_method == "fusion_subset_tau":
        metrics["score_name"] = "raw_fusion_subset_global_tau"
    elif osr_method == "global_fusion_classwise_tau":
        metrics["score_name"] = "global_fusion"
    elif osr_method == "learned_fusion_tau":
        metrics["score_name"] = "learned_fusion_global_tau"
    elif osr_method == "learned_fusion_subset_tau":
        metrics["score_name"] = "learned_fusion_subset_global_tau"
    elif osr_method == "learned_fusion_classwise_tau":
        metrics["score_name"] = "learned_fusion"
    elif osr_method == "learned_fusion_subset_classwise_tau":
        metrics["score_name"] = "learned_fusion_subset_tau"
    elif osr_method == "evt_fusion_classwise_tau":
        metrics["score_name"] = "evt_fusion"
    elif osr_method == "classwise_fusion_tau":
        metrics["score_name"] = "classwise_fusion"
    elif osr_method == "classwise_gaussian":
        metrics["score_name"] = "gaussian"
    elif osr_method == "subset_tau":
        metrics["score_name"] = f"{score_name}_subset_global_tau"
    elif osr_method == "subset_classwise_tau":
        metrics["score_name"] = f"{score_name}_subset_classwise_tau"
    else:
        metrics["score_name"] = score_name
    metrics_df = pd.DataFrame([metrics])
    paths = {
        "val_predictions": out / "val_predictions.csv",
        "test_predictions": out / "test_predictions.csv",
        "metrics": out / "metrics.csv",
        "known_report": out / "known_class_report.csv",
        "unknown_report": out / "unknown_class_report.csv",
        "known_confusion": out / "known_confusion_matrix.csv",
        "unknown_to_known": out / "unknown_to_known_matrix.csv",
        "top_confusions": out / "top_confusion_pairs.csv",
        "reconstruction_report": out / "reconstruction_report.csv",
    }
    if classwise_thresholds is not None:
        paths["classwise_thresholds"] = out / "classwise_thresholds.csv"
    if fusion_params is not None:
        paths["fusion_params"] = out / "fusion_params.csv"
    if evt_params is not None:
        paths["evt_params"] = out / "evt_params.csv"
    if gaussian_params is not None:
        paths["gaussian_params"] = out / "gaussian_params.csv"
        paths["gaussian_model"] = out / "gaussian_model.npz"
    val_df.to_csv(paths["val_predictions"], index=False)
    test_df.to_csv(paths["test_predictions"], index=False)
    metrics_df.to_csv(paths["metrics"], index=False)
    known_class_report(test_df, label_names).to_csv(paths["known_report"], index=False)
    unknown_class_report(test_df, label_names).to_csv(paths["unknown_report"], index=False)
    known_confusion_matrix(test_df, label_names).to_csv(paths["known_confusion"])
    unknown_to_known_matrix(test_df, label_names).to_csv(paths["unknown_to_known"])
    top_confusion_pairs(test_df, label_names).to_csv(paths["top_confusions"], index=False)
    reconstruction_report(test_df).to_csv(paths["reconstruction_report"], index=False)
    if classwise_thresholds is not None:
        pd.DataFrame({"label": label_names, "tau": classwise_thresholds}).to_csv(paths["classwise_thresholds"], index=False)
    if fusion_params is not None:
        fusion_params.to_csv(paths["fusion_params"], index=False)
    if evt_params is not None:
        evt_params.to_csv(paths["evt_params"], index=False)
    if gaussian_params is not None:
        gaussian_params.to_csv(paths["gaussian_params"], index=False)
    if gaussian_model is not None:
        import numpy as np

        np.savez(
            paths["gaussian_model"],
            means=gaussian_model["means"],
            inv_variances=gaussian_model["inv_variances"],
            labels=np.asarray(label_names),
        )

    print(metrics_df.to_string(index=False))
    return paths
