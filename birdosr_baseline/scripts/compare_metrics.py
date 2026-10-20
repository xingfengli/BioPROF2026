#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from birdosr.config import load_config
from birdosr.metrics import unknown_subset_acc_report


def _format_scores(scores: object) -> str:
    if not scores:
        return ""
    if isinstance(scores, str):
        return scores
    return "+".join(str(score) for score in scores)


def _format_weights(weights: object) -> str:
    if not weights:
        return ""
    if isinstance(weights, str):
        return weights
    return "+".join(str(weight) for weight in weights)


def _default_ablation_label(osr_cfg: dict) -> str:
    method = osr_cfg.get("method", "")
    if method == "mls":
        return "MLS"
    if method == "classwise_tau":
        return "MLS + class-wise tau"
    if method == "global_fusion_classwise_tau":
        scores = _format_scores(osr_cfg.get("scores", []))
        return f"{scores} + class-wise tau" if scores else "global fusion + class-wise tau"
    if method == "fusion_subset_tau":
        scores = _format_scores(osr_cfg.get("scores", []))
        return f"raw({scores}) + subset global tau" if scores else "raw fusion + subset global tau"
    if method == "learned_fusion_tau":
        scores = _format_scores(osr_cfg.get("scores", []))
        return f"learned({scores}) + global tau" if scores else "learned fusion + global tau"
    if method == "learned_fusion_subset_tau":
        scores = _format_scores(osr_cfg.get("scores", []))
        return f"learned({scores}) + subset global tau" if scores else "learned fusion + subset global tau"
    if method == "learned_fusion_classwise_tau":
        scores = _format_scores(osr_cfg.get("scores", []))
        return f"learned({scores}) + class-wise tau" if scores else "learned fusion + class-wise tau"
    if method == "learned_fusion_subset_classwise_tau":
        scores = _format_scores(osr_cfg.get("scores", []))
        return f"learned({scores}) + subset class-wise tau" if scores else "learned fusion + subset class-wise tau"
    if method == "evt_fusion_classwise_tau":
        scores = _format_scores(osr_cfg.get("scores", []))
        return f"EVT({scores}) + class-wise tau" if scores else "EVT fusion + class-wise tau"
    return method


def _infer_ablation_group(experiment_name: str, configured_group: str) -> str:
    if configured_group:
        return configured_group
    name = experiment_name.lower()
    if name.endswith("_birdmae_noaug_mls") or name.endswith("_birdmae_noaug_classwise_tau"):
        return "main_evolution"
    if name.endswith("_birdmae_noaug_gfct_energy_msp_margin") or name.endswith("_birdmae_noaug_learned_fusion_tau"):
        return "main_evolution"
    if "nosampler" in name:
        return "training_ablation"
    if "protosphere" in name or name.endswith("_oe") or "_rpl" in name or "_supcon" in name:
        return "training_ablation"
    if "gfct" in name or "score" in name:
        return "scoring_ablation"
    if "classwise_tau" in name or "subset_tau" in name:
        return "threshold_ablation"
    return ""


def _infer_ablation_label(experiment_name: str, configured_label: str | None, osr_cfg: dict) -> str:
    if configured_label:
        return configured_label
    name = experiment_name.lower()
    if name.endswith("_birdmae_noaug_mls"):
        return "baseline: BirdMAE-noaug-MLS"
    if name.endswith("_birdmae_noaug_classwise_tau"):
        return "improvement-1: class-wise tau"
    if name.endswith("_birdmae_noaug_gfct_energy_msp_margin"):
        return "improvement-2: MLS+Energy+MSP+Margin + class-wise tau"
    if name.endswith("_birdmae_noaug_learned_fusion_tau"):
        return "improvement-3: learned fusion + class-wise tau"
    if "nosampler" in name:
        if name.endswith("_mls"):
            return "NoSampler: MLS"
        if name.endswith("_classwise_tau"):
            return "NoSampler: class-wise tau"
        if name.endswith("_fusion"):
            return "NoSampler: fusion"
    return _default_ablation_label(osr_cfg)


def _read_metrics(config_path: str | Path) -> dict:
    cfg = load_config(config_path)
    output_dir = Path(cfg["output_dir"])
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    metrics_path = output_dir / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
    row = pd.read_csv(metrics_path).iloc[0].to_dict()
    osr_cfg = cfg.get("osr", {})
    subset_size = int(osr_cfg.get("unknown_subset_size", cfg.get("evaluation", {}).get("unknown_subset_size", 3)))
    predictions_path = output_dir / "test_predictions.csv"
    if predictions_path.exists():
        subset_metrics = unknown_subset_acc_report(pd.read_csv(predictions_path), subset_size=subset_size)
        row.update(subset_metrics)
    experiment_name = cfg.get("experiment_name", output_dir.name)
    row["experiment_name"] = experiment_name
    row["ablation_label"] = _infer_ablation_label(experiment_name, cfg.get("ablation_label"), osr_cfg)
    row["ablation_group"] = _infer_ablation_group(experiment_name, cfg.get("ablation_group", ""))
    row["ablation_status"] = cfg.get("ablation_status", "")
    row["ablation_note"] = cfg.get("ablation_note", "")
    row["source_checkpoint_experiment"] = cfg.get("source_checkpoint_experiment", "")
    row["scenario"] = cfg.get("scenario", "")
    row["train_site"] = cfg.get("train_site", "")
    row["test_site"] = cfg.get("test_site", "")
    model_cfg = cfg.get("model", {})
    training_cfg = cfg.get("training", {})
    split_cfg = cfg.get("split", {})
    oe_cfg = training_cfg.get("oe", {})
    row["finetune_mode"] = model_cfg.get("finetune_mode", "")
    row["class_balanced_sampler"] = bool(training_cfg.get("class_balanced_sampler", True))
    osr_training_cfg = cfg.get("osr_training", {})
    row["osr_training_method"] = osr_training_cfg.get("method", "none")
    row["prototype_loss_weight"] = osr_training_cfg.get("prototype_loss_weight", "")
    row["radius_loss_weight"] = osr_training_cfg.get("radius_loss_weight", "")
    row["oe_enabled"] = bool(oe_cfg.get("enabled", False))
    row["unknown_oe_fraction"] = split_cfg.get("unknown_oe_fraction", "")
    row["oe_energy_weight"] = oe_cfg.get("energy_weight", "")
    row["oe_uniform_weight"] = oe_cfg.get("uniform_weight", "")
    row["oe_distance_weight"] = oe_cfg.get("distance_weight", "")
    row["osr_config_method"] = osr_cfg.get("method", "")
    row["osr_scores"] = _format_scores(osr_cfg.get("scores") or osr_cfg.get("score", ""))
    row["osr_weights"] = _format_weights(osr_cfg.get("weights", []))
    row["msp_temperature"] = osr_cfg.get("msp_temperature", "")
    row["premax_alpha"] = osr_cfg.get("premax_alpha", "")
    row["subset_tau_objective"] = osr_cfg.get("subset_tau_objective", "")
    row["subset_tau_max_candidates"] = osr_cfg.get("subset_tau_max_candidates", "")
    row["subset_tau_passes"] = osr_cfg.get("subset_tau_passes", "")
    row["evt_tail_fraction"] = osr_cfg.get("evt_tail_fraction", "")
    row["evt_min_tail"] = osr_cfg.get("evt_min_tail", "")
    learned_cfg = osr_cfg.get("learned_fusion", {})
    row["learn_steps"] = learned_cfg.get("steps", osr_cfg.get("learn_steps", ""))
    row["learn_lr"] = learned_cfg.get("learning_rate", osr_cfg.get("learn_lr", ""))
    row["learn_l2"] = learned_cfg.get("l2", osr_cfg.get("learn_l2", ""))
    row["output_dir"] = str(output_dir)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True, help="Config files whose output metrics should be compared.")
    parser.add_argument("--output", default=None, help="Optional CSV path for the merged comparison table.")
    args = parser.parse_args()

    rows = [_read_metrics(path) for path in args.configs]
    df = pd.DataFrame(rows)
    preferred = [
        "experiment_name",
        "ablation_label",
        "ablation_group",
        "ablation_status",
        "ablation_note",
        "source_checkpoint_experiment",
        "scenario",
        "train_site",
        "test_site",
        "finetune_mode",
        "class_balanced_sampler",
        "osr_training_method",
        "prototype_loss_weight",
        "radius_loss_weight",
        "oe_enabled",
        "unknown_oe_fraction",
        "oe_energy_weight",
        "oe_uniform_weight",
        "oe_distance_weight",
        "osr_method",
        "osr_config_method",
        "score_name",
        "osr_scores",
        "osr_weights",
        "msp_temperature",
        "premax_alpha",
        "subset_tau_objective",
        "subset_tau_max_candidates",
        "subset_tau_passes",
        "evt_selected_target_tpr",
        "evt_tail_fraction",
        "evt_min_tail",
        "learn_steps",
        "learn_lr",
        "learn_l2",
        "tau",
        "ACC_k",
        "ACC_u",
        "ACC",
        "unknown_subset_size",
        "unknown_subset_count",
        "ACC_u_subset_mean",
        "ACC_u_subset_std",
        "ACC_subset_mean",
        "ACC_subset_std",
        "AUROC",
        "FPR@95TPR",
        "OSCR",
        "output_dir",
    ]
    columns = [column for column in preferred if column in df.columns]
    df = df[columns + [column for column in df.columns if column not in columns]]

    if args.output:
        output = Path(args.output)
        if not output.is_absolute():
            output = Path.cwd() / output
        output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output, index=False)
        print(f"Wrote comparison table: {output}")

    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
