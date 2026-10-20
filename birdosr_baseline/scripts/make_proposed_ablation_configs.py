#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"

SCENARIOS = ("s1s2", "s2s1")
FULL_SCORES = ["mls", "energy", "msp", "margin", "proto_distance"]
LOGIT_SCORES = ["mls", "energy", "msp", "margin"]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        print(f"Skip existing config: {path}")
        return
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote config: {path}")


def _learned_osr(method: str, scores: list[str]) -> dict:
    cfg = {
        "method": method,
        "scores": scores,
        "msp_temperature": 1.0,
        "learned_fusion": {
            "steps": 2000,
            "learning_rate": 0.05,
            "l2": 0.001,
            "nonnegative": False,
        },
    }
    if "subset" in method:
        cfg.update(
            {
                "unknown_subset_size": 3,
                "subset_tau_objective": "ACC_subset_mean",
                "subset_tau_max_candidates": 200,
                "subset_tau_passes": 2,
            }
        )
    return cfg


def _source_base(scenario: str) -> dict:
    return _load_json(CONFIG_DIR / f"{scenario}_birdmae_noaug_oe_protosphere_subset_tau.json")


def _source_experiment(scenario: str) -> str:
    return f"{scenario}_birdmae_noaug_oe_protosphere"


def _derived_eval_config(
    base_cfg: dict,
    scenario: str,
    suffix: str,
    label: str,
    group: str,
    method: str,
    scores: list[str],
) -> dict:
    cfg = copy.deepcopy(base_cfg)
    experiment_name = f"{scenario}_birdmae_noaug_{suffix}"
    source_experiment = _source_experiment(scenario)
    cfg["experiment_name"] = experiment_name
    cfg["ablation_label"] = label
    cfg["ablation_group"] = group
    cfg["source_checkpoint_experiment"] = source_experiment
    cfg["manifest_path"] = f"birdosr_baseline/outputs/{source_experiment}/manifest.csv"
    cfg["output_dir"] = f"birdosr_baseline/outputs/{experiment_name}"
    cfg["osr"] = _learned_osr(method, scores)
    return cfg


def _training_config(
    base_cfg: dict,
    scenario: str,
    suffix: str,
    label: str,
    group: str,
    *,
    unknown_oe_fraction: float | None = None,
    oe_weights: tuple[float, float, float] | None = None,
    prototype_weight: float | None = None,
    radius_weight: float | None = None,
    radius_reg_weight: float | None = None,
) -> dict:
    cfg = copy.deepcopy(base_cfg)
    experiment_name = f"{scenario}_birdmae_noaug_{suffix}"
    cfg["experiment_name"] = experiment_name
    cfg["ablation_label"] = label
    cfg["ablation_group"] = group
    cfg.pop("source_checkpoint_experiment", None)
    cfg["manifest_path"] = f"birdosr_baseline/outputs/{experiment_name}/manifest.csv"
    cfg["output_dir"] = f"birdosr_baseline/outputs/{experiment_name}"
    cfg["osr"] = _learned_osr("learned_fusion_subset_classwise_tau", FULL_SCORES)
    cfg.setdefault("split", {})
    if unknown_oe_fraction is not None:
        cfg["split"]["unknown_oe_fraction"] = float(unknown_oe_fraction)
    cfg.setdefault("training", {})
    cfg["training"].setdefault("oe", {})
    cfg["training"]["oe"]["enabled"] = True
    cfg["training"]["oe"]["split"] = "oe"
    cfg["training"]["oe"]["batch_size"] = cfg["training"].get("batch_size", 16)
    if oe_weights is not None:
        energy, uniform, distance = oe_weights
        cfg["training"]["oe"]["energy_weight"] = float(energy)
        cfg["training"]["oe"]["uniform_weight"] = float(uniform)
        cfg["training"]["oe"]["distance_weight"] = float(distance)
    cfg.setdefault("osr_training", {})
    cfg["osr_training"]["method"] = "protosphere"
    if prototype_weight is not None:
        cfg["osr_training"]["prototype_loss_weight"] = float(prototype_weight)
    if radius_weight is not None:
        cfg["osr_training"]["radius_loss_weight"] = float(radius_weight)
    if radius_reg_weight is not None:
        cfg["osr_training"]["radius_reg_weight"] = float(radius_reg_weight)
    return cfg


def scoring_configs(base_cfg: dict, scenario: str) -> list[dict]:
    rows = [
        ("score_mls", "ScoreAblation: MLS", ["mls"]),
        ("score_energy", "ScoreAblation: Energy", ["energy"]),
        ("score_msp", "ScoreAblation: MSP", ["msp"]),
        ("score_margin", "ScoreAblation: Margin", ["margin"]),
        ("score_proto", "ScoreAblation: proto_distance", ["proto_distance"]),
        ("score_mls_energy", "ScoreAblation: MLS + Energy", ["mls", "energy"]),
        ("score_mls_msp", "ScoreAblation: MLS + MSP", ["mls", "msp"]),
        ("score_mls_margin", "ScoreAblation: MLS + Margin", ["mls", "margin"]),
        ("score_mls_proto", "ScoreAblation: MLS + proto_distance", ["mls", "proto_distance"]),
        (
            "score_logit4",
            "ScoreAblation: MLS + Energy + MSP + Margin",
            LOGIT_SCORES,
        ),
        (
            "score_full",
            "ScoreAblation: MLS + Energy + MSP + Margin + proto_distance",
            FULL_SCORES,
        ),
    ]
    return [
        _derived_eval_config(
            base_cfg,
            scenario,
            f"oeps_{suffix}",
            label,
            "scoring_ablation",
            "learned_fusion_subset_classwise_tau",
            scores,
        )
        for suffix, label, scores in rows
    ]


def threshold_configs(base_cfg: dict, scenario: str) -> list[dict]:
    rows = [
        ("threshold_global_tau", "ThresholdAblation: learned fusion + global tau", "learned_fusion_tau"),
        (
            "threshold_classwise_tau",
            "ThresholdAblation: learned fusion + class-wise tau",
            "learned_fusion_classwise_tau",
        ),
        (
            "threshold_subset_global_tau",
            "ThresholdAblation: learned fusion + subset global tau",
            "learned_fusion_subset_tau",
        ),
        (
            "threshold_subset_classwise_tau",
            "ThresholdAblation: learned fusion + subset class-wise tau",
            "learned_fusion_subset_classwise_tau",
        ),
    ]
    return [
        _derived_eval_config(
            base_cfg,
            scenario,
            f"oeps_{suffix}",
            label,
            "threshold_ablation",
            method,
            FULL_SCORES,
        )
        for suffix, label, method in rows
    ]


def oe_configs(base_cfg: dict, scenario: str) -> list[dict]:
    rows = [
        ("oe_frac025", "OEAblation: OE fraction 0.25", 0.25, (0.1, 0.1, 0.01)),
        ("oe_frac075", "OEAblation: OE fraction 0.75", 0.75, (0.1, 0.1, 0.01)),
        ("oe_weak", "OEAblation: weak OE loss", 0.5, (0.05, 0.05, 0.005)),
        ("oe_no_distance", "OEAblation: no OE prototype-distance loss", 0.5, (0.1, 0.1, 0.0)),
        ("oe_strong_distance", "OEAblation: stronger OE prototype-distance loss", 0.5, (0.1, 0.1, 0.02)),
        ("oe_high_energy", "OEAblation: higher OE energy loss", 0.5, (0.2, 0.1, 0.01)),
        ("oe_high_uniform", "OEAblation: higher OE uniform loss", 0.5, (0.1, 0.2, 0.01)),
    ]
    return [
        _training_config(
            base_cfg,
            scenario,
            f"oeps_{suffix}",
            label,
            "oe_ablation",
            unknown_oe_fraction=fraction,
            oe_weights=weights,
        )
        for suffix, label, fraction, weights in rows
    ]


def protosphere_configs(base_cfg: dict, scenario: str) -> list[dict]:
    rows = [
        ("proto005_radius0025", "ProtoAblation: prototype 0.05 + radius 0.025", 0.05, 0.025, 0.0001),
        ("proto020_radius005", "ProtoAblation: prototype 0.20 + radius 0.05", 0.2, 0.05, 0.0001),
        ("proto010_radius010", "ProtoAblation: prototype 0.10 + radius 0.10", 0.1, 0.1, 0.0001),
        ("proto010_radius005_reg0", "ProtoAblation: no radius regularization", 0.1, 0.05, 0.0),
        ("proto010_radius005_reg001", "ProtoAblation: stronger radius regularization", 0.1, 0.05, 0.001),
    ]
    return [
        _training_config(
            base_cfg,
            scenario,
            f"oeps_{suffix}",
            label,
            "protosphere_ablation",
            unknown_oe_fraction=0.5,
            oe_weights=(0.1, 0.1, 0.01),
            prototype_weight=prototype,
            radius_weight=radius,
            radius_reg_weight=radius_reg,
        )
        for suffix, label, prototype, radius, radius_reg in rows
    ]


def generate_configs(overwrite: bool) -> list[Path]:
    written: list[Path] = []
    for scenario in SCENARIOS:
        base_cfg = _source_base(scenario)
        configs = (
            scoring_configs(base_cfg, scenario)
            + threshold_configs(base_cfg, scenario)
            + oe_configs(base_cfg, scenario)
            + protosphere_configs(base_cfg, scenario)
        )
        for cfg in configs:
            path = CONFIG_DIR / f"{cfg['experiment_name']}.json"
            _write_json(path, cfg, overwrite=overwrite)
            written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate proposed-method ablation configs.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    generate_configs(overwrite=args.overwrite)


if __name__ == "__main__":
    main()
