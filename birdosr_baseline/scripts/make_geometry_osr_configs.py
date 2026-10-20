#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"

BASE_CONFIGS = {
    "s1s2": CONFIG_DIR / "s1s2_birdmae_noaug_learned_fusion_tau.json",
    "s2s1": CONFIG_DIR / "s2s1_birdmae_noaug_learned_fusion_tau.json",
}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        print(f"Skip existing config: {path}")
        return
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote config: {path}")


def _base_cfg(base_cfg: dict, scenario: str, suffix: str, label: str) -> dict:
    cfg = copy.deepcopy(base_cfg)
    experiment_name = f"{scenario}_birdmae_noaug_{suffix}"
    cfg["experiment_name"] = experiment_name
    cfg["ablation_label"] = label
    cfg["manifest_path"] = f"birdosr_baseline/outputs/{experiment_name}/manifest.csv"
    cfg["output_dir"] = f"birdosr_baseline/outputs/{experiment_name}"
    cfg.setdefault("training", {})
    cfg["training"]["deterministic"] = True
    cfg.setdefault("model", {})
    cfg["model"]["backbone"] = "birdmae:models/Bird-MAE-Base"
    cfg["model"]["pretrained"] = True
    return cfg


def _learned_osr(scores: list[str]) -> dict:
    return {
        "method": "learned_fusion_classwise_tau",
        "scores": scores,
        "msp_temperature": 1.0,
        "learned_fusion": {
            "steps": 2000,
            "learning_rate": 0.05,
            "l2": 0.001,
            "nonnegative": False,
        },
    }


def build_protosphere(base_cfg: dict, scenario: str) -> dict:
    cfg = _base_cfg(base_cfg, scenario, "protosphere", "ProtoSphere-BirdOSR")
    cfg["ablation_group"] = "training_ablation"
    cfg["model"]["finetune_mode"] = "linear_probe"
    cfg["osr_training"] = {
        "method": "protosphere",
        "prototype_loss_weight": 0.1,
        "prototype_margin": 1.0,
        "radius_loss_weight": 0.05,
        "radius_reg_weight": 0.0001,
        "radius_init": 1.0,
    }
    cfg["osr"] = _learned_osr(["mls", "energy", "msp", "margin", "proto_distance"])
    return cfg


def build_oe(base_cfg: dict, scenario: str, unknown_oe_fraction: float) -> dict:
    cfg = _base_cfg(base_cfg, scenario, "oe", "OE-BirdOSR")
    cfg["ablation_group"] = "training_ablation"
    cfg["model"]["finetune_mode"] = "linear_probe"
    cfg.setdefault("split", {})
    cfg["split"]["unknown_oe_fraction"] = float(unknown_oe_fraction)
    cfg["training"]["oe"] = {
        "enabled": True,
        "split": "oe",
        "batch_size": cfg["training"].get("batch_size", 16),
        "energy_weight": 0.1,
        "uniform_weight": 0.1,
        "distance_weight": 0.0,
        "distance_margin": 5.0,
    }
    cfg["osr_training"] = {"method": "oe"}
    cfg["osr"] = _learned_osr(["mls", "energy", "msp", "margin"])
    return cfg


def build_oe_protosphere(base_cfg: dict, scenario: str, unknown_oe_fraction: float) -> dict:
    cfg = _base_cfg(base_cfg, scenario, "oe_protosphere", "OE-ProtoSphere-BirdOSR")
    cfg["ablation_group"] = "training_ablation"
    cfg["model"]["finetune_mode"] = "linear_probe"
    cfg.setdefault("split", {})
    cfg["split"]["unknown_oe_fraction"] = float(unknown_oe_fraction)
    cfg["osr_training"] = {
        "method": "protosphere",
        "prototype_loss_weight": 0.1,
        "prototype_margin": 1.0,
        "radius_loss_weight": 0.05,
        "radius_reg_weight": 0.0001,
        "radius_init": 1.0,
    }
    cfg["training"]["oe"] = {
        "enabled": True,
        "split": "oe",
        "batch_size": cfg["training"].get("batch_size", 16),
        "energy_weight": 0.1,
        "uniform_weight": 0.1,
        "distance_weight": 0.01,
        "distance_margin": 5.0,
    }
    cfg["osr"] = _learned_osr(["mls", "energy", "msp", "margin", "proto_distance"])
    return cfg


def build_oe_protosphere_subset_tau(base_cfg: dict, scenario: str, unknown_oe_fraction: float) -> dict:
    cfg = build_oe_protosphere(base_cfg, scenario, unknown_oe_fraction)
    source_experiment_name = cfg["experiment_name"]
    experiment_name = f"{scenario}_birdmae_noaug_oe_protosphere_subset_tau"
    cfg["experiment_name"] = experiment_name
    cfg["ablation_label"] = "OE-ProtoSphere-SubsetTau"
    cfg["ablation_group"] = "main_evolution"
    cfg["source_checkpoint_experiment"] = source_experiment_name
    cfg["manifest_path"] = f"birdosr_baseline/outputs/{source_experiment_name}/manifest.csv"
    cfg["output_dir"] = f"birdosr_baseline/outputs/{experiment_name}"
    cfg["osr"] = _learned_osr(["mls", "energy", "msp", "margin", "proto_distance"])
    cfg["osr"]["method"] = "learned_fusion_subset_classwise_tau"
    cfg["osr"]["unknown_subset_size"] = 3
    cfg["osr"]["subset_tau_objective"] = "ACC_subset_mean"
    cfg["osr"]["subset_tau_max_candidates"] = 200
    cfg["osr"]["subset_tau_passes"] = 2
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ProtoSphere/OE BirdOSR configs.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--unknown-oe-fraction", type=float, default=0.5)
    args = parser.parse_args()

    for scenario, base_path in BASE_CONFIGS.items():
        base_cfg = _load_json(base_path)
        configs = [
            build_protosphere(base_cfg, scenario),
            build_oe(base_cfg, scenario, args.unknown_oe_fraction),
            build_oe_protosphere(base_cfg, scenario, args.unknown_oe_fraction),
            build_oe_protosphere_subset_tau(base_cfg, scenario, args.unknown_oe_fraction),
        ]
        for cfg in configs:
            _write_json(CONFIG_DIR / f"{cfg['experiment_name']}.json", cfg, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
