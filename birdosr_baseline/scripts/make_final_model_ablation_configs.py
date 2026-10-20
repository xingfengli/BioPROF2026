#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import itertools
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"

SCENARIOS = ("s1s2", "s2s1")
FULL_SCORES = ["mls", "energy", "msp", "margin", "proto_distance"]
LOGIT_SCORES = ["mls", "energy", "msp", "margin"]
SCORE_DISPLAY_NAMES = {
    "mls": "MLS",
    "energy": "Energy",
    "msp": "MSP",
    "margin": "Margin",
    "proto_distance": "proto_distance",
}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        print(f"Skip existing config: {path}")
        return
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote config: {path}")


def _config_name(scenario: str, experiment: str) -> str:
    return f"{scenario}_birdmae_noaug_{experiment}"


def _source_config(scenario: str, experiment: str) -> dict:
    return _load_json(CONFIG_DIR / f"{_config_name(scenario, experiment)}.json")


def _subset_settings() -> dict:
    return {
        "unknown_subset_size": 3,
        "subset_tau_objective": "ACC_subset_mean",
        "subset_tau_max_candidates": 200,
        "subset_tau_passes": 2,
    }


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
        cfg.update(_subset_settings())
    return cfg


def _single_score_osr(method: str, score: str) -> dict:
    cfg = {
        "method": method,
        "score": score,
        "msp_temperature": 1.0,
    }
    if "subset" in method:
        cfg.update(_subset_settings())
    return cfg


def _raw_fusion_osr(method: str, scores: list[str], weights: list[float] | None = None) -> dict:
    cfg = {
        "method": method,
        "scores": scores,
        "msp_temperature": 1.0,
    }
    if weights is not None:
        cfg["weights"] = weights
    if "subset" in method:
        cfg.update(_subset_settings())
    return cfg


def _score_combinations() -> list[tuple[str, ...]]:
    combos: list[tuple[str, ...]] = []
    for size in range(1, len(FULL_SCORES) + 1):
        combos.extend(itertools.combinations(FULL_SCORES, size))
    return combos


def _score_combo_key(scores: tuple[str, ...] | list[str]) -> str:
    return "_".join(scores)


def _score_combo_label(scores: tuple[str, ...] | list[str]) -> str:
    return " + ".join(SCORE_DISPLAY_NAMES.get(score, score) for score in scores)


def _alias_eval_config(
    scenario: str,
    source_experiment: str,
    suffix: str,
    label: str,
    group: str,
    osr: dict,
    *,
    status: str,
    note: str,
) -> dict:
    source = _source_config(scenario, source_experiment)
    cfg = copy.deepcopy(source)
    experiment_name = _config_name(scenario, f"final_{suffix}")
    source_name = _config_name(scenario, source_experiment)
    cfg["experiment_name"] = experiment_name
    cfg["ablation_label"] = label
    cfg["ablation_group"] = group
    cfg["ablation_status"] = status
    cfg["ablation_note"] = note
    cfg["source_checkpoint_experiment"] = source_name
    cfg["manifest_path"] = f"birdosr_baseline/outputs/{source_name}/manifest.csv"
    cfg["output_dir"] = f"birdosr_baseline/outputs/{experiment_name}"
    cfg["osr"] = osr
    return cfg


def _high_energy_oe_only_training_config(scenario: str) -> dict:
    base = _source_config(scenario, "oe")
    cfg = copy.deepcopy(base)
    experiment_name = _config_name(scenario, "final_wo_protosphere")
    cfg["experiment_name"] = experiment_name
    cfg["ablation_label"] = "FinalModule: w/o ProtoSphere"
    cfg["ablation_group"] = "final_module_ablation"
    cfg["ablation_status"] = "new_training_config"
    cfg["ablation_note"] = "Supplement: high-OE-energy OE-only training, because existing OE-only config used default OE energy."
    cfg.pop("source_checkpoint_experiment", None)
    cfg["manifest_path"] = f"birdosr_baseline/outputs/{experiment_name}/manifest.csv"
    cfg["output_dir"] = f"birdosr_baseline/outputs/{experiment_name}"
    cfg.setdefault("split", {})
    cfg["split"]["unknown_oe_fraction"] = 0.5
    cfg.setdefault("training", {})
    cfg["training"]["oe"] = {
        "enabled": True,
        "split": "oe",
        "batch_size": cfg["training"].get("batch_size", 16),
        "energy_weight": 0.2,
        "uniform_weight": 0.1,
        "distance_weight": 0.0,
        "distance_margin": 5.0,
    }
    cfg["osr_training"] = {"method": "oe"}
    cfg["osr"] = _learned_osr("learned_fusion_subset_classwise_tau", LOGIT_SCORES)
    return cfg


def _high_energy_oe_only_global_training_config(scenario: str) -> dict:
    base = _source_config(scenario, "oe")
    cfg = copy.deepcopy(base)
    experiment_name = _config_name(scenario, "final_global_wo_protosphere")
    cfg["experiment_name"] = experiment_name
    cfg["ablation_label"] = "FinalModule: w/o ProtoSphere"
    cfg["ablation_group"] = "final_module_global_ablation"
    cfg["ablation_status"] = "new_training_config"
    cfg["ablation_note"] = (
        "Global-threshold supplement: high-OE-energy OE-only training, without ProtoSphere "
        "or proto_distance scoring."
    )
    cfg.pop("source_checkpoint_experiment", None)
    cfg["manifest_path"] = f"birdosr_baseline/outputs/{experiment_name}/manifest.csv"
    cfg["output_dir"] = f"birdosr_baseline/outputs/{experiment_name}"
    cfg.setdefault("split", {})
    cfg["split"]["unknown_oe_fraction"] = 0.5
    cfg.setdefault("training", {})
    cfg["training"]["oe"] = {
        "enabled": True,
        "split": "oe",
        "batch_size": cfg["training"].get("batch_size", 16),
        "energy_weight": 0.2,
        "uniform_weight": 0.1,
        "distance_weight": 0.0,
        "distance_margin": 5.0,
    }
    cfg["osr_training"] = {"method": "oe"}
    cfg["osr"] = _learned_osr("learned_fusion_subset_tau", LOGIT_SCORES)
    return cfg


def final_main_configs(scenario: str) -> list[dict]:
    return [
        _alias_eval_config(
            scenario,
            "mls",
            "main_mls",
            "FinalMain: BirdMAE-noaug + MLS",
            "final_main_evolution",
            _single_score_osr("mls", "mls"),
            status="existing_reused",
            note="Covered by 01 main evolution; re-evaluated into final suite output.",
        ),
        _alias_eval_config(
            scenario,
            "mls",
            "main_classwise_tau",
            "FinalMain: class-wise tau",
            "final_main_evolution",
            _single_score_osr("classwise_tau", "mls"),
            status="existing_reused",
            note="Covered by 01 main evolution; re-evaluated into final suite output.",
        ),
        _alias_eval_config(
            scenario,
            "mls",
            "main_learned_fusion",
            "FinalMain: learned fusion score",
            "final_main_evolution",
            _learned_osr("learned_fusion_classwise_tau", LOGIT_SCORES),
            status="existing_reused",
            note="Covered by 01 main evolution; re-evaluated into final suite output.",
        ),
        _alias_eval_config(
            scenario,
            "oe",
            "main_oe",
            "FinalMain: OE",
            "final_main_evolution",
            _learned_osr("learned_fusion_classwise_tau", LOGIT_SCORES),
            status="existing_reused",
            note="Covered by 01/02 as OE-BirdOSR with default OE energy.",
        ),
        _alias_eval_config(
            scenario,
            "protosphere",
            "main_protosphere",
            "FinalMain: ProtoSphere",
            "final_main_evolution",
            _learned_osr("learned_fusion_classwise_tau", FULL_SCORES),
            status="existing_reused",
            note="Covered by 01/02 as ProtoSphere-BirdOSR.",
        ),
        _alias_eval_config(
            scenario,
            "oe_protosphere",
            "main_oe_protosphere",
            "FinalMain: OE + ProtoSphere",
            "final_main_evolution",
            _learned_osr("learned_fusion_classwise_tau", FULL_SCORES),
            status="existing_reused",
            note="Covered by 01/02 as OE-ProtoSphere with default OE energy.",
        ),
        _alias_eval_config(
            scenario,
            "oe_protosphere",
            "main_subset_classwise_tau",
            "FinalMain: subset class-wise tau",
            "final_main_evolution",
            _learned_osr("learned_fusion_subset_classwise_tau", FULL_SCORES),
            status="existing_reused",
            note="Covered by 01/02 as OE-ProtoSphere-SubsetTau with default OE energy.",
        ),
        _alias_eval_config(
            scenario,
            "oeps_oe_high_energy",
            "main_higher_oe_energy",
            "FinalMain: higher OE energy loss",
            "final_main_evolution",
            _learned_osr("learned_fusion_subset_classwise_tau", FULL_SCORES),
            status="existing_reused",
            note="Covered by 05 OE ablation; now treated as the final main model.",
        ),
    ]


def final_module_configs(scenario: str) -> list[dict]:
    return [
        _alias_eval_config(
            scenario,
            "oeps_oe_high_energy",
            "module_full",
            "FinalModule: full final model",
            "final_module_ablation",
            _learned_osr("learned_fusion_subset_classwise_tau", FULL_SCORES),
            status="existing_reused",
            note="Existing high-OE-energy final checkpoint from 05.",
        ),
        _alias_eval_config(
            scenario,
            "protosphere",
            "module_wo_oe",
            "FinalModule: w/o OE",
            "final_module_ablation",
            _learned_osr("learned_fusion_subset_classwise_tau", FULL_SCORES),
            status="supplement_eval_config",
            note="New evaluation around existing ProtoSphere checkpoint.",
        ),
        _high_energy_oe_only_training_config(scenario),
        _alias_eval_config(
            scenario,
            "oeps_oe_high_energy",
            "module_wo_subset_classwise_tau",
            "FinalModule: w/o subset class-wise tau",
            "final_module_ablation",
            _learned_osr("learned_fusion_classwise_tau", FULL_SCORES),
            status="supplement_eval_config",
            note="New threshold ablation around the high-OE-energy final checkpoint.",
        ),
        _alias_eval_config(
            scenario,
            "oeps_oe_high_energy",
            "module_wo_learned_fusion_margin",
            "FinalModule: w/o learned fusion, Margin score",
            "final_module_ablation",
            _single_score_osr("subset_classwise_tau", "margin"),
            status="supplement_eval_config",
            note="New true Margin-only score with subset class-wise tau; no learned fusion.",
        ),
        _alias_eval_config(
            scenario,
            "oe_protosphere",
            "module_wo_higher_oe_energy",
            "FinalModule: w/o higher OE energy",
            "final_module_ablation",
            _learned_osr("learned_fusion_subset_classwise_tau", FULL_SCORES),
            status="existing_reused",
            note="Existing default-OE OE-ProtoSphere-SubsetTau comparison.",
        ),
    ]


def final_module_global_configs(scenario: str) -> list[dict]:
    return [
        _alias_eval_config(
            scenario,
            "oeps_oe_high_energy",
            "global_module_full_classwise",
            "FinalModule: full final model_class-wise",
            "final_module_global_ablation",
            _learned_osr("learned_fusion_subset_classwise_tau", FULL_SCORES),
            status="existing_reused",
            note="Reference: high-OE-energy full model with learned fusion and subset class-wise tau.",
        ),
        _alias_eval_config(
            scenario,
            "protosphere",
            "global_module_wo_oe",
            "FinalModule: w/o OE",
            "final_module_global_ablation",
            _learned_osr("learned_fusion_subset_tau", FULL_SCORES),
            status="supplement_eval_config",
            note="Global-threshold module removal: existing ProtoSphere checkpoint, no OE.",
        ),
        _high_energy_oe_only_global_training_config(scenario),
        _alias_eval_config(
            scenario,
            "oeps_oe_high_energy",
            "global_module_wo_subset_classwise_tau",
            "FinalModule: w/o subset class-wise tau",
            "final_module_global_ablation",
            _learned_osr("learned_fusion_classwise_tau", FULL_SCORES),
            status="supplement_eval_config",
            note="Ordinary class-wise threshold around the high-OE-energy full model.",
        ),
        _alias_eval_config(
            scenario,
            "oe_protosphere",
            "global_module_wo_higher_oe_energy",
            "FinalModule: w/o higher OE energy",
            "final_module_global_ablation",
            _learned_osr("learned_fusion_subset_tau", FULL_SCORES),
            status="existing_reused",
            note="Default-OE OE-ProtoSphere model evaluated with subset global tau.",
        ),
        _alias_eval_config(
            scenario,
            "oeps_oe_high_energy",
            "global_module_full_raw_fusion",
            "FinalModule: full final model",
            "final_module_global_ablation",
            _raw_fusion_osr("fusion_subset_tau", FULL_SCORES),
            status="supplement_eval_config",
            note="Full high-OE-energy model with raw equal-weight fusion and subset global tau; no learned fusion.",
        ),
    ]


def high_energy_scoring_configs(scenario: str) -> list[dict]:
    source = "oeps_oe_high_energy"
    rows = [
        ("score_mls", "HighEnergyScore: MLS", _single_score_osr("subset_classwise_tau", "mls")),
        ("score_msp", "HighEnergyScore: MSP", _single_score_osr("subset_classwise_tau", "msp")),
        ("score_margin", "HighEnergyScore: Margin", _single_score_osr("subset_classwise_tau", "margin")),
        ("score_proto", "HighEnergyScore: proto_distance", _single_score_osr("subset_classwise_tau", "proto_distance")),
        ("score_mls_msp", "HighEnergyScore: learned MLS + MSP", _learned_osr("learned_fusion_subset_classwise_tau", ["mls", "msp"])),
        ("score_mls_margin", "HighEnergyScore: learned MLS + Margin", _learned_osr("learned_fusion_subset_classwise_tau", ["mls", "margin"])),
        ("score_full", "HighEnergyScore: full learned fusion", _learned_osr("learned_fusion_subset_classwise_tau", FULL_SCORES)),
    ]
    return [
        _alias_eval_config(
            scenario,
            source,
            f"highoe_{suffix}",
            label,
            "high_oe_scoring_ablation",
            osr,
            status="supplement_eval_config" if suffix != "score_full" else "existing_reused",
            note="High-OE-energy scoring ablation; source checkpoint is the final model.",
        )
        for suffix, label, osr in rows
    ]


def high_energy_scoring_global_configs(scenario: str) -> list[dict]:
    source = "oeps_oe_high_energy"
    configs: list[dict] = []
    for scores in _score_combinations():
        score_list = list(scores)
        combo_key = _score_combo_key(scores)
        combo_label = _score_combo_label(scores)
        configs.append(
            _alias_eval_config(
                scenario,
                source,
                f"highoe_global_score_{combo_key}",
                f"HighEnergyScoreGlobal: learned {combo_label}",
                "high_oe_scoring_global_ablation",
                _learned_osr("learned_fusion_subset_tau", score_list),
                status="supplement_eval_config",
                note=(
                    "BirdOSR-Final scoring ablation: fixed High-Energy OE + ProtoSphere "
                    "checkpoint, learned fusion over this score subset, subset global tau."
                ),
            )
        )
    return configs


def high_energy_threshold_configs(scenario: str) -> list[dict]:
    source = "oeps_oe_high_energy"
    rows = [
        ("threshold_global_tau", "HighEnergyThreshold: learned fusion + global tau", _learned_osr("learned_fusion_tau", FULL_SCORES)),
        ("threshold_classwise_tau", "HighEnergyThreshold: learned fusion + class-wise tau", _learned_osr("learned_fusion_classwise_tau", FULL_SCORES)),
        ("threshold_subset_global_tau", "HighEnergyThreshold: learned fusion + subset global tau", _learned_osr("learned_fusion_subset_tau", FULL_SCORES)),
        ("threshold_subset_classwise_tau", "HighEnergyThreshold: learned fusion + subset class-wise tau", _learned_osr("learned_fusion_subset_classwise_tau", FULL_SCORES)),
    ]
    return [
        _alias_eval_config(
            scenario,
            source,
            f"highoe_{suffix}",
            label,
            "high_oe_threshold_ablation",
            osr,
            status="supplement_eval_config" if suffix != "threshold_subset_classwise_tau" else "existing_reused",
            note="High-OE-energy threshold calibration ablation; source checkpoint is the final model.",
        )
        for suffix, label, osr in rows
    ]


GROUP_BUILDERS = {
    "final_main": final_main_configs,
    "final_module": final_module_configs,
    "final_module_global": final_module_global_configs,
    "high_oe_scoring": high_energy_scoring_configs,
    "high_oe_scoring_global": high_energy_scoring_global_configs,
    "high_oe_threshold": high_energy_threshold_configs,
}


def generate_configs(overwrite: bool, groups: list[str] | None = None) -> dict[str, list[Path]]:
    selected = groups or list(GROUP_BUILDERS)
    written: dict[str, list[Path]] = {group: [] for group in selected}
    for scenario in SCENARIOS:
        for group in selected:
            for cfg in GROUP_BUILDERS[group](scenario):
                path = CONFIG_DIR / f"{cfg['experiment_name']}.json"
                _write_json(path, cfg, overwrite=overwrite)
                written[group].append(path)
    return written


def _parse_groups(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return list(GROUP_BUILDERS)
    groups = [item.strip().lower() for item in value.split(",") if item.strip()]
    invalid = [group for group in groups if group not in GROUP_BUILDERS]
    if invalid:
        valid = ", ".join(["all", *GROUP_BUILDERS])
        raise ValueError(f"Invalid groups {invalid}. Valid groups: {valid}")
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate final-model supplemental ablation configs.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--groups", default="all")
    args = parser.parse_args()
    generate_configs(overwrite=args.overwrite, groups=_parse_groups(args.groups))


if __name__ == "__main__":
    main()
