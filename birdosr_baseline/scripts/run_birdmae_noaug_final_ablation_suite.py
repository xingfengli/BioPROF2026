#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
CONFIG_DIR = ROOT / "configs"
OUTPUT_DIR = ROOT / "outputs"

FULL_SCORES = ["mls", "energy", "msp", "margin", "proto_distance"]


def _score_combo_suffixes() -> list[str]:
    suffixes: list[str] = []
    for size in range(1, len(FULL_SCORES) + 1):
        for scores in itertools.combinations(FULL_SCORES, size):
            suffixes.append("highoe_global_score_" + "_".join(scores))
    return suffixes


GROUP_CHOICES = (
    "final_main",
    "final_module",
    "final_module_global",
    "high_oe_scoring",
    "high_oe_scoring_global",
    "high_oe_threshold",
)
GROUP_SUFFIX_ORDER = {
    "final_main": [
        "main_mls",
        "main_classwise_tau",
        "main_learned_fusion",
        "main_oe",
        "main_protosphere",
        "main_oe_protosphere",
        "main_subset_classwise_tau",
        "main_higher_oe_energy",
    ],
    "final_module": [
        "module_full",
        "module_wo_oe",
        "wo_protosphere",
        "module_wo_subset_classwise_tau",
        "module_wo_learned_fusion_margin",
        "module_wo_higher_oe_energy",
    ],
    "final_module_global": [
        "global_module_full_classwise",
        "global_module_wo_oe",
        "global_wo_protosphere",
        "global_module_wo_subset_classwise_tau",
        "global_module_wo_higher_oe_energy",
        "global_module_full_raw_fusion",
    ],
    "high_oe_scoring": [
        "highoe_score_mls",
        "highoe_score_msp",
        "highoe_score_margin",
        "highoe_score_proto",
        "highoe_score_mls_msp",
        "highoe_score_mls_margin",
        "highoe_score_full",
    ],
    "high_oe_scoring_global": _score_combo_suffixes(),
    "high_oe_threshold": [
        "highoe_threshold_global_tau",
        "highoe_threshold_classwise_tau",
        "highoe_threshold_subset_global_tau",
        "highoe_threshold_subset_classwise_tau",
    ],
}


def _config_path(experiment_name: str) -> Path:
    return CONFIG_DIR / f"{experiment_name}.json"


def _checkpoint_for_experiment(experiment_name: str) -> Path:
    return OUTPUT_DIR / experiment_name / "checkpoint_best.pt"


def _output_dir(config_path: Path) -> Path:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = Path(cfg["output_dir"])
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    return output_dir


def _metrics_path(config_path: Path) -> Path:
    return _output_dir(config_path) / "metrics.csv"


def _own_checkpoint_path(config_path: Path) -> Path:
    return _output_dir(config_path) / "checkpoint_best.pt"


def _run(cmd: list[str], log_path: Path | None = None) -> None:
    print(" ".join(cmd), flush=True)
    if log_path is None:
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)


def _train_and_evaluate(config_path: Path, skip_existing: bool) -> None:
    metrics_path = _metrics_path(config_path)
    checkpoint_path = _own_checkpoint_path(config_path)
    if skip_existing and metrics_path.exists() and checkpoint_path.exists():
        print(f"Skip existing trained metrics: {metrics_path}", flush=True)
        return
    _run(
        [
            sys.executable,
            str(ROOT / "scripts" / "make_splits.py"),
            "--config",
            str(config_path.relative_to(PROJECT_ROOT)),
        ]
    )
    if not (skip_existing and checkpoint_path.exists()):
        _run(
            [
                sys.executable,
                str(ROOT / "scripts" / "train.py"),
                "--config",
                str(config_path.relative_to(PROJECT_ROOT)),
            ],
            log_path=OUTPUT_DIR / config_path.stem / "train.log",
        )
    else:
        print(f"Reuse checkpoint: {checkpoint_path}", flush=True)
    _evaluate(config_path, checkpoint_path, skip_existing=False)


def _ensure_experiment_checkpoint(experiment_name: str) -> Path:
    checkpoint = _checkpoint_for_experiment(experiment_name)
    if checkpoint.exists():
        return checkpoint
    config_path = _config_path(experiment_name)
    if not config_path.exists():
        raise FileNotFoundError(f"Missing source config for checkpoint: {config_path}")
    print(f"Missing checkpoint for {experiment_name}; training source first: {checkpoint}", flush=True)
    _train_and_evaluate(config_path, skip_existing=False)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Training finished but checkpoint was not created: {checkpoint}")
    return checkpoint


def _evaluate(config_path: Path, checkpoint: Path, skip_existing: bool) -> None:
    metrics_path = _metrics_path(config_path)
    if skip_existing and metrics_path.exists():
        print(f"Skip existing metrics: {metrics_path}", flush=True)
        return
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint for {config_path.name}: {checkpoint}")
    _run(
        [
            sys.executable,
            str(ROOT / "scripts" / "evaluate.py"),
            "--config",
            str(config_path.relative_to(PROJECT_ROOT)),
            "--checkpoint",
            str(checkpoint),
        ],
        log_path=OUTPUT_DIR / config_path.stem / "eval.log",
    )


def _run_config(config_path: Path, skip_existing: bool) -> None:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    source = cfg.get("source_checkpoint_experiment")
    if source:
        checkpoint = _ensure_experiment_checkpoint(str(source))
        _evaluate(config_path, checkpoint, skip_existing=skip_existing)
    else:
        _train_and_evaluate(config_path, skip_existing=skip_existing)


def _group_configs(scenario: str, group: str) -> list[Path]:
    paths: list[Path] = []
    for suffix in GROUP_SUFFIX_ORDER[group]:
        path = CONFIG_DIR / f"{scenario}_birdmae_noaug_final_{suffix}.json"
        if path.exists():
            paths.append(path)
    return paths


def _write_comparison(configs: list[Path], output_csv: Path) -> None:
    output_txt = output_csv.with_suffix(".txt")
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "compare_metrics.py"),
        "--configs",
        *[str(path.relative_to(PROJECT_ROOT)) for path in configs],
        "--output",
        str(output_csv.relative_to(PROJECT_ROOT)),
    ]
    print(" ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, capture_output=True, text=True)
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    output_txt.write_text(result.stdout, encoding="utf-8")
    print(result.stdout, flush=True)
    if result.stderr:
        print(result.stderr, file=sys.stderr, flush=True)
    print(f"Wrote comparison text: {output_txt}", flush=True)


def _parse_groups(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return list(GROUP_CHOICES)
    groups = [item.strip().lower() for item in value.split(",") if item.strip()]
    invalid = [group for group in groups if group not in GROUP_CHOICES]
    if invalid:
        valid = ", ".join(["all", *GROUP_CHOICES])
        raise ValueError(f"Invalid groups {invalid}. Valid groups: {valid}")
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description="Run final-model supplemental BirdMAE ablation suites.")
    parser.add_argument("--scenario", choices=["all", "s1s2", "s2s1"], default="all")
    parser.add_argument("--groups", default="all")
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR / "birdmae_noaug_final_ablation_suite_comparison.csv",
    )
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    _run(
        [
            sys.executable,
            str(ROOT / "scripts" / "make_final_model_ablation_configs.py"),
            "--overwrite",
            "--groups",
            args.groups,
        ]
    )

    groups = _parse_groups(args.groups)
    scenarios = ["s1s2", "s2s1"] if args.scenario == "all" else [args.scenario]
    comparison_configs: list[Path] = []
    for scenario in scenarios:
        for group in groups:
            configs = _group_configs(scenario, group)
            if not configs:
                raise FileNotFoundError(f"No configs found for scenario={scenario}, group={group}")
            for config in configs:
                comparison_configs.append(config)
                _run_config(config, skip_existing=args.skip_existing)

    unique_configs: list[Path] = []
    seen = set()
    for config in comparison_configs:
        if config not in seen:
            unique_configs.append(config)
            seen.add(config)

    output_csv = args.output
    if not output_csv.is_absolute():
        output_csv = PROJECT_ROOT / output_csv
    _write_comparison(unique_configs, output_csv)


if __name__ == "__main__":
    main()
