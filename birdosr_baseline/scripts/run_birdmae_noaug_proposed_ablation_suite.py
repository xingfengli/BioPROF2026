#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
CONFIG_DIR = ROOT / "configs"
OUTPUT_DIR = ROOT / "outputs"

GROUP_CHOICES = ("main", "training", "scoring", "threshold", "oe", "protosphere")

MAIN_REFERENCE_CONFIGS = {
    "s1s2": [
        "s1s2_birdmae_noaug_mls.json",
        "s1s2_birdmae_noaug_classwise_tau.json",
        "s1s2_birdmae_noaug_gfct_energy_msp_margin.json",
        "s1s2_birdmae_noaug_learned_fusion_tau.json",
    ],
    "s2s1": [
        "s2s1_birdmae_noaug_mls.json",
        "s2s1_birdmae_noaug_classwise_tau.json",
        "s2s1_birdmae_noaug_gfct_energy_msp_margin.json",
        "s2s1_birdmae_noaug_learned_fusion_tau.json",
    ],
}

MAIN_TRAINING_CONFIGS = {
    "s1s2": [
        "s1s2_birdmae_noaug_protosphere.json",
        "s1s2_birdmae_noaug_oe.json",
        "s1s2_birdmae_noaug_oe_protosphere.json",
    ],
    "s2s1": [
        "s2s1_birdmae_noaug_protosphere.json",
        "s2s1_birdmae_noaug_oe.json",
        "s2s1_birdmae_noaug_oe_protosphere.json",
    ],
}

MAIN_DERIVED_CONFIGS = {
    "s1s2": [
        "s1s2_birdmae_noaug_oe_protosphere_subset_tau.json",
    ],
    "s2s1": [
        "s2s1_birdmae_noaug_oe_protosphere_subset_tau.json",
    ],
}

TRAINING_MODULE_CONFIGS = {
    "s1s2": [
        "s1s2_birdmae_noaug_learned_fusion_tau.json",
        "s1s2_birdmae_noaug_protosphere.json",
        "s1s2_birdmae_noaug_oe.json",
        "s1s2_birdmae_noaug_oe_protosphere.json",
        "s1s2_birdmae_noaug_oe_protosphere_subset_tau.json",
    ],
    "s2s1": [
        "s2s1_birdmae_noaug_learned_fusion_tau.json",
        "s2s1_birdmae_noaug_protosphere.json",
        "s2s1_birdmae_noaug_oe.json",
        "s2s1_birdmae_noaug_oe_protosphere.json",
        "s2s1_birdmae_noaug_oe_protosphere_subset_tau.json",
    ],
}

def _config_path(name: str) -> Path:
    return CONFIG_DIR / name


def _checkpoint_path(scenario: str) -> Path:
    return OUTPUT_DIR / f"{scenario}_birdmae_noaug_mls" / "checkpoint_best.pt"


def _checkpoint_for_experiment(experiment_name: str) -> Path:
    return OUTPUT_DIR / experiment_name / "checkpoint_best.pt"


def _ensure_mls_checkpoint(args: argparse.Namespace, scenario: str) -> Path:
    checkpoint = {"s1s2": args.s1s2_checkpoint, "s2s1": args.s2s1_checkpoint}[scenario]
    if checkpoint.exists():
        return checkpoint

    config = _config_path(f"{scenario}_birdmae_noaug_mls.json")
    print(f"Missing MLS checkpoint for {scenario}; training baseline first: {checkpoint}", flush=True)
    _run(
        [
            sys.executable,
            str(ROOT / "scripts" / "make_splits.py"),
            "--config",
            str(config.relative_to(PROJECT_ROOT)),
        ]
    )
    _run(
        [
            sys.executable,
            str(ROOT / "scripts" / "train.py"),
            "--config",
            str(config.relative_to(PROJECT_ROOT)),
        ],
        log_path=OUTPUT_DIR / config.stem / "train.log",
    )
    if not checkpoint.exists():
        raise FileNotFoundError(f"MLS training finished but checkpoint was not created: {checkpoint}")
    return checkpoint


def _ensure_experiment_checkpoint(experiment_name: str) -> Path:
    checkpoint = _checkpoint_for_experiment(experiment_name)
    if checkpoint.exists():
        return checkpoint

    config = CONFIG_DIR / f"{experiment_name}.json"
    if not config.exists():
        raise FileNotFoundError(
            f"Missing checkpoint for experiment '{experiment_name}' and no source config was found: {config}"
        )
    print(f"Missing checkpoint for {experiment_name}; training source experiment first: {checkpoint}", flush=True)
    _train_and_evaluate(config, skip_existing=False)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Training finished but checkpoint was not created: {checkpoint}")
    return checkpoint


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


def _source_checkpoint_path(config_path: Path) -> Path:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    source = cfg.get("source_checkpoint_experiment")
    if not source:
        source = cfg["experiment_name"]
    return OUTPUT_DIR / str(source) / "checkpoint_best.pt"


def _ensure_source_checkpoint(config_path: Path) -> Path:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    source = cfg.get("source_checkpoint_experiment")
    if not source:
        source = cfg["experiment_name"]
    return _ensure_experiment_checkpoint(str(source))


def _run(cmd: list[str], log_path: Path | None = None) -> None:
    print(" ".join(cmd), flush=True)
    if log_path is None:
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)


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


def _generated_group_configs(scenario: str, group: str) -> list[Path]:
    group_name = {
        "scoring": "scoring_ablation",
        "threshold": "threshold_ablation",
        "oe": "oe_ablation",
        "protosphere": "protosphere_ablation",
    }[group]
    paths = []
    for path in sorted(CONFIG_DIR.glob(f"{scenario}_birdmae_noaug_oeps_*.json")):
        cfg = json.loads(path.read_text(encoding="utf-8"))
        if cfg.get("ablation_group") == group_name:
            paths.append(path)
    return paths


def _main_configs(scenario: str) -> list[Path]:
    return (
        [_config_path(name) for name in MAIN_REFERENCE_CONFIGS[scenario]]
        + [_config_path(name) for name in MAIN_TRAINING_CONFIGS[scenario]]
        + [_config_path(name) for name in MAIN_DERIVED_CONFIGS[scenario]]
    )


def _run_config_with_dependencies(
    config: Path,
    mls_checkpoint: Path,
    skip_existing: bool,
) -> None:
    cfg = json.loads(config.read_text(encoding="utf-8"))
    experiment_name = cfg["experiment_name"]
    if (
        experiment_name.endswith("_mls")
        or experiment_name.endswith("_classwise_tau")
        or "gfct" in experiment_name
        or experiment_name.endswith("_learned_fusion_tau")
    ):
        _evaluate(config, mls_checkpoint, skip_existing=skip_existing)
    elif "source_checkpoint_experiment" in cfg:
        _evaluate(config, _ensure_source_checkpoint(config), skip_existing=skip_existing)
    else:
        _train_and_evaluate(config, skip_existing=skip_existing)


def _run_group(args: argparse.Namespace, scenario: str, group: str) -> list[Path]:
    completed: list[Path] = []
    mls_checkpoint = _ensure_mls_checkpoint(args, scenario)

    if group == "main":
        for name in MAIN_REFERENCE_CONFIGS[scenario]:
            config = _config_path(name)
            completed.append(config)
            _evaluate(config, mls_checkpoint, skip_existing=args.skip_existing)
        for name in MAIN_TRAINING_CONFIGS[scenario]:
            config = _config_path(name)
            completed.append(config)
            _train_and_evaluate(config, skip_existing=args.skip_existing)
        for name in MAIN_DERIVED_CONFIGS[scenario]:
            config = _config_path(name)
            completed.append(config)
            _evaluate(config, _ensure_source_checkpoint(config), skip_existing=args.skip_existing)
        return completed

    if group == "training":
        for name in TRAINING_MODULE_CONFIGS[scenario]:
            config = _config_path(name)
            completed.append(config)
            _run_config_with_dependencies(config, mls_checkpoint, skip_existing=args.skip_existing)
        return completed

    for config in _generated_group_configs(scenario, group):
        completed.append(config)
        cfg = json.loads(config.read_text(encoding="utf-8"))
        if "source_checkpoint_experiment" in cfg:
            _evaluate(config, _ensure_source_checkpoint(config), skip_existing=args.skip_existing)
        else:
            _train_and_evaluate(config, skip_existing=args.skip_existing)
    return completed


def _parse_groups(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return list(GROUP_CHOICES)
    groups = [item.strip().lower() for item in value.split(",") if item.strip()]
    invalid = [item for item in groups if item not in GROUP_CHOICES]
    if invalid:
        valid = ", ".join(("all",) + GROUP_CHOICES)
        raise ValueError(f"Invalid groups {invalid}. Valid groups: {valid}")
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full BirdMAE proposed-method ablation suite.")
    parser.add_argument("--scenario", choices=["all", "s1s2", "s2s1"], default="all")
    parser.add_argument(
        "--groups",
        default="all",
        help="Comma-separated groups: main,training,scoring,threshold,oe,protosphere, or all.",
    )
    parser.add_argument("--s1s2-checkpoint", type=Path, default=_checkpoint_path("s1s2"))
    parser.add_argument("--s2s1-checkpoint", type=Path, default=_checkpoint_path("s2s1"))
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR / "birdmae_noaug_proposed_ablation_suite_comparison.csv",
    )
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    _run([sys.executable, str(ROOT / "scripts" / "make_geometry_osr_configs.py"), "--overwrite"])
    _run([sys.executable, str(ROOT / "scripts" / "make_proposed_ablation_configs.py"), "--overwrite"])

    groups = _parse_groups(args.groups)
    scenarios = ["s1s2", "s2s1"] if args.scenario == "all" else [args.scenario]
    comparison_configs: list[Path] = []
    for scenario in scenarios:
        for group in groups:
            comparison_configs.extend(_run_group(args, scenario, group))

    # Preserve order while removing duplicate configs when groups overlap.
    unique_configs = []
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
