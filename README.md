# BioPROF2026
Bioacoustic Masked Representation with Prototype Sphere Learning and Multi-Score Fusion for Open-Set Bird Species Recognition

BioPROF is a framework for cross-corpus open-set bird vocalization recognition. It combines a frozen Bird-MAE encoder, outlier-exposed prototype sphere learning, learned multi-score fusion, and unknown-subset-aware global threshold calibration to recognize known acoustic classes and reject unknown inputs under corpus shift.

This repository contains the implementation and experimental configurations accompanying the manuscript with the title above.


## Contents

- [Overview](#overview)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training and Evaluation](#training-and-evaluation)
- [Experimental Settings](#experimental-settings)
- [Outputs and Metrics](#outputs-and-metrics)
- [Ablation Studies](#ablation-studies)
- [Reproducibility](#reproducibility)
- [Resource Availability](#resource-availability)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview

Acoustic monitoring systems encounter recording conditions and sound categories that differ from their training data. BioPROF addresses this setting through source-only training and calibration: model optimization, score-fusion learning, and threshold selection use source-corpus data, while target-corpus recordings are reserved exclusively for evaluation.

![BioPROF Blockdiagram](https://github.com/xingfengli/BioPROF2026/blob/main/birdosr_baseline/BioPPROFR3.pdf)

The framework comprises four components:

1. **Bioacoustic masked representation:** A pretrained Bird-MAE encoder extracts audio representations and remains frozen during downstream training.
2. **Outlier-exposed prototype sphere learning:** A known-class classifier and class-specific prototype-sphere parameters are optimized using labeled source known recordings and a separate source outlier-exposure (OE) pool.
3. **Learned multi-score fusion:** Five complementary cues—maximum logit score, energy-related knownness, maximum softmax probability, logit margin, and prototype distance—are standardized and combined through a linear fusion model learned on held-out source calibration data.
4. **Unknown-subset-aware global threshold calibration:** One rejection threshold is selected by maximizing balanced open-set accuracy averaged over unknown-label subsets. The fusion parameters, normalization statistics, and threshold are subsequently fixed for target-corpus evaluation.

Given a fused knownness score, an input is assigned its predicted known class when its score is greater than or equal to the calibrated threshold; otherwise, it is rejected as unknown. 

> **Final method:** `osr.method = learned_fusion_subset_tau`. BioPROF calibrates a single global threshold. Class-wise alternatives retained in the repository are comparison variants.

## Results

The following results are reported in the accompanying manuscript. All values are percentages.

| Scenario | ACC_K ↑ | ACC_U ↑ | Balanced ACC ↑ | AUROC ↑ | FPR@95TPR ↓ | OSCR ↑ |
|:--|--:|--:|--:|--:|--:|--:|
| S1 → S2 | 75.98 | 79.85 ± 6.79 | 77.91 ± 3.40 | 85.40 | 70.53 | 80.81 |
| S2 → S1 | 81.07 | 71.86 ± 6.87 | 76.46 ± 3.44 | 85.82 | 67.34 | 82.29 |

`ACC_U` and balanced `ACC` are reported as mean ± standard deviation over all target unknown-label subsets of size three. These standard deviations reflect variation in unknown-class composition.

Balanced open-set accuracy is defined as:

$$
\mathrm{ACC} = \frac{1}{2}\mathrm{ACC}_{K} + \frac{1}{2}\mathrm{ACC}_{U}.
$$

This metric gives equal weight to known-class recognition and unknown rejection. 


## Repository Structure

| Path | Description |
|:--|:--|
| `birdosr_baseline/src/birdosr/` | Data loading, audio processing, models, training, scoring, calibration, and metrics |
| `birdosr_baseline/configs/` | Training, evaluation, and ablation configurations |
| `birdosr_baseline/scripts/` | Command-line entry points, experiment runners, and visualization utilities |
| `birdosr_baseline/ablation_suites/` | Grouped ablation launchers and supporting documentation |
| `tools/` | Path configuration, proposed-method orchestration, and structural checks |
| `docs/` | Data protocol, reproduction instructions, and method-to-code mapping |

The directory name `birdosr_baseline` is retained for compatibility with the supplied scripts and configuration paths. It contains both the proposed BioPROF method and comparison variants.

## Installation

### Reference Environment

- Linux x86_64
- Python 3.10
- PyTorch 2.5.1 and torchaudio 2.5.1 with CUDA 12.1 wheels
- A compatible NVIDIA GPU driver

Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate

python -m pip install -r birdosr_baseline/requirements.txt
```

Run all subsequent commands from the repository root.

The pinned environment targets CUDA 12.1. Other platforms and dependency combinations have not been validated for numerical reproduction.

Optional visualization dependencies can be installed separately:

```bash
python -m pip install -r requirements-visualization.txt
```

### Pretrained Bird-MAE

Obtain the complete pretrained model from:

[DBD-research-group/Bird-MAE-Base](https://huggingface.co/DBD-research-group/Bird-MAE-Base)

Place the model snapshot, including its weights, configuration, feature extractor, and required custom modeling code, in:

```text
models/Bird-MAE-Base/
```

The corresponding configuration is:

```json
{
  "model": {
    "backbone": "birdmae:models/Bird-MAE-Base",
    "pretrained": true,
    "dropout": 0.2,
    "finetune_mode": "linear_probe"
  }
}
```

The implementation uses `trust_remote_code=True` to load the upstream model. Use a reviewed snapshot and record its exact revision for reproducibility. The upstream model's terms apply independently of this project.

## Data Preparation

### Directory Layout

Arrange the S1 and S2 recordings as follows:

| Partition | Expected path |
|:--|:--|
| S1 known | `data/SCorpora/S1/knownS1/<LABEL>/<audio files>` |
| S1 unknown | `data/SCorpora/S1/unknownS1/<LABEL>/<audio files>` |
| S2 known | `data/SCorpora/S2/knownS2/<LABEL>/<audio files>` |
| S2 unknown | `data/SCorpora/S2/unknownS2/<LABEL>/<audio files>` |

For example:

```text
data/SCorpora/S1/knownS1/AMCR/example.wav
```

Audio discovery is recursive. Recognized file extensions are `.wav`, `.flac`, `.mp3`, `.ogg`, and `.m4a`; decoding support depends on the local audio backend.

### Class Labels

The benchmark contains bird vocalizations and other acoustic categories:

```text
AMCR, AMRO, BLJA, ECMC, ECMK, ENGE, FFCR, RNFL
```

The target unknown-label pools differ by transfer direction:

| Direction | Target unknown labels |
|:--|:--|
| S1 → S2 | BAWW, BCCH, BUBP, EAPH, RBNU |
| S2 → S1 | DDCC, DGBR, DOWO, EGSK, NOCA, RCCR, SREN, WBNU |

Preserve the label identifiers and their ordering in the supplied configurations.

### Cross-Corpus Protocol

For each transfer direction:

- Source known recordings are divided into 90% training and 10% validation.
- Target known recordings and the selected target unknown pools are used exclusively for testing.
- Fusion learning and threshold calibration use the held-out source validation pool.

Splits are constructed using seeded per-class shuffling and rounded sample counts.

Unless `calibration_unknown_labels` is specified, the source unknown-label pool is inferred from all subdirectories under the source unknown directory. Additional folders can therefore change the experiment.

The manuscript reports the following numbers of 8-second clips:

| Direction | Train-K | Val-K | OE-U | Val-U | Test-K | Test-U |
|:--|--:|--:|--:|--:|--:|--:|
| S1 → S2 | 13,137 | 1,461 | 3,266 | 1,306 | 26,710 | 4,079 |
| S2 → S1 | 24,039 | 2,670 | 2,039 | 816 | 14,598 | 6,533 |

Audio is processed at 32 kHz through Bird-MAE's native feature-extraction pipeline. 

Audio recordings are not bundled with this repository. Corpus acquisition details and the exact recording are referred to “Morgan MM, Braasch J. Open set classification strategies for long-term environmental field recordings for bird species recognition. The Journal of the Acoustical Society of America. 2022 Jun 1;151(6):4028-38. Zenodo. https://zenodo.org/records/6456604”.

### Configure Asset Paths

The prepared configurations use the following default paths:

```text
data/SCorpora/
models/Bird-MAE-Base/
```

To use external asset directories:

```bash
python tools/configure_paths.py \
  --data-root /absolute/path/to/SCorpora \
  --birdmae-dir /absolute/path/to/Bird-MAE-Base
```

Both directories must already exist. This helper updates asset paths in the copied configurations without changing experiment identities or hyperparameters.

## Training and Evaluation

### Complete Pipeline

Set the reproducibility environment variables:

```bash
export PYTHONHASHSEED=26
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

Run the proposed method in both directions:

```bash
# Train on S1 and evaluate on S2.
python tools/run_proposed.py --scenario s1s2

# Train on S2 and evaluate on S1.
python tools/run_proposed.py --scenario s2s1
```

For each direction, the runner:

1. Generates the source manifest if it does not already exist.
2. Trains the high-energy-OE source model.
3. Reuses its best checkpoint for final evaluation.
4. Learns source-side score fusion and calibrates one global threshold.
5. Evaluates the fixed decision rule on the target corpus.

The source and final configurations are paired as follows:

| Role | Configuration filename |
|:--|:--|
| S1 → S2 training | `s1s2_birdmae_noaug_oeps_oe_high_energy.json` |
| S1 → S2 evaluation | `s1s2_birdmae_noaug_final_highoe_threshold_subset_global_tau.json` |
| S2 → S1 training | `s2s1_birdmae_noaug_oeps_oe_high_energy.json` |
| S2 → S1 evaluation | `s2s1_birdmae_noaug_final_highoe_threshold_subset_global_tau.json` |

All configurations are located in `birdosr_baseline/configs/`.

**The final evaluation configuration reuses the source-training checkpoint and manifest. No separate final-evaluation checkpoint is trained.**

### Step-by-Step Commands

For S1 → S2:

```bash
# Generate the manifest.
python birdosr_baseline/scripts/make_splits.py \
  --config birdosr_baseline/configs/s1s2_birdmae_noaug_oeps_oe_high_energy.json

# Train the source model.
python birdosr_baseline/scripts/train.py \
  --config birdosr_baseline/configs/s1s2_birdmae_noaug_oeps_oe_high_energy.json

# Calibrate on source data and evaluate on the target corpus.
python birdosr_baseline/scripts/evaluate.py \
  --config birdosr_baseline/configs/s1s2_birdmae_noaug_final_highoe_threshold_subset_global_tau.json \
  --checkpoint birdosr_baseline/outputs/s1s2_birdmae_noaug_oeps_oe_high_energy/checkpoint_best.pt
```

For S2 → S1, replace every `s1s2` occurrence above with `s2s1`.

`make_splits.py` can overwrite the configured manifest. Preserve the manifest associated with any checkpoint that will be reused. The proposed runner refuses to overwrite an existing source checkpoint during training.

### Evaluate Existing Checkpoints

When the corresponding source checkpoint and manifest already exist:

```bash
python tools/run_proposed.py --scenario s1s2 --eval-only
python tools/run_proposed.py --scenario s2s1 --eval-only
```

Evaluation reruns source-side fusion learning and threshold calibration before generating target predictions. This entry point is an experimental evaluation pipeline rather than a standalone deployment interface.

## Experimental Settings

| Setting | Value |
|:--|:--|
| Backbone | Bird-MAE-Base, frozen |
| Classification head | Dropout (0.2) followed by a linear layer |
| Epochs | 50 |
| Optimizer | AdamW |
| Learning rate | 0.0003 |
| Weight decay | 0.01 |
| Known / OE batch size | 16 / 16 |
| Label smoothing | 0.1 |
| Prototype-separation weight / margin | 0.1 / 1.0 |
| Sphere-containment weight | 0.05 |
| Radius-regularization weight | 0.0001 |
| Initial sphere radius | 1.0 |
| OE LSE / uniform-posterior / distance weight | 0.2 / 0.1 / 0.01 |
| OE distance margin | 5.0 |
| Fusion steps / learning rate / L2 | 2,000 / 0.05 / 0.001 |
| Fusion weights | Unconstrained |
| Unknown-subset size | 3 |
| Threshold objective | `ACC_subset_mean` |
| Maximum threshold candidates | 200 |
| Seed | 26 |

The final five-score configuration is:

```json
{
  "osr": {
    "method": "learned_fusion_subset_tau",
    "scores": [
      "mls",
      "energy",
      "msp",
      "margin",
      "proto_distance"
    ],
    "unknown_subset_size": 3,
    "subset_tau_objective": "ACC_subset_mean",
    "subset_tau_max_candidates": 200
  }
}
```

This excerpt shows the main method settings; use the complete supplied configurations for execution.

In the implementation, `energy` denotes the log-sum-exp knownness cue and follows a higher-is-more-known convention.

## Outputs and Metrics

Training saves `checkpoint_best.pt` under:

```text
birdosr_baseline/outputs/<direction>_birdmae_noaug_oeps_oe_high_energy/
```

Final evaluation writes reports under:

```text
birdosr_baseline/outputs/<direction>_birdmae_noaug_final_highoe_threshold_subset_global_tau/
```

Here, `<direction>` is `s1s2` or `s2s1`.

| File | Description |
|:--|:--|
| `metrics.csv` | Aggregate metrics and calibrated global threshold |
| `fusion_params.csv` | Learned fusion and normalization parameters |
| `val_predictions.csv` | Source calibration scores and predictions |
| `test_predictions.csv` | Target test scores and predictions |
| `known_class_report.csv` | Known-class recognition results |
| `unknown_class_report.csv` | Unknown-class rejection results |
| `known_confusion_matrix.csv` | Known-class confusion matrix |
| `unknown_to_known_matrix.csv` | Unknown-to-known confusion counts |

Principal metric fields include:

```text
ACC_k
ACC_u_subset_mean
ACC_u_subset_std
ACC_subset_mean
ACC_subset_std
AUROC
FPR@95TPR
OSCR
```

Subset-aware evaluation considers every three-label combination from the target unknown pool:

- **S1 → S2:** 10 unknown-label subsets.
- **S2 → S1:** 56 unknown-label subsets.

Known-class accuracy remains fixed across these subsets.

The five-score fusion study separately evaluates 31 nonempty score combinations, corresponding to `2^5 − 1`. These score combinations must not be confused with unknown-label subsets.

## Ablation Studies

The supplied runners support comparisons of training components, score combinations, and threshold-calibration strategies.

Inspect the available options:

```bash
python birdosr_baseline/scripts/run_birdmae_noaug_final_ablation_suite.py --help
```

For example, run the nonempty score-combination study under global thresholding:

```bash
python birdosr_baseline/scripts/run_birdmae_noaug_final_ablation_suite.py \
  --scenario all \
  --groups high_oe_scoring_global \
  --skip-existing
```

Suite runners may train missing source models. Historical configurations include class-wise thresholds and earlier method variants. Use the source/final configuration pairing documented above for the complete BioPROF method.

## Reproducibility

Run the dependency-free structural check:

```bash
python tools/check_release.py
```

Verify the reproducibility settings of the original active configuration collection:

```bash
python birdosr_baseline/scripts/verify_proposed_reproducibility.py --seed 26
```

For each experimental run, retain:

- The exact recording inventory and split manifests.
- Training and final evaluation configurations.
- The pretrained Bird-MAE revision and file checksums.
- Source checkpoints, fusion parameters, calibrated thresholds, and evaluation reports.
- Package versions, GPU model, and driver information.

The proposed configurations enable deterministic training. Nevertheless, hardware, GPU kernels, library versions, and upstream model code can affect numerical reproducibility.


## Citation

If you use BioPROF in your research, please cite the accompanying manuscript when its public bibliographic record becomes available. Please also acknowledge the original Bird-MAE and dataset publications as appropriate.

Complete authorship and publication details will be updated when confirmed.

## License

The project code license is pending confirmation by the rights holders.

Third-party datasets, pretrained models, and upstream custom code remain subject to their respective terms.

## Acknowledgments

BioPROF builds on the pretrained [Bird-MAE](https://huggingface.co/DBD-research-group/Bird-MAE-Base) representation and the S1/S2 cross-corpus acoustic recognition benchmark. We acknowledge the authors and maintainers of these resources and the open-source libraries used in this implementation.

## Questions and Contributions

Please use GitHub Issues for questions and bug reports. Include the relevant configuration, command, software environment, and error message to help reproduce the issue.
