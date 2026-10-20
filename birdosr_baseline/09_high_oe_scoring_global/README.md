# 09 Higher-OE Global-Threshold Scoring Ablation

Purpose: run an exhaustive score-combination ablation under the selected
BirdOSR-Final decision framework.

BirdOSR-Final framework:

```text
Input 8s audio
-> BirdMAE frozen encoder
-> Linear known classifier + ProtoSphere head
-> CE + ProtoSphere + High-Energy OE training
-> MLS/Energy/MSP/Margin/proto_distance scoring
-> learned fusion score
-> subset-calibrated global threshold
-> known species or unknown
```

This suite keeps every module fixed except the score subset used by learned
fusion. It evaluates all non-empty combinations of:

```text
MLS
Energy
MSP
Margin
proto_distance
```

Number of settings:

```text
2^5 - 1 = 31 score combinations per scenario
```

Implementation details:

```text
source checkpoint:
  *_birdmae_noaug_oeps_oe_high_energy/checkpoint_best.pt

training modules:
  BirdMAE frozen encoder
  Linear known classifier
  ProtoSphere head
  High-Energy OE loss

OSR scoring:
  learned_fusion_subset_tau

threshold:
  subset-calibrated global tau

changed variable:
  osr.scores only
```

The suite treats score order as order-invariant combinations, not permutations.
Permuting the same score set only changes input column order and does not define
a new OSR method.

Examples:

```text
learned(MLS) + subset global tau
learned(Energy) + subset global tau
learned(MLS + Energy) + subset global tau
learned(MLS + Energy + MSP + Margin + proto_distance) + subset global tau
```

Run:

```bash
bash birdosr_baseline/ablation_suites/09_high_oe_scoring_global/run_background.sh all
```

Tail:

```bash
tail -f birdosr_baseline/outputs/ablation_suites/09_high_oe_scoring_global/run_all.log
```

Result:

```text
birdosr_baseline/outputs/ablation_suites/09_high_oe_scoring_global/all_09_high_oe_scoring_global_comparison.csv
birdosr_baseline/outputs/ablation_suites/09_high_oe_scoring_global/all_09_high_oe_scoring_global_comparison.txt
```
