# 10 Higher-OE Threshold Calibration Ablation

Purpose: repeat threshold calibration ablation under the final high-OE-energy
checkpoint.

This is needed because the earlier `04_threshold_calibration` suite used the
default OE-ProtoSphere checkpoint. The final model changes OE energy weight, so
the threshold distribution must be revalidated.

Compared thresholds:

```text
global tau
class-wise tau
subset global tau
subset class-wise tau
```

All rows use the final full learned fusion score:

```text
MLS + Energy + MSP + Margin + proto_distance
```

Run:

```bash
bash birdosr_baseline/ablation_suites/10_high_oe_threshold/run_background.sh all
```

Tail:

```bash
tail -f birdosr_baseline/outputs/ablation_suites/10_high_oe_threshold/run_all.log
```

Result:

```text
birdosr_baseline/outputs/ablation_suites/10_high_oe_threshold/all_10_high_oe_threshold_comparison.csv
```
