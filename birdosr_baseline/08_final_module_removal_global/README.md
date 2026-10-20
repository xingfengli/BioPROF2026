# 08 Final Module-Removal Global Ablation

Purpose: re-check final module removal under the global-threshold final-model
setting, while keeping this suite separate from the original
`08_final_module_removal` class-wise-threshold suite.

Current final-method reference:

```text
BirdMAE-noaug + OE + ProtoSphere + High-Energy OE + Full Learned Fusion + Subset Global Tau
```

Compared settings:

```text
FinalModule: full final model_class-wise
FinalModule: w/o OE
FinalModule: w/o ProtoSphere
FinalModule: w/o subset class-wise tau
FinalModule: w/o higher OE energy
FinalModule: full final model
```

Experiment definitions:

```text
FinalModule: full final model_class-wise
  Input 8s audio
  -> BirdMAE frozen encoder
  -> Linear known classifier + ProtoSphere head
  -> CE + ProtoSphere + High-Energy OE training
  -> MLS/Energy/MSP/Margin/proto_distance scoring
  -> learned fusion score
  -> subset-calibrated class-wise threshold
  -> known species or unknown

FinalModule: w/o OE
  Input 8s audio
  -> BirdMAE frozen encoder
  -> Linear known classifier + ProtoSphere head
  -> CE + ProtoSphere training
  -> MLS/Energy/MSP/Margin/proto_distance scoring
  -> learned fusion score
  -> subset-calibrated global threshold
  -> known species or unknown

FinalModule: w/o ProtoSphere
  Input 8s audio
  -> BirdMAE frozen encoder
  -> Linear known classifier
  -> CE + High-Energy OE training
  -> MLS/Energy/MSP/Margin scoring
  -> learned fusion score
  -> subset-calibrated global threshold
  -> known species or unknown

FinalModule: w/o subset class-wise tau
  Input 8s audio
  -> BirdMAE frozen encoder
  -> Linear known classifier + ProtoSphere head
  -> CE + ProtoSphere + High-Energy OE training
  -> MLS/Energy/MSP/Margin/proto_distance scoring
  -> learned fusion score
  -> ordinary class-wise threshold
  -> known species or unknown

FinalModule: w/o higher OE energy
  Input 8s audio
  -> BirdMAE frozen encoder
  -> Linear known classifier + ProtoSphere head
  -> CE + ProtoSphere + default OE training
  -> MLS/Energy/MSP/Margin/proto_distance scoring
  -> learned fusion score
  -> subset-calibrated global threshold
  -> known species or unknown

FinalModule: full final model
  Input 8s audio
  -> BirdMAE frozen encoder
  -> Linear known classifier + ProtoSphere head
  -> CE + ProtoSphere + High-Energy OE training
  -> MLS/Energy/MSP/Margin/proto_distance scoring
  -> raw equal-weight fusion score
  -> subset-calibrated global threshold
  -> known species or unknown
```

Coverage and checkpoint behavior:

```text
full final model_class-wise        reuses high-OE-energy full checkpoint
w/o OE                             reuses existing ProtoSphere checkpoint
w/o ProtoSphere                    trains high-OE-energy OE-only checkpoint if missing
w/o subset class-wise tau          reuses high-OE-energy full checkpoint
w/o higher OE energy               reuses default-OE OE-ProtoSphere checkpoint
full final model                   reuses high-OE-energy full checkpoint; raw fusion only
```

Run:

```bash
bash birdosr_baseline/ablation_suites/08_final_module_removal_global/run_background.sh all
```

Tail:

```bash
tail -f birdosr_baseline/outputs/ablation_suites/08_final_module_removal_global/run_all.log
```

Result:

```text
birdosr_baseline/outputs/ablation_suites/08_final_module_removal_global/all_08_final_module_removal_global_comparison.csv
birdosr_baseline/outputs/ablation_suites/08_final_module_removal_global/all_08_final_module_removal_global_comparison.txt
```
