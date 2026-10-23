# Industrial test-rig sensor-fusion protocol

## RQ
Does concatenating basic statistical summaries from pressure, flow, temperature, and vibration improve **valve-condition classification** compared with the best individual modality and 3-sensor ablations?

## Data and authenticity
UCI 447 (Helwig, Pignanelli, Schütze), **real experimental hydraulic rig**, 2205 distinct 60-second cycles recorded with 17 channels of varying sampling rates. Here we use **four** channels: `PS1` pressure 100 Hz, `FS1` flow 10 Hz, `TS1` temperature 1 Hz, `VS1` vibration 1 Hz. Primary label is `profile.txt`, column 2: 100 optimal, 90 small lag, 80 severe lag, 73 near total failure. Other columns: cooler, pump leakage, accumulator and stability flag; **no profile information enters model features**.

## Representation
Six summary statistics per channel per cycle: mean, standard deviation, min, max, RMS, second-half minus first-half mean. These create 24 feature columns. The cycle is the unit of prediction; **a cycle's individual 6000 pressure samples are never split between train and test**. No alignment of raw signals at millisecond scale is claimed.

## Main leakage control
Experiments frequently hold one condition configuration over several contiguous cycles. Ordinary random cycle-level train/test splitting leaks highly similar conditions across partitions and tends to overestimate performance. Build groups from **runs of identical complete profile vectors** (cooler, valve, pump, accumulator, stable); any adjacent change opens a new block. For each seed select a group-disjoint 80/20 holdout, requiring all valve classes in both train and test. Group candidates are selected using label distribution only, NEVER feature values or evaluation metrics.

**Residual risk:** identical operating settings may recur noncontiguously; contiguous block grouping does not ensure cross-hardware or fully out-of-distribution independence. This analysis tests within one rig. Other changing component states can confound valve severity prediction. Holdout is not a new machine.

## Models
Majority-class dummy; 4 single sensors (six features each); all-four feature concatenation (24 features); 4 fusion ablations each excluding one complete modality (18 features). All non-dummy models: fixed Random Forest 160 trees, min leaf 2, balanced class weights, **training-only** median imputation; no hyperparameter search.

## Scores
Primary: Macro-F1 (four classes). Secondary: Balanced Accuracy and ordinary Accuracy. Provide per-class precision/recall/F1 for fused model, grouped test confusion matrix and train/test class counts. Feature importance is descriptive and NOT causal. Optional `--include-naive` calculates a deliberately suspect random cycle split, clearly marked as sensitivity analysis and not included in main comparison.

## Repeated seeds
`--seeds 42 52 62` repeats group-disjoint holdout and reports means + descriptive standard deviation. Overlapping training groups mean these aren't independent experiments or a reliable independent confidence interval.

## Clinical/manufacturing claims NOT supported
Not additive manufacturing/WAAM data, not high-speed real-time deployed inference, not universal sensor fusion, not reliable prediction of future failures, not causal fault localization or validation across different rigs.
