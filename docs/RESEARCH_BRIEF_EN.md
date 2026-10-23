# Mini Research Brief 02 — Leakage-Aware Feature-Level Sensor Fusion

**Type:** independent computational baseline on a public test-rig dataset; not ADaMS-collected data.

## Research motivation

Combining modalities is attractive, but fusion improvements can be overstated when sequential data from the same operating condition appear in both model training and testing. Feature-level fusion provides a small, transparent comparison before more complex architectures are justified.

## RQ and hypothesis

Does cycle-level fusion of pressure, flow, temperature and vibration features improve classification of valve switching condition on a **group-held-out** evaluation, beyond the best single modality? Does removing individual sensors materially change performance?

The hypothesis is unresolved until evaluated on original UCI measurements.

## Dataset

UCI 447 hydraulic test rig (real experiment): 2,205 sixty-second cycles. One target: valve condition (100 optimal, 90 small lag, 80 severe lag, 73 near failure). Four channels selected from the original 17. The other component conditions are possible confounders, not targets or permitted predictors.

## Method

1. Read raw pressure 100 Hz, flow 10 Hz, temperature 1 Hz and vibration 1 Hz at their native rates.
2. Extract six descriptive statistics per **whole cycle**, yielding 24 columns.
3. Identify contiguous runs of identical 5-field experiment annotation vectors; enforce group-disjoint train/test.
4. Compare 4 sensor-only models with all-four fused model and four leave-one-out variants using fixed Random Forest hyperparameters; majority baseline included.
5. Evaluate macro-F1, balanced accuracy and per-class confusion; inspect the held-out counts and number of blocks.

## Risks and interpretation

- Runs of identical condition vectors are **proxies** for experiment blocks; true settings could recur noncontiguously.
- Classifying a labelled rig condition is not predicting future damage or diagnosing real plants.
- Environmental/process confounding remains possible; other component states change with valve state.
- Feature-level fusion does not preserve raw time-aligned relationships and cannot establish the benefit of a temporal Transformer.
- Feature importance from Random Forest is descriptive, not causal.

## Proposed ADaMS extension

Apply a block-aware evaluation and modality ablation protocol to synchronized electrical / optical / thermal WAAM signals when available. Compare robust early fusion to temporal encoders under missing modality tests. Access to manufacturer/process data and domain supervision would be needed for that claim.
