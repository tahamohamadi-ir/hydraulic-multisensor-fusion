"""Real test-rig sensor fusion: pressure, flow, temperature and vibration.

IMPORTANT: Adjacent cycles may reflect the same experimental setting.
Primary evaluation groups contiguous identical profile vectors to mitigate leakage.
"""
from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score,
                             classification_report, ConfusionMatrixDisplay)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline

SENSORS = {"PS1": {"unit": "bar", "hz": 100},
           "FS1": {"unit": "l/min", "hz": 10},
           "TS1": {"unit": "degC", "hz": 1},
           "VS1": {"unit": "mm/s", "hz": 1}}
CLASSES = (73, 80, 90, 100)
LABELS = {73: "near failure", 80: "severe lag", 90: "small lag", 100: "optimal"}
PROFILE_COLUMNS = ("cooler", "valve", "pump", "accumulator", "stable")


def _member_name(zip_file: ZipFile, basename: str) -> str | None:
    names = [s for s in zip_file.namelist() if s.split("/")[-1].lower() == basename.lower()]
    if len(names) > 1:
        raise ValueError(f"Ambiguous duplicate {basename} in ZIP")
    return names[0] if names else None


class SensorSource:
    """Read only requested UCI sensors, directly from folder or (possibly nested) ZIP."""

    def __init__(self, path: Path):
        self.path = path
        self._zips: list[ZipFile] = []
        self._buffers: list[BytesIO] = []
        if path.is_file() and path.suffix.lower() == ".zip":
            top = ZipFile(path)
            self._zips.append(top)
            for item in top.namelist():
                if item.lower().endswith(".zip"):
                    data = BytesIO(top.read(item))
                    self._buffers.append(data)
                    self._zips.append(ZipFile(data))
        elif not path.is_dir():
            raise FileNotFoundError(f"Expected extracted UCI folder or ZIP file: {path}")

    def load(self, basename: str, dtype=np.float32) -> np.ndarray:
        if self.path.is_dir():
            matches = [p for p in self.path.rglob("*") if p.is_file() and p.name.lower() == basename.lower()]
            if len(matches) != 1:
                raise FileNotFoundError(f"Expected exactly one {basename} beneath {self.path}, got {len(matches)}")
            data = np.loadtxt(matches[0], dtype=dtype)
        else:
            matches = [(z, _member_name(z, basename)) for z in self._zips]
            matches = [(z, member) for z, member in matches if member is not None]
            if len(matches) != 1:
                raise FileNotFoundError(f"Expected exactly one {basename} in UCI ZIP(s), got {len(matches)}")
            archive, member = matches[0]
            with archive.open(member) as stream:
                data = np.loadtxt(stream, dtype=dtype)
        if data.ndim != 2 or not np.isfinite(data).all():
            raise ValueError(f"Invalid nonfinite or non-matrix data in {basename}")
        return data

    def close(self) -> None:
        for z in self._zips:
            z.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def extract_features(matrix: np.ndarray, sensor: str) -> pd.DataFrame:
    """Summary features from each complete 60-second cycle, not sample-level splits."""
    if matrix.ndim != 2 or matrix.shape[1] < 2:
        raise ValueError(f"{sensor}: expected at least two samples per cycle")
    center = matrix.shape[1] // 2
    left, right = matrix[:, :center], matrix[:, center:]
    values = {
        f"{sensor}_mean": matrix.mean(axis=1),
        f"{sensor}_std": matrix.std(axis=1),
        f"{sensor}_min": matrix.min(axis=1),
        f"{sensor}_max": matrix.max(axis=1),
        f"{sensor}_rms": np.sqrt((matrix.astype(np.float64) ** 2).mean(axis=1)),
        f"{sensor}_half_mean_delta": right.mean(axis=1) - left.mean(axis=1),
    }
    return pd.DataFrame(values).astype(np.float32)


def consecutive_setting_blocks(profile: np.ndarray) -> np.ndarray:
    """New block whenever *any* profile annotation changes in adjacent cycles.

    The label/condition vector is used ONLY for constructing holdout groups;
    it is never included in model input features.
    """
    if profile.ndim != 2 or profile.shape[1] != 5:
        raise ValueError("profile.txt needs exactly five columns")
    changed = np.any(profile[1:] != profile[:-1], axis=1)
    return np.concatenate(([0], np.cumsum(changed))).astype(int)


def make_group_holdout(y: np.ndarray, groups: np.ndarray, seed: int,
                       test_size: float = .20) -> tuple[np.ndarray, np.ndarray]:
    """Select a group-disjoint split with all 4 valve classes on both sides.

    Candidate selection uses class distribution only, never measured features,
    predictions, or metrics. Avoids accidental absence of a rare class in test.
    """
    if len(y) != len(groups):
        raise ValueError("y/groups length mismatch")
    overall = pd.Series(y).value_counts(normalize=True).reindex(CLASSES, fill_value=0)
    best = None
    split = GroupShuffleSplit(n_splits=100, test_size=test_size, random_state=seed)
    for tr, te in split.split(np.zeros(len(y)), y, groups):
        if not (set(CLASSES) <= set(y[tr]) and set(CLASSES) <= set(y[te])):
            continue
        hist = pd.Series(y[te]).value_counts(normalize=True).reindex(CLASSES, fill_value=0)
        mismatch = float(np.abs(hist - overall).sum() + 0.5 * abs(len(te) / len(y) - test_size))
        if best is None or mismatch < best[0]:
            best = (mismatch, tr, te)
    if best is None:
        raise ValueError("Cannot produce group-disjoint train/test with all valve classes; inspect blocks")
    _, tr, te = best
    if set(groups[tr]) & set(groups[te]):
        raise RuntimeError("BUG: overlapping experimental blocks")
    return tr, te


def classifier(seed: int) -> Pipeline:
    return Pipeline([
        ("train_only_median", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(n_estimators=160, min_samples_leaf=2,
                                      class_weight="balanced_subsample", random_state=seed, n_jobs=-1)),
    ])


def evaluate(y_true: np.ndarray, pred: np.ndarray) -> dict:
    return {"macro_f1": float(f1_score(y_true, pred, labels=CLASSES,
                                       average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
            "accuracy": float(np.mean(y_true == pred)),
            "n_test": int(len(y_true))}


def build_tables(src: SensorSource) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    profile = src.load("profile.txt", dtype=np.float64)
    if profile.shape[1] != 5:
        raise ValueError("Unexpected target profile columns")
    y = profile[:, 1].astype(int)
    if set(y) != set(CLASSES):
        raise ValueError(f"Expected all {CLASSES} labels, saw {sorted(set(y))}")
    frames = []
    for sensor, meta in SENSORS.items():
        print(f"Reading {sensor}.txt ({meta['hz']} Hz)")
        matrix = src.load(f"{sensor}.txt")
        if matrix.shape[0] != len(profile) or matrix.shape[1] != meta["hz"] * 60:
            raise ValueError(f"{sensor}: expected ({len(profile)}, {meta['hz']*60}), got {matrix.shape}")
        frames.append(extract_features(matrix, sensor))
    x = pd.concat(frames, axis=1)
    if x.isna().any().any():
        raise ValueError("Unexpected missing extracted features")
    return x, y, profile, consecutive_setting_blocks(profile)


def one_seed(x: pd.DataFrame, y: np.ndarray, groups: np.ndarray, seed: int,
             report_directory: Path | None = None) -> tuple[list[dict], dict, pd.DataFrame]:
    tr, te = make_group_holdout(y, groups, seed)
    configs: dict[str, list[str]] = {sensor: [f for f in x.columns if f.startswith(f"{sensor}_")]
                                      for sensor in SENSORS}
    configs["fusion_all_four"] = list(x.columns)
    for excluded in SENSORS:
        configs[f"fusion_without_{excluded}"] = [f for f in x.columns if not f.startswith(f"{excluded}_")]
    rows = []
    majority = DummyClassifier(strategy="most_frequent")
    majority.fit(x.iloc[tr], y[tr])
    rows.append({"seed": seed, "model": "majority_class", "n_features": 0,
                 **evaluate(y[te], majority.predict(x.iloc[te]))})
    importances = None
    for name, cols in configs.items():
        estimator = classifier(seed)
        estimator.fit(x.iloc[tr][cols], y[tr])
        pred = estimator.predict(x.iloc[te][cols])
        rows.append({"seed": seed, "model": name, "n_features": len(cols), **evaluate(y[te], pred)})
        if name == "fusion_all_four":
            importances = pd.DataFrame({"seed": seed, "feature": cols,
                                        "importance": estimator.named_steps["rf"].feature_importances_})
            if report_directory:
                # One confusion matrix per seed, grouped holdout only.
                cm = confusion_matrix(y[te], pred, labels=CLASSES)
                display = ConfusionMatrixDisplay(cm, display_labels=[str(c) for c in CLASSES])
                display.plot(cmap="Blues", colorbar=False, values_format="d")
                plt.title(f"Block-held-out valve condition, seed {seed}")
                plt.tight_layout()
                plt.savefig(report_directory / f"confusion_grouped_seed_{seed}.png", dpi=160)
                plt.close()
                report = classification_report(y[te], pred, labels=CLASSES,
                                               target_names=[f"{c}: {LABELS[c]}" for c in CLASSES],
                                               zero_division=0)
                (report_directory / f"classification_report_seed_{seed}.txt").write_text(
                    report, encoding="utf-8")
    audit = {"seed": seed, "n_cycles_train": len(tr), "n_cycles_test": len(te),
             "n_blocks_train": int(len(np.unique(groups[tr]))),
             "n_blocks_test": int(len(np.unique(groups[te]))),
             "classes_train": {str(c): int(np.sum(y[tr] == c)) for c in CLASSES},
             "classes_test": {str(c): int(np.sum(y[te] == c)) for c in CLASSES}}
    assert importances is not None
    return rows, audit, importances


def save_results(records: list[dict], audits: list[dict], importances: list[pd.DataFrame],
                 out: Path, extra_metadata: dict) -> None:
    df = pd.DataFrame(records)
    df.to_csv(out / "results_by_seed.csv", index=False)
    summary = (df.groupby("model", as_index=False)
               .agg(macro_f1_mean=("macro_f1", "mean"),
                    macro_f1_sd=("macro_f1", "std"),
                    balanced_accuracy_mean=("balanced_accuracy", "mean"),
                    accuracy_mean=("accuracy", "mean")))
    summary = summary.sort_values("macro_f1_mean", ascending=False)
    summary.to_csv(out / "model_comparison.csv", index=False)
    importance = pd.concat(importances)
    importance.groupby("feature", as_index=False).importance.mean().sort_values(
        "importance", ascending=False).to_csv(out / "importance_exploratory.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    sorted_summary = summary.sort_values("macro_f1_mean")
    ax.barh(sorted_summary.model, sorted_summary.macro_f1_mean)
    ax.set(xlim=(0, 1), xlabel="Macro-F1 on block-held-out cycles",
           title="Hydraulic test rig: single vs fused vs ablated sensors")
    fig.tight_layout()
    fig.savefig(out / "fusion_comparison.png", dpi=160)
    plt.close(fig)
    metadata = {"generated_utc": datetime.now(timezone.utc).isoformat(),
                "dataset": "UCI 447: real hydraulic test rig (not WAAM)",
                "doi": "10.24432/C5CW21", "evaluation": "group-held-out contiguous full-profile blocks",
                "feature_scope": "six statistical features per 60s cycle per sensor",
                "caveat": "Profiles and machine regimes can recur across blocks; single rig, other component faults co-vary."
                          "No temporal forecasting, causal localization, or new-hardware generalization claim.",
                "seeds": audits, **extra_metadata}
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    lines = ["# Results — UCI Hydraulic Test Rig", "",
             "Generated from a local copy of the original UCI dataset; evaluate only the grouped holdout values.",
             "", summary.to_markdown(index=False, floatfmt=".3f"), "",
             "## Research interpretation", "",
             "- [ ] Does fusion outperform the best single sensor by macro F1? If not, say so.",
             "- [ ] Which leave-one-sensor-out ablation changes performance most?",
             "- [ ] Which valve state is most often confused, particularly 90 vs 100?",
             "- [ ] Single-rig groups may remain dependent even with contiguous block holdout.",
             "- [ ] Other condition labels (cooler/pump/accumulator) vary and can confound classification.",
             "- [ ] Feature importance is not causal; do not infer mechanistic sensor relevance.", ""]
    (out / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(summary.to_string(index=False, float_format=lambda z: f"{z:.3f}"))


def naive_sensitivity(x: pd.DataFrame, y: np.ndarray, seed: int) -> dict:
    """Optional cautionary comparison; NEVER make this the headline result."""
    tr, te = train_test_split(np.arange(len(y)), test_size=.20, stratify=y,
                              random_state=seed)
    est = classifier(seed)
    est.fit(x.iloc[tr], y[tr])
    return {"seed": seed, "method": "NAIVE random-cycle split (potential leakage)",
            **evaluate(y[te], est.predict(x.iloc[te]))}


def file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(input_path: Path, output: Path, seeds: list[int], include_naive: bool,
        stable_only: bool = False) -> None:
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique and nonempty")
    with SensorSource(input_path) as source:
        x, y, profile, groups = build_tables(source)
    initial_n = len(y)
    if stable_only:
        # UCI stable flag: 0 = stable, 1 = conditions not yet stable.
        keep = profile[:, 4] == 0
        x, y, groups = x.loc[keep].reset_index(drop=True), y[keep], groups[keep]
        if not len(y):
            raise ValueError("No stable cycles remain")
    output.mkdir(parents=True, exist_ok=True)
    rows, audits, importances = [], [], []
    for seed in seeds:
        r, a, fi = one_seed(x, y, groups, seed, report_directory=output)
        rows.extend(r); audits.append(a); importances.append(fi)
    metadata = {"raw_cycles": initial_n, "used_cycles": len(y),
                "stable_only": stable_only, "n_experimental_blocks": int(len(np.unique(groups))),
                "class_counts": {str(c): int(np.sum(y == c)) for c in CLASSES},
                "all_profile_columns": PROFILE_COLUMNS, "input_zip_sha256": file_sha256(input_path)}
    save_results(rows, audits, importances, output, metadata)
    if include_naive:
        naive = pd.DataFrame([naive_sensitivity(x, y, seed) for seed in seeds])
        naive.to_csv(output / "CAUTION_naive_cycle_split.csv", index=False)
        print("\nCAUTION: naive cycle split is potentially leaky; NOT a headline result:")
        print(naive.to_string(index=False))
    print("\nWrote outputs to", output.resolve())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=Path("data/hydraulic_uci.zip"),
                   help="UCI ZIP (nested ZIP is supported), or directory with raw .txt files")
    p.add_argument("--output", type=Path, default=Path("results"))
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--include-naive", action="store_true", help="Potentially leaky random-cycle split diagnostic")
    p.add_argument("--stable-only", action="store_true", help="Restrict to UCI stable==0 cycles; sensitivity")
    a = p.parse_args()
    run(a.source, a.output, a.seeds, a.include_naive, a.stable_only)


if __name__ == "__main__":
    main()
