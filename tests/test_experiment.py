"""Small artificial fixtures for CI; they are not evidence about UCI performance."""
import io
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest
from hydraulic.run import (SensorSource, consecutive_setting_blocks,
                           extract_features, make_group_holdout, one_seed,
                           CLASSES)


def toy(n_per_block=6, repetitions=9):
    labels = np.repeat(list(CLASSES) * repetitions, n_per_block)
    profile = np.column_stack([np.full(len(labels), 100), labels,
                               np.zeros(len(labels)), np.full(len(labels), 130),
                               np.zeros(len(labels))]).astype(float)
    group = consecutive_setting_blocks(profile)
    rng = np.random.default_rng(8)
    x = pd.DataFrame(rng.normal(size=(len(labels), 24)),
                     columns=[f"{s}_{stat}" for s in ("PS1", "FS1", "TS1", "VS1")
                              for stat in ("mean", "std", "min", "max", "rms", "half_mean_delta")])
    return x, labels, group


def test_extract_cycle_features():
    raw = np.array([[1, 2, 3, 4], [5, 5, 5, 5]], dtype=np.float32)
    frame = extract_features(raw, "PS1")
    assert len(frame) == 2
    assert frame.loc[0, "PS1_mean"] == 2.5
    assert frame.loc[1, "PS1_std"] == 0
    assert frame.loc[0, "PS1_half_mean_delta"] == 2


def test_consecutive_block_delineation():
    profile = np.array([[100,100,0,130,0], [100,100,0,130,0],
                        [100,90,0,130,0], [100,90,0,130,0],
                        [100,100,0,130,0]], dtype=float)
    assert consecutive_setting_blocks(profile).tolist() == [0, 0, 1, 1, 2]
    with pytest.raises(ValueError):
        consecutive_setting_blocks(profile[:, :4])


def test_group_holdout_never_shares_block():
    x, y, groups = toy()
    train, test = make_group_holdout(y, groups, 42)
    assert not set(groups[train]) & set(groups[test])
    assert set(y[train]) == set(CLASSES)
    assert set(y[test]) == set(CLASSES)


def test_single_vs_fusion_end_to_end():
    x, y, groups = toy()
    records, audit, feat = one_seed(x, y, groups, 42)
    assert len(records) == 10  # dummy, 4 alone, 1 fused, 4 ablations
    assert audit["n_blocks_train"] + audit["n_blocks_test"] == len(set(groups))
    assert len(feat) == 24
    assert all(0 <= r["macro_f1"] <= 1 for r in records)


def test_archive_supports_nested_zip(tmp_path: Path):
    inner = io.BytesIO()
    with ZipFile(inner, "w") as z:
        z.writestr("archive/sub/PS1.txt", "1\t2\n3\t4\n")
    outer = tmp_path / "download.zip"
    with ZipFile(outer, "w") as z:
        z.writestr("data.zip", inner.getvalue())
    with SensorSource(outer) as src:
        matrix = src.load("PS1.txt")
    assert matrix.shape == (2, 2)
    assert float(matrix[1, 0]) == 3
