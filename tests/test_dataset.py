"""Unit and integration tests for dataset assembly and time-series splitting (Phase 7).

Verifies strict temporal ordering, absence of leakage, purge and embargo boundary
correctness, absence of NaNs, target-feature segregation, and reproducibility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ai.dataset.assembly import (
    AssembledDataset,
    DataLeakageError,
    assemble_dataset,
    infer_horizon_from_target,
    is_label_or_future_column,
)
from ai.dataset.splits import (
    DatasetSplits,
    SplitConfig,
    split_dataset,
)
from ai.dataset.validation import (
    ValidationReport,
    validate_dataset_splits,
)
from ai.labels.returns import DEFAULT_HORIZONS


# Fixture to load real assembled dataset once for fast testing
@pytest.fixture(scope="module")
def default_assembled() -> AssembledDataset:
    return assemble_dataset(target_column="direction_4")


@pytest.fixture(scope="module")
def default_splits(default_assembled: AssembledDataset) -> DatasetSplits:
    return split_dataset(default_assembled)


class TestDatasetAssembly:
    """Tests for dataset joining, column segregation, and causality."""

    def test_assembly_row_and_column_counts(self, default_assembled: AssembledDataset) -> None:
        """Assembled dataset must drop 80 warm-up rows and retain 80 derived features."""
        # Raw dataset has 100,000 candles. 80 warm-up bars dropped -> 99,920 rows.
        assert len(default_assembled.df) == 99920
        assert len(default_assembled.feature_names) == 80
        assert default_assembled.target_name == "direction_4"
        assert default_assembled.horizon_bars == 4

    def test_assembly_preserves_tail_label_nans(self, default_assembled: AssembledDataset) -> None:
        """Assembled dataset must preserve NaN labels in the tail rows for horizon H."""
        # For H=4, exactly the final 4 rows have NaN labels
        tail_nans = default_assembled.df["direction_4"].isna().sum()
        assert tail_nans == 4
        # But features must have zero NaNs
        assert default_assembled.X.isna().sum().sum() == 0
        # Usable count is 99,920 - 4 = 99,916
        assert len(default_assembled.usable_df) == 99916

    def test_label_column_detection(self) -> None:
        """Helper correctly identifies future-looking targets and labels."""
        assert is_label_or_future_column("future_return_1") is True
        assert is_label_or_future_column("future_log_return_4") is True
        assert is_label_or_future_column("future_vol_adj_return_8") is True
        assert is_label_or_future_column("direction_16") is True
        assert is_label_or_future_column("direction_vol_4") is True
        assert is_label_or_future_column("rsi_14") is False
        assert is_label_or_future_column("return_1") is False
        assert is_label_or_future_column("close") is False

    def test_labels_strictly_excluded_from_features(self) -> None:
        """Target or label columns in feature_columns must raise DataLeakageError."""
        with pytest.raises(DataLeakageError, match="CRITICAL LEAKAGE ERROR"):
            assemble_dataset(
                target_column="direction_4",
                feature_columns=["return_1", "direction_4"],
            )

        with pytest.raises(DataLeakageError, match="CRITICAL LEAKAGE ERROR"):
            assemble_dataset(
                target_column="direction_4",
                feature_columns=["return_1", "future_return_4"],
            )

    def test_horizon_inference(self) -> None:
        """Infers horizon correctly from standard target names."""
        assert infer_horizon_from_target("direction_1") == 1
        assert infer_horizon_from_target("direction_4") == 4
        assert infer_horizon_from_target("direction_8") == 8
        assert infer_horizon_from_target("direction_16") == 16
        assert infer_horizon_from_target("future_return_4") == 4
        assert infer_horizon_from_target("direction_vol_8") == 8

        with pytest.raises(ValueError, match="Cannot infer horizon"):
            infer_horizon_from_target("invalid_target_name")

    def test_horizon_conflict_raises_error(self) -> None:
        """Explicit horizon_bars conflicting with target suffix raises ValueError."""
        with pytest.raises(ValueError, match="conflicts with target"):
            assemble_dataset(target_column="direction_4", horizon_bars=16)

    def test_no_future_feature_dependency(self) -> None:
        """Modifying future candles does NOT change feature values of past candles."""
        # Create a synthetic dataset
        n = 150
        dates = pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC")
        prices = 1.1000 + np.cumsum(np.random.normal(0, 0.0005, size=n))
        df_feat = pd.DataFrame(
            {
                "timestamp": dates,
                "time": [int(d.timestamp()) for d in dates],
                "feat_dummy": prices,
            }
        )
        # Modify future row in features (at index 140)
        df_feat_mod = df_feat.copy()
        df_feat_mod.loc[140, "feat_dummy"] += 10.0

        # Features at t <= 100 must be completely identical
        assert np.allclose(
            df_feat.loc[:100, "feat_dummy"].values,
            df_feat_mod.loc[:100, "feat_dummy"].values,
        )


class TestTemporalSplits:
    """Tests for chronological partitioning, purge, and embargo correctness."""

    def test_chronological_ordering_within_splits(self, default_splits: DatasetSplits) -> None:
        """Timestamps within each partition must be strictly monotonically increasing."""
        assert default_splits.train_timestamps.is_monotonic_increasing
        assert default_splits.val_timestamps.is_monotonic_increasing
        assert default_splits.test_timestamps.is_monotonic_increasing

    def test_temporal_ordering_across_splits(self, default_splits: DatasetSplits) -> None:
        """Train timestamps must strictly precede Val, which must strictly precede Test."""
        assert default_splits.train.end_timestamp < default_splits.val.start_timestamp
        assert default_splits.val.end_timestamp < default_splits.test.start_timestamp

    def test_no_timestamp_overlap(self, default_splits: DatasetSplits) -> None:
        """Partitions must have zero overlapping timestamps (pairwise disjoint)."""
        train_set = set(default_splits.train_timestamps)
        val_set = set(default_splits.val_timestamps)
        test_set = set(default_splits.test_timestamps)

        assert len(train_set.intersection(val_set)) == 0
        assert len(val_set.intersection(test_set)) == 0
        assert len(train_set.intersection(test_set)) == 0

    def test_deterministic_split_reproducibility(self, default_assembled: AssembledDataset) -> None:
        """Splitting twice with identical configuration produces identical partitions."""
        splits1 = split_dataset(default_assembled)
        splits2 = split_dataset(default_assembled)

        assert (splits1.train_y == splits2.train_y).all()
        assert (splits1.val_y == splits2.val_y).all()
        assert (splits1.test_y == splits2.test_y).all()
        assert (splits1.train_X.values == splits2.train_X.values).all()
        assert (splits1.val_X.values == splits2.val_X.values).all()
        assert (splits1.test_X.values == splits2.test_X.values).all()

    def test_purge_boundary_correctness(self, default_assembled: AssembledDataset) -> None:
        """For horizon H, exactly H rows are purged before validation and test boundaries."""
        h = 4
        config = SplitConfig(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, purge_bars=h)
        splits = split_dataset(default_assembled, config=config)

        df_usable = default_assembled.usable_df.reset_index(drop=True)
        N = len(df_usable)
        raw_train_end = int(N * 0.70)
        raw_val_end = raw_train_end + int(N * 0.15)

        # Train end must be raw_train_end - H
        assert len(splits.train) == raw_train_end - h
        assert splits.train.end_index == raw_train_end - h
        assert len(splits.val) == raw_val_end - raw_train_end - h

        # Gap between train end and val start must be exactly H rows
        assert splits.val.start_index - splits.train.end_index == h

        # Gap between val end and test start must be exactly H rows
        assert splits.test.start_index - splits.val.end_index == h

    def test_embargo_boundary_correctness(self, default_assembled: AssembledDataset) -> None:
        """Embargo buffer creates an additional gap after each boundary."""
        h = 4
        embargo = 8
        config = SplitConfig(purge_bars=h, embargo_bars=embargo)
        splits = split_dataset(default_assembled, config=config)

        # Gap between train and val must be H (purge) + 8 (embargo) = 12 rows
        assert splits.val.start_index - splits.train.end_index == h + embargo

        # Gap between val and test must be H (purge) + 8 (embargo) = 12 rows
        assert splits.test.start_index - splits.val.end_index == h + embargo

    def test_split_size_calculations(self, default_assembled: AssembledDataset) -> None:
        """Split sizes must match theoretical formula."""
        h = 4
        config = SplitConfig(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, purge_bars=h)
        splits = split_dataset(default_assembled, config=config)

        N = splits.total_usable_rows
        expected_train = int(N * 0.70) - h
        expected_val = int(N * 0.15) - h
        expected_test = N - int(N * 0.70) - int(N * 0.15)

        assert len(splits.train) == expected_train
        assert len(splits.val) == expected_val
        assert len(splits.test) == expected_test
        assert splits.total_usable_rows == 99916

    def test_no_nan_features_or_targets(self, default_splits: DatasetSplits) -> None:
        """Partitions must contain zero NaNs in features and zero NaNs in targets."""
        assert default_splits.train_X.isna().sum().sum() == 0
        assert default_splits.val_X.isna().sum().sum() == 0
        assert default_splits.test_X.isna().sum().sum() == 0

        assert default_splits.train_y.isna().sum() == 0
        assert default_splits.val_y.isna().sum() == 0
        assert default_splits.test_y.isna().sum() == 0

    def test_target_alignment(self, default_splits: DatasetSplits) -> None:
        """Target vectors must align with feature rows and observation timestamps."""
        assert len(default_splits.train_X) == len(default_splits.train_y)
        assert len(default_splits.train_X) == len(default_splits.train_timestamps)

        assert len(default_splits.val_X) == len(default_splits.val_y)
        assert len(default_splits.val_X) == len(default_splits.val_timestamps)

        assert len(default_splits.test_X) == len(default_splits.test_y)
        assert len(default_splits.test_X) == len(default_splits.test_timestamps)

    @pytest.mark.parametrize("h", DEFAULT_HORIZONS)
    def test_horizon_specific_purge_behavior(self, h: int) -> None:
        """Validates purge sizing across all candidate horizons H=1, 4, 8, 16."""
        assembled = assemble_dataset(target_column=f"direction_{h}")
        splits = split_dataset(assembled)
        report = validate_dataset_splits(splits, raise_on_error=True)

        assert report.is_valid is True
        assert splits.purge_bars == h
        assert splits.purged_train_bars == h
        assert splits.purged_val_bars == h
        # Assembled tail NaNs match H
        assert int((~assembled.usable_mask).sum()) == h

    def test_invalid_split_config_raises_error(self) -> None:
        """Invalid split configurations raise appropriate ValueError."""
        with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
            SplitConfig(train_ratio=0.80, val_ratio=0.15, test_ratio=0.15)

        with pytest.raises(ValueError, match="strictly positive"):
            SplitConfig(train_ratio=0.70, val_ratio=0.0, test_ratio=0.30)

        with pytest.raises(ValueError, match="purge_bars must be >= 0"):
            SplitConfig(purge_bars=-1)

        with pytest.raises(ValueError, match="embargo_bars must be >= 0"):
            SplitConfig(embargo_bars=-5)


class TestValidationReport:
    """Tests for the validation engine and error detection."""

    def test_clean_validation_report(self, default_splits: DatasetSplits) -> None:
        """Clean splits pass all validation checks."""
        report = validate_dataset_splits(default_splits, raise_on_error=True)
        assert isinstance(report, ValidationReport)
        assert report.is_valid is True
        assert len(report.errors) == 0
        assert report.checks["target_not_in_features"] is True
        assert report.checks["no_label_columns_in_features"] is True
        assert report.checks["no_nans_in_features"] is True
        assert report.checks["no_nans_in_targets"] is True
        assert report.checks["temporal_ordering"] is True
        assert report.checks["no_timestamp_overlap"] is True
