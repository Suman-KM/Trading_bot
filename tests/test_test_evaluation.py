"""Automated tests for Phase 11 Final Out-of-Sample Test Evaluation.

Verifies:
1. TestSetGovernanceGuard prevents test set access without explicit authorization.
2. TestSetGovernanceGuard passes when unlock_test_set=True.
3. Test partition rows are chronologically after Train and Validation.
4. Test partition contains exactly 14,988 rows.
5. Test feature matrix contains exactly 80 derived features.
6. Target is direction_4 with ternary {-1.0, 0.0, 1.0} labels.
7. Scaler parameters are fitted strictly on Train and never modified by Test.
8. Brier calibration scores computation is mathematically correct.
9. Temporal test blocks partition the test set into contiguous non-overlapping segments.
10. Evaluation is deterministically reproducible.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from ai.dataset.assembly import assemble_dataset
from ai.dataset.splits import SplitConfig, split_dataset
from ai.models.baselines import LogisticRegressionBaseline, RandomForestBaseline
from ai.models.test_evaluation import (
    TestSetGovernanceGuard,
    compute_brier_scores,
    compute_test_temporal_blocks,
    evaluate_model_on_test,
)


@pytest.fixture(scope="module")
def default_splits():
    """Module-level fixture providing assembled and split dataset."""
    assembled = assemble_dataset(target_column="direction_4")
    config = SplitConfig(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, purge_bars=4)
    return split_dataset(assembled, config)


class TestTestEvaluation:
    """Rigorous verification test suite for Phase 11 test evaluation."""

    def test_governance_guard_prevents_locked_access(self, default_splits) -> None:
        """TEST 1: TestSetGovernanceGuard must raise PermissionError if unlock_test_set=False."""
        with pytest.raises(PermissionError, match="Test partition is locked"):
            TestSetGovernanceGuard.verify_governance(default_splits, unlock_test_set=False)

    def test_governance_guard_passes_with_authorization(self, default_splits) -> None:
        """TEST 2: TestSetGovernanceGuard passes and audits when authorized."""
        audit = TestSetGovernanceGuard.verify_governance(default_splits, unlock_test_set=True)
        assert audit["authorized"] is True
        assert audit["test_rows"] == 14988
        assert audit["feature_count"] == 80
        assert audit["target"] == "direction_4"

    def test_test_rows_strictly_after_val_and_train(self, default_splits) -> None:
        """TEST 3: Chronological ordering must satisfy Train < Validation < Test."""
        assert default_splits.train.end_timestamp < default_splits.val.start_timestamp
        assert default_splits.val.end_timestamp < default_splits.test.start_timestamp
        assert default_splits.test.timestamps.is_monotonic_increasing

    def test_test_partition_row_count_14988(self, default_splits) -> None:
        """TEST 4: Test partition must contain exactly 14,988 observations."""
        assert len(default_splits.test) == 14988
        assert len(default_splits.test_X) == 14988
        assert len(default_splits.test_y) == 14988

    def test_test_feature_matrix_dimension_80(self, default_splits) -> None:
        """TEST 5: Test feature matrix must have exactly 80 columns."""
        assert default_splits.test.X.shape[1] == 80
        assert len(default_splits.feature_names) == 80
        assert list(default_splits.test.X.columns) == default_splits.feature_names

        forbidden = ["direction_4", "future_return_4", "timestamp", "time", "close"]
        for col in forbidden:
            assert col not in default_splits.test.X.columns

    def test_test_target_is_direction_4(self, default_splits) -> None:
        """TEST 6: Test target must be direction_4 with classes {-1.0, 0.0, 1.0}."""
        assert default_splits.target_name == "direction_4"
        unique_classes = set(np.unique(default_splits.test_y))
        assert unique_classes == {-1.0, 0.0, 1.0}

    def test_scaler_fitted_strictly_on_train_not_test(self, default_splits) -> None:
        """TEST 7: StandardScaler must be fitted strictly on Train, unaffected by Test."""
        model = LogisticRegressionBaseline(random_state=42, class_weight="balanced")
        model.fit(default_splits.train_X, default_splits.train_y)

        ref_train_scaler = StandardScaler()
        ref_train_scaler.fit(default_splits.train_X)

        ref_test_scaler = StandardScaler()
        ref_test_scaler.fit(default_splits.test_X)

        assert model.scaler_ is not None
        assert np.allclose(model.scaler_.mean_, ref_train_scaler.mean_)
        assert np.allclose(model.scaler_.scale_, ref_train_scaler.scale_)
        assert not np.allclose(model.scaler_.mean_, ref_test_scaler.mean_)

    def test_brier_scores_correctness(self) -> None:
        """TEST 8: Multiclass and per-class Brier score calculation correctness."""
        y_true = np.array([-1.0, 0.0, 1.0])
        # Perfect predictions
        y_prob_perfect = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        brier_perf = compute_brier_scores(y_true, y_prob_perfect, classes=[-1.0, 0.0, 1.0])
        assert np.isclose(brier_perf["multiclass_brier"], 0.0)
        assert np.isclose(brier_perf["brier_short"], 0.0)
        assert np.isclose(brier_perf["brier_neutral"], 0.0)
        assert np.isclose(brier_perf["brier_long"], 0.0)

        # Uniform predictions
        y_prob_unif = np.full((3, 3), 1.0 / 3.0)
        brier_unif = compute_brier_scores(y_true, y_prob_unif, classes=[-1.0, 0.0, 1.0])
        # (1 - 1/3)^2 + 2 * (1/3)^2 = 4/9 + 2/9 = 6/9 = 2/3
        expected_multiclass = 2.0 / 3.0
        assert np.isclose(brier_unif["multiclass_brier"], expected_multiclass)

    def test_temporal_blocks_partitioning(self, default_splits) -> None:
        """TEST 9: Temporal blocks divide the test set into 3 contiguous segments."""
        rf = RandomForestBaseline(n_estimators=10, max_depth=5, random_state=42)
        rf.fit(default_splits.train_X, default_splits.train_y)

        df_blocks = compute_test_temporal_blocks(
            test_X=default_splits.test_X,
            test_y=default_splits.test_y,
            test_timestamps=default_splits.test.timestamps,
            models={"RF": rf},
            n_blocks=3,
        )

        assert len(df_blocks) == 3
        assert df_blocks["rows"].sum() == 14988
        assert list(df_blocks["block"]) == ["TEST_BLOCK_A", "TEST_BLOCK_B", "TEST_BLOCK_C"]

    def test_deterministic_reproducibility(self, default_splits) -> None:
        """TEST 10: Fixed seed model evaluation on test partition is deterministic."""
        rf1 = RandomForestBaseline(n_estimators=20, max_depth=6, random_state=42)
        rf1.fit(default_splits.train_X, default_splits.train_y)
        m1 = evaluate_model_on_test(rf1, default_splits.test_X, default_splits.test_y, "RF")

        rf2 = RandomForestBaseline(n_estimators=20, max_depth=6, random_state=42)
        rf2.fit(default_splits.train_X, default_splits.train_y)
        m2 = evaluate_model_on_test(rf2, default_splits.test_X, default_splits.test_y, "RF")

        assert np.isclose(m1.accuracy, m2.accuracy)
        assert np.isclose(m1.balanced_accuracy, m2.balanced_accuracy)
        assert np.isclose(m1.macro_f1, m2.macro_f1)
        assert m1.confusion_matrix == m2.confusion_matrix
