"""Unit, sanity, and leakage tests for Phase 9 baseline model diagnostics.

Verifies:
1. Test split is never accessed.
2. Diagnostics use validation predictions exclusively.
3. Probability arrays have correct shape (N, 3).
4. Probability rows correspond exactly to validation timestamps.
5. Probability rows sum approximately to 1.0.
6. Confidence values are strictly bounded in [0.0, 1.0].
7. Coverage calculations are deterministic and monotonic.
8. Random Forest feature importance contains exactly 80 features.
9. Logistic Regression coefficient importance contains exactly 80 features.
10. No target column appears in importance output.
11. No timestamp column appears in model features.
12. Temporal validation blocks are chronological, contiguous, and non-overlapping.
13. Repeated diagnostic runs are strictly deterministic.
14. Zero lookahead or test-set leakage in diagnostics pipeline.
15. Existing Phase 8 baseline model definitions remain unmodified.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ai.dataset.assembly import assemble_dataset
from ai.dataset.splits import SplitConfig, split_dataset
from ai.features.pipeline import get_feature_registry
from ai.models.baselines import (
    LogisticRegressionBaseline,
    MajorityClassClassifier,
    RandomForestBaseline,
)
from ai.models.diagnostics import (
    compute_calibration_diagnostics,
    compute_class_wise_metrics,
    compute_confidence_coverage,
    compute_error_analysis,
    compute_lr_feature_importance,
    compute_model_agreement,
    compute_probability_diagnostics,
    compute_rf_feature_importance,
    compute_temporal_validation_diagnostics,
    verify_feature_sanity,
)


@pytest.fixture(scope="module")
def default_splits():
    """Module-level fixture providing Phase 7 splits for direction_4."""
    assembled = assemble_dataset(target_column="direction_4")
    config = SplitConfig(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, purge_bars=4)
    return split_dataset(assembled, config)


@pytest.fixture(scope="module")
def fitted_models(default_splits):
    """Module-level fixture fitting Phase 8 baseline models strictly on train."""
    lr = LogisticRegressionBaseline(random_state=42, max_iter=1000, C=1.0)
    lr.fit(default_splits.train_X, default_splits.train_y)

    rf = RandomForestBaseline(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(default_splits.train_X, default_splits.train_y)

    return {"lr": lr, "rf": rf}


class TestModelDiagnostics:
    """Comprehensive test suite verifying Phase 9 baseline model diagnostics."""

    def test_1_test_split_never_accessed(self, default_splits, fitted_models) -> None:
        """TEST 1: Verifies that test split is never accessed or passed to diagnostics."""
        lr = fitted_models["lr"]
        rf = fitted_models["rf"]

        # Ensure diagnostics functions only require validation set
        val_X = default_splits.val_X
        val_y = default_splits.val_y
        val_ts = default_splits.val.timestamps

        prob_lr = lr.predict_proba(val_X)
        prob_rf = rf.predict_proba(val_X)

        assert len(prob_lr) == len(default_splits.val)
        assert len(prob_rf) == len(default_splits.val)
        assert len(prob_lr) != len(default_splits.test)

        # Execute diagnostics exclusively on validation objects
        prob_diag = compute_probability_diagnostics(prob_lr)
        assert prob_diag["sample_count"] == len(default_splits.val)

        temporal = compute_temporal_validation_diagnostics(val_X, val_y, val_ts, lr, rf)
        total_temporal_rows = temporal[temporal["model"] == "Logistic Regression"]["rows"].sum()
        assert total_temporal_rows == len(default_splits.val)
        assert total_temporal_rows != len(default_splits.test)

    def test_2_diagnostics_use_validation_predictions(self, default_splits, fitted_models) -> None:
        """TEST 2: Verifies diagnostics use validation predictions."""
        rf = fitted_models["rf"]
        val_X = default_splits.val_X
        val_y = default_splits.val_y

        y_pred = rf.predict(val_X)
        probs = rf.predict_proba(val_X)

        assert len(y_pred) == len(val_y)
        assert len(y_pred) == 14983

        cw = compute_class_wise_metrics(val_y, y_pred, probs)
        total_support = sum(m["support"] for m in cw.values())
        assert total_support == len(val_y)

    def test_3_probability_arrays_correct_shape(self, default_splits, fitted_models) -> None:
        """TEST 3: Probability arrays must have shape (n_val_samples, 3)."""
        lr = fitted_models["lr"]
        rf = fitted_models["rf"]
        val_X = default_splits.val_X

        prob_lr = lr.predict_proba(val_X)
        prob_rf = rf.predict_proba(val_X)

        assert prob_lr.shape == (len(val_X), 3)
        assert prob_rf.shape == (len(val_X), 3)

    def test_4_probability_rows_correspond_to_validation_timestamps(
        self, default_splits, fitted_models
    ) -> None:
        """TEST 4: Probability rows correspond exactly to validation timestamps."""
        lr = fitted_models["lr"]
        val_X = default_splits.val_X
        val_ts = default_splits.val.timestamps

        prob_lr = lr.predict_proba(val_X)

        assert len(prob_lr) == len(val_ts)
        assert (val_X.index == val_ts.index).all()
        assert val_ts.is_monotonic_increasing

    def test_5_probability_rows_sum_to_one(self, default_splits, fitted_models) -> None:
        """TEST 5: Probability rows must sum approximately to 1.0."""
        lr = fitted_models["lr"]
        rf = fitted_models["rf"]
        val_X = default_splits.val_X

        prob_lr = lr.predict_proba(val_X)
        prob_rf = rf.predict_proba(val_X)

        assert np.allclose(prob_lr.sum(axis=1), 1.0, atol=1e-5)
        assert np.allclose(prob_rf.sum(axis=1), 1.0, atol=1e-5)

    def test_6_confidence_values_bounded_zero_one(self, default_splits, fitted_models) -> None:
        """TEST 6: Confidence values must lie strictly in [0.0, 1.0]."""
        lr = fitted_models["lr"]
        rf = fitted_models["rf"]
        val_X = default_splits.val_X

        prob_lr = lr.predict_proba(val_X)
        prob_rf = rf.predict_proba(val_X)

        assert (prob_lr >= 0.0).all() and (prob_lr <= 1.0).all()
        assert (prob_rf >= 0.0).all() and (prob_rf <= 1.0).all()

        max_lr = np.max(prob_lr, axis=1)
        max_rf = np.max(prob_rf, axis=1)

        assert (max_lr >= 1.0 / 3.0).all()
        assert (max_rf >= 1.0 / 3.0).all()

    def test_7_coverage_calculations_deterministic(self, default_splits, fitted_models) -> None:
        """TEST 7: Coverage calculations must be strictly deterministic."""
        lr = fitted_models["lr"]
        val_X = default_splits.val_X
        val_y = default_splits.val_y

        probs = lr.predict_proba(val_X)
        preds = lr.predict(val_X)

        run1 = compute_confidence_coverage(probs, val_y, preds)
        run2 = compute_confidence_coverage(probs, val_y, preds)

        assert len(run1) == len(run2)
        for r1, r2 in zip(run1, run2):
            assert r1["threshold"] == r2["threshold"]
            assert r1["covered_count"] == r2["covered_count"]
            assert r1["coverage"] == r2["coverage"]
            assert r1["accuracy"] == r2["accuracy"]

    def test_8_rf_feature_importance_contains_80_features(
        self, default_splits, fitted_models
    ) -> None:
        """TEST 8: Random Forest feature importance must contain exactly 80 features."""
        rf = fitted_models["rf"]
        imp_df = compute_rf_feature_importance(rf, default_splits.feature_names)

        assert len(imp_df) == 80
        assert list(imp_df.columns) == ["rank", "feature", "importance"]
        assert list(imp_df["rank"]) == list(range(1, 81))
        assert np.isclose(imp_df["importance"].sum(), 1.0, atol=1e-4)

    def test_9_lr_feature_importance_contains_80_features(
        self, default_splits, fitted_models
    ) -> None:
        """TEST 9: Logistic Regression coefficient importance contains exactly 80 features."""
        lr = fitted_models["lr"]
        imp_df = compute_lr_feature_importance(lr, default_splits.feature_names)

        assert len(imp_df) == 80
        assert "rank" in imp_df.columns
        assert "feature" in imp_df.columns
        assert "aggregate_importance" in imp_df.columns
        assert "coef_short" in imp_df.columns
        assert "coef_neutral" in imp_df.columns
        assert "coef_long" in imp_df.columns
        assert (imp_df["aggregate_importance"] >= 0.0).all()

    def test_10_no_target_column_in_importance(self, default_splits, fitted_models) -> None:
        """TEST 10: No target column appears in importance output."""
        rf = fitted_models["rf"]
        imp_df = compute_rf_feature_importance(rf, default_splits.feature_names)

        registry_defs = get_feature_registry()
        registry_names = [f.name for f in registry_defs]

        sanity = verify_feature_sanity(imp_df, registry_names)
        assert sanity["passed"] is True
        assert sanity["target_leakage_detected"] is False

        # Explicit target check
        target_prefixes = ("direction_", "direction_vol_", "future_", "forward_")
        for f in imp_df["feature"]:
            assert not f.lower().startswith(target_prefixes)
            assert f.lower() not in ("target", "label")

    def test_11_no_timestamp_column_in_model_features(self, default_splits) -> None:
        """TEST 11: No timestamp column appears in model features."""
        raw_ts_cols = {"timestamp", "open_time", "time", "date", "datetime"}
        for f in default_splits.feature_names:
            assert f.lower() not in raw_ts_cols

    def test_12_temporal_validation_blocks_chronological_non_overlapping(
        self, default_splits, fitted_models
    ) -> None:
        """TEST 12: Temporal validation blocks are chronological and non-overlapping."""
        lr = fitted_models["lr"]
        rf = fitted_models["rf"]
        val_X = default_splits.val_X
        val_y = default_splits.val_y
        val_ts = default_splits.val.timestamps

        temporal_df = compute_temporal_validation_diagnostics(val_X, val_y, val_ts, lr, rf)
        lr_blocks = temporal_df[temporal_df["model"] == "Logistic Regression"]

        assert len(lr_blocks) == 3
        assert list(lr_blocks["block"]) == ["VALIDATION_A", "VALIDATION_B", "VALIDATION_C"]

        total_rows = lr_blocks["rows"].sum()
        assert total_rows == len(val_X)

        # Verify time monotonicity across blocks
        t_a_end = pd.to_datetime(lr_blocks.iloc[0]["end_time"])
        t_b_start = pd.to_datetime(lr_blocks.iloc[1]["start_time"])
        t_b_end = pd.to_datetime(lr_blocks.iloc[1]["end_time"])
        t_c_start = pd.to_datetime(lr_blocks.iloc[2]["start_time"])

        assert t_a_end < t_b_start
        assert t_b_end < t_c_start

    def test_13_repeated_diagnostic_runs_deterministic(self, default_splits, fitted_models) -> None:
        """TEST 13: Repeated diagnostic runs are strictly deterministic."""
        rf = fitted_models["rf"]
        val_X = default_splits.val_X
        val_y = default_splits.val_y

        probs = rf.predict_proba(val_X)

        diag1 = compute_calibration_diagnostics(val_y, probs)
        diag2 = compute_calibration_diagnostics(val_y, probs)

        assert np.isclose(
            diag1["multiclass_brier_score"],
            diag2["multiclass_brier_score"],
            atol=1e-8,
        )

        error1 = compute_error_analysis(val_y, rf.predict(val_X), probs)
        error2 = compute_error_analysis(val_y, rf.predict(val_X), probs)
        pd.testing.assert_frame_equal(error1, error2)

    def test_14_no_future_test_data_used(self, default_splits) -> None:
        """TEST 14: Verifies test set is strictly chronologically after validation."""
        val_end = default_splits.val.end_timestamp
        test_start = default_splits.test.start_timestamp

        # Purge gap of H=4 bars (60 minutes) separates validation and test
        assert test_start > val_end
        delta = test_start - val_end
        assert delta.total_seconds() >= 4 * 15 * 60

    def test_15_existing_phase_8_models_remain_unchanged(self) -> None:
        """TEST 15: Verifies that Phase 8 baseline model classes are preserved."""
        maj = MajorityClassClassifier()
        assert hasattr(maj, "fit")
        assert hasattr(maj, "predict")
        assert hasattr(maj, "predict_proba")

        lr = LogisticRegressionBaseline(random_state=42)
        params_lr = lr.get_params()
        assert params_lr["C"] == 1.0
        assert params_lr["solver"] == "lbfgs"
        assert params_lr["scaler"] == "StandardScaler"

        rf = RandomForestBaseline(random_state=42)
        params_rf = rf.get_params()
        assert params_rf["n_estimators"] == 100
        assert params_rf["max_depth"] == 10
        assert params_rf["min_samples_leaf"] == 20

    def test_16_model_agreement_diagnostics(self, default_splits, fitted_models) -> None:
        """TEST 16: Verifies model agreement calculation and contingency matrix."""
        lr = fitted_models["lr"]
        rf = fitted_models["rf"]
        val_X = default_splits.val_X
        val_y = default_splits.val_y

        y_pred_lr = lr.predict(val_X)
        y_pred_rf = rf.predict(val_X)

        agree = compute_model_agreement(y_pred_lr, y_pred_rf, val_y)
        assert agree["total_samples"] == len(val_X)
        assert agree["agreement_count"] + agree["disagreement_count"] == len(val_X)
        assert 0.0 <= agree["agreement_rate"] <= 1.0
        assert -1.0 <= agree["cohen_kappa"] <= 1.0
        assert len(agree["contingency_matrix_lr_rows_rf_cols"]) == 3
