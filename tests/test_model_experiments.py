"""Automated tests for Phase 10 Controlled Model Improvement Study.

Verifies:
1. Training-only scaler fitting.
2. Class weights computed strictly from training labels independent of validation.
3. Test partition remains untouched and never used.
4. Model evaluation and selection strictly uses validation only.
5. Feature matrix dimension remains exactly 80 derived features.
6. Prediction target remains direction_4 with {-1, 0, +1} ternary classes.
7. Chronological splitting integrity and purge boundaries are maintained.
8. Deterministic random seed reproducibility across runs.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from ai.dataset.assembly import assemble_dataset
from ai.dataset.splits import SplitConfig, split_dataset
from ai.models.baselines import LogisticRegressionBaseline, RandomForestBaseline
from ai.models.experiments import (
    ExtraTreesBaseline,
    evaluate_experiment_model,
    run_model_improvement_experiments,
)


@pytest.fixture(scope="module")
def default_splits():
    """Module-level fixture providing assembled and chronologically split EURUSD M15 data."""
    assembled = assemble_dataset(target_column="direction_4")
    config = SplitConfig(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, purge_bars=4)
    return split_dataset(assembled, config)


class TestModelExperiments:
    """Rigorous verification test suite for Phase 10 model improvement experiments."""

    def test_scaler_fitted_on_training_only(self, default_splits) -> None:
        """TEST 1: StandardScaler must be fitted strictly on the training partition."""
        model = LogisticRegressionBaseline(random_state=42, class_weight="balanced")
        model.fit(default_splits.train_X, default_splits.train_y)

        # Fit independent ground truth scalers
        ref_train_scaler = StandardScaler()
        ref_train_scaler.fit(default_splits.train_X)

        ref_val_scaler = StandardScaler()
        ref_val_scaler.fit(default_splits.val_X)

        assert model.scaler_ is not None
        # Mean and scale must match training partition exactly
        assert np.allclose(model.scaler_.mean_, ref_train_scaler.mean_)
        assert np.allclose(model.scaler_.scale_, ref_train_scaler.scale_)

        # Mean and scale must NOT match validation partition
        assert not np.allclose(model.scaler_.mean_, ref_val_scaler.mean_)
        assert not np.allclose(model.scaler_.scale_, ref_val_scaler.scale_)

    def test_class_weights_independent_of_validation(self, default_splits) -> None:
        """TEST 2: Balanced class weights must be computed strictly from training labels."""
        classes = np.array([-1.0, 0.0, 1.0])
        train_weights = compute_class_weight("balanced", classes=classes, y=default_splits.train_y)
        val_weights = compute_class_weight("balanced", classes=classes, y=default_splits.val_y)

        # Weights must be positive and non-zero
        assert len(train_weights) == 3
        assert np.all(train_weights > 0.0)

        # Training class weights should reflect training class balance, not validation
        assert not np.allclose(train_weights, val_weights)

        # Confirm model fitting uses training class weighting
        model = LogisticRegressionBaseline(random_state=42, class_weight="balanced")
        model.fit(default_splits.train_X, default_splits.train_y)
        assert model.class_weight == "balanced"

    def test_test_partition_untouched(self, default_splits) -> None:
        """TEST 3: Test partition must remain strictly untouched and never evaluated."""
        original_test_len = len(default_splits.test)
        original_test_checksum = hash(default_splits.test_X.iloc[0].values.tobytes())

        # Execute experiments with artifacts disabled for test speed
        results = run_model_improvement_experiments(
            splits=default_splits,
            random_seed=42,
            save_artifacts=False,
        )

        # Test set used flag must be explicitly False
        assert results["test_set_used"] is False
        assert results["metadata"]["test_set_used"] is False
        assert results["metadata"]["evaluation_partition"] == "validation"

        # Test partition length and data must remain unchanged
        assert len(default_splits.test) == original_test_len
        current_test_checksum = hash(default_splits.test_X.iloc[0].values.tobytes())
        assert current_test_checksum == original_test_checksum

    def test_model_selection_validation_only(self, default_splits) -> None:
        """TEST 4: All reported metrics must be computed strictly on the validation partition."""
        model_et = ExtraTreesBaseline(n_estimators=50, random_state=42, class_weight="balanced")
        model_et.fit(default_splits.train_X, default_splits.train_y)

        metrics = evaluate_experiment_model(
            model=model_et,
            X_val=default_splits.val_X,
            y_val=default_splits.val_y,
            model_name="Extra Trees Test",
            class_weight="balanced",
        )

        # Verify metrics match validation ground truth
        val_preds = model_et.predict(default_splits.val_X)
        val_acc = np.mean(val_preds == default_splits.val_y)
        assert np.isclose(metrics.accuracy, val_acc)

        # Ensure confusion matrix sum equals validation row count
        cm_sum = sum(sum(row) for row in metrics.confusion_matrix)
        assert cm_sum == len(default_splits.val_y)
        assert cm_sum != len(default_splits.train_y)
        assert cm_sum != len(default_splits.test_y)

    def test_feature_matrix_dimension_80(self, default_splits) -> None:
        """TEST 5: Input feature matrix must contain exactly 80 derived point-in-time features."""
        assert default_splits.train_X.shape[1] == 80
        assert default_splits.val_X.shape[1] == 80
        assert default_splits.test_X.shape[1] == 80
        assert len(default_splits.feature_names) == 80

    def test_target_is_direction_4(self, default_splits) -> None:
        """TEST 6: Prediction target must be direction_4 with ternary {-1.0, 0.0, 1.0} labels."""
        assert default_splits.target_name == "direction_4"
        assert default_splits.horizon_bars == 4

        train_unique = set(np.unique(default_splits.train_y))
        val_unique = set(np.unique(default_splits.val_y))
        test_unique = set(np.unique(default_splits.test_y))

        expected = {-1.0, 0.0, 1.0}
        assert train_unique == expected
        assert val_unique == expected
        assert test_unique == expected

    def test_chronological_split_integrity(self, default_splits) -> None:
        """TEST 7: Chronological split ordering and purge boundaries must be strictly maintained."""
        assert default_splits.train.end_timestamp < default_splits.val.start_timestamp
        assert default_splits.val.end_timestamp < default_splits.test.start_timestamp

        # Purge gap between train end and val start must be at least horizon bars
        train_end_idx = default_splits.train.end_index
        val_start_idx = default_splits.val.start_index
        assert val_start_idx - train_end_idx >= 4

        # Purge gap between val end and test start
        val_end_idx = default_splits.val.end_index
        test_start_idx = default_splits.test.start_index
        assert test_start_idx - val_end_idx >= 4

    def test_experiment_reproducibility(self, default_splits) -> None:
        """TEST 8: Fixed random seeds must produce deterministic, identical model evaluations."""
        rf1 = RandomForestBaseline(
            n_estimators=30, max_depth=8, random_state=42, class_weight="balanced"
        )
        rf1.fit(default_splits.train_X, default_splits.train_y)
        preds1 = rf1.predict(default_splits.val_X)
        probs1 = rf1.predict_proba(default_splits.val_X)

        rf2 = RandomForestBaseline(
            n_estimators=30, max_depth=8, random_state=42, class_weight="balanced"
        )
        rf2.fit(default_splits.train_X, default_splits.train_y)
        preds2 = rf2.predict(default_splits.val_X)
        probs2 = rf2.predict_proba(default_splits.val_X)

        assert np.array_equal(preds1, preds2)
        assert np.allclose(probs1, probs2, atol=1e-10)
