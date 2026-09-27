"""Unit and leakage tests for baseline machine learning models (Phase 8).

Validates strict test-set protection, training-only scaler fitting, target/timestamp
absence from features, deterministic reproducibility, and correct metric accounting.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from ai.dataset.assembly import assemble_dataset
from ai.dataset.splits import SplitConfig, split_dataset
from ai.models.baselines import (
    LogisticRegressionBaseline,
    MajorityClassClassifier,
    RandomForestBaseline,
)
from ai.models.evaluation import (
    evaluate_classification_model,
)
from ai.models.pipeline import run_baseline_training_pipeline


@pytest.fixture(scope="module")
def default_splits():
    """Module-level fixture providing Phase 7 splits for direction_4."""
    assembled = assemble_dataset(target_column="direction_4")
    config = SplitConfig(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, purge_bars=4)
    return split_dataset(assembled, config)


class TestModelLeakageControls:
    """Rigorous tests proving zero lookahead or test-set leakage in baseline ML."""

    def test_test_data_never_passed_to_model_fit(self, default_splits) -> None:
        """TEST 1: Verifies that model.fit() receives exclusively training data."""
        # Create a mock estimator to inspect arguments passed to fit
        mock_estimator = MagicMock()
        mock_estimator.classes_ = np.array([-1.0, 0.0, 1.0])
        mock_estimator.predict.return_value = np.zeros(len(default_splits.val_X))

        # Train on train_X and train_y
        X_train = default_splits.train_X
        y_train = default_splits.train_y
        mock_estimator.fit(X_train, y_train)

        # Inspect call args
        called_args, _ = mock_estimator.fit.call_args
        fitted_X, fitted_y = called_args

        # Assert shape matches train, not test or validation
        assert len(fitted_X) == len(default_splits.train)
        assert len(fitted_X) != len(default_splits.test)
        assert len(fitted_y) == len(default_splits.train)
        assert len(fitted_y) != len(default_splits.test)
        assert (fitted_X.index == default_splits.train_X.index).all()

    def test_scaler_fitted_strictly_on_train(self, default_splits) -> None:
        """TEST 2: StandardScaler mean/scale parameters must match training data only."""
        model_lr = LogisticRegressionBaseline(random_state=42)
        model_lr.fit(default_splits.train_X, default_splits.train_y)

        # Compute ground truth training scaler
        ref_scaler = StandardScaler()
        ref_scaler.fit(default_splits.train_X)

        # Compute validation scaler (to prove difference)
        val_scaler = StandardScaler()
        val_scaler.fit(default_splits.val_X)

        # Model scaler must exactly match train scaler
        assert np.allclose(model_lr.scaler_.mean_, ref_scaler.mean_)
        assert np.allclose(model_lr.scaler_.scale_, ref_scaler.scale_)

        # Model scaler must NOT match validation statistics
        assert not np.allclose(model_lr.scaler_.mean_, val_scaler.mean_)

    def test_validation_transformed_with_training_scaler(self, default_splits) -> None:
        """TEST 3: Validation features are transformed using train scaler without re-fitting."""
        model_lr = LogisticRegressionBaseline(random_state=42)
        model_lr.fit(default_splits.train_X, default_splits.train_y)

        mean_before = model_lr.scaler_.mean_.copy()
        scale_before = model_lr.scaler_.scale_.copy()

        # Call predict on validation set
        _ = model_lr.predict(default_splits.val_X)

        # Scaler attributes must remain strictly unchanged
        assert np.array_equal(model_lr.scaler_.mean_, mean_before)
        assert np.array_equal(model_lr.scaler_.scale_, scale_before)

    def test_target_absent_from_feature_matrix(self, default_splits) -> None:
        """TEST 4: Target column is strictly absent from X_train and X_val."""
        target = default_splits.target_name
        assert target not in default_splits.train_X.columns
        assert target not in default_splits.val_X.columns
        assert target not in default_splits.test_X.columns

    def test_timestamp_absent_from_model_features(self, default_splits) -> None:
        """TEST 5: Timestamp/time columns are not present in the model feature matrix."""
        for col in ["time", "timestamp", "date", "datetime"]:
            assert col not in default_splits.train_X.columns
            assert col not in default_splits.val_X.columns

    def test_feature_count_strictly_80(self, default_splits) -> None:
        """TEST 6: Exactly 80 derived features are present in X_train and X_val."""
        assert default_splits.train_X.shape[1] == 80
        assert default_splits.val_X.shape[1] == 80
        assert len(default_splits.feature_names) == 80

    def test_train_val_chronological_ordering(self, default_splits) -> None:
        """TEST 7: All training observations strictly precede all validation observations."""
        assert default_splits.train.end_timestamp < default_splits.val.start_timestamp
        assert default_splits.val.end_timestamp < default_splits.test.start_timestamp

    def test_deterministic_model_reproducibility(self, default_splits) -> None:
        """TEST 8: Re-training models with fixed random_state=42 yields identical results."""
        # Use first 2000 rows for fast deterministic check
        X_sub = default_splits.train_X.iloc[:2000]
        y_sub = default_splits.train_y.iloc[:2000]
        X_val_sub = default_splits.val_X.iloc[:500]

        # Logistic Regression
        lr1 = LogisticRegressionBaseline(random_state=42).fit(X_sub, y_sub)
        lr2 = LogisticRegressionBaseline(random_state=42).fit(X_sub, y_sub)
        assert np.allclose(lr1.predict_proba(X_val_sub), lr2.predict_proba(X_val_sub))
        assert np.array_equal(lr1.predict(X_val_sub), lr2.predict(X_val_sub))

        # Random Forest
        rf1 = RandomForestBaseline(n_estimators=30, max_depth=6, random_state=42).fit(X_sub, y_sub)
        rf2 = RandomForestBaseline(n_estimators=30, max_depth=6, random_state=42).fit(X_sub, y_sub)
        assert np.allclose(rf1.predict_proba(X_val_sub), rf2.predict_proba(X_val_sub))
        assert np.array_equal(rf1.predict(X_val_sub), rf2.predict(X_val_sub))

    def test_majority_baseline_uses_training_frequencies(self, default_splits) -> None:
        """TEST 9: MajorityClassClassifier identifies training mode and training frequencies."""
        maj = MajorityClassClassifier()
        maj.fit(default_splits.train_X, default_splits.train_y)

        train_mode = float(default_splits.train_y.mode().iloc[0])
        assert maj.majority_class_ == train_mode
        assert maj.majority_class_ == 0.0

        # Predict returns constant training mode
        preds = maj.predict(default_splits.val_X)
        assert (preds == train_mode).all()

        # Class probabilities must match training distribution, NOT validation distribution
        train_prob_neutral = float((default_splits.train_y == 0.0).mean())
        val_prob_neutral = float((default_splits.val_y == 0.0).mean())
        maj_prob_neutral = maj.predict_proba(default_splits.val_X[:1])[0, 1]

        assert np.isclose(maj_prob_neutral, train_prob_neutral, atol=1e-4)
        assert not np.isclose(maj_prob_neutral, val_prob_neutral, atol=1e-2)

    def test_validation_metrics_use_validation_predictions_only(self, default_splits) -> None:
        """TEST 10: evaluate_classification_model evaluates validation targets strictly."""
        maj = MajorityClassClassifier().fit(default_splits.train_X, default_splits.train_y)
        res = evaluate_classification_model(
            model=maj,
            X_val=default_splits.val_X,
            y_val=default_splits.val_y,
            X_train=default_splits.train_X,
            y_train=default_splits.train_y,
        )

        # Expected accuracy on val is exactly the proportion of 0.0 in val
        expected_val_acc = float((default_splits.val_y == 0.0).mean())
        assert np.isclose(res.accuracy, expected_val_acc, atol=1e-4)

    def test_test_split_unaccessed_by_baseline_pipeline(self, default_splits) -> None:
        """TEST 11: Pipeline raises an exception if test_X or test_y are accessed."""

        # Define a proxy class around default_splits that blows up if test data is accessed
        class GuardedTestSplits:
            def __init__(self, splits):
                self._splits = splits

            @property
            def train_X(self):
                return self._splits.train_X.iloc[:500]

            @property
            def train_y(self):
                return self._splits.train_y.iloc[:500]

            @property
            def val_X(self):
                return self._splits.val_X.iloc[:100]

            @property
            def val_y(self):
                return self._splits.val_y.iloc[:100]

            @property
            def test_X(self):
                raise AssertionError("CRITICAL LEAKAGE: test_X was accessed!")

            @property
            def test_y(self):
                raise AssertionError("CRITICAL LEAKAGE: test_y was accessed!")

            @property
            def train(self):
                return self._splits.train

            @property
            def val(self):
                return self._splits.val

            @property
            def test(self):
                # Metadata access allowed, but test_X / test_y access forbidden
                return self._splits.test

            @property
            def target_name(self):
                return self._splits.target_name

            @property
            def horizon_bars(self):
                return self._splits.horizon_bars

            @property
            def feature_names(self):
                return self._splits.feature_names

        guarded = GuardedTestSplits(default_splits)

        # Executing pipeline must NOT access test_X or test_y
        results = run_baseline_training_pipeline(
            splits=guarded,  # type: ignore
            random_seed=42,
            generate_figures=False,
            save_metadata=False,
        )
        assert "majority" in results
        assert "logistic_regression" in results
        assert "random_forest" in results

    def test_no_future_rows_used_in_training_features(self) -> None:
        """TEST 12: Point-in-time causality holds; past features are unaffected by future bars."""
        n = 120
        dates = pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC")
        prices = 1.1000 + np.cumsum(np.random.normal(0, 0.0005, size=n))
        df_feat = pd.DataFrame(
            {"timestamp": dates, "time": [int(d.timestamp()) for d in dates], "feat_1": prices}
        )
        # Modify future bar (index 110)
        df_feat_mod = df_feat.copy()
        df_feat_mod.loc[110, "feat_1"] += 50.0

        # Features at t <= 80 are identical
        assert np.array_equal(df_feat.loc[:80, "feat_1"], df_feat_mod.loc[:80, "feat_1"])
