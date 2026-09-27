"""Final Out-of-Sample Test Evaluation Module (Phase 11).

Provides rigorous test-set governance guards, holdout evaluation metrics,
generalization gap calculations, temporal holdout blocks, confidence filtering,
and probability calibration / Brier score diagnostics on the locked EURUSD M15
test partition (14,988 rows).

CRITICAL GOVERNANCE:
- The TEST partition is unlocked strictly for final holdout measurement.
- Test evaluation results must NEVER be used for model selection, hyperparameter
  tuning, feature selection, threshold modification, or retraining.
- All candidate models are trained strictly on the training partition (69,937 rows).
- Scaler is fitted strictly on the training partition.
- Descriptive reporting only; no claims of profitability or trading viability.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ai.dataset.splits import DatasetSplits
from ai.models.baselines import (
    LogisticRegressionBaseline,
    MajorityClassClassifier,
    RandomForestBaseline,
)
from ai.models.experiments import ExtraTreesBaseline, evaluate_experiment_model

DEFAULT_CLASSES: list[float] = [-1.0, 0.0, 1.0]


class TestSetGovernanceGuard:
    """Security and integrity guard protecting the final holdout test partition."""

    @staticmethod
    def verify_governance(
        splits: DatasetSplits,
        unlock_test_set: bool = False,
    ) -> dict[str, Any]:
        """Verify strict leakage, ordering, and contract assertions on the test partition.

        Parameters
        ----------
        splits : DatasetSplits
            The chronological splits container.
        unlock_test_set : bool, default False
            Explicit authorization flag required to evaluate the test set.

        Returns
        -------
        dict[str, Any]
            Audit summary of verification checks.

        Raises
        ------
        PermissionError
            If unlock_test_set is False.
        AssertionError
            If any leakage or contract rule is violated.
        """
        if not unlock_test_set:
            raise PermissionError(
                "Test partition is locked. Explicit authorization 'unlock_test_set=True' "
                "is required to perform Phase 11 final out-of-sample evaluation."
            )

        assert splits.train is not None, "Training partition is missing."
        assert splits.val is not None, "Validation partition is missing."
        assert splits.test is not None, "Test partition is missing."

        # 1. Row count contracts
        assert splits.train.row_count == 69937, (
            f"Expected 69,937 train rows, got {splits.train.row_count}"
        )
        assert splits.val.row_count == 14983, (
            f"Expected 14,983 val rows, got {splits.val.row_count}"
        )
        assert splits.test.row_count == 14988, (
            f"Expected 14,988 test rows, got {splits.test.row_count}"
        )

        # 2. Chronological order and non-overlap
        assert splits.train.end_timestamp < splits.val.start_timestamp, (
            "Train does not precede Validation."
        )
        assert splits.val.end_timestamp < splits.test.start_timestamp, (
            "Validation does not precede Test."
        )
        assert splits.test.timestamps.is_monotonic_increasing, (
            "Test timestamps are not strictly monotonically increasing."
        )

        # 3. Purge gap boundaries
        purge_gap_train_val = splits.val.start_index - splits.train.end_index
        purge_gap_val_test = splits.test.start_index - splits.val.end_index
        assert purge_gap_train_val >= 4, (
            f"Train-Val purge gap must be >= 4 bars, got {purge_gap_train_val}"
        )
        assert purge_gap_val_test >= 4, (
            f"Val-Test purge gap must be >= 4 bars, got {purge_gap_val_test}"
        )

        # 4. Feature matrix dimension & absence of leakages
        assert splits.test.X.shape[1] == 80, (
            f"Expected exactly 80 features, got {splits.test.X.shape[1]}"
        )
        assert splits.target_name == "direction_4", (
            f"Expected target direction_4, got {splits.target_name}"
        )

        forbidden_cols = ["direction_4", "future_return_4", "timestamp", "time", "close"]
        for col in forbidden_cols:
            assert col not in splits.test.X.columns, (
                f"Forbidden column '{col}' detected in test feature matrix."
            )

        # 5. Target classes contract
        test_classes = set(np.unique(splits.test.y))
        expected_classes = {-1.0, 0.0, 1.0}
        assert test_classes == expected_classes, (
            f"Expected classes {expected_classes}, got {test_classes}"
        )

        return {
            "authorized": True,
            "train_rows": splits.train.row_count,
            "val_rows": splits.val.row_count,
            "test_rows": splits.test.row_count,
            "test_start": str(splits.test.start_timestamp),
            "test_end": str(splits.test.end_timestamp),
            "feature_count": splits.test.X.shape[1],
            "target": splits.target_name,
            "purge_gap_val_test_bars": purge_gap_val_test,
        }


@dataclass(frozen=True)
class FinalTestEvaluationMetrics:
    """Comprehensive performance metrics for out-of-sample test evaluation."""

    model_name: str
    class_weight: str | None
    accuracy: float
    balanced_accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    precision_short: float
    recall_short: float
    f1_short: float
    precision_neutral: float
    recall_neutral: float
    f1_neutral: float
    precision_long: float
    recall_long: float
    f1_long: float
    macro_roc_auc: float | None
    macro_pr_auc: float | None
    brier_score_multiclass: float
    brier_short: float
    brier_neutral: float
    brier_long: float
    confusion_matrix: list[list[int]]
    support_short: int
    support_neutral: int
    support_long: int

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        return asdict(self)


def compute_brier_scores(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    classes: list[float] | np.ndarray = (-1.0, 0.0, 1.0),
) -> dict[str, float]:
    """Compute multiclass and per-class Brier calibration scores.

    Parameters
    ----------
    y_true : np.ndarray
        True class labels.
    y_prob : np.ndarray
        Predicted probability array of shape (N, K).
    classes : list[float] | np.ndarray
        Class labels corresponding to probability columns.

    Returns
    -------
    dict[str, float]
        Dictionary of Brier scores.
    """
    class_list = [float(c) for c in classes]
    n_samples = len(y_true)
    y_onehot = np.zeros_like(y_prob)

    for idx, c in enumerate(class_list):
        y_onehot[:, idx] = (y_true == c).astype(float)

    # Multiclass Brier score: mean squared error across all classes
    diffs = (y_prob - y_onehot) ** 2
    per_class_brier = np.mean(diffs, axis=0)
    multiclass_brier = float(np.sum(diffs) / n_samples)

    return {
        "multiclass_brier": multiclass_brier,
        "brier_short": float(per_class_brier[0]),
        "brier_neutral": float(per_class_brier[1]),
        "brier_long": float(per_class_brier[2]),
    }


def evaluate_model_on_test(
    model: Any,
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray,
    model_name: str,
    class_weight: str | None = None,
    classes: list[float] | np.ndarray = (-1.0, 0.0, 1.0),
) -> FinalTestEvaluationMetrics:
    """Compute standardized evaluation metrics on the final test partition.

    Parameters
    ----------
    model : Any
        Fitted model instance with predict and predict_proba methods.
    X_test : pd.DataFrame | np.ndarray
        Test feature matrix.
    y_test : pd.Series | np.ndarray
        Test target series.
    model_name : str
        Human-readable model identifier.
    class_weight : str | None
        Class weighting configuration.
    classes : list[float] | np.ndarray
        Known class labels.

    Returns
    -------
    FinalTestEvaluationMetrics
        Structured evaluation metrics.
    """
    y_true_arr = np.asarray(y_test, dtype=float)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    class_list = [float(c) for c in classes]

    acc = float(accuracy_score(y_true_arr, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true_arr, y_pred))
    macro_prec = float(precision_score(y_true_arr, y_pred, average="macro", zero_division=0.0))
    macro_rec = float(recall_score(y_true_arr, y_pred, average="macro", zero_division=0.0))
    macro_f1 = float(f1_score(y_true_arr, y_pred, average="macro", zero_division=0.0))

    prec, rec, f1, supp = precision_recall_fscore_support(
        y_true_arr, y_pred, labels=class_list, zero_division=0.0
    )

    # ROC-AUC (OvR)
    try:
        roc_auc = float(roc_auc_score(y_true_arr, y_prob, multi_class="ovr", average="macro"))
    except ValueError:
        roc_auc = 0.5

    # Macro Average Precision (PR-AUC)
    pr_aucs: list[float] = []
    for idx, c in enumerate(class_list):
        y_bin = (y_true_arr == c).astype(int)
        if 0 < np.sum(y_bin) < len(y_bin):
            pr_aucs.append(float(average_precision_score(y_bin, y_prob[:, idx])))
    macro_pr_auc = float(np.mean(pr_aucs)) if pr_aucs else None

    # Brier Scores
    brier = compute_brier_scores(y_true=y_true_arr, y_prob=y_prob, classes=class_list)

    cm = confusion_matrix(y_true_arr, y_pred, labels=class_list).tolist()

    return FinalTestEvaluationMetrics(
        model_name=model_name,
        class_weight=class_weight,
        accuracy=acc,
        balanced_accuracy=bal_acc,
        macro_precision=macro_prec,
        macro_recall=macro_rec,
        macro_f1=macro_f1,
        precision_short=float(prec[0]),
        recall_short=float(rec[0]),
        f1_short=float(f1[0]),
        precision_neutral=float(prec[1]),
        recall_neutral=float(rec[1]),
        f1_neutral=float(f1[1]),
        precision_long=float(prec[2]),
        recall_long=float(rec[2]),
        f1_long=float(f1[2]),
        macro_roc_auc=roc_auc,
        macro_pr_auc=macro_pr_auc,
        brier_score_multiclass=brier["multiclass_brier"],
        brier_short=brier["brier_short"],
        brier_neutral=brier["brier_neutral"],
        brier_long=brier["brier_long"],
        confusion_matrix=cm,
        support_short=int(supp[0]),
        support_neutral=int(supp[1]),
        support_long=int(supp[2]),
    )


def compute_test_temporal_blocks(
    test_X: pd.DataFrame,
    test_y: pd.Series,
    test_timestamps: pd.Series,
    models: dict[str, Any],
    n_blocks: int = 3,
    classes: list[float] | np.ndarray = (-1.0, 0.0, 1.0),
) -> pd.DataFrame:
    """Evaluate performance across contiguous chronological test partition blocks.

    Parameters
    ----------
    test_X : pd.DataFrame
        Test features.
    test_y : pd.Series
        Test ground-truth labels.
    test_timestamps : pd.Series
        Test observation timestamps.
    models : dict[str, Any]
        Dictionary of model_name -> model_instance.
    n_blocks : int, default 3
        Number of contiguous chronological blocks.
    classes : list[float] | np.ndarray
        Class labels.

    Returns
    -------
    pd.DataFrame
        Metrics evaluated per chronological test block.
    """
    n_test = len(test_y)
    block_sz = n_test // n_blocks
    block_defs = [
        ("TEST_BLOCK_A", 0, block_sz),
        ("TEST_BLOCK_B", block_sz, 2 * block_sz),
        ("TEST_BLOCK_C", 2 * block_sz, n_test),
    ]

    records: list[dict[str, Any]] = []

    for model_name, model in models.items():
        preds = model.predict(test_X)
        probs = model.predict_proba(test_X)

        for b_name, s, e in block_defs:
            b_y = np.asarray(test_y.iloc[s:e], dtype=float)
            b_pred = preds[s:e]
            b_prob = probs[s:e]
            b_ts = test_timestamps.iloc[s:e]

            acc = float(accuracy_score(b_y, b_pred))
            bal_acc = float(balanced_accuracy_score(b_y, b_pred))
            m_f1 = float(f1_score(b_y, b_pred, average="macro", zero_division=0.0))

            r_short = float(recall_score(b_y == -1.0, b_pred == -1.0, zero_division=0.0))
            r_neutral = float(recall_score(b_y == 0.0, b_pred == 0.0, zero_division=0.0))
            r_long = float(recall_score(b_y == 1.0, b_pred == 1.0, zero_division=0.0))

            try:
                auc = float(roc_auc_score(b_y, b_prob, multi_class="ovr", average="macro"))
            except ValueError:
                auc = 0.5

            records.append(
                {
                    "model": model_name,
                    "block": b_name,
                    "start_time": str(b_ts.iloc[0]),
                    "end_time": str(b_ts.iloc[-1]),
                    "rows": len(b_y),
                    "accuracy": acc,
                    "balanced_accuracy": bal_acc,
                    "macro_f1": m_f1,
                    "recall_short": r_short,
                    "recall_neutral": r_neutral,
                    "recall_long": r_long,
                    "roc_auc_ovr": auc,
                }
            )

    return pd.DataFrame(records)


def compute_test_confidence_analysis(
    probabilities: np.ndarray,
    predictions: np.ndarray,
    y_true: pd.Series | np.ndarray,
    thresholds: list[float] | None = None,
    model_name: str = "model",
) -> pd.DataFrame:
    """Evaluate performance conditional on prediction confidence on the test partition.

    Parameters
    ----------
    probabilities : np.ndarray
        Predicted class probabilities (N, 3).
    predictions : np.ndarray
        Predicted class labels.
    y_true : pd.Series | np.ndarray
        True target labels.
    thresholds : list[float] | None
        Confidence thresholds.
    model_name : str
        Model identifier.

    Returns
    -------
    pd.DataFrame
        Detailed metrics breakdown per confidence threshold.
    """
    if thresholds is None:
        thresholds = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80]

    prob_arr = np.asarray(probabilities, dtype=float)
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(predictions, dtype=float)
    total_samples = len(y_true_arr)

    max_probs = np.max(prob_arr, axis=1)
    records: list[dict[str, Any]] = []

    for thresh in thresholds:
        mask = max_probs >= thresh
        covered_count = int(np.sum(mask))
        coverage = float(covered_count / total_samples) if total_samples > 0 else 0.0

        if covered_count > 0:
            sub_y = y_true_arr[mask]
            sub_pred = y_pred_arr[mask]

            acc = float(accuracy_score(sub_y, sub_pred))
            unique_classes = np.unique(sub_y)
            bal_acc = (
                float(balanced_accuracy_score(sub_y, sub_pred))
                if len(unique_classes) > 1
                else 0.3333
            )
            macro_f1 = float(f1_score(sub_y, sub_pred, average="macro", zero_division=0.0))

            rec_short = float(recall_score(sub_y == -1.0, sub_pred == -1.0, zero_division=0.0))
            rec_neutral = float(recall_score(sub_y == 0.0, sub_pred == 0.0, zero_division=0.0))
            rec_long = float(recall_score(sub_y == 1.0, sub_pred == 1.0, zero_division=0.0))

            pred_short_count = int(np.sum(sub_pred == -1.0))
            pred_neutral_count = int(np.sum(sub_pred == 0.0))
            pred_long_count = int(np.sum(sub_pred == 1.0))
        else:
            acc = None
            bal_acc = None
            macro_f1 = None
            rec_short = None
            rec_neutral = None
            rec_long = None
            pred_short_count = 0
            pred_neutral_count = 0
            pred_long_count = 0

        records.append(
            {
                "model": model_name,
                "threshold": float(thresh),
                "covered_count": covered_count,
                "coverage_pct": coverage * 100.0,
                "accuracy": acc,
                "balanced_accuracy": bal_acc,
                "macro_f1": macro_f1,
                "subset_recall_short": rec_short,
                "subset_recall_neutral": rec_neutral,
                "subset_recall_long": rec_long,
                "pred_count_short": pred_short_count,
                "pred_count_neutral": pred_neutral_count,
                "pred_count_long": pred_long_count,
            }
        )

    return pd.DataFrame(records)


def run_final_test_evaluation(
    splits: DatasetSplits,
    random_seed: int = 42,
    unlock_test_set: bool = False,
    save_artifacts: bool = True,
    reports_dir: Path | str = "reports",
) -> dict[str, Any]:
    """Execute complete Phase 11 final out-of-sample test evaluation workflow.

    Parameters
    ----------
    splits : DatasetSplits
        Chronological train/val/test splits container.
    random_seed : int, default 42
        Deterministic random seed.
    unlock_test_set : bool, default False
        Explicit authorization flag required to access the test set.
    save_artifacts : bool, default True
        Whether to save CSV and JSON reports to disk.
    reports_dir : Path | str, default "reports"
        Destination directory for report artifacts.

    Returns
    -------
    dict[str, Any]
        Dictionary of test metrics, comparison tables, temporal blocks, and metadata.
    """
    reports_path = Path(reports_dir)

    # 1. Enforce strict governance guard
    guard_audit = TestSetGovernanceGuard.verify_governance(
        splits=splits,
        unlock_test_set=unlock_test_set,
    )

    X_train = splits.train.X
    y_train = splits.train.y
    X_val = splits.val.X
    y_val = splits.val.y
    X_test = splits.test.X
    y_test = splits.test.y
    test_timestamps = splits.test.timestamps

    # 2. Instantiate all 7 candidate models established in Phase 8 / Phase 10
    models_spec: dict[str, tuple[Any, str | None]] = {
        "Majority Baseline": (MajorityClassClassifier(), None),
        "Logistic Regression (unweighted)": (
            LogisticRegressionBaseline(random_state=random_seed, class_weight=None),
            None,
        ),
        "Logistic Regression (balanced)": (
            LogisticRegressionBaseline(random_state=random_seed, class_weight="balanced"),
            "balanced",
        ),
        "Random Forest (unweighted)": (
            RandomForestBaseline(random_state=random_seed, class_weight=None),
            None,
        ),
        "Random Forest (balanced)": (
            RandomForestBaseline(random_state=random_seed, class_weight="balanced"),
            "balanced",
        ),
        "Extra Trees (unweighted)": (
            ExtraTreesBaseline(random_state=random_seed, class_weight=None),
            None,
        ),
        "Extra Trees (balanced)": (
            ExtraTreesBaseline(random_state=random_seed, class_weight="balanced"),
            "balanced",
        ),
    }

    test_metrics_list: list[FinalTestEvaluationMetrics] = []
    val_metrics_list: list[Any] = []
    trained_models: dict[str, Any] = {}

    for name, (model, cweight) in models_spec.items():
        # Train strictly on training data
        model.fit(X_train, y_train)
        trained_models[name] = model

        # Evaluate on validation (to record baseline for generalization gap)
        val_m = evaluate_experiment_model(
            model=model,
            X_val=X_val,
            y_val=y_val,
            model_name=name,
            class_weight=cweight,
        )
        val_metrics_list.append(val_m)

        # Evaluate on test (final holdout)
        test_m = evaluate_model_on_test(
            model=model,
            X_test=X_test,
            y_test=y_test,
            model_name=name,
            class_weight=cweight,
        )
        test_metrics_list.append(test_m)

    # 3. Build Comparison & Generalization Gap DataFrame
    rows = []
    for vm, tm in zip(val_metrics_list, test_metrics_list, strict=True):
        gap_acc = tm.accuracy - vm.accuracy
        gap_bal_acc = tm.balanced_accuracy - vm.balanced_accuracy
        gap_macro_f1 = tm.macro_f1 - vm.macro_f1
        gap_rec_short = tm.recall_short - vm.recall_short
        gap_rec_long = tm.recall_long - vm.recall_long
        gap_auc = (
            (tm.macro_roc_auc - vm.macro_roc_auc)
            if (tm.macro_roc_auc is not None and vm.macro_roc_auc is not None)
            else None
        )

        rows.append(
            {
                "model_name": tm.model_name,
                "class_weight": tm.class_weight if tm.class_weight is not None else "none",
                # Validation Metrics
                "val_accuracy": vm.accuracy,
                "val_balanced_accuracy": vm.balanced_accuracy,
                "val_macro_f1": vm.macro_f1,
                "val_recall_short": vm.recall_short,
                "val_recall_long": vm.recall_long,
                "val_macro_roc_auc": vm.macro_roc_auc,
                # Test Metrics
                "test_accuracy": tm.accuracy,
                "test_balanced_accuracy": tm.balanced_accuracy,
                "test_macro_precision": tm.macro_precision,
                "test_macro_recall": tm.macro_recall,
                "test_macro_f1": tm.macro_f1,
                "test_precision_short": tm.precision_short,
                "test_recall_short": tm.recall_short,
                "test_f1_short": tm.f1_short,
                "test_precision_neutral": tm.precision_neutral,
                "test_recall_neutral": tm.recall_neutral,
                "test_f1_neutral": tm.f1_neutral,
                "test_precision_long": tm.precision_long,
                "test_recall_long": tm.recall_long,
                "test_f1_long": tm.f1_long,
                "test_macro_roc_auc": tm.macro_roc_auc,
                "test_macro_pr_auc": tm.macro_pr_auc,
                "test_brier_multiclass": tm.brier_score_multiclass,
                "test_brier_short": tm.brier_short,
                "test_brier_neutral": tm.brier_neutral,
                "test_brier_long": tm.brier_long,
                # Generalization Gaps (Test - Validation)
                "gap_accuracy": gap_acc,
                "gap_balanced_accuracy": gap_bal_acc,
                "gap_macro_f1": gap_macro_f1,
                "gap_recall_short": gap_rec_short,
                "gap_recall_long": gap_rec_long,
                "gap_macro_roc_auc": gap_auc,
            }
        )
    df_comparison = pd.DataFrame(rows)

    # 4. Temporal Test Analysis across Blocks A, B, C
    temporal_models = {
        "Random Forest (balanced)": trained_models["Random Forest (balanced)"],
        "Logistic Regression (balanced)": trained_models["Logistic Regression (balanced)"],
        "Extra Trees (balanced)": trained_models["Extra Trees (balanced)"],
        "Random Forest (unweighted)": trained_models["Random Forest (unweighted)"],
    }
    df_temporal = compute_test_temporal_blocks(
        test_X=X_test,
        test_y=y_test,
        test_timestamps=test_timestamps,
        models=temporal_models,
        n_blocks=3,
    )

    # 5. Descriptive Confidence Analysis on Test Holdout
    conf_models = [
        "Random Forest (balanced)",
        "Logistic Regression (balanced)",
        "Extra Trees (balanced)",
    ]
    conf_dfs = []
    for m_name in conf_models:
        m_inst = trained_models[m_name]
        m_probs = m_inst.predict_proba(X_test)
        m_preds = m_inst.predict(X_test)
        cdf = compute_test_confidence_analysis(
            probabilities=m_probs,
            predictions=m_preds,
            y_true=y_test,
            model_name=m_name,
        )
        conf_dfs.append(cdf)
    df_confidence = pd.concat(conf_dfs, ignore_index=True)

    # 6. Build Metadata
    train_dist = {
        str(cls): {
            "count": int(np.sum(y_train == cls)),
            "pct": float(np.mean(y_train == cls) * 100.0),
        }
        for cls in DEFAULT_CLASSES
    }
    val_dist = {
        str(cls): {
            "count": int(np.sum(y_val == cls)),
            "pct": float(np.mean(y_val == cls) * 100.0),
        }
        for cls in DEFAULT_CLASSES
    }
    test_dist = {
        str(cls): {
            "count": int(np.sum(y_test == cls)),
            "pct": float(np.mean(y_test == cls) * 100.0),
        }
        for cls in DEFAULT_CLASSES
    }

    metadata: dict[str, Any] = {
        "study": "Phase 11 Final Out-of-Sample Test Evaluation",
        "instrument": "EURUSD",
        "timeframe": "M15",
        "target": splits.target_name,
        "horizon_bars": splits.horizon_bars,
        "fixed_threshold": 0.00050,
        "feature_count": X_train.shape[1],
        "train_rows": splits.train.row_count,
        "val_rows": splits.val.row_count,
        "test_rows": splits.test.row_count,
        "test_set_unlocked": True,
        "governance_audit": guard_audit,
        "class_distributions": {
            "train": train_dist,
            "val": val_dist,
            "test": test_dist,
        },
        "preselected_candidate": "Random Forest (balanced)",
        "models_evaluated": [tm.to_dict() for tm in test_metrics_list],
    }

    # 7. Save Report Artifacts if requested
    if save_artifacts:
        reports_path.mkdir(parents=True, exist_ok=True)

        comp_csv = reports_path / "final_test_comparison.csv"
        df_comparison.to_csv(comp_csv, index=False)

        temp_csv = reports_path / "final_test_temporal.csv"
        df_temporal.to_csv(temp_csv, index=False)

        conf_csv = reports_path / "final_test_confidence.csv"
        df_confidence.to_csv(conf_csv, index=False)

        meta_json = reports_path / "final_test_metadata.json"
        with open(meta_json, "w") as f:
            json.dump(metadata, f, indent=2)

    return {
        "comparison_df": df_comparison,
        "temporal_df": df_temporal,
        "confidence_df": df_confidence,
        "test_metrics": {tm.model_name: tm for tm in test_metrics_list},
        "metadata": metadata,
        "trained_models": trained_models,
        "governance_audit": guard_audit,
    }
