"""Controlled model improvement experiments (Phase 10).

Evaluates controlled class-weighting and alternative tree-based architectures
on EURUSD M15 directional forecasting (H=4, target: direction_4).

Experiments:
- Experiment A: Logistic Regression with class_weight='balanced' vs class_weight=None
- Experiment B: Random Forest with class_weight='balanced' vs class_weight=None
- Experiment C: Extra Trees Classifier with class_weight='balanced' and class_weight=None
- Experiment D: Decision / Confidence filtering analysis across confidence thresholds

CRITICAL SAFETY:
- The TEST partition is strictly locked and NEVER accessed, evaluated, or predicted on.
- Validation is the sole out-of-sample evaluation partition.
- Scaler is fitted strictly on the training partition.
- No broad hyperparameter search, grid search, or external ML frameworks.
- Neutral, descriptive reporting; no claims of profitability or trading viability.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    recall_score,
    roc_auc_score,
)

from ai.dataset.splits import DatasetSplits
from ai.models.baselines import (
    LogisticRegressionBaseline,
    MajorityClassClassifier,
    RandomForestBaseline,
)

DEFAULT_CLASSES: list[float] = [-1.0, 0.0, 1.0]


class ExtraTreesBaseline:
    """Non-linear Extremely Randomized Trees classifier baseline.

    Operates directly on raw features without scaling, evaluating whether extreme
    randomization in split candidate thresholds provides better generalization
    in high-noise financial time series.

    Attributes
    ----------
    n_estimators : int
        Number of decision trees (default 100).
    max_depth : int
        Maximum tree depth limit (default 10).
    min_samples_leaf : int
        Minimum samples required at each leaf node (default 20).
    class_weight : str | dict | None
        Class weighting strategy (None or 'balanced').
    random_state : int
        Deterministic random seed.
    n_jobs : int
        Parallel worker threads (-1 for all cores).
    model_ : ExtraTreesClassifier
        Fitted underlying scikit-learn ExtraTreesClassifier.
    classes_ : np.ndarray
        Class labels known to the classifier.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        min_samples_leaf: int = 20,
        class_weight: str | dict | None = None,
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.class_weight = class_weight
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.model_: ExtraTreesClassifier | None = None
        self.classes_: np.ndarray | None = None

    def get_params(self) -> dict[str, Any]:
        """Return model hyperparameters dictionary."""
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "class_weight": self.class_weight,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
        }

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
    ) -> ExtraTreesBaseline:
        """Fit Extra Trees classifier directly on training data.

        Parameters
        ----------
        X_train : pd.DataFrame | np.ndarray
            Training feature matrix.
        y_train : pd.Series | np.ndarray
            Training target labels.

        Returns
        -------
        ExtraTreesBaseline
            Fitted model instance.
        """
        self.model_ = ExtraTreesClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.model_.fit(X_train, y_train)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return predicted class labels for input observations.

        Parameters
        ----------
        X : pd.DataFrame | np.ndarray
            Input feature matrix.

        Returns
        -------
        np.ndarray
            Predicted class labels.
        """
        if self.model_ is None:
            raise ValueError("ExtraTreesBaseline is not fitted yet.")
        return self.model_.predict(X)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return predicted class probabilities for input observations.

        Parameters
        ----------
        X : pd.DataFrame | np.ndarray
            Input feature matrix.

        Returns
        -------
        np.ndarray
            Predicted class probabilities.
        """
        if self.model_ is None:
            raise ValueError("ExtraTreesBaseline is not fitted yet.")
        return self.model_.predict_proba(X)


@dataclass(frozen=True)
class ExperimentMetrics:
    """Comprehensive performance metrics for an evaluated model."""

    model_name: str
    class_weight: str | None
    accuracy: float
    balanced_accuracy: float
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
    confusion_matrix: list[list[int]]

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        return asdict(self)


def evaluate_experiment_model(
    model: Any,
    X_val: pd.DataFrame | np.ndarray,
    y_val: pd.Series | np.ndarray,
    model_name: str,
    class_weight: str | None = None,
    classes: list[float] | np.ndarray = (-1.0, 0.0, 1.0),
) -> ExperimentMetrics:
    """Compute detailed evaluation metrics on validation set.

    Parameters
    ----------
    model : Any
        Fitted classifier with predict and predict_proba methods.
    X_val : pd.DataFrame | np.ndarray
        Validation feature matrix.
    y_val : pd.Series | np.ndarray
        Validation true targets.
    model_name : str
        Readable model name.
    class_weight : str | None
        Class weighting configuration.
    classes : list[float] | np.ndarray
        Known class labels.

    Returns
    -------
    ExperimentMetrics
        Structured evaluation metrics.
    """
    y_true_arr = np.asarray(y_val, dtype=float)
    y_pred = model.predict(X_val)
    prob = model.predict_proba(X_val)

    class_list = [float(c) for c in classes]

    acc = float(accuracy_score(y_true_arr, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true_arr, y_pred))
    macro_f1 = float(f1_score(y_true_arr, y_pred, average="macro", zero_division=0.0))

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true_arr, y_pred, labels=class_list, zero_division=0.0
    )

    # OvR ROC-AUC
    try:
        roc_auc = float(roc_auc_score(y_true_arr, prob, multi_class="ovr", average="macro"))
    except ValueError:
        roc_auc = None

    # Macro Average Precision (PR-AUC)
    pr_aucs: list[float] = []
    for idx, c in enumerate(class_list):
        y_bin = (y_true_arr == c).astype(int)
        if 0 < np.sum(y_bin) < len(y_bin):
            pr_aucs.append(float(average_precision_score(y_bin, prob[:, idx])))
    macro_pr_auc = float(np.mean(pr_aucs)) if pr_aucs else None

    cm = confusion_matrix(y_true_arr, y_pred, labels=class_list).tolist()

    return ExperimentMetrics(
        model_name=model_name,
        class_weight=class_weight,
        accuracy=acc,
        balanced_accuracy=bal_acc,
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
        confusion_matrix=cm,
    )


def compute_temporal_robustness(
    val_X: pd.DataFrame,
    val_y: pd.Series,
    val_timestamps: pd.Series,
    models: dict[str, Any],
    classes: list[float] | np.ndarray = (-1.0, 0.0, 1.0),
) -> pd.DataFrame:
    """Evaluate performance across 3 contiguous chronological validation blocks.

    Parameters
    ----------
    val_X : pd.DataFrame
        Validation features.
    val_y : pd.Series
        Validation ground-truth targets.
    val_timestamps : pd.Series
        Validation timestamps.
    models : dict[str, Any]
        Dictionary of model_name -> model_instance (or (model, transformed_features)).
    classes : list[float] | np.ndarray
        Class labels.

    Returns
    -------
    pd.DataFrame
        Temporal validation metrics across Blocks A, B, and C.
    """
    n_val = len(val_y)
    block_sz = n_val // 3
    blocks = [
        ("VALIDATION_A", 0, block_sz),
        ("VALIDATION_B", block_sz, 2 * block_sz),
        ("VALIDATION_C", 2 * block_sz, n_val),
    ]

    records: list[dict[str, Any]] = []

    for model_name, model_item in models.items():
        if isinstance(model_item, tuple):
            model, val_feat = model_item
        else:
            model = model_item
            val_feat = val_X

        preds = model.predict(val_feat)
        probs = model.predict_proba(val_feat)

        for b_name, s, e in blocks:
            b_y = np.asarray(val_y.iloc[s:e], dtype=float)
            b_pred = preds[s:e]
            b_prob = probs[s:e]
            b_ts = val_timestamps.iloc[s:e]

            acc = float(accuracy_score(b_y, b_pred))
            bal_acc = float(balanced_accuracy_score(b_y, b_pred))
            m_f1 = float(f1_score(b_y, b_pred, average="macro", zero_division=0.0))

            r_short = float(recall_score(b_y == -1.0, b_pred == -1.0, zero_division=0.0))
            r_neutral = float(recall_score(b_y == 0.0, b_pred == 0.0, zero_division=0.0))
            r_long = float(recall_score(b_y == 1.0, b_pred == 1.0, zero_division=0.0))

            try:
                auc = float(roc_auc_score(b_y, b_prob, multi_class="ovr", average="macro"))
            except ValueError:
                auc = None

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


def compute_confidence_directional_analysis(
    probabilities: np.ndarray,
    predictions: np.ndarray,
    y_true: pd.Series | np.ndarray,
    thresholds: list[float] | None = None,
    classes: list[float] | np.ndarray = (-1.0, 0.0, 1.0),
    model_name: str = "model",
) -> pd.DataFrame:
    """Evaluate directional coverage and performance conditional on prediction confidence.

    Parameters
    ----------
    probabilities : np.ndarray
        Predicted class probabilities of shape (N, 3).
    predictions : np.ndarray
        Predicted class labels.
    y_true : pd.Series | np.ndarray
        True target labels.
    thresholds : list[float] | None
        Confidence thresholds to evaluate.
    classes : list[float] | np.ndarray
        Class labels.
    model_name : str
        Model identifier.

    Returns
    -------
    pd.DataFrame
        Detailed metrics breakdown per threshold.
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

            # Subset recalls
            rec_short = float(recall_score(sub_y == -1.0, sub_pred == -1.0, zero_division=0.0))
            rec_neutral = float(recall_score(sub_y == 0.0, sub_pred == 0.0, zero_division=0.0))
            rec_long = float(recall_score(sub_y == 1.0, sub_pred == 1.0, zero_division=0.0))

            # Global signal retention (fraction of ALL validation directional events captured)
            n_tot_short = np.sum(y_true_arr == -1.0)
            n_tot_long = np.sum(y_true_arr == 1.0)

            glob_ret_short = (
                float(np.sum((sub_y == -1.0) & (sub_pred == -1.0)) / n_tot_short)
                if n_tot_short > 0
                else 0.0
            )
            glob_ret_long = (
                float(np.sum((sub_y == 1.0) & (sub_pred == 1.0)) / n_tot_long)
                if n_tot_long > 0
                else 0.0
            )

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
            glob_ret_short = 0.0
            glob_ret_long = 0.0
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
                "global_retention_short": glob_ret_short,
                "global_retention_long": glob_ret_long,
                "pred_count_short": pred_short_count,
                "pred_count_neutral": pred_neutral_count,
                "pred_count_long": pred_long_count,
            }
        )

    return pd.DataFrame(records)


def run_model_improvement_experiments(
    splits: DatasetSplits,
    random_seed: int = 42,
    save_artifacts: bool = True,
    reports_dir: Path | str = "reports",
) -> dict[str, Any]:
    """Execute controlled Phase 10 model improvement study.

    Evaluates class-weighting and ExtraTrees architectures strictly on the
    chronological validation set. The test partition is strictly protected and
    never accessed, predicted on, or evaluated.

    Parameters
    ----------
    splits : DatasetSplits
        Chronological train/val/test splits container.
    random_seed : int, default 42
        Deterministic random seed.
    save_artifacts : bool, default True
        Whether to write CSV and JSON reports to disk.
    reports_dir : Path | str, default "reports"
        Destination directory for report artifacts.

    Returns
    -------
    dict[str, Any]
        Dictionary of experiment results, metrics, dataframes, and metadata.
    """
    reports_path = Path(reports_dir)

    # 1. Strict Safety & Contract Assertions
    assert splits.train is not None, "Training split is missing"
    assert splits.val is not None, "Validation split is missing"
    assert splits.train.X.shape[1] == 80, (
        f"Expected exactly 80 features, got {splits.train.X.shape[1]}"
    )
    assert splits.target_name == "direction_4", (
        f"Expected target direction_4, got {splits.target_name}"
    )

    X_train = splits.train.X
    y_train = splits.train.y
    X_val = splits.val.X
    y_val = splits.val.y
    val_timestamps = splits.val.timestamps

    # 2. Instantiate Candidate Baseline & Improved Models
    models: dict[str, tuple[Any, str | None]] = {
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

    metrics_list: list[ExperimentMetrics] = []
    trained_models: dict[str, Any] = {}

    for name, (model, cweight) in models.items():
        model.fit(X_train, y_train)

        metrics = evaluate_experiment_model(
            model=model,
            X_val=X_val,
            y_val=y_val,
            model_name=name,
            class_weight=cweight,
        )
        metrics_list.append(metrics)
        trained_models[name] = model

    # 3. Build Comparison DataFrame
    comparison_rows = []
    for m in metrics_list:
        comparison_rows.append(
            {
                "model_name": m.model_name,
                "class_weight": m.class_weight if m.class_weight is not None else "none",
                "accuracy": m.accuracy,
                "balanced_accuracy": m.balanced_accuracy,
                "macro_f1": m.macro_f1,
                "precision_short": m.precision_short,
                "recall_short": m.recall_short,
                "f1_short": m.f1_short,
                "precision_neutral": m.precision_neutral,
                "recall_neutral": m.recall_neutral,
                "f1_neutral": m.f1_neutral,
                "precision_long": m.precision_long,
                "recall_long": m.recall_long,
                "f1_long": m.f1_long,
                "macro_roc_auc": m.macro_roc_auc,
                "macro_pr_auc": m.macro_pr_auc,
            }
        )
    df_comparison = pd.DataFrame(comparison_rows)

    # 4. Temporal Robustness Analysis across Validation Blocks A, B, C
    temporal_models = {
        name: trained_models[name] for name in trained_models if name != "Majority Baseline"
    }
    df_temporal = compute_temporal_robustness(
        val_X=X_val,
        val_y=y_val,
        val_timestamps=val_timestamps,
        models=temporal_models,
    )

    # 5. Confidence / Coverage Analysis
    confidence_dfs = []
    key_models_confidence = [
        "Logistic Regression (balanced)",
        "Random Forest (balanced)",
        "Extra Trees (balanced)",
        "Random Forest (unweighted)",
    ]
    for m_name in key_models_confidence:
        m_inst = trained_models[m_name]
        probs = m_inst.predict_proba(X_val)
        preds = m_inst.predict(X_val)
        cdf = compute_confidence_directional_analysis(
            probabilities=probs,
            predictions=preds,
            y_true=y_val,
            model_name=m_name,
        )
        confidence_dfs.append(cdf)
    df_confidence = pd.concat(confidence_dfs, ignore_index=True)

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

    metadata: dict[str, Any] = {
        "study": "Phase 10 Controlled Model Improvement Study",
        "instrument": "EURUSD",
        "timeframe": "M15",
        "target": splits.target_name,
        "horizon_bars": splits.horizon_bars,
        "feature_count": X_train.shape[1],
        "train_rows": splits.train.row_count,
        "val_rows": splits.val.row_count,
        "test_rows": splits.test.row_count,
        "test_set_used": False,
        "evaluation_partition": "validation",
        "random_seed": random_seed,
        "class_distributions": {
            "train": train_dist,
            "val": val_dist,
        },
        "models_evaluated": [m.to_dict() for m in metrics_list],
    }

    # 7. Save Artifacts if Requested
    if save_artifacts:
        reports_path.mkdir(parents=True, exist_ok=True)

        comp_csv_path = reports_path / "model_improvement_comparison.csv"
        df_comparison.to_csv(comp_csv_path, index=False)

        temp_csv_path = reports_path / "model_improvement_temporal.csv"
        df_temporal.to_csv(temp_csv_path, index=False)

        conf_csv_path = reports_path / "model_improvement_confidence.csv"
        df_confidence.to_csv(conf_csv_path, index=False)

        meta_json_path = reports_path / "model_experiments_metadata.json"
        with open(meta_json_path, "w") as f:
            json.dump(metadata, f, indent=2)

    return {
        "metrics": {m.model_name: m for m in metrics_list},
        "comparison_df": df_comparison,
        "temporal_df": df_temporal,
        "confidence_df": df_confidence,
        "metadata": metadata,
        "trained_models": trained_models,
        "test_set_used": False,
    }
