"""Baseline model training and validation pipeline orchestration.

Coordinates fitting of Majority, Logistic Regression, and Random Forest baselines strictly
on the training set, computes comprehensive validation metrics, generates confusion matrix
visualizations, and enforces strict test set protection.
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import sklearn

from ai.dataset.splits import DatasetSplits
from ai.models.baselines import (
    LogisticRegressionBaseline,
    MajorityClassClassifier,
    RandomForestBaseline,
)
from ai.models.evaluation import (
    ModelEvaluationResult,
    evaluate_classification_model,
    plot_and_save_confusion_matrix,
)

FIGURES_DIR = Path("reports/figures")
METADATA_PATH = Path("reports/baseline_model_metadata.json")


def run_baseline_training_pipeline(
    splits: DatasetSplits,
    random_seed: int = 42,
    generate_figures: bool = True,
    save_metadata: bool = True,
    figures_dir: Path | str = FIGURES_DIR,
    metadata_path: Path | str = METADATA_PATH,
) -> dict[str, ModelEvaluationResult]:
    """Execute baseline model training and validation evaluation.

    CRITICAL SAFETY & LEAKAGE CONTROL:
    - Models are fitted strictly on `splits.train_X` and `splits.train_y`.
    - Evaluation is performed strictly on `splits.val_X` and `splits.val_y`.
    - The TEST set (`splits.test_X`, `splits.test_y`) is NEVER accessed or evaluated.
      Test set is strictly reserved for future final verification.

    Parameters
    ----------
    splits : DatasetSplits
        Chronological dataset partitions from Phase 7.
    random_seed : int, default 42
        Deterministic random seed for stochastic algorithms.
    generate_figures : bool, default True
        If True, saves confusion matrix heatmaps to disk.
    save_metadata : bool, default True
        If True, saves machine-readable evaluation contract to JSON.
    figures_dir : Path | str
        Output directory for figure plots.
    metadata_path : Path | str
        Output destination for metadata JSON file.

    Returns
    -------
    dict[str, ModelEvaluationResult]
        Dictionary mapping model keys ('majority', 'logistic_regression', 'random_forest')
        to their evaluation results.
    """
    # 1. Test set protection verification
    # Note: Test set is reserved for later final evaluation and is not used during Phase 8.
    X_train = splits.train_X
    y_train = splits.train_y
    X_val = splits.val_X
    y_val = splits.val_y

    classes = [-1.0, 0.0, 1.0]
    p_figures = Path(figures_dir)
    p_figures.mkdir(parents=True, exist_ok=True)

    results: dict[str, ModelEvaluationResult] = {}
    models_metadata: dict[str, Any] = {}

    # ----------------------------------------------------
    # Model A: Majority Class Baseline
    # ----------------------------------------------------
    model_majority = MajorityClassClassifier()
    model_majority.fit(X_train, y_train)

    eval_majority = evaluate_classification_model(
        model=model_majority,
        X_val=X_val,
        y_val=y_val,
        X_train=X_train,
        y_train=y_train,
        model_name="Majority Class Baseline",
        classes=classes,
    )
    results["majority"] = eval_majority
    models_metadata["majority"] = {
        "model_type": "MajorityClassClassifier",
        "parameters": {"majority_class": model_majority.majority_class_},
        "validation_metrics": eval_majority.to_dict(),
    }

    if generate_figures:
        plot_and_save_confusion_matrix(
            cm=eval_majority.confusion_matrix,
            classes=classes,
            model_name="Majority Class Baseline",
            output_path=p_figures / "confusion_matrix_majority.png",
        )

    # ----------------------------------------------------
    # Model B: Logistic Regression (StandardScaler on train)
    # ----------------------------------------------------
    model_lr = LogisticRegressionBaseline(
        random_state=random_seed,
        max_iter=1000,
        C=1.0,
        solver="lbfgs",
    )
    model_lr.fit(X_train, y_train)

    eval_lr = evaluate_classification_model(
        model=model_lr,
        X_val=X_val,
        y_val=y_val,
        X_train=X_train,
        y_train=y_train,
        model_name="Logistic Regression",
        classes=classes,
    )
    results["logistic_regression"] = eval_lr
    models_metadata["logistic_regression"] = {
        "model_type": "LogisticRegressionBaseline",
        "parameters": model_lr.get_params(),
        "validation_metrics": eval_lr.to_dict(),
    }

    if generate_figures:
        plot_and_save_confusion_matrix(
            cm=eval_lr.confusion_matrix,
            classes=classes,
            model_name="Logistic Regression",
            output_path=p_figures / "confusion_matrix_logistic_regression.png",
        )

    # ----------------------------------------------------
    # Model C: Random Forest
    # ----------------------------------------------------
    model_rf = RandomForestBaseline(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=20,
        random_state=random_seed,
        n_jobs=-1,
    )
    model_rf.fit(X_train, y_train)

    eval_rf = evaluate_classification_model(
        model=model_rf,
        X_val=X_val,
        y_val=y_val,
        X_train=X_train,
        y_train=y_train,
        model_name="Random Forest",
        classes=classes,
    )
    results["random_forest"] = eval_rf
    models_metadata["random_forest"] = {
        "model_type": "RandomForestBaseline",
        "parameters": model_rf.get_params(),
        "validation_metrics": eval_rf.to_dict(),
    }

    if generate_figures:
        plot_and_save_confusion_matrix(
            cm=eval_rf.confusion_matrix,
            classes=classes,
            model_name="Random Forest",
            output_path=p_figures / "confusion_matrix_random_forest.png",
        )

    # ----------------------------------------------------
    # Compile Master Metadata Contract
    # ----------------------------------------------------
    metadata: dict[str, Any] = {
        "pipeline_phase": "Phase 8 — Baseline Machine Learning Models",
        "instrument": "EURUSD",
        "timeframe": "M15",
        "target_name": splits.target_name,
        "horizon_bars": splits.horizon_bars,
        "feature_count": len(splits.feature_names),
        "train_rows": len(splits.train),
        "validation_rows": len(splits.val),
        "test_rows": len(splits.test),
        "random_seed": random_seed,
        "test_set_used_for_selection": False,
        "test_set_status": "Strictly untouched / reserved for future out-of-sample evaluation",
        "software_environment": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "scikit_learn_version": sklearn.__version__,
            "scipy_version": scipy.__version__,
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
        },
        "models": models_metadata,
    }

    if save_metadata:
        p_meta = Path(metadata_path)
        p_meta.parent.mkdir(parents=True, exist_ok=True)
        with open(p_meta, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    return results
