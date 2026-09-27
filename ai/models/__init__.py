"""Baseline Machine Learning models package for EURUSD M15 directional forecasting.

Provides naive Majority Class reference, regularized Logistic Regression with strictly
train-fitted scaling, and non-linear Random Forest baselines along with standardized
validation metrics and test-set protection.
"""

from __future__ import annotations

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
from ai.models.pipeline import run_baseline_training_pipeline

__all__ = [
    "LogisticRegressionBaseline",
    "MajorityClassClassifier",
    "ModelEvaluationResult",
    "RandomForestBaseline",
    "evaluate_classification_model",
    "plot_and_save_confusion_matrix",
    "run_baseline_training_pipeline",
]
