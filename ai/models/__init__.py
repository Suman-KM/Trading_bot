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
from ai.models.evaluation import (
    ModelEvaluationResult,
    evaluate_classification_model,
    plot_and_save_confusion_matrix,
)
from ai.models.experiments import (
    ExperimentMetrics,
    ExtraTreesBaseline,
    compute_confidence_directional_analysis,
    compute_temporal_robustness,
    evaluate_experiment_model,
    run_model_improvement_experiments,
)
from ai.models.pipeline import run_baseline_training_pipeline

__all__ = [
    "ExperimentMetrics",
    "ExtraTreesBaseline",
    "LogisticRegressionBaseline",
    "MajorityClassClassifier",
    "ModelEvaluationResult",
    "RandomForestBaseline",
    "compute_calibration_diagnostics",
    "compute_class_wise_metrics",
    "compute_confidence_coverage",
    "compute_confidence_directional_analysis",
    "compute_error_analysis",
    "compute_lr_feature_importance",
    "compute_model_agreement",
    "compute_probability_diagnostics",
    "compute_rf_feature_importance",
    "compute_temporal_robustness",
    "compute_temporal_validation_diagnostics",
    "evaluate_classification_model",
    "evaluate_experiment_model",
    "plot_and_save_confusion_matrix",
    "run_baseline_training_pipeline",
    "run_model_improvement_experiments",
    "verify_feature_sanity",
]
