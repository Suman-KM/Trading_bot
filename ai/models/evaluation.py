"""Comprehensive evaluation metrics and confusion matrix visualization for ML models.

Computes multi-class classification metrics: accuracy, balanced accuracy, macro F1/precision/recall,
per-class statistics, ROC-AUC (OvR), and renders non-interactive confusion matrix plots.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

# Use non-GUI Agg backend for headless environments
matplotlib.use("Agg")


CLASS_NAME_MAP = {
    -1.0: "SHORT (-1)",
    0.0: "NEUTRAL (0)",
    1.0: "LONG (+1)",
}


@dataclass(frozen=True)
class ModelEvaluationResult:
    """Standardized quantitative validation evaluation metrics.

    Attributes
    ----------
    model_name : str
        Identifier of the evaluated model.
    accuracy : float
        Overall classification accuracy on validation set.
    balanced_accuracy : float
        Mean of recall obtained on each class.
    macro_precision : float
        Unweighted mean precision across classes.
    macro_recall : float
        Unweighted mean recall across classes.
    macro_f1 : float
        Unweighted mean F1 score across classes.
    per_class_precision : dict[str, float]
        Precision mapped by readable class name.
    per_class_recall : dict[str, float]
        Recall mapped by readable class name.
    per_class_f1 : dict[str, float]
        F1 score mapped by readable class name.
    per_class_support : dict[str, int]
        True validation support count for each class.
    confusion_matrix : list[list[int]]
        Row: true class, Column: predicted class.
    roc_auc_ovr : float | None
        Multiclass One-vs-Rest ROC-AUC score, or None if unavailable.
    train_rows : int
        Number of observations in training set.
    val_rows : int
        Number of observations in validation set.
    feature_count : int
        Number of input features.
    train_class_distribution : dict[str, float]
        Percentage distribution of classes in training set.
    val_class_distribution : dict[str, float]
        Percentage distribution of classes in validation set.
    pred_class_distribution : dict[str, float]
        Percentage distribution of predicted classes on validation set.
    """

    model_name: str
    accuracy: float
    balanced_accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_class_precision: dict[str, float]
    per_class_recall: dict[str, float]
    per_class_f1: dict[str, float]
    per_class_support: dict[str, int]
    confusion_matrix: list[list[int]]
    roc_auc_ovr: float | None
    train_rows: int
    val_rows: int
    feature_count: int
    train_class_distribution: dict[str, float]
    val_class_distribution: dict[str, float]
    pred_class_distribution: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        return asdict(self)


def evaluate_classification_model(
    model: Any,
    X_val: pd.DataFrame | np.ndarray,
    y_val: pd.Series | np.ndarray,
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    model_name: str = "model",
    classes: list[float] | None = None,
) -> ModelEvaluationResult:
    """Evaluate a trained model strictly on the validation set.

    Parameters
    ----------
    model : Any
        Fitted classifier with predict and optionally predict_proba methods.
    X_val : pd.DataFrame | np.ndarray
        Validation feature matrix.
    y_val : pd.Series | np.ndarray
        Validation ground-truth target vector.
    X_train : pd.DataFrame | np.ndarray
        Training feature matrix (for metadata record only).
    y_train : pd.Series | np.ndarray
        Training target vector (for class distribution calculation).
    model_name : str, default 'model'
        Name identifier for the model.
    classes : list[float] | None, optional
        Explicit class list ordered as [-1.0, 0.0, 1.0].

    Returns
    -------
    ModelEvaluationResult
        Complete evaluation metrics object.
    """
    if classes is None:
        classes = [-1.0, 0.0, 1.0]

    y_val_arr = np.asarray(y_val, dtype=float)
    y_train_arr = np.asarray(y_train, dtype=float)

    # 1. Predictions
    y_pred = model.predict(X_val)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    # 2. Basic multi-class metrics
    acc = float(accuracy_score(y_val_arr, y_pred_arr))
    bal_acc = float(balanced_accuracy_score(y_val_arr, y_pred_arr))

    # 3. Macro metrics
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_val_arr,
        y_pred_arr,
        labels=classes,
        average="macro",
        zero_division=0.0,
    )

    # 4. Per-class metrics
    p_per, r_per, f1_per, s_per = precision_recall_fscore_support(
        y_val_arr,
        y_pred_arr,
        labels=classes,
        average=None,
        zero_division=0.0,
    )

    per_class_p: dict[str, float] = {}
    per_class_r: dict[str, float] = {}
    per_class_f1: dict[str, float] = {}
    per_class_supp: dict[str, int] = {}

    for cls_val, p, r, f, s in zip(classes, p_per, r_per, f1_per, s_per):
        cls_name = CLASS_NAME_MAP.get(cls_val, str(cls_val))
        per_class_p[cls_name] = round(float(p), 4)
        per_class_r[cls_name] = round(float(r), 4)
        per_class_f1[cls_name] = round(float(f), 4)
        per_class_supp[cls_name] = int(s)

    # 5. Confusion Matrix
    cm_arr = confusion_matrix(y_val_arr, y_pred_arr, labels=classes)
    cm_list: list[list[int]] = [[int(val) for val in row] for row in cm_arr]

    # 6. ROC-AUC (OvR)
    roc_auc_val: float | None = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_val)
            # Ensure probabilities match classes
            if y_proba.shape[1] == len(classes):
                roc_auc_val = float(
                    roc_auc_score(
                        y_val_arr,
                        y_proba,
                        labels=classes,
                        multi_class="ovr",
                        average="macro",
                    )
                )
                roc_auc_val = round(roc_auc_val, 4)
        except Exception:
            roc_auc_val = None

    # 7. Class Distributions
    train_dist: dict[str, float] = {}
    val_dist: dict[str, float] = {}
    pred_dist: dict[str, float] = {}

    total_tr = len(y_train_arr)
    total_val = len(y_val_arr)
    total_pred = len(y_pred_arr)

    for cls_val in classes:
        cls_name = CLASS_NAME_MAP.get(cls_val, str(cls_val))
        tr_cnt = int(np.sum(y_train_arr == cls_val))
        v_cnt = int(np.sum(y_val_arr == cls_val))
        p_cnt = int(np.sum(y_pred_arr == cls_val))

        train_dist[cls_name] = round((tr_cnt / total_tr) * 100.0, 2)
        val_dist[cls_name] = round((v_cnt / total_val) * 100.0, 2)
        pred_dist[cls_name] = round((p_cnt / total_pred) * 100.0, 2)

    return ModelEvaluationResult(
        model_name=model_name,
        accuracy=round(acc, 4),
        balanced_accuracy=round(bal_acc, 4),
        macro_precision=round(float(p_macro), 4),
        macro_recall=round(float(r_macro), 4),
        macro_f1=round(float(f1_macro), 4),
        per_class_precision=per_class_p,
        per_class_recall=per_class_r,
        per_class_f1=per_class_f1,
        per_class_support=per_class_supp,
        confusion_matrix=cm_list,
        roc_auc_ovr=roc_auc_val,
        train_rows=len(X_train),
        val_rows=len(X_val),
        feature_count=X_train.shape[1],
        train_class_distribution=train_dist,
        val_class_distribution=val_dist,
        pred_class_distribution=pred_dist,
    )


def plot_and_save_confusion_matrix(
    cm: list[list[int]] | np.ndarray,
    classes: list[float] | None = None,
    model_name: str = "Model",
    output_path: Path | str = "reports/figures/confusion_matrix.png",
) -> None:
    """Render and save a formatted confusion matrix heatmap to disk.

    Parameters
    ----------
    cm : list[list[int]] | np.ndarray
        Confusion matrix array.
    classes : list[float] | None, optional
        Classes corresponding to rows and columns.
    model_name : str, default 'Model'
        Display title for the model.
    output_path : Path | str
        Destination PNG image path.
    """
    if classes is None:
        classes = [-1.0, 0.0, 1.0]

    class_labels = [CLASS_NAME_MAP.get(c, str(c)) for c in classes]
    cm_arr = np.array(cm, dtype=int)

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(cm_arr, cmap=plt.cm.Blues, alpha=0.85)

    # Format ticks
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(class_labels, fontsize=10)
    ax.set_yticklabels(class_labels, fontsize=10)

    # Move x ticks to bottom
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)

    # Annotate numbers in each cell
    total_val = np.sum(cm_arr)
    thresh = cm_arr.max() / 2.0
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            val = cm_arr[i, j]
            pct = (val / total_val) * 100.0 if total_val > 0 else 0.0
            color = "white" if val > thresh else "black"
            ax.text(
                j,
                i,
                f"{val:,}\n({pct:.1f}%)",
                ha="center",
                va="center",
                color=color,
                fontsize=10,
                fontweight="bold",
            )

    fig.colorbar(cax, fraction=0.046, pad=0.04)
    ax.set_title(f"Validation Confusion Matrix — {model_name}", fontsize=12, pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11, labelpad=8)
    ax.set_ylabel("True Label", fontsize=11, labelpad=8)
    plt.tight_layout()

    plt.savefig(out_p, dpi=150, bbox_inches="tight")
    plt.close(fig)
