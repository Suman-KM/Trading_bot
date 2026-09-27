"""Baseline machine learning classification models.

Implements three reference models for EURUSD M15 directional forecasting:
1. MajorityClassClassifier: Naive constant reference predicting training modal class.
2. LogisticRegressionBaseline: Linear model with strictly train-fitted StandardScaler.
3. RandomForestBaseline: Non-linear ensemble operating on raw features.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class MajorityClassClassifier(BaseEstimator, ClassifierMixin):
    """Naive reference classifier that always predicts the training majority class.

    Attributes
    ----------
    classes_ : np.ndarray
        Sorted unique classes observed during training.
    majority_class_ : float
        The class label with the highest frequency in the training data.
    class_probabilities_ : np.ndarray
        Empirical training class probabilities aligned with classes_.
    """

    def __init__(self) -> None:
        self.classes_: np.ndarray | None = None
        self.majority_class_: float | None = None
        self.class_probabilities_: np.ndarray | None = None

    def fit(
        self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray
    ) -> MajorityClassClassifier:
        """Fit baseline by determining the majority class in y.

        Parameters
        ----------
        X : pd.DataFrame | np.ndarray
            Training features (ignored by majority baseline, but shape verified).
        y : pd.Series | np.ndarray
            Training ground-truth targets.

        Returns
        -------
        MajorityClassClassifier
            Fitted classifier instance.
        """
        y_arr = np.asarray(y)
        if len(y_arr) == 0:
            raise ValueError("Training targets y must not be empty.")

        unique_classes, counts = np.unique(y_arr, return_counts=True)
        self.classes_ = np.sort(unique_classes)

        # Majority class is the class with maximum count
        max_idx = int(np.argmax(counts))
        self.majority_class_ = float(unique_classes[max_idx])

        # Compute empirical class frequencies aligned with sorted classes_
        total_samples = len(y_arr)
        prob_dict = {cls: count / total_samples for cls, count in zip(unique_classes, counts)}
        self.class_probabilities_ = np.array([prob_dict[cls] for cls in self.classes_], dtype=float)

        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict the training majority class for all rows in X.

        Parameters
        ----------
        X : pd.DataFrame | np.ndarray
            Input observations.

        Returns
        -------
        np.ndarray
            Array of constant predictions matching the training majority class.
        """
        if self.majority_class_ is None:
            raise ValueError("MajorityClassClassifier is not fitted yet.")
        n_samples = len(X)
        return np.full(n_samples, self.majority_class_, dtype=float)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return empirical training class probabilities for all rows in X.

        Parameters
        ----------
        X : pd.DataFrame | np.ndarray
            Input observations.

        Returns
        -------
        np.ndarray
            Array of shape (n_samples, n_classes) with constant empirical probabilities.
        """
        if self.class_probabilities_ is None:
            raise ValueError("MajorityClassClassifier is not fitted yet.")
        n_samples = len(X)
        return np.tile(self.class_probabilities_, (n_samples, 1))


class LogisticRegressionBaseline:
    """Multiclass Logistic Regression baseline with train-fitted feature standardization.

    The StandardScaler is fitted strictly on the training set. Validation and test
    matrices are transformed exclusively using the fitted training scaler parameters.

    Attributes
    ----------
    random_state : int
        Deterministic random seed.
    max_iter : int
        Maximum optimization iterations.
    C : float
        Inverse regularization strength.
    solver : str
        Optimization algorithm (default 'lbfgs').
    scaler_ : StandardScaler
        Scaler fitted strictly on training data.
    model_ : LogisticRegression
        Fitted underlying scikit-learn LogisticRegression model.
    classes_ : np.ndarray
        Class labels known to the classifier.
    """

    def __init__(
        self,
        random_state: int = 42,
        max_iter: int = 1000,
        C: float = 1.0,
        solver: str = "lbfgs",
    ) -> None:
        self.random_state = random_state
        self.max_iter = max_iter
        self.C = C
        self.solver = solver

        self.scaler_: StandardScaler | None = None
        self.model_: LogisticRegression | None = None
        self.classes_: np.ndarray | None = None

    def get_params(self) -> dict[str, Any]:
        """Return model hyperparameters dictionary."""
        return {
            "random_state": self.random_state,
            "max_iter": self.max_iter,
            "C": self.C,
            "solver": self.solver,
            "scaler": "StandardScaler",
        }

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
    ) -> LogisticRegressionBaseline:
        """Fit scaler on training data, transform X_train, and fit LogisticRegression.

        Parameters
        ----------
        X_train : pd.DataFrame | np.ndarray
            Training feature matrix.
        y_train : pd.Series | np.ndarray
            Training target labels.

        Returns
        -------
        LogisticRegressionBaseline
            Fitted baseline instance.
        """
        self.scaler_ = StandardScaler()
        X_train_scaled = self.scaler_.fit_transform(X_train)

        self.model_ = LogisticRegression(
            random_state=self.random_state,
            max_iter=self.max_iter,
            C=self.C,
            solver=self.solver,
        )
        self.model_.fit(X_train_scaled, y_train)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Transform X using the training-fitted scaler and return predicted class labels.

        Parameters
        ----------
        X : pd.DataFrame | np.ndarray
            Input feature matrix.

        Returns
        -------
        np.ndarray
            Predicted class labels.
        """
        if self.scaler_ is None or self.model_ is None:
            raise ValueError("LogisticRegressionBaseline is not fitted yet.")
        X_scaled = self.scaler_.transform(X)
        return self.model_.predict(X_scaled)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Transform X using the training-fitted scaler and return predicted class probabilities.

        Parameters
        ----------
        X : pd.DataFrame | np.ndarray
            Input feature matrix.

        Returns
        -------
        np.ndarray
            Predicted class probabilities.
        """
        if self.scaler_ is None or self.model_ is None:
            raise ValueError("LogisticRegressionBaseline is not fitted yet.")
        X_scaled = self.scaler_.transform(X)
        return self.model_.predict_proba(X_scaled)


class RandomForestBaseline:
    """Non-linear Random Forest classification baseline.

    Operates directly on raw features without scaling, using bounded tree depth
    to prevent overfitting on high-frequency financial noise.

    Attributes
    ----------
    n_estimators : int
        Number of decision trees (default 100).
    max_depth : int
        Maximum tree depth limit (default 10).
    min_samples_leaf : int
        Minimum samples required at each leaf node (default 20).
    random_state : int
        Deterministic random seed.
    n_jobs : int
        Parallel worker threads (-1 for all cores).
    model_ : RandomForestClassifier
        Fitted underlying scikit-learn RandomForestClassifier.
    classes_ : np.ndarray
        Class labels known to the classifier.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        min_samples_leaf: int = 20,
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.model_: RandomForestClassifier | None = None
        self.classes_: np.ndarray | None = None

    def get_params(self) -> dict[str, Any]:
        """Return model hyperparameters dictionary."""
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
        }

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
    ) -> RandomForestBaseline:
        """Fit Random Forest classifier directly on training data.

        Parameters
        ----------
        X_train : pd.DataFrame | np.ndarray
            Training feature matrix.
        y_train : pd.Series | np.ndarray
            Training target labels.

        Returns
        -------
        RandomForestBaseline
            Fitted baseline instance.
        """
        self.model_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
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
            raise ValueError("RandomForestBaseline is not fitted yet.")
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
            raise ValueError("RandomForestBaseline is not fitted yet.")
        return self.model_.predict_proba(X)
