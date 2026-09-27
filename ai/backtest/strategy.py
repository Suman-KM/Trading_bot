"""ML-assisted research trading strategy generating candidate signals.

Translates multi-class directional predictions (H=4, target: direction_4)
into strictly validated trading signals with volatility-based ATR stop loss and
take profit targets.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from ai.backtest.models import BacktestConfig
from trading.models.signal import Signal, SignalAction


class MLAssistedStrategy:
    """Strategy generating candidate entry signals from point-in-time features
    and fitted ML model.
    """

    def __init__(
        self,
        model: Any,
        config: BacktestConfig,
        model_version: str = "RandomForest_balanced_H4",
        feature_version: str = "phase5_features_v1",
        timeframe: str = "15m",
        symbol: str = "EURUSD",
    ) -> None:
        self.model = model
        self.config = config
        self.model_version = model_version
        self.feature_version = feature_version
        self.timeframe = timeframe
        self.symbol = symbol

        # Infer class index mapping from fitted model
        classes = list(getattr(model, "classes_", [-1.0, 0.0, 1.0]))
        self.classes = classes
        self.short_idx = classes.index(-1.0) if -1.0 in classes else None
        self.neutral_idx = classes.index(0.0) if 0.0 in classes else None
        self.long_idx = classes.index(1.0) if 1.0 in classes else None

    def evaluate_bar(
        self,
        features_row: pd.Series | pd.DataFrame | np.ndarray,
        current_close: float,
        current_atr: float,
        timestamp: datetime,
    ) -> tuple[Signal, dict[str, float]]:
        """Evaluate point-in-time features at the close of candle t to generate signal for t+1.

        Parameters
        ----------
        features_row : pd.Series | pd.DataFrame | np.ndarray
            Single row of 80 point-in-time features for bar t.
        current_close : float
            Closing price of bar t (Bid price).
        current_atr : float
            ATR14 value at bar t.
        timestamp : datetime
            Close timestamp of bar t.

        Returns
        -------
        tuple[Signal, dict[str, float]]
            (Candidate Signal, class probability mapping)
        """
        # Format feature matrix for prediction
        if isinstance(features_row, pd.Series):
            X = features_row.to_frame().T
        elif isinstance(features_row, np.ndarray):
            X = features_row.reshape(1, -1) if features_row.ndim == 1 else features_row
        else:
            X = features_row

        # Obtain class probabilities
        probs = self.model.predict_proba(X)[0]
        prob_dict = {
            "p_short": float(probs[self.short_idx]) if self.short_idx is not None else 0.0,
            "p_neutral": float(probs[self.neutral_idx]) if self.neutral_idx is not None else 0.0,
            "p_long": float(probs[self.long_idx]) if self.long_idx is not None else 0.0,
        }

        # Argmax class and confidence
        max_idx = int(np.argmax(probs))
        pred_class = self.classes[max_idx]
        confidence = float(probs[max_idx])

        # Enforce strategy confidence threshold (RiskEngine will enforce MIN_SIGNAL_CONFIDENCE)
        effective_threshold = self.config.confidence_threshold

        stop_distance = float(self.config.stop_loss_atr_multiple * current_atr)
        take_profit_distance = float(self.config.take_profit_atr_multiple * current_atr)

        if pred_class == 1.0 and confidence >= effective_threshold:
            action = SignalAction.BUY
            suggested_stop_loss = current_close - stop_distance
            suggested_take_profit = current_close + take_profit_distance
            expected_return = 0.00050  # Canonical H=4 fixed threshold
        elif pred_class == -1.0 and confidence >= effective_threshold:
            action = SignalAction.SELL
            suggested_stop_loss = current_close + stop_distance
            suggested_take_profit = current_close - take_profit_distance
            expected_return = -0.00050
        else:
            action = SignalAction.HOLD
            suggested_stop_loss = None
            suggested_take_profit = None
            expected_return = 0.0

        signal = Signal(
            symbol=self.symbol,
            action=action,
            confidence=confidence,
            timestamp=timestamp,
            model_version=self.model_version,
            timeframe=self.timeframe,
            expected_return=expected_return,
            feature_version=self.feature_version,
            suggested_entry_price=current_close,
            suggested_stop_loss=suggested_stop_loss,
            suggested_take_profit=suggested_take_profit,
        )

        return signal, prob_dict
