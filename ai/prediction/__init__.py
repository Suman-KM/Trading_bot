"""Inference and signal generation module.

Formats model outputs into standardized signal payloads:
{
    "symbol": str,
    "timestamp": str (ISO 8601 UTC),
    "action": "BUY" | "SELL" | "HOLD",
    "confidence": float (0.0 to 1.0),
    "model_version": str,
    "timeframe": str,
    "expected_return": float,
    "feature_version": str
}
Confidence represents model certainty, NOT account risk or position size.
"""
