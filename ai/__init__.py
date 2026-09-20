"""AI Autonomous Trading System - Member 2 Research & Signal Engine.

This package contains the research, modeling, backtesting, and signal generation
components owned by Member 2.

Architectural Rule:
The models and signal engines in this package NEVER directly place orders or control
account balances. They generate structured signals which are forwarded to Member 1's
risk engine for validation and execution.
"""

__version__ = "0.1.0"
