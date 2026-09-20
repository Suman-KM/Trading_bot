"""Environment, dependency, and project structure smoke tests.

Validates that Member 2's Python 3.12 environment on Apple Silicon is correctly configured
and that all foundational dependencies and internal packages import without error.
"""

from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path

import pytest


def test_python_version() -> None:
    """Verify that Python version satisfies >=3.12 and <3.13."""
    major = sys.version_info.major
    minor = sys.version_info.minor
    assert (major, minor) >= (3, 12), f"Expected Python >= 3.12, got {major}.{minor}"
    assert (major, minor) < (3, 13), f"Expected Python < 3.13, got {major}.{minor}"


def test_system_architecture() -> None:
    """Verify that the host environment is macOS Apple Silicon (arm64)."""
    assert platform.system() == "Darwin", f"Expected Darwin, got {platform.system()}"
    assert platform.machine() == "arm64", f"Expected arm64, got {platform.machine()}"


@pytest.mark.parametrize(
    "package_name",
    [
        "numpy",
        "pandas",
        "pyarrow",
        "sklearn",
        "xgboost",
        "scipy",
        "matplotlib",
        "streamlit",
        "fastapi",
        "pydantic",
        "httpx",
        "pytest",
    ],
)
def test_core_dependencies_importable(package_name: str) -> None:
    """Verify that each core dependency can be imported and exposes a version."""
    module = importlib.import_module(package_name)
    assert module is not None
    assert hasattr(module, "__version__"), f"{package_name} missing __version__ attribute"
    assert isinstance(module.__version__, str)
    assert len(module.__version__) > 0


@pytest.mark.parametrize(
    "submodule_name",
    [
        "ai",
        "ai.features",
        "ai.models",
        "ai.training",
        "ai.prediction",
        "ai.evaluation",
        "ai.backtesting",
    ],
)
def test_ai_package_structure(submodule_name: str) -> None:
    """Verify that the internal ai package and its submodules are discoverable and importable."""
    module = importlib.import_module(submodule_name)
    assert module is not None


def test_directory_scaffolding() -> None:
    """Verify that the required directory structure exists with .gitkeep markers."""
    repo_root = Path(__file__).resolve().parent.parent
    expected_dirs = [
        repo_root / "data" / "raw",
        repo_root / "data" / "processed",
        repo_root / "data" / "external",
        repo_root / "dashboard",
        repo_root / "notebooks",
        repo_root / "scripts",
        repo_root / "tests",
        repo_root / "docs",
    ]
    for directory in expected_dirs:
        assert directory.is_dir(), f"Expected directory missing: {directory}"
        gitkeep = directory / ".gitkeep"
        # Only directories without source files strictly require .gitkeep
        if directory.name in ("raw", "processed", "external", "dashboard", "scripts", "docs"):
            assert gitkeep.is_file(), f"Missing .gitkeep in {directory}"
