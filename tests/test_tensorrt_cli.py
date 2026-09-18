from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = (
    "export_onnx_tensorrt.py",
    "tensorrt_build_engine.py",
    "tensorrt_infer.py",
    "tensorrt_validate.py",
)


@pytest.mark.parametrize("script", SCRIPTS)
def test_cli_help_is_cpu_safe(script: str) -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout.lower()


def test_runtime_module_import_does_not_require_tensorrt() -> None:
    completed = subprocess.run(
        [sys.executable, "-c", "import deployment.tensorrt.runtime"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
