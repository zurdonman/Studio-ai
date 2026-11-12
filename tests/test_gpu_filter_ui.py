"""UI regression via jsdom when Node/jsdom are installed (no browser/GPU)."""
import shutil
import subprocess
from pathlib import Path

import pytest


def test_gpu_filter_ui():
    root = Path(__file__).resolve().parents[1]
    if not shutil.which("node"):
        pytest.skip("Node not installed")
    probe = subprocess.run(["node", "-e", "require('jsdom')"], cwd=root, capture_output=True)
    if probe.returncode:
        pytest.skip("jsdom not installed")
    result = subprocess.run(["node", "tests/ui_l4_filter.cjs"], cwd=root, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
