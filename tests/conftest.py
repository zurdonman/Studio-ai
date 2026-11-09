"""One isolated Studio data root shared by CPU and metadata-only integration tests."""
import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def temp_studio_env(tmp_path_factory):
    root = tmp_path_factory.mktemp("studio_test_data")
    os.environ["STUDIO_DATA_DIR"] = str(root)
    os.environ["STUDIO_FORCE_LOCAL"] = "1"
    # Tests explicitly exercise a private updater instance; never call the real
    # public Hub because a test suite happens to run for more than a minute.
    os.environ["STUDIO_MODEL_AUTO_REFRESH"] = "0"
    # Private watchdog tests tick explicitly. Do not race hardware monkeypatches
    # in unrelated unit tests; all admission and allocation guards stay enabled.
    os.environ["STUDIO_MEMORY_POLL_SECONDS"] = "60"
    from app import app

    with TestClient(app) as client:
        yield client, root


@pytest.fixture
def colab_memory(temp_studio_env, monkeypatch):
    """Complete, mutable hardware doubles, not a bypass of admission/watchdog."""
    import types
    import studio.memory_manager as mm
    import studio.memory_policy as policy

    def configure(*, ram=64, available=52, gpu=True, vram=22, vram_free=21):
        state = {"ram": ram, "available": available, "gpu": gpu, "vram": vram, "vram_free": vram_free}
        def vm():
            total, free = state["ram"] * 1024**3, state["available"] * 1024**3
            return types.SimpleNamespace(total=total, available=free, used=total-free,
                                         percent=100*(total-free)/total)
        def graphics():
            return {"has_gpu": state["gpu"], "gpu_name": "NVIDIA L4" if state["gpu"] else "CPU",
                    "gpu_tier": "L4" if state["gpu"] else "CPU", "gpu_count": int(state["gpu"]),
                    "vram_total_gb": state["vram"] if state["gpu"] else 0,
                    "vram_free_gb": state["vram_free"] if state["gpu"] else 0,
                    "vram_used_gb": state["vram"]-state["vram_free"] if state["gpu"] else 0,
                    "bf16_supported": state["gpu"]}
        monkeypatch.setattr(policy.psutil, "virtual_memory", vm)
        monkeypatch.setattr(policy, "read_cgroup_memory", lambda: {})
        monkeypatch.setattr(mm, "query_gpu_info", graphics)
        return state
    return configure
