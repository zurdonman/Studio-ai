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
    from app import app

    with TestClient(app) as client:
        yield client, root
