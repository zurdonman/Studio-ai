"""MiniMax is metadata only until its modular backend is actually validated."""
import json

import pytest


MODEL_ID = "minimax-h3-local-candidate"


def test_candidate_inspection_and_no_claim_of_gpu_support(temp_studio_env):
    import studio.mcp_server as mcp
    from studio.catalog import catalog_manager
    model = catalog_manager.get_definition(MODEL_ID)
    assert model["validation_state"] == "candidate"
    assert model["gpu_tested"] is False
    assert model["engine"] == "unvalidated"
    assert model["artifacts"][0]["repo_id"] == "MiniMaxAI/MiniMax-H3"
    result = mcp._execute_mcp_tool("studio_inspect_model", {"model_id": MODEL_ID})
    payload = json.loads(result["content"][0]["text"])
    assert payload["metadata_only"] is True
    assert payload["automatic_weight_downloads"] is False
    assert payload["assessment"]["support"] == "candidate"
    assert payload["assessment"]["compatible"] is False
    assert payload["assessment"]["runnable_now"] is False
    assert payload["assessment"]["resources"]["ram"]["required_gb"] is None


def test_candidate_rejected_on_all_loading_paths(temp_studio_env, monkeypatch):
    import studio.mcp_server as mcp
    from studio.catalog import catalog_manager
    from studio.engines.video_engine import load_video_model_into_memory
    from studio.jobs import JobRecord
    client, _ = temp_studio_env
    monkeypatch.setattr(catalog_manager, "is_downloaded", lambda mid: True)
    assert client.post(f"/api/catalog/{MODEL_ID}/download").status_code == 400
    assert client.post("/api/memory/load", json={"model_id": MODEL_ID, "auto_download": True}).status_code == 400
    assert mcp._execute_mcp_tool("studio_prepare_model", {"model_id": MODEL_ID})["isError"]
    assert mcp._execute_mcp_tool("studio_generate_video", {"model_id": MODEL_ID, "mode": "ai", "video_submode": "t2v", "prompt": "Una escena"})["isError"]
    with pytest.raises(ValueError, match="MODEL_RECIPE_UNVALIDATED"):
        catalog_manager.download_model(MODEL_ID, JobRecord("download", "test"))
    with pytest.raises(ValueError, match="MODEL_RECIPE_UNVALIDATED"):
        load_video_model_into_memory(MODEL_ID, auto_download=True)
    schemas = {item["name"]: item for item in mcp._build_mcp_tools_schema()}
    assert MODEL_ID not in schemas["studio_prepare_model"]["inputSchema"]["properties"]["model_id"]["enum"]
