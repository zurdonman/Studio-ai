"""Until ID-LoRA's conditioning exists, reject requests instead of faking voice identity."""
import json
from pathlib import Path

import pytest


def test_id_lora_is_a_blocked_informational_candidate(temp_studio_env):
    import studio.mcp_server as mcp
    from studio.catalog import catalog_manager
    model = catalog_manager.get_definition("lora-ltx23-id-talkvid-candidate")
    assert model["validation_state"] == "candidate"
    assert model["gpu_tested"] is False
    assert model["compatible_models"] == []
    assert model["artifacts"][0]["repo_id"] == "AviadDahan/LTX-2.3-ID-LoRA-TalkVid-3K"
    result = mcp._execute_mcp_tool("studio_inspect_model", {"model_id": model["id"]})
    payload = json.loads(result["content"][0]["text"])
    assert payload["assessment"]["runnable_now"] is False
    assert mcp._execute_mcp_tool("studio_prepare_model", {"model_id": model["id"]})["isError"]


def test_reference_voice_never_silently_becomes_audio_mux(temp_studio_env, monkeypatch):
    import studio.mcp_server as mcp
    from studio.engines.video_engine import generate_video_clip
    client, _ = temp_studio_env
    monkeypatch.setattr(mcp.job_manager, "submit", lambda *a, **kw: pytest.fail("A job should not be started"))
    response = mcp._execute_mcp_tool("studio_generate_video", {
        "video_submode": "i2v", "model_id": "ltx-2.3-distilled-av", "reference_voice_rel": "uploads/voice.wav", "prompt": "test"
    })
    assert response["isError"] and "VOICE_REFERENCE_BACKEND_UNAVAILABLE" in response["content"][0]["text"]
    api = client.post("/api/video/generate", json={"reference_voice_rel": "uploads/voice.wav"})
    assert api.status_code == 400 and "VOICE_REFERENCE_BACKEND_UNAVAILABLE" in api.text
    with pytest.raises(RuntimeError, match="VOICE_REFERENCE_BACKEND_UNAVAILABLE"):
        generate_video_clip(reference_voice_path=Path("voice.wav"))
    schema = {item["name"]: item for item in mcp._build_mcp_tools_schema()}
    props = schema["studio_generate_video"]["inputSchema"]["properties"]
    assert "reference_voice_rel" in props
    assert "NO es referencia de voz" in props["audio_rel"]["description"]
