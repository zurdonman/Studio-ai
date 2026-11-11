"""The audiovisual guide is shipped with the MCP server, not an external document."""
import json


def test_audiovisual_guide_available_through_tool_and_resource(temp_studio_env, monkeypatch):
    import studio.mcp_server as mcp
    from studio.catalog import catalog_manager
    client, _ = temp_studio_env

    def no_download(*args, **kwargs):
        raise AssertionError("Reading the guide must not download weights")
    monkeypatch.setattr(catalog_manager, "download_model", no_download)

    schema = {item["name"]: item for item in mcp._build_mcp_tools_schema()}
    assert "av" in schema["studio_get_model_guide"]["inputSchema"]["properties"]["guide"]["enum"]
    response = mcp._execute_mcp_tool("studio_get_model_guide", {"guide": "av"})
    assert not response.get("isError", False)
    payload = json.loads(response["content"][0]["text"])
    assert payload["uri"] == "studio://manual/audiovisual"
    for text in ("ltx-2.3-distilled-av", "ltx-2.5-distilled-av", "minimax-h3-local-candidate", "NO implementa", "audio_rel"):
        assert text in payload["markdown"]
    assert "studio_get_model_guide(guide='av')" in mcp.MCP_SERVER_INSTRUCTIONS

    headers = {"X-Studio-MCP-Key": client.get("/api/mcp/info").json()["token"]}
    request = {"jsonrpc": "2.0", "id": 1, "method": "resources/read", "params": {"uri": payload["uri"]}}
    resource = client.post("/mcp", headers=headers, json=request).json()["result"]["contents"][0]
    assert resource["text"] == payload["markdown"]


def test_existing_guide_choices_and_invalid_guide_still_work(temp_studio_env):
    import studio.mcp_server as mcp
    for guide, uri in [("operations", "studio://manual/model-oracle"), ("research", "studio://manual/model-research")]:
        response = mcp._execute_mcp_tool("studio_get_model_guide", {"guide": guide})
        assert json.loads(response["content"][0]["text"])["uri"] == uri
    assert mcp._execute_mcp_tool("studio_get_model_guide", {"guide": "invalid"})["isError"]
