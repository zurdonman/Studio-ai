"""Crash-path regressions: no GPU, weight downloads, external tunnel or live Hub."""
import io
import json
import sys
import time
import types
import zipfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.usefixtures("temp_studio_env")


def test_ram_guard_fails_before_native_allocations(monkeypatch, colab_memory):
    colab_memory(ram=53, available=39, gpu=False)
    from studio import memory_manager as mm
    monkeypatch.setattr(mm.psutil, "virtual_memory", lambda: types.SimpleNamespace(total=53 * 1024**3, available=39 * 1024**3))
    with pytest.raises(RuntimeError, match="MEMORY_BUDGET_EXCEEDED"):
        mm.require_available_host_ram(40, stage="14B")
    monkeypatch.setattr(mm.psutil, "virtual_memory", lambda: types.SimpleNamespace(total=53 * 1024**3, available=50 * 1024**3))
    assert mm.require_available_host_ram(40, stage="14B")["available_gb"] == 50


def test_video_ram_guard_runs_after_release_before_download_or_load(monkeypatch, tmp_path, colab_memory):
    colab_memory(ram=53, available=20)
    from studio.engines import video_engine as engine
    from studio import memory_manager as mm
    from studio.catalog import DEFAULT_CATALOG
    definition = next(d for d in DEFAULT_CATALOG if d["id"] == "wan21-vace-14b-fp8")
    events = []
    old_handle = {"pipe": object()}
    manager = mm.RuntimeMemoryManager()
    monkeypatch.setattr(mm, "flush_ram_and_vram", lambda: None)
    monkeypatch.setattr(mm, "query_gpu_info", lambda: {"vram_used_gb": 0})
    monkeypatch.setattr(mm.psutil, "virtual_memory", lambda: types.SimpleNamespace(total=53 * 1024**3, available=20 * 1024**3, used=33 * 1024**3))
    manager.register_loaded("image", "qwen", "Qwen", old_handle)
    monkeypatch.setattr(engine, "memory_manager", manager)
    monkeypatch.setattr(engine, "catalog_manager", types.SimpleNamespace(
        get_definition=lambda _: definition, is_downloaded=lambda _: False,
        download_model=lambda *a, **k: events.append("download")))
    monkeypatch.setattr(engine, "flush_ram_and_vram", lambda: None)
    monkeypatch.setattr(engine, "_instantiate_video_pipeline", lambda *a, **k: events.append("instantiate"))
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True)))
    monkeypatch.setitem(sys.modules, "diffusers", types.ModuleType("diffusers"))
    with pytest.raises(RuntimeError, match="MEMORY_BUDGET_EXCEEDED"):
        engine.load_video_model_into_memory(definition["id"], auto_download=True)
    assert manager.get_slot("image") is None and old_handle == {}
    assert events == []


def test_fp8_runtime_missing_does_not_expand_or_read_checkpoint(monkeypatch, tmp_path):
    from studio.engines.video_engine import _load_wan_fp8_single_file_transformer
    st = types.ModuleType("safetensors")
    st.torch = types.ModuleType("safetensors.torch")
    def forbidden(*a, **k):
        pytest.fail("Unsupported FP8 must stop before reading weights")
    st.torch.load_file = forbidden
    monkeypatch.setitem(sys.modules, "safetensors", st)
    monkeypatch.setitem(sys.modules, "safetensors.torch", st.torch)
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    with pytest.raises(RuntimeError, match="FP8_RUNTIME_UNSUPPORTED"):
        _load_wan_fp8_single_file_transformer(object, tmp_path / "14b.safetensors", tmp_path)


@pytest.mark.parametrize("original_format", [True, False])
def test_fp8_loader_restores_original_precision_of_excluded_modules(monkeypatch, tmp_path, original_format):
    from studio.engines import video_engine as engine
    class Tensor:
        def __init__(self, value, dtype):
            self.value, self.dtype, self.device = value, dtype, "cpu"
            self.data = self
        def to(self, *, device, dtype):
            return Tensor(self.value, dtype)
        def copy_(self, other):
            self.value = other.value
    original = Tensor(0.123456, "bf16")
    restored = Tensor(0.125, "bf16")
    torch = types.ModuleType("torch")
    torch.float8_e4m3fn, torch.bfloat16, torch.float16, torch.float32 = "fp8", "bf16", "fp16", "fp32"
    st = types.ModuleType("safetensors")
    st.torch = types.ModuleType("safetensors.torch")
    st.torch.load_file = lambda *a, **k: {
        ("blocks.0.norm2.weight" if original_format else "blocks.0.norm3.weight"): original,
        ("blocks.0.self_attn.q.weight" if original_format else "blocks.0.attn1.to_q.weight"): Tensor(1, "fp8"),
    }
    utils = types.ModuleType("diffusers.loaders.single_file_utils")
    utils.convert_wan_transformer_to_diffusers = lambda sd: {"blocks.0.norm3.weight": sd.pop("blocks.0.norm2.weight")}
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "safetensors", st)
    monkeypatch.setitem(sys.modules, "safetensors.torch", st.torch)
    monkeypatch.setitem(sys.modules, "diffusers.loaders.single_file_utils", utils)
    monkeypatch.setattr(engine, "_enable_fp8_layerwise_casting_safe", lambda tr: tr)
    class Transformer:
        enable_layerwise_casting = lambda *a, **k: None
        @classmethod
        def from_single_file(cls, sd, **kw):
            assert kw["torch_dtype"] == "fp8"
            assert kw["device"] == "cpu" and kw["low_cpu_mem_usage"] and kw["local_files_only"]
            sd.clear()  # Upstream conversion consumes the input mapping.
            return cls()
        def named_parameters(self):
            return [("blocks.0.norm3.weight", restored)]
        def named_buffers(self):
            return []
    engine._load_wan_fp8_single_file_transformer(Transformer, tmp_path / "fp8.safetensors", tmp_path)
    assert restored.value == original.value


@pytest.fixture
def fake_vision_runtime(monkeypatch, tmp_path, colab_memory):
    colab_memory()
    from studio.engines import coherence_engine as engine
    calls, registered = [], []
    # Deliberately has no nonexistent local_artifact_dir method.
    monkeypatch.setattr(engine, "catalog_manager", types.SimpleNamespace(
        get_definition=lambda _: {"id": "qwen25-vl-3b-coherence", "name": "Qwen VL", "engine": "qwen25_vl_coherence",
                                  "min_vram_gb": 3.2, "artifacts": [{"role": "base", "subdir": "base"}]},
        is_downloaded=lambda _: True, model_dir=lambda _: tmp_path))
    from studio.memory_manager import RuntimeMemoryManager
    manager = RuntimeMemoryManager()
    register = manager.register_loaded
    def register_and_capture(**kw):
        registered.append(kw)
        return register(**kw)
    monkeypatch.setattr(manager, "register_loaded", register_and_capture)
    monkeypatch.setattr(engine, "memory_manager", manager)
    monkeypatch.setattr(engine, "ensure_patchable_runtime",
                        lambda **_kw: {"ok": True, "reimported": False, "notice": ""})
    monkeypatch.setattr(engine, "query_gpu_info", lambda: {"vram_free_gb": 15, "vram_used_gb": 3})
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True), bfloat16="bf16"))
    transformers = types.ModuleType("transformers")
    def load_processor(path, **kw):
        calls.append(("processor", path, kw))
        return object()
    def load_model(path, **kw):
        calls.append(("model", path, kw))
        return types.SimpleNamespace(eval=lambda: None)
    transformers.AutoProcessor = types.SimpleNamespace(from_pretrained=load_processor)
    transformers.BitsAndBytesConfig = lambda **kw: kw
    transformers.Qwen2_5_VLForConditionalGeneration = types.SimpleNamespace(from_pretrained=load_model)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    return engine, calls, registered, transformers, tmp_path


def test_vision_loader_uses_real_catalog_api_and_local_artifacts(fake_vision_runtime):
    engine, calls, registered, _, root = fake_vision_runtime
    result = engine.load_coherence_vlm_into_memory()
    assert result["model_id"] == "qwen25-vl-3b-coherence"
    assert len(registered) == 1
    assert all(path == str(root / "base") and kw["local_files_only"] and kw["trust_remote_code"] is False
               for _, path, kw in calls)
    assert calls[1][2]["quantization_config"]["load_in_4bit"] is True


def test_vision_quantization_failure_does_not_retry_in_bf16(fake_vision_runtime):
    engine, calls, registered, transformers, _ = fake_vision_runtime
    def fail(*a, **kw):
        calls.append(("failed", a, kw))
        raise RuntimeError("quantization incompatible")
    transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained = fail
    with pytest.raises(RuntimeError, match="quantization incompatible"):
        engine.load_coherence_vlm_into_memory()
    assert len([c for c in calls if c[0] == "failed"]) == 1 and registered == []


@pytest.mark.parametrize("mode,all_consistent", [("hybrid_auto", False), ("fast_cv", True)])
def test_cv_pass_does_not_claim_identity_or_clear_semantic_gate(monkeypatch, tmp_path, mode, all_consistent):
    from studio.engines import coherence_engine as engine
    monkeypatch.setattr(engine, "query_gpu_info", lambda: {"has_gpu": True})
    monkeypatch.setattr(engine, "analyze_pair_fast_cv", lambda *a: {
        "spatial_displacement_pct": 68.2, "cv_overall_score": 74.5,
        "color_palette_score": 64.4, "lighting_exposure_score": 97.6, "aspect_ratio_compatible": True})
    def unavailable(**kw):
        raise AttributeError("no local_artifact_dir")
    monkeypatch.setattr(engine, "load_coherence_vlm_into_memory", unavailable)
    result = engine.run_coherence_analysis(tmp_path / "anchor.png", [tmp_path / "candidate.png"], mode=mode,
                                           expect_spatial_progression=True)
    assert result["comparisons"][0]["verdict"] == "PASS"
    assert result["effective_comparison_threshold"] == 62
    assert result["all_consistent"] is all_consistent
    assert result["photometric_consistent"] is True
    assert not result["identity_verified"] and not result["ready_for_video"]
    assert result["assessment_scope"] == "photometric_only_not_identity"
    assert "NO identidad" in result["agent_next_step"]


def test_rest_coherence_preserves_spatial_progression_flag(temp_studio_env, monkeypatch):
    from PIL import Image
    import app
    client, _ = temp_studio_env
    image = io.BytesIO()
    Image.new("RGB", (32, 32)).save(image, format="PNG")
    uploaded = client.post("/api/assets/upload", files={"file": ("audit.png", image.getvalue(), "image/png")}).json()["asset"]["rel_path"]
    monkeypatch.setattr(app, "run_coherence_analysis", lambda **kw: {"spatial": kw["expect_spatial_progression"]})
    response = client.post("/api/coherence/analyze", json={"anchor_image_rel": uploaded,
        "candidate_image_rels": [uploaded], "expect_spatial_progression": True}).json()
    job_id = response["job"]["id"]
    for _ in range(100):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("completed", "failed"):
            break
        time.sleep(0.01)
    assert job["result"] == {"spatial": True}


def test_diagnostic_bundle_is_offline_redacted_and_excludes_weights(monkeypatch, tmp_path):
    from studio import diagnostics as diag
    monkeypatch.setattr(diag, "_command", lambda _: "offline")
    monkeypatch.setattr(diag.importlib.metadata, "distributions", lambda: [])
    session = diag.DiagnosticSession(tmp_path)
    secret = "st_mcp_not_a_real_credential"
    session.write("server.log", f"GET /mcp?token={secret} Authorization: Bearer private-session\n")
    session.write("tunnel.log", "origin exited unexpectedly\n")
    session.sample(99999999)
    session.event("server_exited", status="unexpected_exit", exit=diag.describe_exit(-9))
    (tmp_path / "state" / "session_console.log").write_text('old token hf_abcdefgh12345678 {"token":"custom-private-value"}')
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "14b.safetensors").write_bytes(b"NOT_FOR_BUNDLE")
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs" / "frame.jpg").write_bytes(b"IMAGE_NOT_FOR_BUNDLE")
    bundle = diag.export_diagnostic_bundle(tmp_path)
    with zipfile.ZipFile(bundle) as zf:
        payload = "\n".join(zf.read(n).decode() for n in zf.namelist())
        assert secret not in payload and "private-session" not in payload and "custom-private-value" not in payload
        assert "hf_abcdefgh12345678" not in payload and "NOT_FOR_BUNDLE" not in payload
        assert not any(n.startswith("models/") or n.endswith(".jpg") for n in zf.namelist())
        assert "frame.jpg" in zf.read("outputs_inventory.json").decode()
        assert any(n.endswith("server.log") for n in zf.namelist())
        assert any(n.endswith("telemetry.jsonl") for n in zf.namelist())


def test_diagnostic_streams_rotate_and_exit_classification_does_not_invent_oom(monkeypatch, tmp_path):
    from studio import diagnostics as diag
    monkeypatch.setattr(diag, "MAX_LOG_BYTES", 128)
    session = diag.DiagnosticSession(tmp_path)
    session.write("server.log", "x" * 128)
    session.write("server.log", "tail")
    assert (session.folder / "server.log.1").stat().st_size == 128
    assert (session.folder / "server.log").read_text() == "tail"
    assert "no confirmado" in diag.describe_exit(-9)["explanation"]
    before = {"cgroup": {"path": "/group", "events": {"oom_kill": 1}}}
    after = {"cgroup": {"path": "/group", "events": {"oom_kill": 3}}}
    killed = diag.describe_exit(-9, before, after)
    assert killed["cgroup_oom_kill_delta"] == 2
    assert "no identifica" in killed["explanation"]
    assert diag.describe_exit(-11)["signal"] == "SIGSEGV"


@pytest.mark.parametrize("returncode", [17, -9])
def test_launcher_reports_real_child_exit_and_saves_raw_stderr(monkeypatch, tmp_path, capsys, returncode):
    import launch_colab as launcher
    from studio import diagnostics as diag
    monkeypatch.setenv("STUDIO_DATA_DIR", str(tmp_path / "data"))
    repo = tmp_path / "repo"
    repo.mkdir()
    termination = "import os, signal; os.kill(os.getpid(), signal.SIGKILL)" if returncode == -9 else "sys.exit(17)"
    (repo / "app.py").write_text("import sys\nprint('native load failed', file=sys.stderr, flush=True)\n" + termination + "\n")
    monkeypatch.setattr(launcher, "ROOT_DIR", repo)
    monkeypatch.setattr(launcher, "ensure_dependencies", lambda: None)
    monkeypatch.setattr(launcher, "wait_for_local_server", lambda *a, **k: {"hardware": {}})
    monkeypatch.setattr(launcher, "start_cloudflared_tunnel", lambda *a, **k: (None, None))
    monkeypatch.setattr(launcher, "try_colab_proxy_url", lambda _: None)
    monkeypatch.setattr(launcher, "_print_box", lambda _: None)
    monkeypatch.setattr(diag, "_command", lambda _: "offline")
    monkeypatch.setattr(diag.importlib.metadata, "distributions", lambda: [])
    launcher.main()
    folders = list((tmp_path / "data" / "state" / "diagnostics").iterdir())
    status = json.loads((folders[0] / "status.json").read_text())
    assert status["status"] == "unexpected_exit" and status["exit"]["returncode"] == returncode
    assert "native load failed" in (folders[0] / "server.log").read_text()
    output = capsys.readouterr().out
    assert "terminó" in output and "detenido limpiamente" not in output
    assert list((tmp_path / "data" / "temp").glob("*.zip"))


def test_bridge_retries_only_reads_and_preserves_same_job_after_502(monkeypatch):
    import studio_mcp as bridge
    client = bridge.StudioColabBridge("https://example.invalid")
    calls = []
    def http(method, path, **kw):
        calls.append((method, path))
        if method == "POST":
            return {"result": {"content": [{"type": "text", "text": json.dumps({"job_id": "job-existing", "status": "running"})}]}}
        raise bridge.BridgeTransportError("502 then tunnel unavailable", status_code=502)
    monkeypatch.setattr(client, "_http_json", http)
    monkeypatch.setattr(bridge.time, "sleep", lambda _: None)
    response = client._handle_tool_call({"id": 1, "params": {"name": "studio_generate_video", "arguments": {"wait": True}}})
    payload = json.loads(response["result"]["content"][0]["text"])
    assert payload["job_id"] == "job-existing" and payload["do_not_resubmit"] is True
    assert payload["status"] == "unknown_not_failed"
    assert payload["resume_tool_call"]["arguments"] == {"job_id": "job-existing"}
    assert len([c for c in calls if c[0] == "POST"]) == 1
    assert len([c for c in calls if c[0] == "GET"]) == 4


def test_bridge_can_recover_transient_502_without_replaying_generation(monkeypatch):
    import studio_mcp as bridge
    client = bridge.StudioColabBridge("https://example.invalid")
    responses = [bridge.BridgeTransportError("502", status_code=502), {"id": "same", "status": "completed", "progress": 100}]
    def http(method, path, **kw):
        assert method == "GET" and path.endswith("same")
        result = responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result
    monkeypatch.setattr(client, "_http_json", http)
    monkeypatch.setattr(bridge.time, "sleep", lambda _: None)
    assert client._poll_job_until_done("same")["status"] == "completed"
    assert not responses


def test_bridge_post_transport_error_reports_uncertain_submission_without_retry(monkeypatch):
    import studio_mcp as bridge
    client = bridge.StudioColabBridge("https://example.invalid")
    calls = []
    def http(method, path, **kw):
        calls.append(method)
        raise bridge.BridgeTransportError("530", status_code=530)
    monkeypatch.setattr(client, "_http_json", http)
    response = client._handle_tool_call({"id": 1, "params": {"name": "studio_generate_video", "arguments": {}}})
    payload = json.loads(response["result"]["content"][0]["text"])
    assert payload["error_code"] == "SUBMISSION_OUTCOME_UNKNOWN"
    assert payload["do_not_resubmit"] and payload["job_id"] is None
    assert calls == ["POST"]


def test_diagnostics_http_zip_and_notebook_recovery_cell(temp_studio_env):
    client, _ = temp_studio_env
    response = client.get("/api/logs/diagnostics")
    assert response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert "environment.json" in archive.namelist()
        assert "outputs_inventory.json" in archive.namelist()
    notebook = json.loads((Path(__file__).resolve().parents[1] / "Studio_IA_Colab.ipynb").read_text())
    sources = ["".join(c["source"]) for c in notebook["cells"]]
    assert "--ff-only" in sources[1] and "--single-branch" in sources[1]
    assert "studio.diagnostics" in sources[-1] and "files.download" in sources[-1]
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "colab_cell", "exec")


def test_quick_tunnel_url_does_not_treat_cloudflare_api_as_a_public_tunnel():
    import launch_colab as launcher
    assert launcher._quick_tunnel_url('Request failed: POST https://api.trycloudflare.com/tunnel') is None
    assert launcher._quick_tunnel_url('GET https://API.trycloudflare.com/v1: timeout') is None
    assert launcher._quick_tunnel_url('| https://apple-ocean-flower-lake.trycloudflare.com |') == 'https://apple-ocean-flower-lake.trycloudflare.com'
    assert launcher._quick_tunnel_url('https://api.trycloudflare.com and https://fresh-quick.trycloudflare.com') == 'https://fresh-quick.trycloudflare.com'
    assert launcher._quick_tunnel_url('https://fresh-quick.trycloudflare.com.evil.example') is None


def test_tunnel_start_rejects_api_error_and_accepts_real_url(monkeypatch, tmp_path):
    import launch_colab as launcher
    monkeypatch.setattr(launcher, 'ROOT_DIR', tmp_path)
    binary = tmp_path / 'cloudflared'
    binary.write_text('#!/bin/sh\necho "error connecting to https://api.trycloudflare.com"\nexit 1\n')
    binary.chmod(0o755)
    proc, url = launcher.start_cloudflared_tunnel(7860)
    assert proc is None and url is None
    binary.write_text('#!/bin/sh\necho "| https://apple-ocean-flower-lake.trycloudflare.com |"\nsleep 5\n')
    proc, url = launcher.start_cloudflared_tunnel(7860)
    try:
        assert proc is not None and proc.poll() is None
        assert url == 'https://apple-ocean-flower-lake.trycloudflare.com'
    finally:
        launcher._stop_process(proc)


def test_launcher_reconnects_tunnel_without_restarting_server(monkeypatch, tmp_path, capsys):
    import subprocess
    import launch_colab as launcher
    from studio import diagnostics as diag
    monkeypatch.setenv('STUDIO_DATA_DIR', str(tmp_path / 'data'))
    repo = tmp_path / 'repo'
    repo.mkdir()
    (repo / 'app.py').write_text('import time\nprint("SERVER_ONCE", flush=True)\ntime.sleep(1.0)\n')
    monkeypatch.setattr(launcher, 'ROOT_DIR', repo)
    monkeypatch.setattr(launcher, 'ensure_dependencies', lambda: None)
    monkeypatch.setattr(launcher, 'wait_for_local_server', lambda *a, **k: {'hardware': {}})
    monkeypatch.setattr(launcher, 'try_colab_proxy_url', lambda _: None)
    monkeypatch.setattr(launcher, 'TUNNEL_RETRY_INITIAL_SECONDS', .02)
    monkeypatch.setattr(launcher, 'TUNNEL_RETRY_SECONDS', .02)
    monkeypatch.setattr(launcher, 'MONITOR_INTERVAL_SECONDS', .02)
    monkeypatch.setattr(diag, '_command', lambda _: 'offline')
    monkeypatch.setattr(diag.importlib.metadata, 'distributions', lambda: [])
    calls, registrations = [], []
    def fake_start(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            return None, None
        return subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(20)']), 'https://fresh-quick.trycloudflare.com'
    monkeypatch.setattr(launcher, 'start_cloudflared_tunnel', fake_start)
    monkeypatch.setattr(launcher, '_register_tunnel', lambda port, url: registrations.append(url))
    launcher.main()
    output = capsys.readouterr().out
    assert output.count('SERVER_ONCE') == 1
    assert 'Sin enlace público' in output and 'Túnel reconectado' in output
    assert registrations == ['https://fresh-quick.trycloudflare.com']
    assert len(calls) == 2
    events_file = next((tmp_path / 'data' / 'state' / 'diagnostics').iterdir()) / 'events.jsonl'
    assert 'tunnel_reconnected' in events_file.read_text()


def test_launcher_recovers_tunnel_exit_while_server_keeps_running(monkeypatch, tmp_path, capsys):
    import subprocess
    import launch_colab as launcher
    from studio import diagnostics as diag
    monkeypatch.setenv('STUDIO_DATA_DIR', str(tmp_path / 'data'))
    repo = tmp_path / 'repo'
    repo.mkdir()
    (repo / 'app.py').write_text('import time\nprint("ONE_SERVER", flush=True)\ntime.sleep(1.2)\n')
    monkeypatch.setattr(launcher, 'ROOT_DIR', repo)
    monkeypatch.setattr(launcher, 'ensure_dependencies', lambda: None)
    monkeypatch.setattr(launcher, 'wait_for_local_server', lambda *a, **k: {'hardware': {}})
    monkeypatch.setattr(launcher, 'try_colab_proxy_url', lambda _: None)
    monkeypatch.setattr(launcher, 'TUNNEL_RETRY_INITIAL_SECONDS', .02)
    monkeypatch.setattr(launcher, 'MONITOR_INTERVAL_SECONDS', .02)
    monkeypatch.setattr(diag, '_command', lambda _: 'offline')
    monkeypatch.setattr(diag.importlib.metadata, 'distributions', lambda: [])
    calls, registrations = [], []
    def fake_start(*args, **kwargs):
        calls.append(1)
        sleep_time = '.2' if len(calls) == 1 else '20'
        proc = subprocess.Popen([sys.executable, '-c', f'import time; time.sleep({sleep_time})'])
        return proc, f'https://quick-{len(calls)}.trycloudflare.com'
    monkeypatch.setattr(launcher, 'start_cloudflared_tunnel', fake_start)
    monkeypatch.setattr(launcher, '_register_tunnel', lambda port, url: registrations.append(url))
    launcher.main()
    output = capsys.readouterr().out
    assert output.count('ONE_SERVER') == 1 and len(calls) == 2
    assert 'El túnel terminó' in output and 'Túnel reconectado' in output
    assert registrations == ['https://quick-1.trycloudflare.com', 'https://quick-2.trycloudflare.com']


def test_retry_closing_while_start_in_flight_stops_late_process(monkeypatch):
    import subprocess
    import threading
    import launch_colab as launcher
    gate, entered = threading.Event(), threading.Event()
    processes = []
    def delayed(*args):
        entered.set()
        gate.wait(timeout=3)
        proc = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(20)'])
        processes.append(proc)
        return proc, 'https://quick-late.trycloudflare.com'
    monkeypatch.setattr(launcher, 'start_cloudflared_tunnel', delayed)
    retry = launcher.TunnelRetry(7860, types.SimpleNamespace(write=lambda *args: None))
    assert retry.start() and entered.wait(timeout=3)
    assert not retry.start()
    retry.close()
    gate.set()
    retry._worker.join(timeout=3)
    assert not retry._worker.is_alive() and processes[0].poll() is not None
    assert retry.result() is None


def test_failed_tunnel_download_does_not_leave_binary_for_next_retry(monkeypatch, tmp_path):
    import launch_colab as launcher
    monkeypatch.setattr(launcher, 'ROOT_DIR', tmp_path)
    def incomplete(url, path):
        Path(path).write_text('partial')
        raise OSError('network interrupted')
    monkeypatch.setattr(launcher.urllib.request, 'urlretrieve', incomplete)
    assert launcher.start_cloudflared_tunnel(7860) == (None, None)
    assert not (tmp_path / 'cloudflared').exists()
    assert not (tmp_path / 'cloudflared.download').exists()
