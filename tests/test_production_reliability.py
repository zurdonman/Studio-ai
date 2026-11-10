"""Regressions for the three failures observed in the 30 s Colab run.

1. Wan LoRA stack aborted on a truncated `ing_pose.*` key (Diffusers conversion).
2. A failed load left `vid_speed_lora` attached, so the retry raised
   "Adapter name ... already in use" instead of starting clean.
3. The 4-bit inspector raised a bare bitsandbytes AssertionError when the
   compatibility patch was installed after the classes were created.

Plus the agent-guidance contract: ask once about fast LoRAs, never guess, and
never accept a "fast" request without the speed LoRA.
"""
import json
import sys
import types

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _isolated_studio(temp_studio_env):
    """Internal modules are imported below, after the shared data root is set."""
    return temp_studio_env


def _wandb_lora_state_dict():
    """Shape of the LoRA that failed: junk keys + truncated prefix + valid tensors."""
    return {
        ".0.bias": np.zeros((1,), dtype=np.float32),
        ".0.weight": np.zeros((2,), dtype=np.float32),
        "ing_pose.0.bias": np.ones((4,), dtype=np.float32),
        "ing_pose.0.weight": np.ones((4, 4), dtype=np.float32),
        "diffusion_model.blocks.0.self_attn.q.lora_down.weight": np.ones((4, 16), dtype=np.float32),
        "diffusion_model.blocks.0.self_attn.q.lora_up.weight": np.ones((16, 4), dtype=np.float32),
        "diffusion_model.blocks.0.self_attn.q.alpha": np.float32(4.0),
    }


def test_wan_lora_state_dict_never_returns_keys_diffusers_cannot_convert():
    from studio.engines.video_engine import _LORA_REPORT, _prepare_wan_lora_state_dict

    clean = _prepare_wan_lora_state_dict(_wandb_lora_state_dict())
    # Everything handed to `load_lora_weights` is already native PEFT format.
    assert clean, "the valid tensors must survive"
    assert all(k.startswith("transformer.") for k in clean)
    assert "transformer.blocks.0.attn1.q.lora_down.weight" in clean
    assert "transformer.blocks.0.attn1.q.lora_up.weight" in clean
    # The unusable export leftovers are reported, not silently fatal.
    assert getattr(_LORA_REPORT, "dropped_from_file", []) == [".0.bias", ".0.weight"]
    assert any("pose_patch_embedding" in k for k in getattr(_LORA_REPORT, "ignored", []))
    # alpha/rank was folded into the down matrix, keeping the effective scale.
    assert float(clean["transformer.blocks.0.attn1.q.lora_down.weight"][0, 0]) == pytest.approx(1.0)


def test_supported_wan_lora_is_left_to_diffusers_conversion():
    from studio.engines.video_engine import _prepare_wan_lora_state_dict

    raw = {
        "blocks.0.self_attn.q.lora_down.weight": np.ones((4, 16), dtype=np.float32),
        "blocks.0.self_attn.q.lora_up.weight": np.ones((16, 4), dtype=np.float32),
        "text_embedding.0.lora_down.weight": np.ones((4, 16), dtype=np.float32),
        "text_embedding.0.lora_up.weight": np.ones((16, 4), dtype=np.float32),
        "text_embedding.0.alpha": np.float32(2.0),
        "vace_blocks.0.proj.lora_down.weight": np.ones((4, 4), dtype=np.float32),
    }
    clean = _prepare_wan_lora_state_dict(raw)
    assert "diffusion_model.vace_blocks.0.proj.lora_down.weight" not in clean
    assert "diffusion_model.text_embedding.0.alpha" not in clean
    assert np.allclose(clean["diffusion_model.text_embedding.0.lora_down.weight"], 0.5)


class _PeftPipe:
    """Minimal stand-in with the real adapter bookkeeping of a Diffusers Wan pipeline."""

    def __init__(self, fail_on=()):
        self.adapters = {}
        self.weights = {}
        self.fail_on = set(fail_on)
        self.hooks_removed = 0

    class _Transformer:
        def __init__(self, pipe):
            self.peft_config = {}
            self._pipe = pipe

        def named_modules(self):
            return []

        def modules(self):
            return []

    @property
    def transformer(self):
        return self._Transformer(self)

    def remove_all_hooks(self):
        self.hooks_removed += 1

    def load_lora_weights(self, sd, adapter_name=None, **_kw):
        if adapter_name in self.fail_on:
            raise ValueError("state_dict should be empty at this point")
        self.adapters[adapter_name] = True

    def delete_adapters(self, names):
        for name in names:
            self.adapters.pop(name, None)
            self.weights.pop(name, None)

    def unload_lora_weights(self):
        self.adapters.clear()
        self.weights.clear()

    def set_adapters(self, names, adapter_weights=None):
        for name, weight in zip(names, adapter_weights or []):
            self.weights[name] = weight


@pytest.fixture
def lora_runtime(temp_studio_env, monkeypatch, tmp_path):
    import types as _types

    from studio.catalog import catalog_manager
    from studio.engines import video_engine

    def prepare(*lora_ids):
        for lid in lora_ids:
            defn = catalog_manager.get_definition(lid)
            art = defn["artifacts"][0]
            target = catalog_manager.model_dir(lid) / art["subdir"] / art["filename"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"fake")
            catalog_manager.marker_file(lid).write_text("{}", encoding="utf-8")

    fake_st = _types.ModuleType("safetensors")
    fake_st_torch = _types.ModuleType("safetensors.torch")
    fake_st_torch.load_file = lambda _p, device="cpu": {}
    fake_st.torch = fake_st_torch
    monkeypatch.setitem(sys.modules, "safetensors", fake_st)
    monkeypatch.setitem(sys.modules, "safetensors.torch", fake_st_torch)
    monkeypatch.setattr(video_engine, "_apply_best_offload", lambda *a, **k: None)
    monkeypatch.setattr(video_engine, "_maybe_set_wan_flow_shift", lambda *a, **k: None)
    return video_engine, catalog_manager, prepare


def test_failed_adapter_load_leaves_no_stale_adapter_name(lora_runtime, colab_memory):
    colab_memory()
    video_engine, catalog_manager, prepare = lora_runtime
    prepare("lora-wan21-14b-fusionx-phantom", "lora-wan21-14b-unianimate-char")
    pipe = _PeftPipe(fail_on={"wan21_unianimate_char"})
    handle = {"engine": "wan_vace_fp8", "defn": catalog_manager.get_definition("wan21-vace-14b-fp8")}
    with pytest.raises(ValueError):
        video_engine._sync_video_lora_on_handle(
            handle, pipe, "lora-wan21-14b-fusionx-phantom", 0.9,
            extra_video_lora_ids=["lora-wan21-14b-unianimate-char"],
        )
    # Nothing stays attached, so the next attempt cannot hit "already in use".
    assert pipe.adapters == {}
    assert handle["active_lora_id"] is None and handle["active_lora_signature"] is None
    partial = handle["lora_load_partial"]
    assert partial["failed_lora_id"] == "lora-wan21-14b-unianimate-char"
    assert partial["loaded_before_failure"] == ["vid_speed_lora"]

    # The retry (with the problem gone) succeeds on a clean model and clears the flag.
    pipe.fail_on = set()
    video_engine._sync_video_lora_on_handle(
        handle, pipe, "lora-wan21-14b-fusionx-phantom", 0.9,
        extra_video_lora_ids=["lora-wan21-14b-unianimate-char"],
    )
    assert "lora_load_partial" not in handle
    assert handle["active_lora_signature"] == (
        "lora-wan21-14b-fusionx-phantom",
        "lora-wan21-14b-unianimate-char",
    )
    assert set(pipe.adapters) == {"vid_speed_lora", "wan21_unianimate_char"}


def test_unknown_adapter_name_from_previous_attempt_is_recovered(lora_runtime, colab_memory):
    colab_memory()
    video_engine, catalog_manager, prepare = lora_runtime
    prepare("lora-wan21-14b-fusionx-phantom")
    pipe = _PeftPipe()
    # Simulates the state left by the old code: adapter attached, handle unaware.
    pipe.adapters["vid_speed_lora"] = True
    handle = {"engine": "wan_vace_fp8", "defn": catalog_manager.get_definition("wan21-vace-14b-fp8")}
    video_engine._sync_video_lora_on_handle(handle, pipe, "lora-wan21-14b-fusionx-phantom", 0.9)
    assert handle["active_lora_id"] == "lora-wan21-14b-fusionx-phantom"
    assert set(pipe.adapters) == {"vid_speed_lora"}


def test_four_bit_inspector_reports_stale_patches_instead_of_asserting(monkeypatch):
    from studio.engines import coherence_engine

    class _Params4bit:
        def __new__(cls, *a, **k):
            return super().__new__(cls)

    class _Linear4bit:
        pass

    bnb = types.SimpleNamespace(
        nn=types.SimpleNamespace(Params4bit=_Params4bit, Linear4bit=_Linear4bit)
    )
    monkeypatch.setitem(sys.modules, "bitsandbytes", bnb)
    with pytest.raises(RuntimeError, match="RUNTIME_PATCHES_STALE"):
        coherence_engine._check_four_bit_modules_patchable(object())

    # Once the guard is installed the check passes.
    def _guarded(*a, **k):
        return None

    _guarded._studio_safe_wrapped = True
    bnb.nn.Params4bit.__new__ = staticmethod(_guarded)
    coherence_engine._check_four_bit_modules_patchable(object())


def test_runtime_rebuild_is_reported_without_touching_weights(monkeypatch):
    from studio import runtime_flags
    from studio.compat_patches import ensure_patchable_runtime

    monkeypatch.setitem(runtime_flags._STATE, "rebuilt_modules", ())
    monkeypatch.setitem(runtime_flags._STATE, "rebuild_notice", "")
    report = ensure_patchable_runtime(modules=("json",))
    assert report["ok"] is True
    assert "reiniciarse el proceso del servidor" in report["notice"]
    assert runtime_flags.snapshot()["runtime_rebuilt_for_compatibility"] is True
    # A second call in the same process must not reload again; it reports the blocker.
    again = ensure_patchable_runtime(modules=("json",))
    assert again["ok"] is False and again["reimported"] is False


def test_preferences_are_stored_once_and_reused_by_the_plan(temp_studio_env, monkeypatch):
    from studio import mcp_preferences
    from studio.agent_presets import compile_agent_workflow_preset
    from studio.mcp_server import _execute_mcp_tool

    monkeypatch.setattr(mcp_preferences, "PREF_FILE", mcp_preferences.DATA_ROOT / "prefs_test.json")
    saved = _execute_mcp_tool("studio_answer_preference", {"key": "use_fast_loras", "value": True})
    payload = json.loads(saved["content"][0]["text"])
    assert payload["saved"] == {"use_fast_loras": True}
    assert all(q["key"] != "use_fast_loras" for q in payload["pending_questions"])

    ask = json.loads(_execute_mcp_tool("studio_ask_user", {"preference_key": "use_fast_loras"})["content"][0]["text"])
    assert ask["action"] == "ASK_USER_ONCE" and ask["recommended"] is True and ask["blocking"] is False

    plan = compile_agent_workflow_preset(
        scene_description="dos personas andando por un parque hablando, 30 segundos",
        gpu_override={"gpu_name": "NVIDIA L4", "vram_total_gb": 22.5, "has_gpu": True},
    )
    stack = plan["detected_gpu"]["selected_video_stack"]
    assert stack["fast_loras"] is True
    assert stack["video_lora_id"] == "lora-wan21-14b-fusionx-phantom"
    assert plan["production_preferences"]["applied"]["use_fast_loras"] is True

    # Quality preference: same GPU, no speed LoRA and no guessing.
    _execute_mcp_tool("studio_answer_preference", {"key": "use_fast_loras", "value": False})
    plan_q = compile_agent_workflow_preset(
        scene_description="dos personas andando por un parque hablando",
        gpu_override={"gpu_name": "NVIDIA L4", "vram_total_gb": 22.5, "has_gpu": True},
    )
    stack_q = plan_q["detected_gpu"]["selected_video_stack"]
    assert stack_q["video_lora_id"] == "" and stack_q["fast_loras"] is False
    assert stack_q["extra_video_lora_ids"], "coherence adapters stay enabled"


def test_fast_request_without_speed_lora_is_rejected_with_guidance(temp_studio_env, colab_memory):
    import io

    from PIL import Image

    from studio.mcp_server import _execute_mcp_tool

    colab_memory(ram=64, available=52, gpu=True, vram=22, vram_free=21)
    client, _ = temp_studio_env
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (30, 60, 90)).save(buf, format="PNG")
    rels = [
        client.post("/api/assets/upload", files={"file": (name, buf.getvalue(), "image/png")}).json()["asset"]["rel_path"]
        for name in ("a.png", "b.png")
    ]
    result = _execute_mcp_tool(
        "studio_generate_video",
        {
            "video_submode": "flf2v",
            "mode": "ai",
            "model_id": "wan21-vace-14b-fp8",
            "video_lora_id": "",
            "extra_video_lora_ids": ["lora-wan21-14b-unianimate-char"],
            "image_rel": rels[0],
            "end_image_rel": rels[1],
            "prompt": "two people walking",
            "duration_sec": 5,
        },
    )
    payload = json.loads(result["content"][0]["text"])
    assert payload["ok"] is False and payload["error_code"] == "SPEED_LORA_MISSING"
    assert "lora-wan21-14b-fusionx-phantom" in payload["exact_fix_action"]
    assert payload["detected_extra_loras"] == ["lora-wan21-14b-unianimate-char"]


def test_work_state_exposes_partial_adapter_cleanup(temp_studio_env):
    from studio.memory_manager import memory_manager
    from studio.mcp_server import _execute_mcp_tool

    handle = {"lora_load_partial": {"failed_lora_id": "lora-x", "cleanup_errors": []}}
    memory_manager.register_loaded("video", "wan21-vace-14b-fp8", "Wan", handle)
    try:
        state = json.loads(_execute_mcp_tool("studio_get_work_state", {})["content"][0]["text"])
    finally:
        memory_manager._slots.pop("video", None)
    assert state["video_model"]["adapters_partial_failure"]["failed_lora_id"] == "lora-x"
    assert "no se reintenta a ciegas" in state["retry_policy"]


def test_overlong_clip_is_refused_with_a_split_plan(temp_studio_env, colab_memory):
    """Ask for 30 s in one call: never return a silently shorter video."""
    import io

    from PIL import Image

    from studio.mcp_server import _execute_mcp_tool

    colab_memory(ram=64, available=52, gpu=True, vram=22, vram_free=21)
    client, _ = temp_studio_env
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (30, 60, 90)).save(buf, format="PNG")
    rels = [
        client.post("/api/assets/upload", files={"file": (name, buf.getvalue(), "image/png")}).json()["asset"]["rel_path"]
        for name in ("a.png", "b.png")
    ]
    result = _execute_mcp_tool(
        "studio_generate_video",
        {
            "video_submode": "flf2v",
            "mode": "ai",
            "model_id": "wan21-vace-14b-fp8",
            "video_lora_id": "lora-wan21-14b-fusionx-phantom",
            "image_rel": rels[0],
            "end_image_rel": rels[1],
            "prompt": "two people walking",
            "duration_sec": 30,
        },
    )
    payload = json.loads(result["content"][0]["text"])
    assert payload["ok"] is False and payload["error_code"] == "DURATION_EXCEEDS_SINGLE_CLIP"
    assert payload["max_seconds_per_clip"] == pytest.approx(10.0, abs=0.1)
    assert payload["recommended_clips"] >= 2 and payload["recommended_clip_duration_sec"] <= 8
    assert "studio_render_montage" in payload["exact_fix_action"]


def test_park_scene_keeps_the_park_and_drops_duration_meta_instructions():
    from studio.agent_presets import compile_agent_workflow_preset

    plan = compile_agent_workflow_preset(
        "coherencia_personajes_entorno_avance",
        scene_description=(
            "dos personas andando por un parque hablando de algo polémico y haciendo una parodia, "
            "el vídeo debe durar 30 segundos"
        ),
        gpu_override={"has_gpu": True, "gpu_name": "L4", "vram_total_gb": 24, "vram_free_gb": 22},
    )
    summary = plan["planned_duration_summary"]
    assert summary["num_clips"] == 6 and summary["num_keyframes"] == 7
    assert summary["estimated_total_seconds"] == pytest.approx(30.0)
    anchor = plan["execution_plan"][0]["arguments"]["prompt"]
    video = next(s for s in plan["execution_plan"] if s["tool"] == "studio_generate_video")["arguments"]["prompt"]
    for text in (anchor, video):
        assert "parque" in text.lower() or "park" in text.lower()
        assert "debe durar" not in text and "30 segundos" not in text
        assert "sidewalk" not in text.lower() and "storefront" not in text.lower()


def test_build_identity_is_published_without_touching_the_checkout(temp_studio_env):
    """The agent must be able to tell a stale deploy from a fresh one."""
    from studio import runtime_flags

    build = runtime_flags.build_identity()
    assert build["source"] in ("git", "copy")
    assert build["root"] and "mtime" in build or build["commit"]
    if build["source"] == "git":
        assert len(build["commit"]) == 40 and build["commit_short"] == build["commit"][:7]
        assert isinstance(build["dirty"], bool)
    snap = runtime_flags.snapshot()
    assert snap["build"]["source"] == build["source"]  # same cached identity


def test_audit_script_accepts_the_link_the_user_pastes():
    """`--url https://host/mcp?token=...` must work and never print the token."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "tools" / "mcp_audit.py"
    spec = importlib.util.spec_from_file_location("mcp_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    url, token = module._split_link("https://abc.trycloudflare.com/mcp?token=st_mcp_deadbeef", "")
    assert url == "https://abc.trycloudflare.com" and token == "st_mcp_deadbeef"
    url2, token2 = module._split_link("https://abc.trycloudflare.com", "st_mcp_x")
    assert url2 == "https://abc.trycloudflare.com" and token2 == "st_mcp_x"
    assert "st_mcp" not in module._redact("token=st_mcp_secret and st_mcp_secret")


def test_public_mcp_info_never_hands_the_token_to_a_stranger():
    """The tunnel URL is public: /api/mcp/info must not leak the token to anyone."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from studio import mcp_server

    token = "st_mcp_auditsecret0123456789"
    previous = __import__("os").environ.get("STUDIO_MCP_TOKEN")
    __import__("os").environ["STUDIO_MCP_TOKEN"] = token
    try:
        probe = FastAPI()
        probe.include_router(mcp_server.mcp_router)
        client = TestClient(probe)
        # 1) tercero anónimo, sin cabeceras de UI
        stranger = client.get("/api/mcp/info", headers={"host": "abc.trycloudflare.com"}).json()
        assert stranger["ok"] is True
        assert stranger["token"] == "" and stranger["token_exposed"] is False
        assert token not in json.dumps(stranger)
        assert "/mcp?token=" not in json.dumps(stranger)
        assert all("token" not in str(v) for v in stranger["commands"].values())
        # 2) UI del mismo host: Origin coincide y manda X-Studio-UI
        trusted = client.get(
            "/api/mcp/info",
            headers={
                "x-forwarded-host": "abc.trycloudflare.com",
                "x-forwarded-proto": "https",
                "origin": "https://abc.trycloudflare.com",
                "x-studio-ui": "1",
            },
        ).json()
        assert trusted["token_exposed"] is True and trusted["token"] == token
        # 3) mismo Origin pero sin la cabecera de la UI: no es la interfaz
        forced = client.get(
            "/api/mcp/info",
            headers={
                "x-forwarded-host": "abc.trycloudflare.com",
                "x-forwarded-proto": "https",
                "origin": "https://abc.trycloudflare.com",
            },
        ).json()
        assert forced["token"] == "" and forced["token_exposed"] is False
        # 4) Origin de otro sitio (CSRF desde una página ajena)
        foreign = client.get(
            "/api/mcp/info",
            headers={
                "x-forwarded-host": "abc.trycloudflare.com",
                "x-forwarded-proto": "https",
                "origin": "https://evil.example",
                "x-studio-ui": "1",
            },
        ).json()
        assert foreign["token"] == "" and foreign["token_exposed"] is False
        # 5) un cliente que ya presenta el token sí lo recupera
        bearer = client.get(
            "/api/mcp/info",
            headers={"host": "abc.trycloudflare.com", "authorization": f"Bearer {token}"},
        ).json()
        assert bearer["token_exposed"] is True
    finally:
        if previous is None:
            __import__("os").environ.pop("STUDIO_MCP_TOKEN", None)
        else:
            __import__("os").environ["STUDIO_MCP_TOKEN"] = previous


def test_explicit_clip_count_beats_a_duration_mentioned_in_the_text():
    """Pedir 2 clips y decir '10 segundos' debe dar 2 clips, no lo que diga el texto."""
    from studio.agent_presets import _infer_clip_plan_from_text as infer

    assert infer("dos personas andando, el vídeo debe durar 10 segundos", 2) == (2, 5.0)
    assert infer("dos vecinos, el vídeo debe durar 30 segundos", None)[0] == 6
    assert infer("un paseo de un minuto", None)[0] == 12
    assert infer("lo que sea, 30 segundos", 6)[0] == 6
    assert infer("dos personas andando", None)[0] == 2
    assert infer("x", 99)[0] == 12  # acotado al máximo del preset


def test_small_video_model_stays_on_gpu_instead_of_cpu_offload(monkeypatch):
    """Un 1.3B (~2.7 GB) en una L4 de 22 GB no debe pagar offload por paso.

    El umbral usaba `min_vram_gb` de la RECETA (8 GB) como si fuera el tamaño del
    modelo: eso activaba CPU offload con 21.8 GB libres y cada paso tardaba ~20 s.
    """
    import types

    from studio.engines import video_engine

    class _P:
        def __init__(self, n):
            self.n = n

        def numel(self):
            return self.n

        def element_size(self):
            return 2

    calls = []

    def _fake_module(n_bytes):
        mod = types.SimpleNamespace()
        mod.parameters = lambda: [_P(n_bytes // 2)]
        return mod

    pipe = types.SimpleNamespace(
        transformer=_fake_module(int(1.9 * 1024**3)),
        vae=_fake_module(int(0.25 * 1024**3)),
        text_encoder=_fake_module(int(0.5 * 1024**3)),
        to=lambda dev: calls.append(("to", dev)),
        enable_model_cpu_offload=lambda: calls.append(("model_offload",)),
        enable_sequential_cpu_offload=lambda: calls.append(("sequential_offload",)),
        remove_all_hooks=lambda: calls.append(("remove_hooks",)),
        vae2=None,
    )
    video_engine._apply_best_offload(pipe, vram_total_gb=22.03, min_vram_gb=8.0, job=None)
    assert ("to", "cuda") in calls
    assert not any(c[0].endswith("offload") for c in calls), calls

    # Un 14B FP8 (~15.8 GB de pesos) sí necesita offload en 22 GB.
    calls.clear()
    big = types.SimpleNamespace(
        transformer=_fake_module(int(14.0 * 1024**3)),
        vae=_fake_module(int(0.6 * 1024**3)),
        to=lambda dev: calls.append(("to", dev)),
        enable_model_cpu_offload=lambda: calls.append(("model_offload",)),
        enable_sequential_cpu_offload=lambda: calls.append(("sequential_offload",)),
        remove_all_hooks=lambda: calls.append(("remove_hooks",)),
    )
    video_engine._apply_best_offload(big, vram_total_gb=22.03, min_vram_gb=15.8, job=None)
    assert any(c[0].endswith("offload") for c in calls), calls


def test_invalid_lora_checkpoint_is_retried_without_orphan_alpha():
    """El ValueError de Diffusers se arregla quitando el `.alpha` suelto, no el adaptador."""
    from studio.engines.video_engine import (
        _drop_pure_alpha_keys,
        _is_invalid_lora_checkpoint,
    )
    import numpy as np

    exc = ValueError("Invalid LoRA checkpoint. Make sure all LoRA param names contain `'lora'` substring.")
    assert _is_invalid_lora_checkpoint(exc)
    assert not _is_invalid_lora_checkpoint(ValueError("otra cosa"))

    sd = {
        "diffusion_model.blocks.0.self_attn.q.lora_down.weight": np.ones((4, 4)),
        "diffusion_model.blocks.0.self_attn.q.lora_up.weight": np.ones((4, 4)),
        "diffusion_model.blocks.0.self_attn.q.alpha": np.float32(4.0),
        "diffusion_model.bogus.alpha": np.float32(1.0),
    }
    clean = _drop_pure_alpha_keys(sd)
    assert "diffusion_model.blocks.0.self_attn.q.lora_down.weight" in clean
    assert not any(k.endswith(".alpha") for k in clean)


def test_credits_are_labelled_as_an_estimate_and_autoshutdown_fires():
    """Colab no expone el saldo: la app debe decirlo y apagar una GPU ociosa."""
    import types

    from studio import hardware

    calls = []

    class _Tracker:
        def __init__(self, due, minutes=90, jobs=0):
            self.due, self.minutes, self.jobs = due, minutes, jobs

        def get_snapshot(self, active_jobs_count=0):
            return {
                "is_colab": True,
                "credits": {
                    "is_estimate": True,
                    "autoshutdown_due": self.due and active_jobs_count == 0,
                    "idle_minutes": self.minutes,
                    "idle_autoshutdown_minutes": 45,
                },
            }

    original = hardware.request_colab_shutdown
    hardware.request_colab_shutdown = lambda: calls.append("shutdown")
    try:
        assert hardware.enforce_idle_shutdown(_Tracker(due=True), 0) is True
        assert calls == ["shutdown"]
        # Con trabajo en curso NO se apaga nada, aunque lleve horas.
        assert hardware.enforce_idle_shutdown(_Tracker(due=True), 1) is False
        assert calls == ["shutdown"]
        # Sin vencimiento, tampoco.
        assert hardware.enforce_idle_shutdown(_Tracker(due=False), 0) is False
    finally:
        hardware.request_colab_shutdown = original

    snap = hardware.hardware_tracker.get_snapshot(active_jobs_count=0)
    credits = snap["credits"]
    assert credits["is_estimate"] is True
    assert credits["source"] == "estimacion_local_tarifa_fija"
    assert "no publica tu saldo" in credits["notice"]
    assert "idle_autoshutdown_minutes" in credits


def test_github_channel_only_runs_closed_verbs_and_never_forwards_credentials():
    """El canal ejecuta verbos permitidos; el resto se rechaza y no toca nada."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "tools" / "mcp_github_channel.py"
    spec = importlib.util.spec_from_file_location("mcp_github_channel", path)
    ch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ch)

    # La lista es cerrada y NO incluye nada destructivo ni descargas.
    assert "generate_video" in ch.ALLOWED_VERBS
    for forbidden in ("prepare_model", "free_memory_or_disk", "delete_model", "shell", "download"):
        assert forbidden not in ch.ALLOWED_VERBS
    assert set(ch.ALLOWED_VERBS) == set(ch.VERB_TO_TOOL)

    # Un argumento con credencial se rechaza ANTES de hablar con el servidor.
    sent = []
    ch._rpc = lambda base, token, payload, session: sent.append(payload) or {"result": {"content": []}}

    session = {"ready": "1", "seq": "1", "id": "s"}
    try:
        ch._dispatch("get_state", {"include_recent_assets": True, "token": "st_mcp_x"},
                     "http://127.0.0.1:7860", "t", session)
        raise AssertionError("debía rechazar el token")
    except ValueError:
        pass
    assert sent == []

    ch._dispatch("get_state", {"include_recent_assets": False}, "http://127.0.0.1:7860", "t", session)
    tool_call = sent[-1]["params"]
    assert tool_call["name"] == "studio_get_state_and_manual"
    assert "token" not in tool_call["arguments"]


def test_github_channel_talks_to_the_local_server_without_duplicating_queues():
    """El canal debe ir por HTTP al servidor local, no importar su cola de trabajos."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "tools" / "mcp_github_channel.py"
    source = path.read_text(encoding="utf-8")
    # Nunca importa la cola del servidor: eso crearía trabajos invisibles para la UI.
    assert "from studio.mcp_server import _execute_mcp_tool" not in source
    assert "from studio.mcp_server import _wait_for_job_bounded" not in source
    assert "127.0.0.1:7860" in source

    spec = importlib.util.spec_from_file_location("mcp_github_channel_http", path)
    ch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ch)
    assert callable(ch._rpc)
    assert "wait_for_job" in ch.ALLOWED_VERBS


def test_github_channel_starts_from_the_app_and_publishes_results(monkeypatch, tmp_path):
    """Un clic en el panel debe encender el canal, ejecutar y publicar; sin comandos."""
    import studio.github_channel_service as svc

    store = {
        "ordenes.json": {"orders": [
            {"id": "x1", "verb": "get_state", "args": {"include_recent_assets": False}, "wait": False},
            {"id": "x2", "verb": "shell", "args": {"cmd": "ls"}},
        ]},
        "resultados.json": {"executed_ids": [], "results": []},
    }

    class FakeGist:
        def __init__(self, gist_id, token):
            self.gist_id = gist_id

        @staticmethod
        def create(token, description=""):
            return "gist_test"

        def read(self, name):
            return store.get(name, {})

        def write(self, name, body):
            store[name] = body

    ch = svc._channel_module()
    ch.GistMailbox = FakeGist
    monkeypatch.setattr(svc, "_channel_module", lambda: ch)
    monkeypatch.setattr(svc, "load_saved_token", lambda: "tok")
    monkeypatch.setattr(svc, "save_token", lambda token: None)
    monkeypatch.setattr(svc, "_token_path", lambda: tmp_path / "ci.json")

    out = svc.start(gh_token="tok", base="http://127.0.0.1:9", mcp_token="t", interval=5)
    # El sondeo corre en un hilo; esperamos a que procese las dos órdenes.
    for _ in range(60):
        if len(store["resultados.json"].get("results") or []) >= 2:
            break
        import time

        time.sleep(0.1)
    svc.stop()

    assert out["backend"] == "gist"
    assert out["gist_id"] == "gist_test"
    ids = [r["id"] for r in store["resultados.json"]["results"]]
    assert "x1" in ids and "x2" in ids
    bad = [r for r in store["resultados.json"]["results"] if r["id"] == "x2"][0]
    assert bad["error"] == "VERB_NOT_ALLOWED"
    state = svc.snapshot()
    assert state["executed"] == 2, "el contador no debe inflarse al republicar"
    # Al agente se le da el buzón, nunca el token.
    handoff = svc.agent_handoff("gist_test", "r", "b", "gist")
    assert "tok" not in str(handoff)
    assert handoff["gist_id"] == "gist_test"
