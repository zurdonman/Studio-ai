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
    import time

    import studio.github_channel_service as svc

    store = {
        "agent": {"orders": [
            {"id": "x1", "verb": "get_state", "args": {"include_recent_assets": False}, "wait": False},
            {"id": "x2", "verb": "shell", "args": {"cmd": "ls"}},
        ]},
        "app": {"executed_ids": [], "results": []},
    }

    ch = svc._channel_module()
    real_mailbox = ch._Mailbox

    class FakeMailbox:
        def __init__(self, *a, **k):
            self._inner = real_mailbox(*a, **k)
            self.token = k.get("token", a[5] if len(a) > 5 else "")

        def read_orders(self):
            return {"orders": store["agent"]["orders"]}

        def read_results(self):
            return store["app"]

        def write_results(self, ids, results):
            store["app"] = {"executed_ids": ids, "results": results}

    monkeypatch.setattr(ch, "_Mailbox", FakeMailbox)
    monkeypatch.setattr(svc, "_channel_module", lambda: ch)
    monkeypatch.setattr(svc, "load_saved_token", lambda: "tok")
    monkeypatch.setattr(svc, "save_token", lambda token: None)
    monkeypatch.setattr(svc, "_token_path", lambda: tmp_path / "ci.json")

    out = svc.start(gh_token="tok", base="http://127.0.0.1:9", mcp_token="t", interval=5)
    for _ in range(60):
        if len(store["app"].get("results") or []) >= 2:
            break
        time.sleep(0.1)
    state = svc.snapshot()
    svc.stop()

    assert out["backend"] == "state"
    ids = [r["id"] for r in store["app"]["results"]]
    assert ids[:2] == ["x1", "x2"]
    bad = [r for r in store["app"]["results"] if r["id"] == "x2"][0]
    assert bad["error"] == "VERB_NOT_ALLOWED"
    assert state["executed"] == 2, "el contador no debe inflarse al republicar"
    handoff = svc.agent_handoff("", "zurdonman/Studio-ai", "main", "state")
    assert handoff["write_needs"].startswith("un token")
    assert "ghp_" not in str(handoff)
    assert handoff["anonymous_read"] is True
    assert "studio-channel" in handoff["read_url"]


def test_github_channel_reads_orders_without_any_token():
    """La app debe poder leer órdenes sin credenciales: el buzón es público."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "tools" / "mcp_github_channel.py"
    spec = importlib.util.spec_from_file_location("mcp_github_channel_state", path)
    ch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ch)

    box = ch._Mailbox("state", "zurdonman/Studio-ai", "studio-channel", "", "", token="",
                      state_branch="studio-channel", state_path="channel/agent_state.json")
    assert box.state is not None
    assert box.token == ""

    # Lectura anónima: la petición no lleva cabecera de autorización.
    seen = {}

    def fake_api(url, token, method="GET", payload=None):
        seen["url"], seen["token"], seen["method"] = url, token, method
        import base64

        body = base64.b64encode(json.dumps(
            {"agent": {"orders": [{"id": "z1", "verb": "get_state"}]}, "app": {"results": []}}
        ).encode()).decode()
        return {"content": body, "sha": "abc"}

    ch._gh_api = fake_api
    orders = box.read_orders()
    assert [o["id"] for o in orders["orders"]] == ["z1"]
    assert seen["token"] == ""
    assert "studio-channel" in seen["url"]

    # Publicar resultados sin token debe fallar explícitamente, no en silencio.
    try:
        box.write_results(["z1"], [{"id": "z1", "ok": True}])
        raise AssertionError("sin token no se debe poder publicar")
    except RuntimeError:
        pass


def test_github_channel_does_not_commit_to_git_unless_something_changed():
    """Publicar resultados no debe crear un comité por sondeo (ensuciaría el repo)."""
    import importlib.util
    import time
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "tools" / "mcp_github_channel.py"
    spec = importlib.util.spec_from_file_location("mcp_gh_beat", path)
    ch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ch)

    writes = []
    box = ch.StateMailbox("zurdonman/Studio-ai", "studio-channel", "channel/agent_state.json", "tok")
    box.read = lambda: {"schema": 1, "agent": {"orders": []},
                        "app": {"executed_ids": ["a"], "results": [{"id": "a"}],
                                "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}}
    # El GET previo (para el sha) no cuenta: solo interesa el PUT que crea comité.
    ch._gh_api = lambda url, token, method="GET", payload=None: (
        writes.append(payload) if method == "PUT" else {"sha": "x"})

    # Mismo contenido y latido reciente → no se escribe nada.
    box.write_app(["a"], [{"id": "a"}])
    assert writes == []

    # Contenido nuevo → se escribe una vez.
    box.write_app(["a", "b"], [{"id": "a"}, {"id": "b"}])
    assert len(writes) == 1


def test_github_channel_can_autostart_without_user_actions(monkeypatch):
    """Con credenciales ya guardadas, cada sesión de Colab queda conectada sola."""
    import studio.github_channel_service as svc

    monkeypatch.setattr(svc, "load_saved_token", lambda: "tok")
    started = {}

    def fake_start(**kwargs):
        started.update(kwargs)
        return {"ok": True, "mode": "activo"}

    monkeypatch.setattr(svc, "start", fake_start)
    monkeypatch.setenv("STUDIO_CHANNEL_AUTOSTART", "1")
    out = svc.autostart_if_possible(mcp_token="t")
    assert out["ok"] is True

    # Sin credenciales no arranca nada (ni falla).
    monkeypatch.setattr(svc, "load_saved_token", lambda: "")
    started.clear()
    out = svc.autostart_if_possible(mcp_token="t")
    assert out["ok"] is False and "skipped" in out
    assert started == {}

    # Por defecto está apagado: es un canal de diagnóstico, no parte del producto.
    monkeypatch.delenv("STUDIO_CHANNEL_AUTOSTART", raising=False)
    monkeypatch.setattr(svc, "load_saved_token", lambda: "tok")
    assert svc.autostart_if_possible(mcp_token="t")["ok"] is False

    # Y se puede desactivar explícitamente.
    monkeypatch.setenv("STUDIO_CHANNEL_AUTOSTART", "0")
    monkeypatch.setattr(svc, "load_saved_token", lambda: "tok")
    out = svc.autostart_if_possible(mcp_token="t")
    assert out["ok"] is False and "skipped" in out


def test_colab_secret_is_used_when_no_env_token(monkeypatch):
    """El secreto de Colab (🔑) debe ser suficiente: una configuración por cuenta."""
    import sys
    import types

    import studio.github_channel_service as svc

    monkeypatch.delenv("STUDIO_CHANNEL_GH_TOKEN", raising=False)
    fake_userdata = types.SimpleNamespace(get=lambda name: "ghp_desde_secreto" if name == "GH_TOKEN" else None)
    monkeypatch.setitem(sys.modules, "google.colab", types.SimpleNamespace(userdata=fake_userdata))
    monkeypatch.setitem(sys.modules, "google.colab.userdata", fake_userdata)
    monkeypatch.setattr(svc, "_token_path", lambda: __import__("pathlib").Path("/nonexistent/x.json"))

    assert svc.load_saved_token() == "ghp_desde_secreto"


def test_channel_status_endpoint_never_waits_for_colab_secrets(monkeypatch):
    """El panel no puede quedarse en «comprobando» porque Colab pida autorización."""
    import sys
    import time
    import types

    import studio.github_channel_service as svc

    def blocking_get(name):
        time.sleep(30)  # simula la espera de autorización de Colab
        return "ghp_nunca"

    fake = types.SimpleNamespace(get=blocking_get)
    monkeypatch.setitem(sys.modules, "google.colab", types.SimpleNamespace(userdata=fake))
    monkeypatch.setitem(sys.modules, "google.colab.userdata", fake)
    monkeypatch.delenv("STUDIO_CHANNEL_GH_TOKEN", raising=False)
    monkeypatch.setattr(svc, "_token_path", lambda: __import__("pathlib").Path("/nonexistent/x.json"))
    monkeypatch.setattr(svc, "_secret_cache", {"checked": False, "value": ""})

    started = time.time()
    assert svc.has_token() is False
    assert svc.token_source() == "ninguna"
    assert time.time() - started < 1.0, "la comprobación de estado debe ser instantánea"


def test_github_channel_ui_lives_inside_the_app_closure():
    """Regresión: las funciones del canal deben estar dentro del IIFE donde vive api().

    Si se declaran fuera, al pulsar el botón salta «api is not defined», el error se
    lo traga el try/catch del modal y el panel se queda en «Comprobando…» para
    siempre. Pasó exactamente eso.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    js = (root / "studio" / "static" / "app.js").read_text(encoding="utf-8")
    end = js.rindex("})();")
    for fn in ("refreshGitHubChannel", "startGitHubChannel", "stopGitHubChannel", "channelFetch"):
        pos = js.index(f"function {fn}")
        assert pos < end, f"{fn} está fuera del cierre y rompería el panel"

    html = (root / "studio" / "static" / "index.html").read_text(encoding="utf-8")
    # Versión en el script: sin esto el navegador reutiliza el JS viejo y el usuario
    # sigue viendo el fallo después de arreglarlo.
    assert 'app.js?v=' in html


def test_channel_actions_are_trusted_behind_a_tunnel_proxy():
    """Detrás del túnel, Host=127.0.0.1 pero Origin=https://<túnel>: debe aceptarse.

    Sin esto, el panel recibe 403 en /probe y /start y el botón parece no hacer nada.
    """
    from starlette.requests import Request

    import studio.mcp_server as srv

    def make(headers):
        raw = [(k.lower().encode(), v.encode()) for k, v in headers.items()]
        return Request({"type": "http", "method": "POST", "path": "/", "headers": raw,
                        "query_string": b"", "scheme": "http", "server": ("127.0.0.1", 7860)})

    tunnel = "https://cambridge-comics-constraint-gather.trycloudflare.com"
    # Caso real: el proxy no reenvía el host, pero el navegador manda Origin + X-Studio-UI.
    assert srv._caller_is_trusted(make({"host": "127.0.0.1:7860", "origin": tunnel,
                                        "x-studio-ui": "1"})) is True
    # Sin la cabecera del panel, cualquiera podría: se sigue rechazando.
    assert srv._caller_is_trusted(make({"host": "127.0.0.1:7860", "origin": tunnel})) is False
    # Y con x-forwarded-host también.
    assert srv._caller_is_trusted(make({"host": "127.0.0.1:7860", "x-forwarded-host":
                                        "cambridge-comics-constraint-gather.trycloudflare.com",
                                        "x-forwarded-proto": "https", "origin": tunnel,
                                        "x-studio-ui": "1"})) is True


def test_channel_result_keeps_the_diagnostics_that_matter():
    """Regresión: get_state devolvía sólo {"ok": true} y dejaba al agente ciego."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "tools" / "mcp_github_channel.py"
    spec = importlib.util.spec_from_file_location("mcp_gh_compact", path)
    ch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ch)

    fake_state = {
        "ok": True,
        "hardware": {
            "gpu": {"gpu_name": "Tesla T4", "has_gpu": True, "vram_free_gb": 14.2,
                    "measurement_source": "nvidia_smi_driver"},
            "ram": {"total_gb": 12.7, "available_gb": 9.1},
            "disk": {"free_gb": 60.5},
            "credits": {"hourly_rate": 1.2, "is_estimate": True},
        },
        "build": {"commit": "c6d7271", "branch": "arena/x", "dirty": False},
        "loaded_in_vram": [{"name": "qwen-image"}],
        "active_jobs": [{"id": "j1", "status": "running", "progress_pct": 40}],
        "manual_index": [{"uri": "x", "name": "y"}] * 50,
        "golden_rules": ["regla"] * 20,
    }
    out = ch._compact(fake_state, "get_state")
    state = out["state"]
    assert state["gpu"]["gpu_name"] == "Tesla T4"
    assert state["gpu"]["vram_free_gb"] == 14.2
    assert state["build"]["commit"] == "c6d7271"
    assert state["loaded_in_vram"] == ["qwen-image"]
    assert state["active_jobs"][0]["status"] == "running"
    assert "manual_index" in state["payload_keys"]  # se sabe que existe, sin volcarlo


def test_channel_result_is_capped_so_the_buzon_stays_small():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "tools" / "mcp_github_channel.py"
    spec = importlib.util.spec_from_file_location("mcp_gh_cap", path)
    ch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ch)

    huge = {"ok": True, "phases": ["x" * 500] * 200}
    out = ch._compact(huge, "get_workflow_preset")
    assert len(json.dumps(out, ensure_ascii=False)) < ch.MAX_PUBLISHED_CHARS
    assert out.get("truncated") is True


def test_compat_patches_never_block_boot(tmp_path, monkeypatch):
    """Un parche opcional que falle no puede impedir que arranque el servidor."""
    import socket
    import sys

    import studio.compat_patches as cp
    import studio.memory_safety_hooks as hooks

    def boom():
        socket.create_connection(("127.0.0.1", 1), timeout=0.2)

    # peft no se importa durante el arranque: era lo que abría conexiones de red.
    monkeypatch.delitem(sys.modules, "peft", raising=False)
    cp._patch_peft_integrations()
    assert "peft" not in sys.modules

    monkeypatch.setattr(cp, "_patch_peft_integrations", boom)
    monkeypatch.setattr(hooks, "install_memory_safety_hooks", boom)
    cp.apply_runtime_compatibility_patches()  # no debe lanzar


def test_i2v_fast_preference_rejects_omitted_accelerator(temp_studio_env, colab_memory, monkeypatch):
    import io
    from PIL import Image
    from studio.mcp_server import _execute_mcp_tool
    import studio.mcp_preferences as prefs
    colab_memory(ram=64, available=52, gpu=True, vram=22, vram_free=21)
    client, _ = temp_studio_env
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (30, 60, 90)).save(buf, format="PNG")
    rel = client.post("/api/assets/upload", files={"file": ("a.png", buf.getvalue(), "image/png")}).json()["asset"]["rel_path"]
    monkeypatch.setattr(prefs, "PREF_FILE", prefs.PREF_FILE.parent / "fast-only-test.json")
    prefs.set_preferences({"use_fast_loras": True})
    result = _execute_mcp_tool("studio_generate_video", {"video_submode": "i2v", "mode": "ai",
        "model_id": "wan21-vace-1.3b", "image_rel": rel, "prompt": "two people boxing"})
    payload = json.loads(result["content"][0]["text"])
    assert payload["error_code"] == "SPEED_LORA_MISSING"
    assert "lora-wan21-1.3b-bidirect2" in payload["speed_lora_candidates"]


def test_i2v_explicit_fast_rejects_full_steps(temp_studio_env, colab_memory):
    import io
    from PIL import Image
    from studio.mcp_server import _execute_mcp_tool
    colab_memory(ram=64, available=52, gpu=True, vram=22, vram_free=21)
    client, _ = temp_studio_env
    buf = io.BytesIO()
    Image.new("RGB", (64, 64)).save(buf, format="PNG")
    rel = client.post("/api/assets/upload", files={"file": ("a.png", buf.getvalue(), "image/png")}).json()["asset"]["rel_path"]
    result = _execute_mcp_tool("studio_generate_video", {"video_submode": "i2v", "mode": "ai",
        "model_id": "wan21-vace-1.3b", "video_lora_id": "lora-wan21-1.3b-bidirect2", "require_fast": True,
        "use_official_defaults": False, "steps": 30, "image_rel": rel, "prompt": "two people boxing"})
    payload = json.loads(result["content"][0]["text"])
    assert payload["error_code"] == "FAST_STEPS_EXCEEDED"


def _tiny_wan_mixed_fixture(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    pytest.importorskip("peft")
    from diffusers import WanImageToVideoPipeline, WanTransformer3DModel
    from safetensors.torch import save_file

    model = WanTransformer3DModel(patch_size=(1, 2, 2), num_attention_heads=2,
        attention_head_dim=8, in_channels=4, out_channels=4, text_dim=16,
        freq_dim=8, ffn_dim=32, num_layers=1, cross_attn_norm=True)
    pipe = WanImageToVideoPipeline(tokenizer=None, text_encoder=None, vae=None,
                                    scheduler=None, transformer=model)
    raw = {
        "diffusion_model.blocks.0.self_attn.q.lora_down.weight": torch.full((2, 16), 0.1),
        "diffusion_model.blocks.0.self_attn.q.lora_up.weight": torch.full((16, 2), 0.2),
        "diffusion_model.blocks.0.self_attn.q.diff_b": torch.full((16,), 0.4),
        "diffusion_model.blocks.0.self_attn.norm_q.diff": torch.full((16,), 0.3),
        "diffusion_model.blocks.0.norm3.diff": torch.full((16,), 0.2),
        "diffusion_model.patch_embedding.diff_b": torch.full((16,), 0.25),
    }
    path = tmp_path / "mixed.safetensors"
    save_file(raw, str(path))
    return torch, pipe, raw, path


def test_wan_ti2v_mixed_lora_real_diffusers_and_peft(tmp_path):
    """LoRA + full-rank deltas must both affect the output and be reversible."""
    from studio.engines.video_engine import (_load_single_video_lora_onto_pipe,
        _apply_wan_direct_deltas, _restore_wan_direct_deltas)

    torch, pipe, raw, path = _tiny_wan_mixed_fixture(tmp_path)
    model = pipe.transformer
    originals = {name: param.detach().clone() for name, param in model.named_parameters()}
    x = torch.ones((1, 16))
    base_output = model.blocks[0].attn1.to_q(x).detach().clone()
    deltas = _load_single_video_lora_onto_pipe(pipe, "wan_ti2v", path, "test", False, None)
    pipe.set_adapters(["test"], adapter_weights=[0.75])
    handle = {}
    assert _apply_wan_direct_deltas(handle, pipe, [(deltas, 0.75)]) == 4
    # Independent float32 paths can differ by a few ulps, especially near zero.
    assert torch.allclose(model.blocks[0].attn1.to_q(x), base_output + 0.75 * (0.4 + 16 * 0.1 * 2 * 0.2), atol=1e-6)
    assert torch.allclose(model.blocks[0].attn1.norm_q.weight, originals["blocks.0.attn1.norm_q.weight"] + 0.75 * 0.3)
    assert torch.allclose(model.blocks[0].norm2.weight, originals["blocks.0.norm2.weight"] + 0.75 * 0.2)
    assert torch.allclose(model.patch_embedding.bias, originals["patch_embedding.bias"] + 0.75 * 0.25)
    _restore_wan_direct_deltas(handle, pipe)
    pipe.unload_lora_weights()
    assert all(torch.equal(param, originals[name]) for name, param in model.named_parameters())


def test_wan_direct_deltas_reject_bad_target_before_modifying_model(tmp_path):
    from studio.engines.video_engine import _apply_wan_direct_deltas
    torch, pipe, _, _ = _tiny_wan_mixed_fixture(tmp_path)
    before = pipe.transformer.patch_embedding.bias.detach().clone()
    handle = {}
    with pytest.raises(ValueError, match="WAN_LORA_DELTA_SHAPE_MISMATCH"):
        _apply_wan_direct_deltas(handle, pipe, [({"patch_embedding.bias": torch.ones(16),
                                                  "blocks.0.attn1.norm_q.weight": torch.ones(3)}, 0.75)])
    assert torch.equal(before, pipe.transformer.patch_embedding.bias)
    assert not handle.get("wan_direct_delta_originals")


def test_wan_ti2v_mixed_lora_rejects_unknown_or_incomplete_weights(tmp_path):
    from studio.engines.video_engine import _split_wan_ti2v_direct_deltas, _wan_ti2v_expected_lora_modules
    torch, _, raw, _ = _tiny_wan_mixed_fixture(tmp_path)
    low_rank, _ = _split_wan_ti2v_direct_deltas(raw)
    incomplete = dict(low_rank)
    incomplete.pop("diffusion_model.blocks.0.self_attn.q.lora_up.weight")
    with pytest.raises(ValueError, match="WAN_LORA_INCOMPLETE"):
        _wan_ti2v_expected_lora_modules(incomplete)
    with pytest.raises(ValueError, match="WAN_LORA_UNSUPPORTED_KEY"):
        _wan_ti2v_expected_lora_modules({**low_rank, "diffusion_model.unknown.weight": torch.ones(1)})


def test_wan_ti2v_delta_stack_scale_change_unload_and_failure_roll_back(tmp_path, monkeypatch):
    from studio.engines import video_engine as engine
    from safetensors.torch import save_file
    torch, pipe, raw, path = _tiny_wan_mixed_fixture(tmp_path)
    original = {name: param.detach().clone() for name, param in pipe.transformer.named_parameters()}
    lora = {"id": "test-fast", "name": "Fast test", "category": "lora", "compatible_models": ["base"], "max_steps": 6}
    monkeypatch.setattr(engine.catalog_manager, "get_definition", lambda lid: lora if lid == "test-fast" else {"id": "base"})
    monkeypatch.setattr(engine.catalog_manager, "is_downloaded", lambda lid: True)
    monkeypatch.setattr(engine.memory_manager, "prepare_for_adapters", lambda *a, **k: None)
    monkeypatch.setattr(engine.memory_manager, "update_adapters", lambda *a, **k: None)
    monkeypatch.setattr(engine, "query_gpu_info", lambda: {"vram_total_gb": 22})
    monkeypatch.setattr(engine, "_apply_best_offload", lambda *a, **k: None)
    monkeypatch.setattr(engine, "_resolve_lora_safetensors_file", lambda lid: path)
    handle = {"engine": "wan_ti2v", "defn": {"id": "base"}, "min_vram_gb": 1}
    engine._sync_video_lora_on_handle(handle, pipe, "test-fast", 0.75)
    assert torch.allclose(pipe.transformer.patch_embedding.bias, original["patch_embedding.bias"] + 0.75 * 0.25)
    engine._sync_video_lora_on_handle(handle, pipe, "test-fast", 0.25)
    assert torch.allclose(pipe.transformer.patch_embedding.bias, original["patch_embedding.bias"] + 0.25 * 0.25)
    engine._sync_video_lora_on_handle(handle, pipe, None)
    assert all(torch.equal(param, original[name]) for name, param in pipe.transformer.named_parameters())
    assert not handle.get("wan_direct_delta_originals")
    broken = dict(raw)
    broken["diffusion_model.patch_embedding.diff_b"] = torch.ones(3)
    save_file(broken, str(path))
    with pytest.raises(ValueError, match="WAN_LORA_DELTA_SHAPE_MISMATCH"):
        engine._sync_video_lora_on_handle(handle, pipe, "test-fast", 0.75)
    assert not handle.get("active_lora_id")
    assert all(torch.equal(param, original[name]) for name, param in pipe.transformer.named_parameters())


def test_wan_ti2v_restores_deltas_if_offload_fails(tmp_path, monkeypatch):
    from studio.engines import video_engine as engine
    torch, pipe, _, path = _tiny_wan_mixed_fixture(tmp_path)
    original = {name: param.detach().clone() for name, param in pipe.transformer.named_parameters()}
    lora = {"id": "test-fast", "name": "Fast test", "compatible_models": ["base"], "max_steps": 6}
    monkeypatch.setattr(engine.catalog_manager, "get_definition", lambda lid: lora if lid == "test-fast" else {"id": "base"})
    monkeypatch.setattr(engine.catalog_manager, "is_downloaded", lambda lid: True)
    monkeypatch.setattr(engine.memory_manager, "prepare_for_adapters", lambda *a, **k: None)
    monkeypatch.setattr(engine.memory_manager, "update_adapters", lambda *a, **k: None)
    monkeypatch.setattr(engine, "query_gpu_info", lambda: {"vram_total_gb": 22})
    def broken_offload(*args, **kwargs):
        raise RuntimeError("offload failed")
    monkeypatch.setattr(engine, "_apply_best_offload", broken_offload)
    monkeypatch.setattr(engine, "_resolve_lora_safetensors_file", lambda lid: path)
    handle = {"engine": "wan_ti2v", "defn": {"id": "base"}}
    with pytest.raises(RuntimeError, match="offload failed"):
        engine._sync_video_lora_on_handle(handle, pipe, "test-fast", 0.75)
    assert not handle.get("wan_direct_delta_originals")
    assert all(torch.equal(param, original[name]) for name, param in pipe.transformer.named_parameters())
