"""Contract tests for the experimental LTX audiovisual recipes (no GPU/weights)."""
import sys
import types

import pytest



@pytest.mark.parametrize("model_id,repo", [
    ("ltx-2.3-distilled-av", "diffusers/LTX-2.3-Distilled-Diffusers"),
    ("ltx-2.5-distilled-av", "Lightricks/LTX-2.5-Diffusers"),
])
def test_recipe_is_native_distilled_and_not_a_claim_of_l4_success(temp_studio_env, model_id, repo):
    from studio.catalog import catalog_manager, _matches_artifact_filters
    from studio.model_oracle import fast_video_recipe
    from studio.memory_policy import model_budget
    d = catalog_manager.get_definition(model_id)
    assert d["engine"] == "ltx2_distilled_av" and d["gpu_tested"] is False
    assert d["artifacts"][0]["repo_id"] == repo
    assert {"t2v", "i2v"} <= set(d["video_modes"])
    assert fast_video_recipe(d, "i2v", catalog_manager)["kind"] == "distilled"
    assert model_budget(d, has_gpu=True)["ram_gb"] == 80
    pats = d["artifacts"][0]["allow_patterns"]
    assert _matches_artifact_filters("audio_vae/config.json", pats, None, True)
    assert _matches_artifact_filters("connectors/config.json", pats, None, True)
    assert not _matches_artifact_filters("transformer_full/model.safetensors", pats, None, True)
    assert not _matches_artifact_filters("diffusion_decoder/model.safetensors", pats, None, True)


def test_fixed_sigmas_and_guidance_are_not_silently_skipped(temp_studio_env, monkeypatch):
    from studio.engines.video_engine import _ltx_distilled_call_settings
    class Scheduler:
        @classmethod
        def from_config(cls, config, **kwargs):
            return (config, kwargs)
    modules = {
        "diffusers": types.ModuleType("diffusers"),
        "diffusers.pipelines": types.ModuleType("diffusers.pipelines"),
        "diffusers.pipelines.ltx2": types.ModuleType("diffusers.pipelines.ltx2"),
        "diffusers.pipelines.ltx2.utils": types.ModuleType("diffusers.pipelines.ltx2.utils"),
    }
    modules["diffusers"].FlowMatchEulerDiscreteScheduler = Scheduler
    modules["diffusers.pipelines.ltx2.utils"].DISTILLED_SIGMA_VALUES = [1, .9, .7, .4]
    for name, mod in modules.items():
        monkeypatch.setitem(sys.modules, name, mod)
    pipe = types.SimpleNamespace(scheduler=types.SimpleNamespace(config={"name": "old"}))
    sig = {"sigmas", "guidance_scale", "audio_guidance_scale", "stg_scale", "audio_stg_scale", "modality_scale", "audio_modality_scale"}
    kwargs = _ltx_distilled_call_settings(pipe, sig)
    assert kwargs == {"sigmas": [1, .9, .7, .4], "guidance_scale": 1., "audio_guidance_scale": 1.,
                      "stg_scale": 0., "audio_stg_scale": 0., "modality_scale": 1., "audio_modality_scale": 1.}
    assert pipe.scheduler[1] == {"use_dynamic_shifting": False, "shift_terminal": None}
    with pytest.raises(RuntimeError, match="LTX_DISTILLED_RUNTIME_INCOMPATIBLE"):
        _ltx_distilled_call_settings(pipe, {"guidance_scale"})

@pytest.mark.parametrize("model_id,repo,transformer", [
    ("ltx-2.3-full-av", "diffusers/LTX-2.3-Diffusers", "transformer"),
    ("ltx-2.5-full-av", "Lightricks/LTX-2.5-Diffusers", "transformer_full"),
])
def test_full_models_have_separate_sharded_downloads(temp_studio_env, model_id, repo, transformer):
    from studio.catalog import catalog_manager, _matches_artifact_filters
    from studio.model_oracle import fast_video_recipe
    d = catalog_manager.get_definition(model_id)
    art = d["artifacts"][0]
    assert art["repo_id"] == repo and d["engine"] == "ltx2_full_av"
    assert d["gpu_tested"] is False and not d.get("ltx_distilled")
    assert fast_video_recipe(d, "i2v", catalog_manager) is None
    assert d["default_steps"] >= 30 and d["default_cfg"] > 1
    pats = art["allow_patterns"]
    assert _matches_artifact_filters(f"{transformer}/diffusion_pytorch_model-00001-of-00004.safetensors", pats, None, True)
    assert _matches_artifact_filters("text_encoder/model-00001-of-00005.safetensors", pats, None, True)
    assert not _matches_artifact_filters("diffusion_decoder/model.safetensors", pats, None, True)
    if transformer == "transformer_full":
        assert not _matches_artifact_filters("transformer/diffusion_pytorch_model.safetensors", pats, None, True)


def test_full_model_shard_indexes_require_all_nonempty_shards(temp_studio_env, monkeypatch, tmp_path):
    import json
    import studio.catalog as catalog
    catalog_manager = catalog.catalog_manager
    monkeypatch.setattr(catalog, "MODELS_DIR", tmp_path)
    model_id = "ltx-2.3-full-av"
    base = catalog_manager.model_dir(model_id) / "base"
    base.mkdir(parents=True, exist_ok=True)
    (base / "model_index.json").write_text('{}')
    catalog_manager.marker_file(model_id).write_text('{}')
    indexes = catalog_manager.get_definition(model_id)["artifacts"][0]["required_shard_indexes"]
    for index in indexes:
        path = base / index
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"weight_map": {"one": "a.safetensors", "two": "b.safetensors"}}))
        (path.parent / "a.safetensors").write_bytes(b'x')
        (path.parent / "b.safetensors").write_bytes(b'x')
    assert catalog_manager.is_downloaded(model_id)
    (base / "transformer" / "b.safetensors").write_bytes(b'')
    assert not catalog_manager.is_downloaded(model_id)
    (base / "transformer" / indexes[1].split('/')[-1]).write_text(json.dumps({"weight_map": {"bad": "../outside"}}))
    assert not catalog_manager.is_downloaded(model_id)


def test_ltx25_full_loader_uses_full_transformer_and_sft_scheduler(temp_studio_env, monkeypatch, tmp_path):
    import torch
    from studio.engines import video_engine as ve
    events = []
    class Transformer:
        @classmethod
        def from_pretrained(cls, path, **kw):
            events.append(("transformer", path, kw))
            return "full-weights"
    class Scheduler:
        @classmethod
        def from_config(cls, cfg, **kw):
            events.append(("scheduler", cfg, kw))
            return "sft-scheduler"
    class Condition:
        @classmethod
        def from_pretrained(cls, path, **kw):
            events.append(("pipe", path, kw))
            obj = types.SimpleNamespace(scheduler=types.SimpleNamespace(config={"old": True}))
            obj.components = {"transformer": kw.get("transformer")}
            return obj
    fake = types.ModuleType("diffusers")
    fake.LTX2ConditionPipeline = Condition
    fake.LTX2Pipeline = type("Text", (), {})
    fake.FlowMatchEulerDiscreteScheduler = Scheduler
    fake.LTX2VideoTransformer3DModel = Transformer
    monkeypatch.setitem(sys.modules, "diffusers", fake)
    monkeypatch.setattr(ve, "apply_runtime_compatibility_patches", lambda: None)
    monkeypatch.setattr(ve, "guarded_model_load", lambda f, *a, **kw: f(*a, **kw))
    (tmp_path / "model_index.json").write_text('{}')
    out = ve._instantiate_video_pipeline("ltx2_full_av", tmp_path, {"id": "ltx-2.5-full-av", "ltx_full_transformer": True})
    assert events[0][2] == {"subfolder": "transformer_full", "torch_dtype": torch.bfloat16, "local_files_only": True}
    assert events[1][2]["transformer"] == "full-weights" and events[1][2]["local_files_only"]
    assert events[1][2]["prompt_enhancer"] is None and events[1][2]["duration_head"] is None
    assert events[2][2] == {"use_dynamic_shifting": True, "shift_terminal": 0.1}
    assert out["pipe"].scheduler == "sft-scheduler"
    assert out["all_components"]["transformer"] == "full-weights"


def test_ltx23_full_loader_does_not_switch_to_sft_or_distilled(temp_studio_env, monkeypatch, tmp_path):
    from studio.engines import video_engine as ve
    calls = []
    class Condition:
        @classmethod
        def from_pretrained(cls, path, **kw):
            calls.append(kw)
            return types.SimpleNamespace(components={"transformer": "original"})
    fake = types.ModuleType("diffusers")
    fake.LTX2ConditionPipeline = Condition
    fake.LTX2Pipeline = type("Text", (), {})
    monkeypatch.setitem(sys.modules, "diffusers", fake)
    monkeypatch.setattr(ve, "apply_runtime_compatibility_patches", lambda: None)
    monkeypatch.setattr(ve, "guarded_model_load", lambda f, *a, **kw: f(*a, **kw))
    (tmp_path / "model_index.json").write_text('{}')
    out = ve._instantiate_video_pipeline("ltx2_full_av", tmp_path, {})
    assert calls[0]["local_files_only"] and "transformer" not in calls[0]
    assert out["all_components"]["transformer"] == "original"
