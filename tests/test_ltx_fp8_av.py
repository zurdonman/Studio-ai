"""FP8 storage recipes: CPU toy models and mocked IO only, NOT GPU validation."""
import json
import sys
import types

import pytest


@pytest.mark.parametrize("version,flavor", [(v, f) for v in ("2.3", "2.5") for f in ("full", "distilled")])
def test_fp8_catalog_keeps_audio_and_original_recipe(temp_studio_env, version, flavor):
    from studio.catalog import catalog_manager
    from studio.model_oracle import fast_video_recipe
    from studio.model_compatibility import assess_definition, assessment_context
    model_id = f"ltx-{version}-{flavor}-fp8-av"
    d = catalog_manager.get_definition(model_id)
    base = catalog_manager.get_definition(f"ltx-{version}-{flavor}-av")
    assert d["ltx_fp8_cast"] and d["gpu_tested"] is False
    assert d["artifacts"][0]["repo_id"] == base["artifacts"][0]["repo_id"]
    assert d["default_steps"] == base["default_steps"] and d["default_cfg"] == base["default_cfg"]
    assert d["video_modes"] == base["video_modes"] and d["ltx_large"]
    assert bool(d.get("ltx_distilled")) == (flavor == "distilled")
    assert bool(fast_video_recipe(d, "i2v", catalog_manager)) == (flavor == "distilled")
    snapshot = {"gpu": {"has_gpu": True, "gpu_name": "L4", "gpu_tier": "L4", "vram_total_gb": 24, "vram_free_gb": 23, "bf16_supported": True},
                "ram": {"total_gb": 128, "available_gb": 120}, "disk": {"free_gb": 200}}
    ctx = assessment_context(snapshot=snapshot, detected_gpu=snapshot["gpu"], runtime={"versions": {}, "executables": {}})
    assessment = assess_definition(d, ctx, catalog=catalog_manager)
    assert "FP8" in assessment["memory_strategy"] and not assessment["gpu_tested"]
    assert "BF16" in assessment["memory_strategy"]


@pytest.mark.parametrize("full", [False, True])
def test_quant_loader_casts_before_loading_other_components(temp_studio_env, monkeypatch, tmp_path, full):
    from studio.engines import video_engine as ve
    calls = []
    transformer = object()
    class Transformer:
        @classmethod
        def from_pretrained(cls, *a, **kw):
            calls.append(("load_transformer", kw)); return transformer
    class Scheduler:
        @classmethod
        def from_config(cls, config, **kw): calls.append(("scheduler", kw)); return "scheduler"
    class Condition:
        @classmethod
        def from_pretrained(cls, *a, **kw):
            calls.append(("pipeline", kw))
            return types.SimpleNamespace(components={"transformer": kw["transformer"]}, scheduler=types.SimpleNamespace(config={}))
    mod = types.ModuleType("diffusers")
    mod.LTX2VideoTransformer3DModel = Transformer
    mod.LTX2ConditionPipeline = Condition
    mod.LTX2Pipeline = type("Text", (), {})
    mod.FlowMatchEulerDiscreteScheduler = Scheduler
    monkeypatch.setitem(sys.modules, "diffusers", mod)
    monkeypatch.setattr(ve, "apply_runtime_compatibility_patches", lambda: None)
    monkeypatch.setattr(ve, "guarded_model_load", lambda f, *a, **kw: f(*a, **kw))
    monkeypatch.setattr(ve, "_enable_fp8_layerwise_casting_safe", lambda t: calls.append(("cast", t)))
    monkeypatch.setattr(ve, "_is_module_layerwise_fp8", lambda t: True)
    (tmp_path / "model_index.json").write_text('{}')
    d = {"ltx25": True, "ltx_fp8_cast": True, "ltx_full_transformer": full}
    bundle = ve._instantiate_video_pipeline("ltx2_full_av" if full else "ltx2_distilled_av", tmp_path, d)
    assert [e[0] for e in calls] == (["load_transformer", "cast", "pipeline", "scheduler"] if full else ["load_transformer", "cast", "pipeline"])
    assert calls[0][1]["subfolder"] == ("transformer_full" if full else "transformer")
    assert calls[2][1]["transformer"] is transformer
    assert calls[2][1]["prompt_enhancer"] is None and calls[2][1]["duration_head"] is None
    assert bundle["all_components"]["transformer"] is transformer
    if full:
        assert calls[-1][1] == {"use_dynamic_shifting": True, "shift_terminal": 0.1}
    monkeypatch.setattr(ve, "_is_module_layerwise_fp8", lambda t: False)
    with pytest.raises(RuntimeError, match="LTX_FP8_CAST_FAILED"):
        ve._instantiate_video_pipeline("ltx2_full_av", tmp_path, d)


@pytest.mark.parametrize("version", ["2.3", "2.5"])
def test_real_tiny_ltx_transformer_fp8_keeps_both_streams_with_cpu_offload(temp_studio_env, version):
    # Exercises actual Diffusers/Accelerate hooks, with random weights and CPU only.
    torch = pytest.importorskip("torch")
    diffusers = pytest.importorskip("diffusers")
    pytest.importorskip("accelerate")
    from accelerate import cpu_offload
    from studio.engines.video_engine import _enable_fp8_layerwise_casting_safe, _is_module_layerwise_fp8
    cls = getattr(diffusers, "LTX2VideoTransformer3DModel", None)
    if cls is None or not hasattr(cls, "enable_layerwise_casting") or not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("Requires recent Diffusers and Torch FP8")
    cfg = dict(in_channels=4, out_channels=4, num_attention_heads=2, attention_head_dim=8, cross_attention_dim=16,
               audio_in_channels=4, audio_out_channels=4, audio_num_attention_heads=2, audio_attention_head_dim=4,
               audio_cross_attention_dim=8, num_layers=1, caption_channels=16, rope_double_precision=False,
               use_prompt_embeddings=False, cross_attn_mod=True, audio_cross_attn_mod=True)
    if version == "2.5":
        cfg.update(use_prompt_adaln_single=False, ff_bias=False, use_keyframes_abs_pos_embedding=True)
    model = cls(**cfg).to(torch.bfloat16).eval()
    _enable_fp8_layerwise_casting_safe(model)
    assert _is_module_layerwise_fp8(model)
    assert any(p.dtype == torch.float8_e4m3fn for p in model.parameters())
    assert any(p.dtype == torch.bfloat16 for p in model.parameters())
    kwargs = dict(hidden_states=torch.randn(1, 8, 4, dtype=torch.bfloat16),
                  audio_hidden_states=torch.randn(1, 5, 4, dtype=torch.bfloat16),
                  encoder_hidden_states=torch.randn(1, 4, 16, dtype=torch.bfloat16),
                  audio_encoder_hidden_states=torch.randn(1, 4, 8, dtype=torch.bfloat16),
                  timestep=torch.tensor([10.]), sigma=torch.tensor([10.]), num_frames=2, height=2, width=2,
                  audio_num_frames=5, fps=25., return_dict=False)
    cpu_offload(model, execution_device=torch.device("cpu"))
    with torch.no_grad():
        for _ in range(2):
            video, audio = model(**kwargs)
            assert video.shape == (1, 8, 4) and audio.shape == (1, 5, 4)
            assert torch.isfinite(video).all() and torch.isfinite(audio).all()


def test_25_distilled_download_uses_only_indexed_shards_and_pinned_revision(temp_studio_env, monkeypatch, tmp_path):
    import studio.catalog as catalog
    from studio.jobs import JobRecord
    model_id = "ltx-2.5-distilled-fp8-av"
    monkeypatch.setattr(catalog, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(catalog.catalog_manager, "check_disk_space", lambda *a, **kw: {"ok": True})
    t_idx = "transformer/diffusion_pytorch_model.safetensors.index.json"
    e_idx = "text_encoder/model.safetensors.index.json"
    files = {"model_index.json": "{}", t_idx: json.dumps({"weight_map": {"a": "part-00001-of-00004.safetensors"}}),
             e_idx: json.dumps({"weight_map": {"b": "encoder.safetensors"}}),
             "text_encoder/encoder.safetensors": "x", "transformer/part-00001-of-00004.safetensors": "x",
             "transformer/unused-00001-of-00008.safetensors": "x"}
    class Api:
        def __init__(self, **kw): pass
        def model_info(self, **kw):
            return types.SimpleNamespace(sha="abc123", siblings=[types.SimpleNamespace(rfilename=k, size=len(v)) for k,v in files.items()])
    downloaded = []
    def download(repo_id, filename, local_dir, **kwargs):
        assert kwargs["revision"] == "abc123"
        downloaded.append(filename)
        p = __import__("pathlib").Path(local_dir) / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(files[filename])
        return str(p)
    monkeypatch.setattr(catalog, "HfApi", Api)
    monkeypatch.setattr(catalog, "hf_hub_download", download)
    catalog.catalog_manager.download_model(model_id, JobRecord("download", "test"))
    assert "transformer/unused-00001-of-00008.safetensors" not in downloaded
    assert "transformer/part-00001-of-00004.safetensors" in downloaded
    assert catalog.catalog_manager.is_downloaded(model_id)
    target = catalog.catalog_manager.model_dir(model_id) / "base" / "transformer" / "part-00001-of-00004.safetensors"
    target.write_text("incorrect length")
    assert not catalog.catalog_manager.is_downloaded(model_id)


def test_t4_fp8_is_blocked_before_download(temp_studio_env, monkeypatch):
    import torch
    from studio.engines import video_engine as ve
    monkeypatch.setattr(ve, "apply_runtime_compatibility_patches", lambda: None)
    monkeypatch.setattr(ve.memory_manager, "get_slot", lambda *a: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (7, 5))
    monkeypatch.setattr(ve.catalog_manager, "download_model", lambda *a, **kw: pytest.fail("download started"))
    with pytest.raises(RuntimeError, match="LTX_BF16_UNSUPPORTED"):
        ve._load_video_model_into_memory_locked("ltx-2.3-full-fp8-av", auto_download=True)


def test_fp8_memory_admission_blocks_download(temp_studio_env, monkeypatch, colab_memory):
    import torch
    from studio.engines import video_engine as ve
    from studio.memory_policy import MemoryPressureError
    colab_memory(ram=53, available=48, vram=24, vram_free=23)
    monkeypatch.setattr(ve, "apply_runtime_compatibility_patches", lambda: None)
    monkeypatch.setattr(ve.memory_manager, "get_slot", lambda *a: None)
    monkeypatch.setattr(ve, "flush_ram_and_vram", lambda: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 9))
    monkeypatch.setattr(ve.catalog_manager, "download_model", lambda *a, **kw: pytest.fail("download started"))
    with pytest.raises(MemoryPressureError):
        ve._load_video_model_into_memory_locked("ltx-2.3-full-fp8-av", auto_download=True)
