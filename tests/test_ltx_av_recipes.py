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
