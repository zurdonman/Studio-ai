"""Reviewed loader contracts, not a universal parameter-count -> VRAM calculator.

Adding an upstream model to discovery never adds a loader here. Resource numbers
are conservative planning estimates for the specified recipe, NOT GPU benchmarks.
See docs/MODEL_RESEARCH.md for sources and docs/MCP_MODEL_ORACLE.md for semantics.
This module deliberately imports no GPU framework and has no network side effects.
"""
from __future__ import annotations

from typing import Any, Dict

DIFFUSERS_DOCS = "https://github.com/huggingface/diffusers/blob/v0.42.1/docs/source/en/api/pipelines/"
RESEARCH_DATE = "2026-11-05"

# Exact classes used by Studio's loaders. A class being present in Diffusers is
# necessary, but NOT sufficient: input conventions, companions and formats matter.
ENGINE_CONTRACTS: Dict[str, Dict[str, Any]] = {
    "qwen_image_edit": {"classes": ["QwenImageEditPlusPipeline"], "diffusers": "0.36.0", "ram_gb": 28, "quantization": "prequantized bitsandbytes 4-bit; Lightning before offload", "packages": ["bitsandbytes", "peft"]},
    "qwen_image_t2i": {"classes": ["QwenImagePipeline"], "diffusers": "0.36.0", "ram_gb": 28, "quantization": "prequantized bitsandbytes 4-bit; Lightning before offload", "packages": ["bitsandbytes", "peft"]},
    "sdxl_turbo": {"classes": ["AutoPipelineForText2Image"], "diffusers": "0.30.0", "ram_gb": 12, "quantization": "FP16; model CPU offload"},
    "sdxl_base": {"classes": ["StableDiffusionXLPipeline"], "diffusers": "0.30.0", "ram_gb": 16, "dtype": "float16"},
    "flux_schnell": {"classes": ["FluxPipeline"], "diffusers": "0.31.0", "ram_gb": 48, "dtype": "bfloat16", "native_bf16": True},
    "flux2_klein": {"classes": ["Flux2KleinPipeline"], "diffusers": "0.37.1", "transformers": "4.57.1", "ram_gb": 32, "dtype": "bfloat16", "native_bf16": True},
    "z_image_turbo": {"classes": ["ZImagePipeline"], "diffusers": "0.36.0", "transformers": "4.57.1", "ram_gb": 32, "dtype": "bfloat16", "native_bf16": True},
    "pixart_sigma": {"classes": ["PixArtSigmaPipeline"], "diffusers": "0.30.0", "ram_gb": 32, "dtype": "float16"},
    "sana_image": {"classes": ["SanaPipeline"], "diffusers": "0.32.0", "ram_gb": 24, "dtype": "bfloat16", "native_bf16": True},
    "sana_sprint": {"classes": ["SanaSprintPipeline"], "diffusers": "0.33.0", "ram_gb": 24, "dtype": "bfloat16", "native_bf16": True},
    "wan_ti2v": {"classes": ["WanImageToVideoPipeline", "WanPipeline"], "diffusers": "0.35.0", "ram_gb": 40},
    "wan_t2v": {"classes": ["WanPipeline"], "diffusers": "0.33.0", "ram_gb": 32},
    "wan_vace": {"classes": ["WanVACEPipeline"], "diffusers": "0.33.0", "ram_gb": 28},
    "wan_vace_fp8": {"classes": ["WanVACEPipeline", "WanVACETransformer3DModel"], "diffusers": "0.35.0", "ram_gb": 40, "quantization": "explicit FP8/meta single-file denoiser + BF16 encoder + FP32 VAE; CPU offload"},
    "wan_i2v_14b": {"classes": ["WanImageToVideoPipeline"], "diffusers": "0.35.0", "ram_gb": 64},
    "wan_i2v_14b_fp8": {"classes": ["WanImageToVideoPipeline", "WanTransformer3DModel"], "diffusers": "0.35.0", "ram_gb": 40, "quantization": "explicit FP8/meta single-file denoiser + companion encoder and VAE"},
    "ltx_video": {"classes": ["LTXConditionPipeline"], "diffusers": "0.33.0", "ram_gb": 32},
    "ltx_video_distilled": {"classes": ["LTXConditionPipeline"], "diffusers": "0.35.0", "ram_gb": 48, "quantization": "FP8 layerwise casting of denoiser; BF16 companions"},
    "ltx2_av": {"classes": ["LTX2ConditionPipeline", "LTX2Pipeline"], "diffusers": "0.36.0", "ram_gb": 80},
    "ltx2_full_av": {"classes": ["LTX2ConditionPipeline", "LTX2Pipeline", "LTX2VideoTransformer3DModel"], "diffusers": "0.42.0.dev0", "transformers": "5.5.0", "ram_gb": 100, "native_bf16": True},
    "ltx2_distilled_av": {"classes": ["LTX2ConditionPipeline", "LTX2Pipeline"], "diffusers": "0.42.0.dev0", "transformers": "5.5.0", "ram_gb": 80},
    "sana_video": {"classes": ["SanaImageToVideoPipeline", "SanaVideoPipeline"], "diffusers": "0.36.0", "ram_gb": 28},
    "skyreels_v2": {"classes": ["SkyReelsV2ImageToVideoPipeline"], "diffusers": "0.35.0", "ram_gb": 36},
    "cogvideox_i2v": {"classes": ["CogVideoXImageToVideoPipeline"], "diffusers": "0.31.0", "ram_gb": 32},
    "mms_tts": {"classes": [], "ram_gb": 2, "cpu_capable": True, "ffmpeg": True, "packages": ["torch", "transformers", "scipy"]},
    "kokoro_tts": {"classes": [], "ram_gb": 3, "cpu": True, "ffmpeg": True, "packages": ["torch", "kokoro", "soundfile"], "executables": ["espeak-ng"]},
    "bark_tts": {"classes": [], "ram_gb": 8, "cpu_capable": True, "ffmpeg": True, "packages": ["torch", "transformers", "scipy"]},
    "qwen25_vl_coherence": {"classes": [], "ram_gb": 12, "packages": ["torch", "transformers", "accelerate", "bitsandbytes"]},
    "edtalk_onnx_hd": {"classes": [], "ram_gb": 4, "cpu_capable": True, "ffmpeg": True, "packages": ["onnxruntime-gpu", "opencv-python-headless", "scipy"]},
    "musicgen": {"classes": [], "ram_gb": 6, "cpu_capable": True, "packages": ["torch", "transformers", "scipy"]},
    "audioldm2": {"classes": ["AudioLDM2Pipeline"], "diffusers": "0.30.0", "ram_gb": 10, "cpu_capable": True},
    "ffmpeg_motion": {"classes": [], "ram_gb": 1, "cpu": True, "ffmpeg": True},
    "edge_tts": {"classes": [], "ram_gb": 0.5, "cpu": True, "ffmpeg": True, "packages": ["edge-tts"], "network_service": True},
    "opencv_fast": {"classes": [], "ram_gb": 1, "cpu": True, "packages": ["opencv-python-headless"]},
    "procedural_dsp": {"classes": [], "ram_gb": 1, "cpu": True, "ffmpeg": True},
}

# Only these additional engines go through the explicit, local-only image factory.
REVIEWED_IMAGE_ENGINES = {
    key: value for key, value in ENGINE_CONTRACTS.items()
    if key in {"sdxl_base", "flux_schnell", "flux2_klein", "z_image_turbo", "pixart_sigma", "sana_image", "sana_sprint"}
}


def _image(model_id: str, name: str, repo: str, engine: str, disk: float, vram: float,
           recommended: float, steps: int, cfg: float, doc: str, *, edit: bool = False,
           license_id: str | None = None, variant: str | None = None) -> Dict[str, Any]:
    return {
        "id": model_id, "name": name, "category": "image", "engine": engine,
        "is_default": False, "badge": "Receta documentada · GPU pendiente de validar",
        "validation_state": "reviewed_recipe", "gpu_tested": False,
        "capabilities": ["t2i", "i2i", "multi_ref"] if edit else ["t2i"],
        "estimated_gb": disk, "min_vram_gb": vram, "recommended_vram_gb": recommended,
        "min_ram_gb": ENGINE_CONTRACTS[engine]["ram_gb"],
        "resource_confidence": "engineering_estimate", "license": license_id,
        "source_url": DIFFUSERS_DOCS + doc + ".md", "researched_at": RESEARCH_DATE,
        "description": "Pipeline Diffusers explícito con CPU offload. Estimaciones de planificación, no benchmark ni garantía de memoria. Sin cuantización implícita.",
        "official_spec": {"default_steps": steps, "default_cfg": cfg, "default_width": 1024,
                          "default_height": 1024, "prompt_lang": "en", "variant": variant},
        "artifacts": [{"role": "base", "repo_id": repo, "subdir": "base", "verify_file": "model_index.json",
                       "allow_patterns": ["model_index.json", "scheduler/*.json", "tokenizer/*", "tokenizer_2/*"] + [
                           f"{component}/{pattern}"
                           for component in ("text_encoder", "text_encoder_2", "transformer", "unet", "vae")
                           for pattern in (["*.fp16.safetensors", "*.json"] if variant == "fp16" else ["*.safetensors", "*.json"])
                       ],
                       "ignore_patterns": ["*.py", "*.so", "*.pyd", "*.sh", "*.ckpt", "*.pth", "*.pt", "*.bin", "*.gguf", "*.onnx", "*.onnx_data",
                                           "*.msgpack", "*.h5", "*.ot", "*.fp32.*"] + (["*.bf16.*"] if variant == "fp16" else ["*.fp16.*", "*.bf16.*"])}],
    }


EXTRA_IMAGE_MODELS = [
    _image("sdxl-base-1.0", "SDXL Base 1.0 · FP16", "stabilityai/stable-diffusion-xl-base-1.0", "sdxl_base", 9.0, 7.0, 12.0, 30, 5.0, "stable_diffusion/stable_diffusion_xl", variant="fp16"),
    _image("flux1-schnell", "FLUX.1 Schnell · BF16 + CPU offload", "black-forest-labs/FLUX.1-schnell", "flux_schnell", 36.0, 24.0, 32.0, 4, 0.0, "flux", license_id="apache-2.0"),
    _image("flux2-klein-4b", "FLUX.2 Klein 4B · creación y referencias", "black-forest-labs/FLUX.2-klein-4B", "flux2_klein", 19.0, 14.0, 20.0, 4, 1.0, "flux2", edit=True, license_id="apache-2.0"),
    _image("z-image-turbo", "Z-Image Turbo · BF16 · 9 pasos / 8 NFE", "Tongyi-MAI/Z-Image-Turbo", "z_image_turbo", 23.0, 16.0, 22.0, 9, 0.0, "z_image", license_id="apache-2.0"),
    _image("pixart-sigma-xl-1024", "PixArt Sigma XL 1024 · FP16 (T5 incluido)", "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS", "pixart_sigma", 24.0, 12.0, 16.0, 20, 4.5, "pixart_sigma"),
    _image("sana-600m-1024", "SANA 600M 1024 · FP16 + encoder BF16 / VAE FP32", "Efficient-Large-Model/Sana_600M_1024px_diffusers", "sana_image", 20.0, 10.0, 16.0, 20, 4.5, "sana"),
    _image("sana-1600m-1024", "SANA 1600M 1024 BF16", "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers", "sana_image", 24.0, 12.0, 18.0, 20, 4.5, "sana"),
    _image("sana-sprint-600m", "SANA Sprint 600M · destilado", "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers", "sana_sprint", 20.0, 10.0, 16.0, 2, 0.0, "sana_sprint"),
]


# SANA's mixed-precision contract is not a blanket pipe.to(float16).
for _definition in EXTRA_IMAGE_MODELS:
    if _definition["engine"] in ("sana_image", "sana_sprint"):
        _definition["official_spec"]["component_dtypes"] = {
            "transformer": "float16" if _definition["id"] == "sana-600m-1024" else "bfloat16",
            "text_encoder": "bfloat16", "vae": "float32",
        }
        _definition["description"] += " SANA: encoder BF16 y VAE FP32; archivos canónicos sin variante, almacenamiento posiblemente FP32."


def image_defaults(definition: Dict[str, Any], use_lightning: bool = True) -> Dict[str, Any]:
    """Do not apply Qwen's four-step Lightning schedule to unrelated pipelines."""
    engine = definition.get("engine")
    spec = definition.get("official_spec") or {}
    if engine in ("qwen_image_edit", "qwen_image_t2i"):
        return {"steps": 4 if use_lightning else 28, "guidance_scale": 4.0}
    if engine == "sdxl_turbo":
        return {"steps": 4, "guidance_scale": 0.0}
    return {"steps": int(spec.get("default_steps", 30)), "guidance_scale": float(spec.get("default_cfg", 4.0))}


def image_recipe_files_complete(definition: Dict[str, Any], base_dir: Any) -> bool:
    """Presence/completeness, NOT deserialization or GPU validation of weights.

    A completion marker plus model_index.json alone cannot prepare a new image
    pipeline. Each declared model component and each indexed shard must exist.
    This never deletes files or changes to a different model/format.
    """
    import json
    from pathlib import Path, PurePosixPath
    if definition.get("engine") not in REVIEWED_IMAGE_ENGINES:
        return True
    base_dir = Path(base_dir)
    try:
        manifest_file = base_dir / "model_index.json"
        if manifest_file.stat().st_size > 262144:
            return False
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        expected = REVIEWED_IMAGE_ENGINES[definition["engine"]]["classes"][0]
        if not isinstance(manifest, dict) or manifest.get("_class_name") != expected:
            return False
        components = [name for name in ("transformer", "unet", "vae", "text_encoder", "text_encoder_2")
                      if isinstance(manifest.get(name), list) and len(manifest[name]) == 2 and manifest[name][0] is not None]
        if not components:
            return False
        for name in components:
            folder = base_dir / name
            variant = (definition.get("official_spec") or {}).get("variant")
            def selected(file: Any) -> bool:
                if variant:
                    return f".{variant}." in file.name or f".{variant}-" in file.name
                return not any(f".{v}." in file.name or f".{v}-" in file.name for v in ("fp16", "bf16", "fp32"))
            if (not (folder / "config.json").is_file() or not (folder / "config.json").stat().st_size
                    or not any(selected(f) and f.is_file() and f.stat().st_size for f in folder.glob("*.safetensors"))):
                return False
            for index_file in folder.glob("*.safetensors.index.json"):
                if not selected(index_file):
                    continue
                if index_file.stat().st_size > 4_000_000:
                    return False
                index = json.loads(index_file.read_text(encoding="utf-8"))
                mapping = index.get("weight_map") if isinstance(index, dict) else None
                if not isinstance(mapping, dict) or not mapping:
                    return False
                for shard in set(mapping.values()):
                    if not isinstance(shard, str) or "\\" in shard or PurePosixPath(shard).is_absolute() or any(p in (".", "..") for p in shard.split("/")):
                        return False
                    file = folder / shard
                    if not file.is_file() or not file.stat().st_size:
                        return False
        return True
    except (OSError, ValueError, TypeError):
        return False
