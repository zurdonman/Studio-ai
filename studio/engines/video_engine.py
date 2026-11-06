import inspect
import json
import math
import os
import random
import shutil
import subprocess
import tempfile
import time
import types
import wave
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from studio.catalog import catalog_manager
from studio.config import VIDEOS_OUT_DIR
from studio.engines.ffmpeg_utils import (
    create_motion_clip_from_image,
    get_ffmpeg_bin,
    probe_media_info,
)
from studio.hardware import query_gpu_info
from studio.jobs import JobRecord
from studio.memory_manager import (
    flush_ram_and_vram,
    memory_manager,
    run_with_live_memory_progress,
)

# Prompts negativos oficiales verificados en las model cards de cada fabricante
OFFICIAL_NEGATIVE_PROMPTS: Dict[str, str] = {
    "wan": (
        "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, "
        "overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, "
        "poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
        "still picture, messy background, three legs, many people in the background, walking backwards"
    ),
    "sana": (
        "A chaotic sequence with misshapen, deformed limbs in heavy motion blur, sudden disappearance, "
        "jump cuts, jerky movements, rapid shot changes, frames out of sync, inconsistent character shapes, "
        "temporal artifacts, jitter, and ghosting effects, creating a disorienting visual experience."
    ),
    "ltx": (
        "shaky, glitchy, low quality, worst quality, inconsistent motion, blurry, jittery, distorted, "
        "deformed, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly"
    ),
    "default": (
        "worst quality, low quality, blurry, jittery, distorted, deformed, disfigured, misshapen limbs, "
        "extra limbs, bad anatomy, artifacts, overexposed"
    ),
}


def _apply_best_offload(
    pipe: Any,
    vram_total_gb: float,
    min_vram_gb: float,
    job: Optional[JobRecord] = None,
    pct: float = 90.0,
) -> None:
    if hasattr(pipe, "remove_all_hooks"):
        try:
            pipe.remove_all_hooks()
        except Exception:
            pass

    if vram_total_gb < min_vram_gb + 1.5 and hasattr(pipe, "enable_sequential_cpu_offload"):
        if job:
            job.log(
                f"Activando Sequential CPU Offload para ajustar en {vram_total_gb:.1f} GB VRAM...",
                progress=pct,
                step="Activando Sequential Offload...",
            )
        pipe.enable_sequential_cpu_offload()
    elif hasattr(pipe, "enable_model_cpu_offload"):
        if job:
            job.log("Activando CPU Offload inteligente...", progress=pct, step="Activando CPU Offload...")
        pipe.enable_model_cpu_offload()
    elif hasattr(pipe, "to"):
        pipe.to("cuda")

    if hasattr(pipe, "vae") and pipe.vae is not None:
        if hasattr(pipe.vae, "enable_slicing"):
            try:
                pipe.vae.enable_slicing()
            except Exception:
                pass
        if hasattr(pipe.vae, "enable_tiling"):
            try:
                sig_tile = inspect.signature(pipe.vae.enable_tiling).parameters
                if "tile_sample_min_height" in sig_tile:
                    if vram_total_gb >= 20.0:
                        # En L4 (22.5 GB) / A100 / H100 quedan >21 GB libres durante VAE encode/decode con CPU Offload.
                        # Evita partir 800×1088 en 30 mosaicos de 256×256 (~3m33s -> ~12s).
                        pipe.vae.enable_tiling(
                            tile_sample_min_height=1280,
                            tile_sample_min_width=1280,
                            tile_sample_stride_height=960,
                            tile_sample_stride_width=960,
                        )
                    else:
                        pipe.vae.enable_tiling(
                            tile_sample_min_height=704,
                            tile_sample_min_width=704,
                            tile_sample_stride_height=512,
                            tile_sample_stride_width=512,
                        )
                else:
                    pipe.vae.enable_tiling()
            except Exception:
                try:
                    pipe.vae.enable_tiling()
                except Exception:
                    pass


def _attach_wan22_flf2v_support(pipe: Any) -> Any:
    """
    Habilita soporte nativo Frame a Frame (Inicio → Fin, `last_image`) en `WanImageToVideoPipeline`
    cuando `expand_timesteps=True` (arquitectura Wan 2.2 TI2V-5B).

    En Diffusers <=0.42.1, `WanImageToVideoPipeline.prepare_latents` ignora `last_image` cuando
    `self.config.expand_timesteps` es True (`video_condition = image` de 1 solo frame).
    Sin embargo, `WanImageToVideoPipeline.__call__` soporta tensores `condition` y `first_frame_mask`
    de longitud `num_latent_frames` completa:
      - `latent_model_input = (1 - mask) * condition + mask * latents`
      - `temp_ts = (mask[0][0][:, ::2, ::2] * t).flatten()` (fija `t=0` en los fotogramas anclados)
    Codificamos `image` en `[:, :, 0:1]` (`mask=0`) y `last_image` en bloque causal estacionario
    de 4 fotogramas (`repeat(1,1,5,1,1)` -> `[:, :, -1:]`, `mask=0`) para clavar exactamente
    tanto el fotograma inicial como el fotograma final sin parpadeo final.
    """
    if getattr(pipe, "_studio_wan22_flf2v_patched", False):
        return pipe
    cfg = getattr(pipe, "config", None)
    exp_ts = cfg.get("expand_timesteps", False) if isinstance(cfg, dict) else getattr(cfg, "expand_timesteps", False)
    if not bool(exp_ts):
        return pipe
    orig_prepare_latents = getattr(pipe, "prepare_latents", None)
    if not callable(orig_prepare_latents):
        return pipe

    def _wan22_flf2v_prepare_latents(
        self: Any,
        image: Any,
        batch_size: int,
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        dtype: Any = None,
        device: Any = None,
        generator: Any = None,
        latents: Any = None,
        last_image: Any = None,
    ) -> Any:
        scfg = getattr(self, "config", None)
        s_exp = scfg.get("expand_timesteps", False) if isinstance(scfg, dict) else getattr(scfg, "expand_timesteps", False)
        if last_image is None or not bool(s_exp):
            return orig_prepare_latents(
                image,
                batch_size,
                num_channels_latents,
                height,
                width,
                num_frames,
                dtype,
                device,
                generator,
                latents,
                last_image,
            )

        import torch  # type: ignore

        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        if num_latent_frames < 2:
            return orig_prepare_latents(
                image,
                batch_size,
                num_channels_latents,
                height,
                width,
                num_frames,
                dtype,
                device,
                generator,
                latents,
                None,
            )

        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial
        shape = (batch_size, num_channels_latents, num_latent_frames, latent_height, latent_width)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(f"Lista de generadores ({len(generator)}) != batch_size ({batch_size}).")

        try:
            from diffusers.utils.torch_utils import get_module_execution_device, randn_tensor  # type: ignore

            vae_device = get_module_execution_device(self.vae)
            if latents is None:
                latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            else:
                latents = latents.to(device=device, dtype=dtype)
        except ImportError:
            vae_device = device
            if latents is None:
                latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)
            else:
                latents = latents.to(device=device, dtype=dtype)

        vae_dtype = getattr(self.vae, "dtype", dtype)

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(device=device, dtype=dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
            1, self.vae.config.z_dim, 1, 1, 1
        ).to(device=device, dtype=dtype)

        def _extract_mode_latents(enc_out: Any) -> Any:
            if hasattr(enc_out, "latent_dist"):
                return enc_out.latent_dist.mode()
            if hasattr(enc_out, "latents"):
                return enc_out.latents
            return enc_out[0].mode()

        # 1) Fotograma inicial (modo causal primer frame t=0)
        start_cond = image.unsqueeze(2).to(device=vae_device, dtype=vae_dtype)
        z_start = _extract_mode_latents(self.vae.encode(start_cond)).to(device=device, dtype=dtype)
        z_start = (z_start - latents_mean) * latents_std

        # 2) Fotograma final (5 repeticiones -> bloque causal estacionario 4-frame `[:, :, -1:]`)
        end_cond = last_image.unsqueeze(2).repeat(1, 1, 5, 1, 1).to(device=vae_device, dtype=vae_dtype)
        z_end = _extract_mode_latents(self.vae.encode(end_cond))[:, :, -1:].to(device=device, dtype=dtype)
        z_end = (z_end - latents_mean) * latents_std

        if batch_size > 1:
            if z_start.shape[0] == 1:
                z_start = z_start.repeat(batch_size, 1, 1, 1, 1)
            if z_end.shape[0] == 1:
                z_end = z_end.repeat(batch_size, 1, 1, 1, 1)

        latent_condition = torch.zeros(shape, dtype=dtype, device=device)
        latent_condition[:, :, 0:1] = z_start[:, :, 0:1]
        latent_condition[:, :, -1:] = z_end[:, :, -1:]

        first_frame_mask = torch.ones(
            1, 1, num_latent_frames, latent_height, latent_width, dtype=dtype, device=device
        )
        first_frame_mask[:, :, 0] = 0.0
        first_frame_mask[:, :, -1] = 0.0

        return latents, latent_condition, first_frame_mask

    pipe.prepare_latents = types.MethodType(_wan22_flf2v_prepare_latents, pipe)
    pipe._studio_wan22_flf2v_patched = True
    return pipe


def _switch_pipeline_class(
    handle: Dict[str, Any],
    target_cls: Any,
    vram_total_gb: Optional[float] = None,
) -> Any:
    """
    Conmuta entre clases hermanas de Diffusers (ej. WanImageToVideoPipeline <-> WanPipeline)
    filtrando estrictamente los argumentos admitidos por target_cls.__init__ para evitar
    TypeError ('image_encoder', 'image_processor', etc.), conservando dtypes mixtos
    (VAE float32 + Transformer bfloat16), preservando flags de configuración como
    expand_timesteps=True / boundary_ratio, y reactivando CPU Offload sin duplicar RAM.
    """
    pipe = handle["pipe"]
    if isinstance(pipe, target_cls):
        _attach_wan22_flf2v_support(pipe)
        return pipe

    if hasattr(pipe, "remove_all_hooks"):
        try:
            pipe.remove_all_hooks()
        except Exception:
            pass

    saved_components: Dict[str, Any] = dict(handle.get("all_components") or {})
    cur_components: Dict[str, Any] = dict(getattr(pipe, "components", {}) or {})
    for k, v in cur_components.items():
        if v is not None or k not in saved_components:
            saved_components[k] = v
    handle["all_components"] = saved_components

    cfg_dict: Dict[str, Any] = {}
    raw_cfg = getattr(pipe, "config", None)
    if isinstance(raw_cfg, dict):
        for k, v in raw_cfg.items():
            if not str(k).startswith("_"):
                cfg_dict[k] = v

    extra_init: Dict[str, Any] = dict(handle.get("extra_init_kwargs") or {})
    merged_candidates: Dict[str, Any] = {
        **cfg_dict,
        **extra_init,
        **saved_components,
    }

    sig = inspect.signature(target_cls.__init__)
    accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if accepts_var_kw:
        valid_kwargs = merged_candidates
    else:
        allowed = {
            name
            for name, param in sig.parameters.items()
            if name != "self"
            and param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }
        valid_kwargs = {k: v for k, v in merged_candidates.items() if k in allowed}

    new_pipe = target_cls(**valid_kwargs)
    _attach_wan22_flf2v_support(new_pipe)
    if vram_total_gb is None:
        vram_total_gb = float(query_gpu_info().get("vram_total_gb", 22.0))
    _apply_best_offload(new_pipe, vram_total_gb, float(handle.get("min_vram_gb", 14.0)))
    handle["pipe"] = new_pipe
    return new_pipe


def _read_model_index_json(base_dir: Path) -> Dict[str, Any]:
    idx_file = base_dir / "model_index.json"
    if idx_file.exists():
        try:
            return json.loads(idx_file.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _prepare_wan_lora_state_dict(raw_sd: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanea y normaliza un state_dict LoRA de Wan 2.1 / Wan 2.2 (ComfyUI / Kijai / LightX2V / FastWan / rCM / Diffusers)
    para que `WanLoraLoaderMixin.load_lora_weights(sd)` en Diffusers >=0.42 nunca falle por:
      - claves sin prefijo `diffusion_model.` (`blocks.0.self_attn...`)
      - escalares `.alpha` en capas fuera de `blocks.*` (`text_embedding`, `time_embedding`, `time_projection`, `head.head`, `img_emb`)
        que `_convert_non_diffusers_wan_lora_to_diffusers` no consume y provocarían ValueError al final
      - claves `vace_blocks.*` cuando se aplica un LoRA sobre un transformer sin bloques VACE o viceversa.
    """
    sd: Dict[str, Any] = dict(raw_sd)

    # Si ya está en formato nativo PEFT/Diffusers (`transformer.blocks...`), devolver tal cual
    if any(k.startswith("transformer.") and "lora_" in k for k in sd):
        return {k: v for k, v in sd.items() if "dora_scale" not in k}

    # Si está en formato Musubi (`lora_unet_...`), Diffusers tiene conversor propio
    if any(k.startswith("lora_unet_") for k in sd):
        return {k: v for k, v in sd.items() if "dora_scale" not in k}

    # Añadir prefijo `diffusion_model.` si viene con claves desnudas `blocks.0.self_attn...`
    if not any(k.startswith("diffusion_model.") for k in sd) and any(
        k.startswith("blocks.") and (".self_attn." in k or ".cross_attn." in k or ".ffn." in k)
        for k in sd
    ):
        sd = {f"diffusion_model.{k}": v for k, v in sd.items()}

    if not any(k.startswith("diffusion_model.") for k in sd):
        return sd

    # Pre-aplicar `.alpha` en capas fuera de `diffusion_model.blocks.*` (ej. rCM / LightX2V tienen
    # `diffusion_model.text_embedding.0.alpha` y `time_embedding.0.alpha` que Diffusers no hace .pop())
    for k in list(sd.keys()):
        if "dora_scale" in k or k.startswith("diffusion_model.vace_blocks."):
            sd.pop(k, None)
            continue
        if k.endswith(".alpha") and not k.startswith("diffusion_model.blocks."):
            alpha_tensor = sd.pop(k)
            base_prefix = k[: -len(".alpha")]
            down_key = (
                f"{base_prefix}.lora_A.weight"
                if f"{base_prefix}.lora_A.weight" in sd
                else f"{base_prefix}.lora_down.weight"
            )
            if down_key in sd:
                down_w = sd[down_key]
                rank = int(down_w.shape[0]) if hasattr(down_w, "shape") and len(down_w.shape) > 0 else 1
                alpha_val = float(alpha_tensor.item()) if hasattr(alpha_tensor, "item") else float(alpha_tensor)
                if rank > 0:
                    sd[down_key] = down_w * (alpha_val / float(rank))

    return sd


def _resolve_lora_safetensors_file(lora_id: str) -> Path:
    defn = catalog_manager.get_definition(lora_id)
    if not defn:
        raise ValueError(f"LoRA de vídeo no encontrado en el catálogo: {lora_id}")
    m_dir = catalog_manager.model_dir(lora_id)
    for art in defn.get("artifacts", []):
        sub = m_dir / art["subdir"]
        fn = art.get("filename") or art.get("verify_file")
        if fn:
            cand = sub / fn
            if cand.exists():
                return cand
        if sub.exists():
            found = sorted(sub.rglob("*.safetensors"))
            if found:
                return found[0]
    raise FileNotFoundError(f"No se encontró archivo .safetensors para el LoRA {lora_id}")


def _sync_video_lora_on_handle(
    handle: Dict[str, Any],
    pipe: Any,
    video_lora_id: Optional[str],
    video_lora_scale: float = 1.0,
    job: Optional[JobRecord] = None,
    pct_base: float = 8.0,
) -> Optional[Dict[str, Any]]:
    """
    Activa, ajusta escala o desactiva en caliente un LoRA acelerador en el pipeline de vídeo,
    quitando temporalmente los hooks de CPU Offload para evitar conflictos de dispositivo en PEFT.
    """
    clean_id = (video_lora_id or "").strip() or None
    cur_id = handle.get("active_lora_id")
    cur_scale = handle.get("active_lora_scale")
    vram_total = float(query_gpu_info().get("vram_total_gb", 22.0))
    min_vram = float(handle.get("min_vram_gb", 14.0))

    if not clean_id:
        if cur_id and hasattr(pipe, "unload_lora_weights"):
            if job:
                job.log("Desactivando LoRA acelerador para volver al modo estándar...", progress=pct_base)
            if hasattr(pipe, "remove_all_hooks"):
                try:
                    pipe.remove_all_hooks()
                except Exception:
                    pass
            pipe.unload_lora_weights()
            _apply_best_offload(pipe, vram_total, min_vram)
            handle["active_lora_id"] = None
            handle["active_lora_scale"] = None
        return None

    lora_defn = catalog_manager.get_definition(clean_id)
    if not lora_defn:
        raise ValueError(f"LoRA desconocido: {clean_id}")

    compat_models = lora_defn.get("compatible_models")
    loaded_model_id = (handle.get("defn") or {}).get("id")
    if compat_models and loaded_model_id and loaded_model_id not in compat_models:
        raise ValueError(
            f"El LoRA «{lora_defn['name']}» está diseñado para {', '.join(compat_models)} "
            f"y no es compatible con «{loaded_model_id}»."
        )

    if not catalog_manager.is_downloaded(clean_id):
        if job is not None:
            job.log(
                f"Descargando acelerador ligero «{lora_defn['name']}» ({lora_defn.get('estimated_gb', 0.3)} GB)...",
                progress=max(1.0, pct_base - 4.0),
                step="Descargando LoRA Turbo...",
            )
            catalog_manager.download_model(
                clean_id,
                job,
                progress_range=(max(1.0, pct_base - 4.0), pct_base),
            )
        else:
            raise RuntimeError(f"El LoRA «{lora_defn['name']}» aún no está descargado.")

    eff_scale = float(video_lora_scale) if video_lora_scale is not None else float(lora_defn.get("default_lora_scale", 1.0))

    if cur_id == clean_id:
        if cur_scale != eff_scale and hasattr(pipe, "set_adapters"):
            pipe.set_adapters(["vid_speed_lora"], adapter_weights=[eff_scale])
            handle["active_lora_scale"] = eff_scale
        return lora_defn

    lora_file = _resolve_lora_safetensors_file(clean_id)
    if job:
        job.log(
            f"Activando LoRA acelerador «{lora_defn['name']}» (escala {eff_scale:.2f})...",
            progress=pct_base,
            step="Aplicando LoRA Turbo...",
        )

    if hasattr(pipe, "remove_all_hooks"):
        try:
            pipe.remove_all_hooks()
        except Exception:
            pass

    if cur_id and hasattr(pipe, "unload_lora_weights"):
        pipe.unload_lora_weights()

    engine = handle.get("engine", "")
    if engine in ("wan_ti2v", "wan_t2v", "wan_vace", "wan_i2v_14b", "skyreels_v2"):
        import safetensors.torch  # type: ignore

        raw_sd = safetensors.torch.load_file(str(lora_file), device="cpu")
        clean_sd = _prepare_wan_lora_state_dict(raw_sd)
        pipe.load_lora_weights(clean_sd, adapter_name="vid_speed_lora")
    else:
        pipe.load_lora_weights(
            str(lora_file.parent),
            weight_name=lora_file.name,
            adapter_name="vid_speed_lora",
        )

    if hasattr(pipe, "set_adapters"):
        pipe.set_adapters(["vid_speed_lora"], adapter_weights=[eff_scale])

    _apply_best_offload(pipe, vram_total, min_vram)
    handle["active_lora_id"] = clean_id
    handle["active_lora_scale"] = eff_scale
    return lora_defn


def compute_official_video_params(
    defn: Optional[Dict[str, Any]],
    req_width: int,
    req_height: int,
    req_duration_sec: float,
    req_fps: int,
    req_steps: int,
    req_cfg: Optional[float],
    img_size: Optional[Tuple[int, int]] = None,  # (img_w, img_h)
) -> Dict[str, Any]:
    """
    Aplica las reglas geométricas y temporales oficiales de cada arquitectura (Wan 2.2 720p VAE 16×16,
    Wan 2.1 VACE 480p, SkyReels 540p, LTX-Video CFG 3.0, SANA motion score, CogVideoX 720×480@8fps)
    para evitar deformaciones, sub-muestreo de tokens o alucinaciones.
    """
    spec: Dict[str, Any] = dict((defn or {}).get("official_spec") or {})
    mod_val = int(spec.get("mod_value", 32))
    frame_rule = str(spec.get("frame_rule", "8k+1"))

    # 1) Resolución fija estricta (ej. CogVideoX-5B-I2V solo admite 720×480)
    fixed_res = spec.get("fixed_resolution")
    adjusted_notes: List[str] = []
    if isinstance(fixed_res, list) and len(fixed_res) == 2:
        width, height = int(fixed_res[0]), int(fixed_res[1])
        if (req_width, req_height) != (width, height):
            adjusted_notes.append(f"resolución fija oficial {width}×{height}")
    elif img_size is not None and img_size[0] > 0 and img_size[1] > 0 and spec.get("max_area"):
        ar = float(img_size[1]) / float(img_size[0])
        if ar >= 1.62 and isinstance(spec.get("portrait"), list):
            width, height = int(spec["portrait"][0]), int(spec["portrait"][1])
        elif ar <= (1.0 / 1.62) and isinstance(spec.get("landscape"), list):
            width, height = int(spec["landscape"][0]), int(spec["landscape"][1])
        elif 0.90 <= ar <= 1.11 and isinstance(spec.get("square"), list):
            width, height = int(spec["square"][0]), int(spec["square"][1])
        else:
            # Fórmula exacta de Diffusers: round(sqrt(max_area * ar)) // mod_val * mod_val
            max_area = float(spec["max_area"])
            ar = max(9.0 / 16.0, min(16.0 / 9.0, ar))
            height = max(mod_val * 8, (int(round(math.sqrt(max_area * ar))) // mod_val) * mod_val)
            width = max(mod_val * 8, (int(round(math.sqrt(max_area / ar))) // mod_val) * mod_val)
        if abs(width * height - req_width * req_height) > 65536:
            adjusted_notes.append(f"resolución nativa óptima {width}×{height} conservando proporción")
    else:
        width = max(256, min(1280, (int(req_width) // mod_val) * mod_val))
        height = max(256, min(1280, (int(req_height) // mod_val) * mod_val))
        min_area = spec.get("min_area")
        # Ej. Wan 2.2 TI2V-5B exige 720P (1280×704 o 704×1280); si llega 480×832 subir al bucket oficial
        if min_area and (width * height) < int(min_area):
            if height > width * 1.12 and isinstance(spec.get("portrait"), list):
                width, height = int(spec["portrait"][0]), int(spec["portrait"][1])
            elif width > height * 1.12 and isinstance(spec.get("landscape"), list):
                width, height = int(spec["landscape"][0]), int(spec["landscape"][1])
            elif isinstance(spec.get("square"), list):
                width, height = int(spec["square"][0]), int(spec["square"][1])
            adjusted_notes.append(f"escalado al bucket oficial 720P ({width}×{height})")

    # 2) FPS y número de fotogramas exacto por familia
    if frame_rule == "cogvideo49":
        fps = 8
        num_frames = 49
        if req_fps != 8:
            adjusted_notes.append("49 fotogramas @ 8 fps oficiales de CogVideoX")
    else:
        fps = int(req_fps) if int(req_fps) > 0 else int(spec.get("default_fps", 24))
        fps = max(8, min(30, fps))
        raw_frames = max(17, int(round(float(req_duration_sec) * fps)))
        if frame_rule == "4k+1":
            num_frames = int(round((raw_frames - 1) / 4.0)) * 4 + 1
            num_frames = max(17, min(161, num_frames))
        else:
            num_frames = int(round((raw_frames - 1) / 8.0)) * 8 + 1
            num_frames = max(17, min(161, num_frames))

    # 3) Pasos e intensidad de guía (CFG)
    steps = max(4, min(60, int(req_steps)))
    if req_cfg is not None and float(req_cfg) > 0:
        cfg = float(req_cfg)
    else:
        cfg = float(spec.get("default_cfg", (defn or {}).get("default_cfg", 5.0)))

    return {
        "width": width,
        "height": height,
        "fps": fps,
        "num_frames": num_frames,
        "steps": steps,
        "guidance_scale": cfg,
        "adjusted_notes": adjusted_notes,
    }


def _instantiate_video_pipeline(
    engine: str,
    base_dir: Path,
    defn: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Instancia la clase exacta de Diffusers con los tipos y schedulers oficiales verificados."""
    import torch  # type: ignore
    import diffusers  # type: ignore

    bdir = str(base_dir)
    idx_cfg = _read_model_index_json(base_dir)
    spec: Dict[str, Any] = dict((defn or {}).get("official_spec") or {})

    if engine == "wan_ti2v":
        from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanImageToVideoPipeline, WanPipeline  # type: ignore

        vae = AutoencoderKLWan.from_pretrained(bdir, subfolder="vae", torch_dtype=torch.float32)
        exp_ts = bool(idx_cfg.get("expand_timesteps", True))
        b_ratio = idx_cfg.get("boundary_ratio", None)
        pipe_i2v = WanImageToVideoPipeline.from_pretrained(
            bdir,
            vae=vae,
            torch_dtype=torch.bfloat16,
            expand_timesteps=exp_ts,
            boundary_ratio=b_ratio,
        )
        flow_shift = float(spec.get("flow_shift", 5.0))
        pipe_i2v.scheduler = UniPCMultistepScheduler.from_config(pipe_i2v.scheduler.config, flow_shift=flow_shift)
        _attach_wan22_flf2v_support(pipe_i2v)
        return {
            "pipe": pipe_i2v,
            "all_components": dict(pipe_i2v.components),
            "extra_init_kwargs": {"expand_timesteps": exp_ts, "boundary_ratio": b_ratio},
            "i2v_cls": WanImageToVideoPipeline,
            "t2v_cls": WanPipeline,
            "flf2v_cls": WanImageToVideoPipeline,
            "supported_submodes": ["t2v", "i2v", "flf2v"],
            "active_mode": "i2v",
        }

    if engine == "wan_t2v":
        from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanPipeline  # type: ignore

        vae = AutoencoderKLWan.from_pretrained(bdir, subfolder="vae", torch_dtype=torch.float32)
        pipe = WanPipeline.from_pretrained(bdir, vae=vae, torch_dtype=torch.bfloat16)
        flow_shift = float(spec.get("flow_shift", 3.0))
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=flow_shift)
        return {
            "pipe": pipe,
            "all_components": dict(pipe.components),
            "extra_init_kwargs": {},
            "i2v_cls": None,
            "t2v_cls": WanPipeline,
            "flf2v_cls": None,
            "supported_submodes": ["t2v"],
            "active_mode": "t2v",
        }

    if engine == "wan_vace":
        from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanVACEPipeline  # type: ignore

        vae = AutoencoderKLWan.from_pretrained(bdir, subfolder="vae", torch_dtype=torch.float32)
        pipe = WanVACEPipeline.from_pretrained(bdir, vae=vae, torch_dtype=torch.bfloat16)
        flow_shift = float(spec.get("flow_shift", 3.0))
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=flow_shift)
        return {
            "pipe": pipe,
            "all_components": dict(pipe.components),
            "extra_init_kwargs": {},
            "i2v_cls": WanVACEPipeline,
            "t2v_cls": WanVACEPipeline,
            "flf2v_cls": WanVACEPipeline,
            "supported_submodes": ["t2v", "i2v", "flf2v"],
            "active_mode": "unified",
        }

    if engine == "wan_i2v_14b":
        from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanImageToVideoPipeline  # type: ignore

        vae = AutoencoderKLWan.from_pretrained(bdir, subfolder="vae", torch_dtype=torch.float32)
        extra_kw: Dict[str, Any] = {}
        if (base_dir / "image_encoder").exists():
            from transformers import CLIPVisionModel  # type: ignore

            extra_kw["image_encoder"] = CLIPVisionModel.from_pretrained(
                bdir, subfolder="image_encoder", torch_dtype=torch.float32
            )
        pipe = WanImageToVideoPipeline.from_pretrained(bdir, vae=vae, torch_dtype=torch.bfloat16, **extra_kw)
        flow_shift = float(spec.get("flow_shift", 5.0))
        if pipe.config.boundary_ratio is None:
            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=flow_shift)
        return {
            "pipe": pipe,
            "all_components": dict(pipe.components),
            "extra_init_kwargs": {},
            "i2v_cls": WanImageToVideoPipeline,
            "t2v_cls": None,
            "flf2v_cls": WanImageToVideoPipeline,
            "supported_submodes": ["i2v", "flf2v"],
            "active_mode": "i2v",
        }

    if engine == "ltx_video":
        from diffusers import LTXConditionPipeline  # type: ignore

        pipe = LTXConditionPipeline.from_pretrained(bdir, torch_dtype=torch.bfloat16)
        return {
            "pipe": pipe,
            "all_components": dict(pipe.components),
            "extra_init_kwargs": {},
            "i2v_cls": LTXConditionPipeline,
            "t2v_cls": LTXConditionPipeline,
            "flf2v_cls": LTXConditionPipeline,
            "supported_submodes": ["t2v", "i2v", "flf2v"],
            "active_mode": "unified",
        }

    if engine == "ltx2_av":
        from diffusers import LTX2ConditionPipeline, LTX2Pipeline  # type: ignore

        pipe_cond = LTX2ConditionPipeline.from_pretrained(bdir, torch_dtype=torch.bfloat16)
        return {
            "pipe": pipe_cond,
            "all_components": dict(pipe_cond.components),
            "extra_init_kwargs": {},
            "i2v_cls": LTX2ConditionPipeline,
            "t2v_cls": LTX2Pipeline,
            "flf2v_cls": LTX2ConditionPipeline,
            "supported_submodes": ["t2v", "i2v", "flf2v"],
            "active_mode": "i2v",
        }

    if engine == "sana_video":
        from diffusers import SanaImageToVideoPipeline, SanaVideoPipeline  # type: ignore

        pipe_i2v = SanaImageToVideoPipeline.from_pretrained(bdir, torch_dtype=torch.bfloat16)
        if hasattr(pipe_i2v, "vae") and pipe_i2v.vae is not None:
            pipe_i2v.vae.to(torch.float32)
        if hasattr(pipe_i2v, "text_encoder") and pipe_i2v.text_encoder is not None:
            pipe_i2v.text_encoder.to(torch.bfloat16)
        return {
            "pipe": pipe_i2v,
            "all_components": dict(pipe_i2v.components),
            "extra_init_kwargs": {},
            "i2v_cls": SanaImageToVideoPipeline,
            "t2v_cls": SanaVideoPipeline,
            "flf2v_cls": None,
            "supported_submodes": ["t2v", "i2v"],
            "active_mode": "i2v",
        }

    if engine == "skyreels_v2":
        from diffusers import AutoencoderKLWan, SkyReelsV2ImageToVideoPipeline, UniPCMultistepScheduler  # type: ignore

        vae = AutoencoderKLWan.from_pretrained(bdir, subfolder="vae", torch_dtype=torch.float32)
        pipe = SkyReelsV2ImageToVideoPipeline.from_pretrained(bdir, vae=vae, torch_dtype=torch.bfloat16)
        flow_shift = float(spec.get("flow_shift", 5.0))
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=flow_shift)
        return {
            "pipe": pipe,
            "all_components": dict(pipe.components),
            "extra_init_kwargs": {},
            "i2v_cls": SkyReelsV2ImageToVideoPipeline,
            "t2v_cls": None,
            "flf2v_cls": SkyReelsV2ImageToVideoPipeline,
            "supported_submodes": ["i2v", "flf2v"],
            "active_mode": "i2v",
        }

    if engine == "cogvideox_i2v":
        from diffusers import CogVideoXImageToVideoPipeline  # type: ignore

        pipe = CogVideoXImageToVideoPipeline.from_pretrained(bdir, torch_dtype=torch.bfloat16)
        return {
            "pipe": pipe,
            "all_components": dict(pipe.components),
            "extra_init_kwargs": {},
            "i2v_cls": CogVideoXImageToVideoPipeline,
            "t2v_cls": None,
            "flf2v_cls": None,
            "supported_submodes": ["i2v"],
            "active_mode": "i2v",
        }

    # Fallback genérico para modelos Diffusers importados desde el Buscador HF en vivo
    pipe = diffusers.DiffusionPipeline.from_pretrained(bdir, torch_dtype=torch.bfloat16)
    call_params = inspect.signature(pipe.__call__).parameters
    sup: List[str] = []
    if "image" in call_params or "conditions" in call_params or "video" in call_params:
        sup.append("i2v")
    if "last_image" in call_params or "conditions" in call_params:
        sup.append("flf2v")
    img_p = call_params.get("image")
    if img_p is None or img_p.default is not inspect.Parameter.empty:
        sup.append("t2v")
    return {
        "pipe": pipe,
        "all_components": dict(getattr(pipe, "components", {}) or {}),
        "extra_init_kwargs": {},
        "i2v_cls": type(pipe) if "i2v" in sup else None,
        "t2v_cls": type(pipe) if "t2v" in sup else None,
        "flf2v_cls": type(pipe) if "flf2v" in sup else None,
        "supported_submodes": sup or ["t2v", "i2v"],
        "active_mode": "unified",
    }


def load_video_model_into_memory(
    model_id: str = "wan22-ti2v-5b",
    auto_download: bool = False,
    job: Optional[JobRecord] = None,
    progress_range: Tuple[float, float] = (0.0, 100.0),
) -> Dict[str, Any]:
    """Carga cualquier modelo de vídeo del catálogo liberando antes otros modelos de VRAM."""
    p_lo, p_hi = progress_range

    def _span(frac: float) -> float:
        return round(p_lo + max(0.0, min(1.0, frac)) * (p_hi - p_lo), 1)

    defn = catalog_manager.get_definition(model_id)
    if not defn:
        raise ValueError(f"Modelo de vídeo desconocido: {model_id}")

    if defn.get("builtin_no_download") or defn.get("engine") == "ffmpeg_motion":
        if job:
            job.log(
                "Animación FFmpeg está integrada y no necesita cargarse en VRAM.",
                level="success",
                progress=_span(1.0),
                step="Listo (0 GB VRAM)",
            )
        return {"model_id": model_id, "vram_allocated_gb": 0.0}

    cur_vid = memory_manager.get_slot("video")
    if cur_vid and cur_vid.get("model_id") == model_id:
        vram_now = query_gpu_info()["vram_used_gb"]
        if job:
            job.log(
                f"«{defn['name']}» ya está cargado y listo en RAM/VRAM ({vram_now} GB).",
                level="success",
                progress=_span(1.0),
                step="Modelo ya activo en memoria",
            )
        return {"model_id": model_id, "vram_allocated_gb": vram_now}

    try:
        import torch  # type: ignore
        import diffusers  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            f"Faltan librerías de vídeo/GPU ({exc}). En Colab se instalan automáticamente."
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No hay GPU activa en este entorno (modo CPU). "
            "Puedes usar «Cámara FFmpeg (0 GB)» o ejecutar en Google Colab con GPU."
        )

    if not catalog_manager.is_downloaded(model_id):
        if auto_download and job is not None:
            dl_start = _span(0.02)
            dl_end = _span(0.72)
            job.log(
                f"Descargando «{defn['name']}» de Hugging Face...",
                progress=dl_start,
                step="Descargando modelo de vídeo...",
            )
            catalog_manager.download_model(model_id, job, progress_range=(dl_start, dl_end))
            load_frac_start = 0.72
        else:
            raise RuntimeError(
                f"«{defn['name']}» no está descargado todavía. Pulsa «Descargar y Cargar» arriba."
            )
    else:
        load_frac_start = 0.04
        if job:
            job.log(
                "Archivos verificados en disco. Cargando tensores en RAM y VRAM...",
                progress=_span(load_frac_start),
                step="Preparando memoria de vídeo...",
            )

    def _map_pct(frac: float) -> float:
        return _span(load_frac_start + frac * (1.0 - load_frac_start))

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    for prev_slot in ("video", "image"):
        ps = memory_manager.get_slot(prev_slot)
        if ps:
            if job:
                job.log(
                    f"Liberando «{ps['name']}» de RAM/VRAM (sus archivos siguen en disco)...",
                    progress=_map_pct(0.08),
                    step=f"Liberando VRAM ({prev_slot})...",
                )
            memory_manager.unload_slot(prev_slot)

    flush_ram_and_vram()

    art = defn["artifacts"][0]
    base_dir = catalog_manager.model_dir(model_id) / art["subdir"]
    engine = defn.get("engine", "generic_diffusers_video")

    p_s = _map_pct(0.14)
    p_e = _map_pct(0.84)
    if job:
        job.log(
            f"Cargando pipeline «{defn['name']}» en bfloat16...",
            progress=p_s,
            step="Cargando tensores de vídeo...",
        )

    bundle: Optional[Dict[str, Any]] = None
    try:
        bundle = run_with_live_memory_progress(
            job=job,
            start_pct=p_s,
            end_pct=p_e,
            step_label=f"Cargando {defn['name']}",
            fn=lambda: _instantiate_video_pipeline(engine, base_dir, defn=defn),
            expected_seconds=55.0,
        )
        pipe = bundle["pipe"]
        gpu_info = query_gpu_info()
        vram_total = float(gpu_info.get("vram_total_gb", 0.0))
        min_vram = float(defn.get("min_vram_gb", 14.0))

        _apply_best_offload(pipe, vram_total, min_vram, job=job, pct=_map_pct(0.92))

        vram_gb = query_gpu_info()["vram_used_gb"]
        bundle["engine"] = engine
        bundle["min_vram_gb"] = min_vram
        bundle["defn"] = defn
        memory_manager.register_loaded(
            slot="video",
            model_id=model_id,
            name=defn["name"],
            handle=bundle,
        )
    except Exception:
        if bundle is not None:
            bundle.clear()
        flush_ram_and_vram()
        raise

    if job:
        job.log(
            f"Modelo de vídeo «{defn['name']}» listo en RAM/VRAM ({vram_gb} GB VRAM en reposo).",
            level="success",
            progress=_span(1.0),
            step="Modelo de vídeo listo",
        )
    return {"model_id": model_id, "vram_allocated_gb": vram_gb}


def _write_float_audio_wav(audio_arr: Any, sample_rate: int, wav_path: Path) -> None:
    import numpy as np  # type: ignore

    if hasattr(audio_arr, "detach"):
        arr = audio_arr.detach().float().cpu().numpy()
    else:
        arr = np.asarray(audio_arr, dtype=np.float32)

    arr = np.squeeze(arr)
    if arr.ndim == 2:
        if arr.shape[0] <= 2 and arr.shape[1] > arr.shape[0]:
            arr = arr.T
        channels = arr.shape[1]
    else:
        channels = 1

    pcm16 = (np.clip(arr, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm16.tobytes())


def _frame_to_uint8_rgb_array(frame: Any) -> Any:
    import numpy as np  # type: ignore

    if isinstance(frame, Image.Image):
        return np.asarray(frame.convert("RGB"), dtype=np.uint8)
    if hasattr(frame, "detach"):
        arr = frame.detach().float().cpu().numpy()
    else:
        arr = np.asarray(frame)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[2] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        if arr.min() < -0.05:
            arr = (arr + 1.0) * 0.5
        arr = (np.clip(arr, 0.0, 1.0) * 255.0).round().astype("uint8")
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.shape[2] == 4:
        arr = arr[:, :, :3]
    elif arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    return np.ascontiguousarray(arr, dtype=np.uint8)


def _frames_to_mp4(
    frames: Any,
    output_mp4: Path,
    fps: int = 16,
    audio_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Codifica fotogramas (np.ndarray 4D, torch.Tensor o lista PIL/np) a MP4 H.264 BT.709 vía pipe RGB24 directo."""
    import numpy as np  # type: ignore

    if frames is None:
        raise ValueError("La lista de fotogramas generados está vacía.")
    if hasattr(frames, "detach"):
        frames = frames.detach().float().cpu().numpy()
    if len(frames) == 0:
        raise ValueError("La lista de fotogramas generados está vacía.")

    ffmpeg_bin = get_ffmpeg_bin()
    output_mp4.parent.mkdir(parents=True, exist_ok=True)

    rgb_arrays: List[Any]
    raw_bytes_fast: Optional[bytes] = None
    h0 = 0
    w0 = 0

    if isinstance(frames, np.ndarray) and frames.ndim == 4:
        arr4 = frames
        if arr4.shape[-1] not in (1, 3, 4) and arr4.shape[1] in (1, 3, 4):
            arr4 = np.transpose(arr4, (0, 2, 3, 1))
        if arr4.dtype != np.uint8:
            if float(arr4.min()) < -0.05:
                arr4 = (arr4 + 1.0) * 0.5
            arr4 = (np.clip(arr4, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        if arr4.shape[-1] == 4:
            arr4 = arr4[:, :, :, :3]
        elif arr4.shape[-1] == 1:
            arr4 = np.repeat(arr4, 3, axis=3)
        arr4 = np.ascontiguousarray(arr4, dtype=np.uint8)
        h0, w0 = int(arr4.shape[1]), int(arr4.shape[2])
        raw_bytes_fast = arr4.tobytes()
        rgb_arrays = [arr4[i] for i in range(int(arr4.shape[0]))]
    else:
        rgb_arrays = [_frame_to_uint8_rgb_array(f) for f in frames]
        h0, w0 = int(rgb_arrays[0].shape[0]), int(rgb_arrays[0].shape[1])
        if all(int(a.shape[0]) == h0 and int(a.shape[1]) == w0 for a in rgb_arrays):
            raw_bytes_fast = b"".join(a.tobytes() for a in rgb_arrays)

    # Vía rápida: envío directo de memoria RGB24 por stdin (~2s en lugar de ~55s escribiendo PNGs)
    if raw_bytes_fast is not None and h0 > 0 and w0 > 0:
        cmd_raw = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{w0}x{h0}",
            "-framerate",
            str(fps),
            "-i",
            "-",
        ]
        if audio_path and audio_path.exists():
            cmd_raw += ["-i", str(audio_path), "-shortest", "-c:a", "aac", "-b:a", "192k"]
        cmd_raw += [
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-colorspace",
            "bt709",
            "-movflags",
            "+faststart",
            "-y",
            str(output_mp4),
        ]
        res_raw = subprocess.run(cmd_raw, input=raw_bytes_fast, capture_output=True, timeout=300)
        if res_raw.returncode == 0 and output_mp4.exists():
            return probe_media_info(output_mp4)

    # Fallback seguro en disco (compress_level=1 rápido)
    tmp_dir = Path(tempfile.mkdtemp(prefix="vframes_"))
    try:
        for i, arr in enumerate(rgb_arrays):
            Image.fromarray(arr, mode="RGB").save(
                tmp_dir / f"frame_{i:05d}.png", "PNG", compress_level=1
            )

        cmd = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-i",
            str(tmp_dir / "frame_%05d.png"),
        ]
        if audio_path and audio_path.exists():
            cmd += ["-i", str(audio_path), "-shortest", "-c:a", "aac", "-b:a", "192k"]
        cmd += [
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-colorspace",
            "bt709",
            "-movflags",
            "+faststart",
            "-y",
            str(output_mp4),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if res.returncode != 0:
            raise RuntimeError(f"Error codificando fotogramas a MP4: {res.stderr.strip()}")
        return probe_media_info(output_mp4)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def generate_video_clip(
    mode: str = "ai",  # "ai" | "ffmpeg_motion"
    video_submode: str = "auto",  # "auto" | "t2v" | "i2v" | "flf2v"
    model_id: Optional[str] = None,
    video_lora_id: Optional[str] = None,
    video_lora_scale: float = 1.0,
    prompt: str = "",
    negative_prompt: str = "",
    image_path: Optional[Path] = None,
    end_image_path: Optional[Path] = None,
    audio_path: Optional[Path] = None,
    width: int = 1280,
    height: int = 704,
    duration_sec: float = 5.0,
    fps: int = 24,
    steps: int = 45,
    guidance_scale: Optional[float] = None,
    seed: int = 0,
    motion_style: str = "zoom_in",
    job: Optional[JobRecord] = None,
) -> Dict[str, Any]:
    """
    Genera un clip de vídeo respetando los parámetros oficiales de cada arquitectura:
      - Texto a Vídeo (t2v)
      - Imagen a Vídeo simple (i2v)
      - Frame a Frame: Inicio → Fin (flf2v)
    Si `model_id` se indica y no coincide con el modelo en VRAM, valida primero compatibilidad de modo
    y lo descarga/carga automáticamente antes de inferir.
    """
    t0 = time.time()
    VIDEOS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts_slug = datetime.now().strftime("%Y%m%d_%H%M%S")

    has_start_img = bool(image_path and image_path.exists())
    has_end_img = bool(end_image_path and end_image_path.exists())

    submode = (video_submode or "auto").strip().lower()
    if submode not in ("t2v", "i2v", "flf2v"):
        if has_start_img and has_end_img:
            submode = "flf2v"
        elif has_start_img:
            submode = "i2v"
        else:
            submode = "t2v"

    if submode == "t2v":
        has_start_img = False
        has_end_img = False
    elif submode == "i2v":
        if not has_start_img:
            raise ValueError(
                "Modo «Imagen a Vídeo»: sube o selecciona una imagen inicial antes de generar "
                "(o cambia arriba a la pestaña «Texto a Vídeo»)."
            )
        has_end_img = False
    elif submode == "flf2v":
        if not has_start_img or not has_end_img:
            raise ValueError(
                "Modo «Frame a Frame (Inicio → Fin)»: sube o selecciona tanto el Fotograma Inicial como el Fotograma Final."
            )

    if mode == "ffmpeg_motion" or model_id == "ffmpeg-motion-cpu":
        if not has_start_img or image_path is None:
            raise ValueError(
                "La animación de cámara con FFmpeg necesita al menos una imagen inicial."
            )
        out_name = f"clip_ffmpeg_{ts_slug}.mp4"
        out_path = VIDEOS_OUT_DIR / out_name
        info = create_motion_clip_from_image(
            image_path=image_path,
            end_image_path=end_image_path if has_end_img else None,
            output_mp4=out_path,
            duration_sec=duration_sec,
            width=width,
            height=height,
            fps=fps,
            motion=motion_style,
            audio_path=audio_path,
            job=job,
        )
        label_mode = "Transición Frame→Frame" if has_end_img else f"Movimiento {motion_style}"
        meta = {
            "id": out_path.stem,
            "filename": out_name,
            "media_url": f"/media/outputs/videos/{out_name}",
            "type": "video",
            "video_submode": "flf2v" if has_end_img else "i2v",
            "model_id": "ffmpeg-motion-cpu",
            "model_name": f"Animación FFmpeg ({label_mode})",
            "prompt": prompt or f"{label_mode} sobre {image_path.name}",
            "duration_seconds": info["duration_seconds"],
            "width": info["width"] or width,
            "height": info["height"] or height,
            "fps": fps,
            "seed": 0,
            "steps": 1,
            "elapsed_seconds": round(time.time() - t0, 2),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        (VIDEOS_OUT_DIR / f"{out_path.stem}.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return meta

    # Comprobar compatibilidad del modelo elegido ANTES de gastar tiempo cargándolo
    slot = memory_manager.get_slot("video")
    target_model_id = (model_id or "").strip() or (slot["model_id"] if slot else "")
    if not target_model_id:
        raise RuntimeError(
            "No hay ningún modelo de vídeo seleccionado ni cargado en VRAM. "
            "Elige un modelo arriba o usa «Cámara FFmpeg 0 GB»."
        )

    target_defn = catalog_manager.get_definition(target_model_id)
    if target_defn:
        declared_modes = target_defn.get("video_modes") or ["i2v", "t2v"]
        if submode not in declared_modes:
            sub_es = {
                "flf2v": "Frame a Frame (Inicio → Fin)",
                "i2v": "Imagen a Vídeo",
                "t2v": "Texto a Vídeo",
            }.get(submode, submode)
            raise ValueError(
                f"El modelo «{target_defn['name']}» no soporta el modo «{sub_es}». "
                "Para Frame a Frame con IA elige arriba «Wan 2.2 TI2V-5B HD 720P», «Wan 2.1 VACE 1.3B», «SkyReels V2 1.3B 540P» o «LTX-Video 0.9.5 2B»."
            )

    infer_pct_start = 10.0
    if slot is None or slot.get("model_id") != target_model_id:
        if job:
            job.log(
                f"Preparando automáticamente el modelo seleccionado («{target_defn['name'] if target_defn else target_model_id}») antes de generar...",
                progress=1.0,
                step="Preparando modelo de vídeo...",
            )
        load_video_model_into_memory(
            model_id=target_model_id,
            auto_download=True,
            job=job,
            progress_range=(1.0, 42.0),
        )
        slot = memory_manager.get_slot("video")
        infer_pct_start = 44.0

    if not slot:
        raise RuntimeError("No fue posible activar el modelo de vídeo en memoria.")

    import torch  # type: ignore

    handle = slot["handle"]
    defn = handle.get("defn") or catalog_manager.get_definition(slot["model_id"]) or {}
    engine = handle.get("engine", "generic_diffusers_video")
    supported = handle.get("supported_submodes") or ["t2v", "i2v"]

    if submode == "flf2v":
        target_cls = handle.get("flf2v_cls")
        if target_cls is None or "flf2v" not in supported:
            raise ValueError(
                f"El modelo cargado («{slot['name']}») no soporta Frame a Frame (Inicio → Fin) nativo. "
                "Elige «Wan 2.1 VACE 1.3B», «SkyReels V2 1.3B 540P» o «LTX-Video 0.9.5 2B»."
            )
        pipe = _switch_pipeline_class(handle, target_cls)
        handle["active_mode"] = "flf2v"
    elif submode == "i2v":
        target_cls = handle.get("i2v_cls")
        if target_cls is None or "i2v" not in supported:
            raise ValueError(
                f"El modelo cargado («{slot['name']}») es exclusivo de Texto a Vídeo y no recibe imagen inicial."
            )
        pipe = _switch_pipeline_class(handle, target_cls)
        handle["active_mode"] = "i2v"
    else:
        target_cls = handle.get("t2v_cls")
        if target_cls is None or "t2v" not in supported:
            raise ValueError(
                f"El modelo cargado («{slot['name']}») requiere una imagen inicial y no soporta Texto a Vídeo puro."
            )
        pipe = _switch_pipeline_class(handle, target_cls)
        handle["active_mode"] = "t2v"

    active_lora_defn = _sync_video_lora_on_handle(
        handle=handle,
        pipe=pipe,
        video_lora_id=video_lora_id,
        video_lora_scale=video_lora_scale,
        job=job,
        pct_base=max(4.0, infer_pct_start - 2.0),
    )

    actual_seed = int(seed) if int(seed) > 0 else random.randint(1, 2_147_483_647)

    if audio_path and audio_path.exists():
        a_info = probe_media_info(audio_path)
        if a_info["duration_seconds"] > 0.3:
            duration_sec = min(8.0, a_info["duration_seconds"])

    raw_start_img: Optional[Image.Image] = None
    img_size: Optional[Tuple[int, int]] = None
    if has_start_img and image_path is not None:
        raw_start_img = Image.open(image_path).convert("RGB")
        img_size = raw_start_img.size

    eff_req_cfg = (
        guidance_scale
        if guidance_scale is not None
        else (
            float(active_lora_defn["recommended_cfg"])
            if active_lora_defn and active_lora_defn.get("recommended_cfg") is not None
            else None
        )
    )
    eff = compute_official_video_params(
        defn=defn,
        req_width=width,
        req_height=height,
        req_duration_sec=duration_sec,
        req_fps=fps,
        req_steps=steps,
        req_cfg=eff_req_cfg,
        img_size=img_size,
    )
    width = eff["width"]
    height = eff["height"]
    fps = eff["fps"]
    num_frames = eff["num_frames"]
    steps = eff["steps"]
    eff_cfg = eff["guidance_scale"]

    if eff["adjusted_notes"] and job:
        job.log(
            f"Ajuste automático según documentación oficial de «{slot['name']}»: {', '.join(eff['adjusted_notes'])}.",
            progress=infer_pct_start - 1.0,
        )

    # Redimensionado center-crop sin deformar proporciones ni caras/extremidades
    img_pil: Optional[Image.Image] = None
    end_pil: Optional[Image.Image] = None
    if raw_start_img is not None:
        img_pil = _fit_and_crop_rgb(raw_start_img, width, height)
    if has_end_img and end_image_path is not None:
        raw_end = Image.open(end_image_path).convert("RGB")
        end_pil = _fit_and_crop_rgb(raw_end, width, height)

    flush_ram_and_vram()
    generator = torch.Generator(device="cuda").manual_seed(actual_seed)
    step_ts = {"t": time.time()}

    submode_es = {
        "t2v": "Texto → Vídeo",
        "i2v": "Imagen → Vídeo",
        "flf2v": "Frame a Frame (Inicio → Fin)",
    }.get(submode, submode)

    pct_span = 90.0 - infer_pct_start

    def _step_cb(_p: Any, idx: int, _ts: Any, kw: Dict[str, Any]) -> Dict[str, Any]:
        if job:
            job.check_cancelled()
            now = time.time()
            dt = max(0.01, now - step_ts["t"])
            step_ts["t"] = now
            cur = idx + 1
            pct = infer_pct_start + (cur / steps) * pct_span
            job.log(
                f"Vídeo IA [{submode_es}] paso {cur}/{steps} ({dt:.2f} s/paso)",
                progress=pct,
                step=f"{submode_es}: paso {cur}/{steps}",
                speed_text=f"{dt:.2f} s/paso",
            )
            if cur >= steps:
                job.log(
                    f"Difusión completada. Decodificando {num_frames} fotogramas HD en el VAE 3D...",
                    progress=90.5,
                    step="Decodificando vídeo en VAE 3D...",
                )
        return kw

    if job:
        lora_tag = f" + LoRA «{active_lora_defn['name'].split('(')[0].strip()}»" if active_lora_defn else ""
        if submode == "flf2v" and image_path is not None and end_image_path is not None:
            job.log(
                f"🎞️ Frame a Frame activo: Inicio=«{image_path.name}» → Final=«{end_image_path.name}» (ambos fotogramas anclados en t=0).",
                progress=max(1.0, infer_pct_start - 0.5),
            )
        job.log(
            f"Generando [{submode_es}] con «{slot['name']}»{lora_tag} ({width}×{height}, {num_frames} fotogramas @ {fps} fps, {steps} pasos, CFG {eff_cfg}, semilla {actual_seed})...",
            progress=infer_pct_start,
            step=f"Inferencia {submode_es} 0/{steps}...",
        )

    call_sig = inspect.signature(pipe.__call__).parameters
    default_prompt_by_mode = {
        "t2v": "cinematic high quality realistic video, natural smooth coherent motion, sharp focus",
        "i2v": "natural coherent motion, smooth realistic movement, consistent anatomy and lighting",
        "flf2v": "smooth natural coherent transition from the starting pose and scene to the final pose and scene",
    }
    clean_prompt = (prompt or "").strip() or default_prompt_by_mode.get(submode, default_prompt_by_mode["i2v"])

    # SANA-Video exige sufijo oficial " motion score: 30." en su entrenamiento
    spec = defn.get("official_spec") or {}
    if engine == "sana_video" and "motion score:" not in clean_prompt.lower():
        m_score = int(spec.get("motion_score", 30))
        clean_prompt = f"{clean_prompt.rstrip('. ')}. motion score: {m_score}."

    # Elegir negative_prompt oficial específico de la familia del modelo
    if (negative_prompt or "").strip():
        eff_neg = negative_prompt.strip()
    elif engine in ("wan_ti2v", "wan_t2v", "wan_vace", "wan_i2v_14b", "skyreels_v2"):
        eff_neg = OFFICIAL_NEGATIVE_PROMPTS["wan"]
    elif engine == "sana_video":
        eff_neg = OFFICIAL_NEGATIVE_PROMPTS["sana"]
    elif engine in ("ltx_video", "ltx2_av"):
        eff_neg = OFFICIAL_NEGATIVE_PROMPTS["ltx"]
    else:
        eff_neg = OFFICIAL_NEGATIVE_PROMPTS["default"]

    kwargs: Dict[str, Any] = {
        "prompt": clean_prompt,
        "height": height,
        "width": width,
        "num_inference_steps": int(steps),
        "guidance_scale": float(eff_cfg),
        "generator": generator,
    }
    if "negative_prompt" in call_sig:
        kwargs["negative_prompt"] = eff_neg
    if "frames" in call_sig and "num_frames" not in call_sig:
        kwargs["frames"] = num_frames
    elif "num_frames" in call_sig:
        kwargs["num_frames"] = num_frames
    if "frame_rate" in call_sig:
        kwargs["frame_rate"] = float(fps) if engine == "ltx2_av" else int(fps)
    if engine == "cogvideox_i2v" and "use_dynamic_cfg" in call_sig:
        kwargs["use_dynamic_cfg"] = True
    if "callback_on_step_end" in call_sig:
        kwargs["callback_on_step_end"] = _step_cb

    # Condicionamiento específico verificado para cada arquitectura
    if engine == "wan_vace":
        if submode == "i2v" and img_pil is not None:
            neutral = Image.new("RGB", (width, height), (128, 128, 128))
            m_keep = Image.new("L", (width, height), 0)
            m_gen = Image.new("L", (width, height), 255)
            kwargs["video"] = [img_pil] + [neutral] * (num_frames - 1)
            kwargs["mask"] = [m_keep] + [m_gen] * (num_frames - 1)
        elif submode == "flf2v" and img_pil is not None and end_pil is not None:
            neutral = Image.new("RGB", (width, height), (128, 128, 128))
            m_keep = Image.new("L", (width, height), 0)
            m_gen = Image.new("L", (width, height), 255)
            kwargs["video"] = [img_pil] + [neutral] * max(1, num_frames - 2) + [end_pil]
            kwargs["mask"] = [m_keep] + [m_gen] * max(1, num_frames - 2) + [m_keep]
    elif engine == "ltx_video":
        from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition  # type: ignore

        if submode == "i2v" and img_pil is not None:
            kwargs["conditions"] = [LTXVideoCondition(image=img_pil, frame_index=0)]
        elif submode == "flf2v" and img_pil is not None and end_pil is not None:
            kwargs["conditions"] = [
                LTXVideoCondition(image=img_pil, frame_index=0),
                LTXVideoCondition(image=end_pil, frame_index=num_frames - 1),
            ]
    elif engine == "ltx2_av":
        if submode in ("i2v", "flf2v") and img_pil is not None:
            from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition  # type: ignore

            conds = [LTX2VideoCondition(frames=img_pil, index=0, strength=1.0)]
            if submode == "flf2v" and end_pil is not None:
                conds.append(LTX2VideoCondition(frames=end_pil, index=-1, strength=1.0))
            kwargs["conditions"] = conds
    else:
        if submode in ("i2v", "flf2v") and img_pil is not None:
            if "reference_images" in call_sig and "image" not in call_sig:
                kwargs["reference_images"] = [img_pil]
            elif "image" in call_sig:
                kwargs["image"] = img_pil
        if submode == "flf2v" and end_pil is not None and "last_image" in call_sig:
            kwargs["last_image"] = end_pil

    generated_audio_wav: Optional[Path] = None
    try:
        if engine == "ltx2_av":
            kwargs["output_type"] = "np"
            kwargs["return_dict"] = False
            res_tuple = pipe(**kwargs)
            video_out = res_tuple[0]
            audio_out = res_tuple[1] if len(res_tuple) > 1 else None
            frames = video_out[0]
            if audio_out is not None and not (audio_path and audio_path.exists()):
                sr = int(
                    getattr(getattr(getattr(pipe, "vocoder", None), "config", None), "output_sampling_rate", 24000)
                )
                generated_audio_wav = VIDEOS_OUT_DIR / f"_tmp_ltx2_audio_{ts_slug}.wav"
                _write_float_audio_wav(audio_out[0], sr, generated_audio_wav)
        else:
            try:
                output = pipe(**kwargs)
            except TypeError as exc:
                if "callback_on_step_end" in str(exc) and "callback_on_step_end" in kwargs:
                    kwargs.pop("callback_on_step_end", None)
                    output = pipe(**kwargs)
                else:
                    raise
            frames = output.frames[0] if hasattr(output, "frames") else output.images

        out_name = f"vid_{ts_slug}_{actual_seed % 10000:04d}.mp4"
        out_path = VIDEOS_OUT_DIR / out_name

        if job:
            job.log("Codificando fotogramas a MP4 H.264 BT.709 con FFmpeg...", progress=92.0, step="Codificando MP4...")

        mux_audio = audio_path if (audio_path and audio_path.exists()) else generated_audio_wav
        info = _frames_to_mp4(frames=frames, output_mp4=out_path, fps=fps, audio_path=mux_audio)
    finally:
        if generated_audio_wav and generated_audio_wav.exists():
            generated_audio_wav.unlink(missing_ok=True)
        flush_ram_and_vram()

    meta = {
        "id": out_path.stem,
        "filename": out_name,
        "media_url": f"/media/outputs/videos/{out_name}",
        "type": "video",
        "video_submode": submode,
        "model_id": slot["model_id"],
        "model_name": slot["name"],
        "active_lora_id": active_lora_defn["id"] if active_lora_defn else None,
        "active_lora_name": active_lora_defn["name"] if active_lora_defn else None,
        "prompt": clean_prompt,
        "negative_prompt": eff_neg,
        "duration_seconds": info["duration_seconds"],
        "num_frames": num_frames,
        "width": info["width"] or width,
        "height": info["height"] or height,
        "fps": fps,
        "guidance_scale": eff_cfg,
        "seed": actual_seed,
        "steps": steps,
        "elapsed_seconds": round(time.time() - t0, 2),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    (VIDEOS_OUT_DIR / f"{out_path.stem}.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return meta


def _fit_and_crop_rgb(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Ajusta una imagen al tamaño exacto (center-crop suave) sin estirar caras ni extremidades."""
    src_w, src_h = img.size
    if src_w <= 0 or src_h <= 0:
        return img.resize((target_w, target_h), Image.Resampling.LANCZOS)
    scale = max(target_w / src_w, target_h / src_h)
    new_w = max(target_w, int(round(src_w * scale)))
    new_h = max(target_h, int(round(src_h * scale)))
    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    left = max(0, (new_w - target_w) // 2)
    top = max(0, (new_h - target_h) // 2)
    return resized.crop((left, top, left + target_w, top + target_h))
