import inspect
import json
import os
import random
import shutil
import subprocess
import tempfile
import time
import wave
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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
                pipe.vae.enable_tiling()
            except Exception:
                pass


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


def _instantiate_video_pipeline(engine: str, base_dir: Path) -> Dict[str, Any]:
    """Instancia la clase exacta de Diffusers con los tipos precisos (ej. VAE float32 en Wan/Sana/SkyReels)."""
    import torch  # type: ignore
    import diffusers  # type: ignore

    bdir = str(base_dir)
    idx_cfg = _read_model_index_json(base_dir)

    if engine == "wan_ti2v":
        from diffusers import AutoencoderKLWan, WanImageToVideoPipeline, WanPipeline  # type: ignore

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
        return {
            "pipe": pipe_i2v,
            "all_components": dict(pipe_i2v.components),
            "extra_init_kwargs": {"expand_timesteps": exp_ts, "boundary_ratio": b_ratio},
            "i2v_cls": WanImageToVideoPipeline,
            "t2v_cls": WanPipeline,
            "flf2v_cls": None,  # Wan 2.2 TI2V (expand_timesteps=True) no condiciona last_image
            "supported_submodes": ["t2v", "i2v"],
            "active_mode": "i2v",
        }

    if engine == "wan_t2v":
        from diffusers import AutoencoderKLWan, WanPipeline  # type: ignore

        vae = AutoencoderKLWan.from_pretrained(bdir, subfolder="vae", torch_dtype=torch.float32)
        pipe = WanPipeline.from_pretrained(bdir, vae=vae, torch_dtype=torch.bfloat16)
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
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)
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
        from diffusers import AutoencoderKLWan, WanImageToVideoPipeline  # type: ignore

        vae = AutoencoderKLWan.from_pretrained(bdir, subfolder="vae", torch_dtype=torch.float32)
        extra_kw: Dict[str, Any] = {}
        if (base_dir / "image_encoder").exists():
            from transformers import CLIPVisionModel  # type: ignore

            extra_kw["image_encoder"] = CLIPVisionModel.from_pretrained(
                bdir, subfolder="image_encoder", torch_dtype=torch.float32
            )
        pipe = WanImageToVideoPipeline.from_pretrained(bdir, vae=vae, torch_dtype=torch.bfloat16, **extra_kw)
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

        # LTXConditionPipeline soporta nativamente T2V (sin conditions), I2V (frame 0) y Frame a Frame (frame 0 + N-1)
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
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=5.0)
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

    # Fallback genérico para cualquier modelo Diffusers importado desde el Explorador HF en vivo
    pipe = diffusers.DiffusionPipeline.from_pretrained(bdir, torch_dtype=torch.bfloat16)
    call_params = inspect.signature(pipe.__call__).parameters
    sup: List[str] = []
    if "image" in call_params or "conditions" in call_params or "video" in call_params:
        sup.append("i2v")
    if "last_image" in call_params or "conditions" in call_params:
        sup.append("flf2v")
    # Comprobar si image es opcional para permitir t2v
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
) -> Dict[str, Any]:
    """Carga cualquier modelo de vídeo del catálogo (0-100%) liberando antes otros modelos de VRAM."""
    defn = catalog_manager.get_definition(model_id)
    if not defn:
        raise ValueError(f"Modelo de vídeo desconocido: {model_id}")

    if defn.get("builtin_no_download") or defn.get("engine") == "ffmpeg_motion":
        if job:
            job.log(
                "Animación FFmpeg está integrada y no necesita cargarse en VRAM.",
                level="success",
                progress=100.0,
                step="Listo (0 GB VRAM)",
            )
        return {"model_id": model_id, "vram_allocated_gb": 0.0}

    # Si este mismo modelo ya está cargado en memoria, no duplicar RAM
    cur_vid = memory_manager.get_slot("video")
    if cur_vid and cur_vid.get("model_id") == model_id:
        vram_now = query_gpu_info()["vram_used_gb"]
        if job:
            job.log(
                f"«{defn['name']}» ya está cargado y listo en RAM/VRAM ({vram_now} GB).",
                level="success",
                progress=100.0,
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
            "Puedes usar «Animación Instantánea FFmpeg (0 GB)» o ejecutar en Google Colab con GPU."
        )

    if not catalog_manager.is_downloaded(model_id):
        if auto_download and job is not None:
            job.log(
                f"Fase 1/2: Descargando «{defn['name']}» de Hugging Face (0% → 72%)...",
                progress=1.0,
                step="Fase 1/2: Descargando modelo de vídeo...",
            )
            catalog_manager.download_model(model_id, job, progress_range=(1.0, 72.0))
            load_p_start = 72.0
        else:
            raise RuntimeError(
                f"«{defn['name']}» no está descargado todavía. Pulsa «Descargar y Cargar» arriba."
            )
    else:
        load_p_start = 4.0
        if job:
            job.log(
                "Archivos verificados en disco. Cargando tensores en RAM y VRAM...",
                progress=load_p_start,
                step="Preparando memoria de vídeo...",
            )

    def _map_pct(frac: float) -> float:
        return round(load_p_start + frac * (100.0 - load_p_start), 1)

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Liberar cualquier modelo previo (imagen o vídeo anterior) antes de alojar el nuevo
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
    p_e = _map_pct(0.82)
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
            fn=lambda: _instantiate_video_pipeline(engine, base_dir),
            expected_seconds=55.0,
        )
        pipe = bundle["pipe"]
        gpu_info = query_gpu_info()
        vram_total = float(gpu_info.get("vram_total_gb", 0.0))
        min_vram = float(defn.get("min_vram_gb", 14.0))

        _apply_best_offload(pipe, vram_total, min_vram, job=job, pct=_map_pct(0.90))

        vram_gb = query_gpu_info()["vram_used_gb"]
        bundle["engine"] = engine
        bundle["min_vram_gb"] = min_vram
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
            progress=100.0,
            step="Modelo de vídeo listo (100%)",
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


def _frames_to_mp4(
    frames: List[Any],
    output_mp4: Path,
    fps: int = 16,
    audio_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Codifica una secuencia de fotogramas PIL/numpy/tensor a MP4 H.264 BT.709 con FFmpeg."""
    import numpy as np  # type: ignore

    ffmpeg_bin = get_ffmpeg_bin()
    tmp_dir = Path(tempfile.mkdtemp(prefix="vframes_"))
    try:
        for i, frame in enumerate(frames):
            if isinstance(frame, Image.Image):
                pil_f = frame.convert("RGB")
            else:
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
                pil_f = Image.fromarray(arr).convert("RGB")
            pil_f.save(tmp_dir / f"frame_{i:05d}.png", "PNG")

        output_mp4.parent.mkdir(parents=True, exist_ok=True)
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
            "19",
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
    prompt: str = "",
    negative_prompt: str = "",
    image_path: Optional[Path] = None,
    end_image_path: Optional[Path] = None,
    audio_path: Optional[Path] = None,
    width: int = 832,
    height: int = 480,
    duration_sec: float = 4.0,
    fps: int = 16,
    steps: int = 24,
    guidance_scale: float = 5.0,
    seed: int = 0,
    motion_style: str = "zoom_in",
    job: Optional[JobRecord] = None,
) -> Dict[str, Any]:
    """
    Genera un clip de vídeo soportando los 3 modos creativos:
      - Texto a Vídeo (t2v)
      - Imagen a Vídeo simple (i2v)
      - Frame a Frame: Inicio → Fin (flf2v)
    además de Animación de Cámara / Transición instantánea con FFmpeg (0 GB VRAM).
    """
    t0 = time.time()
    VIDEOS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts_slug = datetime.now().strftime("%Y%m%d_%H%M%S")

    has_start_img = bool(image_path and image_path.exists())
    has_end_img = bool(end_image_path and end_image_path.exists())

    # Resolver submodo efectivo cuando llega en "auto"
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

    if mode == "ffmpeg_motion":
        if not has_start_img or image_path is None:
            raise ValueError(
                "La animación de cámara con FFmpeg necesita al menos una imagen inicial (sube una foto o cambia a IA Texto a Vídeo)."
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

    slot = memory_manager.get_slot("video")
    if not slot:
        raise RuntimeError(
            "No hay ningún modelo de vídeo cargado en RAM/VRAM. "
            "Elige un modelo arriba y pulsa «Descargar y Cargar» (o usa «Cámara FFmpeg 0 GB»)."
        )

    import torch  # type: ignore

    handle = slot["handle"]
    engine = handle.get("engine", "generic_diffusers_video")
    supported = handle.get("supported_submodes") or ["t2v", "i2v"]

    # Validar y conmutar la clase de pipeline exacta según submodo ("t2v", "i2v", "flf2v")
    if submode == "flf2v":
        target_cls = handle.get("flf2v_cls")
        if target_cls is None or "flf2v" not in supported:
            raise ValueError(
                f"El modelo cargado («{slot['name']}») no soporta Frame a Frame (Inicio → Fin) nativo. "
                "Para Frame a Frame con IA carga «LTX-Video 0.9.5 2B», «Wan 2.1 VACE 1.3B» o «SkyReels V2 1.3B» "
                "(o usa «Cámara FFmpeg 0 GB» para transición instantánea)."
            )
        pipe = _switch_pipeline_class(handle, target_cls)
        handle["active_mode"] = "flf2v"
    elif submode == "i2v":
        target_cls = handle.get("i2v_cls")
        if target_cls is None or "i2v" not in supported:
            raise ValueError(
                f"El modelo cargado («{slot['name']}») es exclusivo de Texto a Vídeo y no recibe imagen inicial. "
                "Cambia a «Texto a Vídeo» o carga «Wan 2.2 TI2V-5B», «LTX-Video 0.9.5 2B» o «Wan 2.1 VACE 1.3B»."
            )
        pipe = _switch_pipeline_class(handle, target_cls)
        handle["active_mode"] = "i2v"
    else:  # submode == "t2v"
        target_cls = handle.get("t2v_cls")
        if target_cls is None or "t2v" not in supported:
            raise ValueError(
                f"El modelo cargado («{slot['name']}») requiere una imagen inicial y no soporta Texto a Vídeo puro. "
                "Cambia a «Imagen a Vídeo» o carga «Wan 2.2 TI2V-5B», «LTX-Video 0.9.5 2B» o «SANA-Video 2B»."
            )
        pipe = _switch_pipeline_class(handle, target_cls)
        handle["active_mode"] = "t2v"

    actual_seed = int(seed) if int(seed) > 0 else random.randint(1, 2_147_483_647)

    if audio_path and audio_path.exists():
        a_info = probe_media_info(audio_path)
        if a_info["duration_seconds"] > 0.3:
            duration_sec = min(10.0, a_info["duration_seconds"])

    # Alinear ancho y alto a múltiplos de 32 (compatible con Wan 2.2 VAE 16x2=32, LTX 32, Sana, CogVideoX)
    width = max(256, min(1280, (int(width) // 32) * 32))
    height = max(256, min(1280, (int(height) // 32) * 32))
    fps = max(8, min(30, int(fps)))
    steps = max(4, min(60, int(steps)))

    img_pil: Optional[Image.Image] = None
    end_pil: Optional[Image.Image] = None
    if has_start_img and image_path is not None:
        raw_img = Image.open(image_path).convert("RGB")
        if raw_img.height > raw_img.width * 1.12 and width > height:
            width, height = height, width
        elif raw_img.width > raw_img.height * 1.12 and height > width:
            width, height = height, width
        img_pil = raw_img.resize((width, height), Image.Resampling.LANCZOS)

    if has_end_img and end_image_path is not None:
        raw_end = Image.open(end_image_path).convert("RGB")
        end_pil = _fit_and_crop_rgb(raw_end, width, height)

    # Alinear fotogramas a 8*k + 1 (cumple simultáneamente 8k+1 de LTX/LTX-2 y 4k+1 de Wan/Sana/CogVideoX/SkyReels)
    raw_frames = max(17, int(round(duration_sec * fps)))
    num_frames = ((raw_frames - 1) // 8) * 8 + 1

    flush_ram_and_vram()
    generator = torch.Generator(device="cuda").manual_seed(actual_seed)
    step_ts = {"t": time.time()}

    submode_es = {
        "t2v": "Texto → Vídeo",
        "i2v": "Imagen → Vídeo",
        "flf2v": "Frame a Frame (Inicio → Fin)",
    }.get(submode, submode)

    def _step_cb(_p: Any, idx: int, _ts: Any, kw: Dict[str, Any]) -> Dict[str, Any]:
        if job:
            job.check_cancelled()
            now = time.time()
            dt = max(0.01, now - step_ts["t"])
            step_ts["t"] = now
            cur = idx + 1
            pct = 12.0 + (cur / steps) * 76.0
            job.log(
                f"Vídeo IA [{submode_es}] paso {cur}/{steps} ({dt:.2f} s/paso)",
                progress=pct,
                step=f"{submode_es}: paso {cur}/{steps}",
                speed_text=f"{dt:.2f} s/paso",
            )
        return kw

    if job:
        job.log(
            f"Generando [{submode_es}] con «{slot['name']}» ({width}×{height}, {num_frames} fotogramas @ {fps} fps, {steps} pasos, semilla {actual_seed})...",
            progress=10.0,
            step=f"Inferencia {submode_es} 0/{steps}...",
        )

    call_sig = inspect.signature(pipe.__call__).parameters
    default_prompt_by_mode = {
        "t2v": "cinematic high quality video, natural smooth motion, rich detail",
        "i2v": "cinematic smooth camera motion, natural subtle movement, high detail",
        "flf2v": "smooth natural cinematic transition from the first scene to the final scene, coherent motion",
    }
    clean_prompt = (prompt or "").strip() or default_prompt_by_mode.get(submode, default_prompt_by_mode["i2v"])
    default_neg = (
        negative_prompt.strip()
        or "worst quality, blurry, jittery, distorted, static, deformed, overexposed, artifacts"
    )

    kwargs: Dict[str, Any] = {
        "prompt": clean_prompt,
        "height": height,
        "width": width,
        "num_inference_steps": int(steps),
        "guidance_scale": float(guidance_scale),
        "generator": generator,
    }
    if "negative_prompt" in call_sig:
        kwargs["negative_prompt"] = default_neg
    if "frames" in call_sig and "num_frames" not in call_sig:
        kwargs["frames"] = num_frames
    elif "num_frames" in call_sig:
        kwargs["num_frames"] = num_frames
    if "frame_rate" in call_sig:
        kwargs["frame_rate"] = float(fps) if engine == "ltx2_av" else int(fps)
    if "callback_on_step_end" in call_sig:
        kwargs["callback_on_step_end"] = _step_cb

    # Condicionamiento específico según arquitectura y submodo (T2V / I2V / FLF2V)
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
                # Solo reintentar sin callback_on_step_end si la firma de una clase personalizada no lo aceptó
                if "callback_on_step_end" in str(exc) and "callback_on_step_end" in kwargs:
                    kwargs.pop("callback_on_step_end", None)
                    output = pipe(**kwargs)
                else:
                    raise
            frames = output.frames[0] if hasattr(output, "frames") else output.images

        out_name = f"vid_{ts_slug}_{actual_seed % 10000:04d}.mp4"
        out_path = VIDEOS_OUT_DIR / out_name

        if job:
            job.log("Codificando fotogramas a MP4 H.264 BT.709 con FFmpeg...", progress=90.0, step="Codificando MP4...")

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
        "prompt": clean_prompt,
        "negative_prompt": negative_prompt,
        "duration_seconds": info["duration_seconds"],
        "width": info["width"] or width,
        "height": info["height"] or height,
        "fps": fps,
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
    """Ajusta una imagen al aspect-ratio exacto (center-crop suave) sin deformarla para Frame Final."""
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
