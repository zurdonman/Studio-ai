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


def _apply_best_offload(pipe: Any, vram_total_gb: float, min_vram_gb: float, job: Optional[JobRecord] = None, pct: float = 90.0) -> None:
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
    else:
        pipe.to("cuda")

    if hasattr(pipe, "vae"):
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


def _instantiate_video_pipeline(engine: str, base_dir: Path) -> Dict[str, Any]:
    """Instancia la clase exacta de Diffusers con los tipos precisos (ej. VAE float32 en Wan/Sana/SkyReels)."""
    import torch  # type: ignore
    import diffusers  # type: ignore

    bdir = str(base_dir)

    if engine == "wan_ti2v":
        from diffusers import AutoencoderKLWan, WanImageToVideoPipeline, WanPipeline  # type: ignore

        vae = AutoencoderKLWan.from_pretrained(bdir, subfolder="vae", torch_dtype=torch.float32)
        # Wan2.2-TI2V-5B soporta tanto I2V (expand_timesteps=True) como T2V compartiendo componentes
        pipe_i2v = WanImageToVideoPipeline.from_pretrained(bdir, vae=vae, torch_dtype=torch.bfloat16)
        return {"pipe": pipe_i2v, "i2v_cls": WanImageToVideoPipeline, "t2v_cls": WanPipeline, "active_mode": "i2v"}

    if engine == "wan_t2v":
        from diffusers import AutoencoderKLWan, WanPipeline  # type: ignore

        vae = AutoencoderKLWan.from_pretrained(bdir, subfolder="vae", torch_dtype=torch.float32)
        pipe = WanPipeline.from_pretrained(bdir, vae=vae, torch_dtype=torch.bfloat16)
        return {"pipe": pipe, "i2v_cls": None, "t2v_cls": WanPipeline, "active_mode": "t2v"}

    if engine == "wan_vace":
        from diffusers import AutoencoderKLWan, WanVACEPipeline  # type: ignore

        vae = AutoencoderKLWan.from_pretrained(bdir, subfolder="vae", torch_dtype=torch.float32)
        pipe = WanVACEPipeline.from_pretrained(bdir, vae=vae, torch_dtype=torch.bfloat16)
        return {"pipe": pipe, "i2v_cls": WanVACEPipeline, "t2v_cls": WanVACEPipeline, "active_mode": "both"}

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
        return {"pipe": pipe, "i2v_cls": WanImageToVideoPipeline, "t2v_cls": None, "active_mode": "i2v"}

    if engine == "ltx_video":
        from diffusers import LTXImageToVideoPipeline, LTXPipeline  # type: ignore

        pipe_i2v = LTXImageToVideoPipeline.from_pretrained(bdir, torch_dtype=torch.bfloat16)
        return {"pipe": pipe_i2v, "i2v_cls": LTXImageToVideoPipeline, "t2v_cls": LTXPipeline, "active_mode": "i2v"}

    if engine == "ltx2_av":
        from diffusers import LTX2ImageToVideoPipeline, LTX2Pipeline  # type: ignore

        pipe_i2v = LTX2ImageToVideoPipeline.from_pretrained(bdir, torch_dtype=torch.bfloat16)
        return {"pipe": pipe_i2v, "i2v_cls": LTX2ImageToVideoPipeline, "t2v_cls": LTX2Pipeline, "active_mode": "i2v"}

    if engine == "sana_video":
        from diffusers import SanaImageToVideoPipeline, SanaVideoPipeline  # type: ignore

        pipe_i2v = SanaImageToVideoPipeline.from_pretrained(bdir, torch_dtype=torch.bfloat16)
        if hasattr(pipe_i2v, "vae"):
            pipe_i2v.vae.to(torch.float32)
        return {"pipe": pipe_i2v, "i2v_cls": SanaImageToVideoPipeline, "t2v_cls": SanaVideoPipeline, "active_mode": "i2v"}

    if engine == "skyreels_v2":
        from diffusers import AutoencoderKLWan, SkyReelsV2ImageToVideoPipeline, UniPCMultistepScheduler  # type: ignore

        vae = AutoencoderKLWan.from_pretrained(bdir, subfolder="vae", torch_dtype=torch.float32)
        pipe = SkyReelsV2ImageToVideoPipeline.from_pretrained(bdir, vae=vae, torch_dtype=torch.bfloat16)
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=5.0)
        return {"pipe": pipe, "i2v_cls": SkyReelsV2ImageToVideoPipeline, "t2v_cls": None, "active_mode": "i2v"}

    if engine == "cogvideox_i2v":
        from diffusers import CogVideoXImageToVideoPipeline  # type: ignore

        pipe = CogVideoXImageToVideoPipeline.from_pretrained(bdir, torch_dtype=torch.bfloat16)
        return {"pipe": pipe, "i2v_cls": CogVideoXImageToVideoPipeline, "t2v_cls": None, "active_mode": "i2v"}

    # Fallback genérico para cualquier modelo Diffusers importado desde el Explorador HF
    pipe = diffusers.DiffusionPipeline.from_pretrained(bdir, torch_dtype=torch.bfloat16)
    return {"pipe": pipe, "i2v_cls": None, "t2v_cls": None, "active_mode": "auto"}


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
            "Puedes usar «Animación de Cámara Instantánea FFmpeg (0 GB)» o ejecutar en Google Colab con GPU."
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
        # Asegurar forma (samples, channels)
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
    prompt: str = "",
    negative_prompt: str = "",
    image_path: Optional[Path] = None,
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
    """Genera un clip de vídeo con el modelo de IA activo o con animación FFmpeg (CPU, 0 VRAM)."""
    t0 = time.time()
    VIDEOS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts_slug = datetime.now().strftime("%Y%m%d_%H%M%S")

    if mode == "ffmpeg_motion":
        if not image_path or not image_path.exists():
            raise ValueError("Selecciona o sube una imagen inicial para crear el clip animado con FFmpeg.")
        out_name = f"clip_ffmpeg_{ts_slug}.mp4"
        out_path = VIDEOS_OUT_DIR / out_name
        info = create_motion_clip_from_image(
            image_path=image_path,
            output_mp4=out_path,
            duration_sec=duration_sec,
            width=width,
            height=height,
            fps=fps,
            motion=motion_style,
            audio_path=audio_path,
            job=job,
        )
        meta = {
            "id": out_path.stem,
            "filename": out_name,
            "media_url": f"/media/outputs/videos/{out_name}",
            "type": "video",
            "model_id": "ffmpeg-motion-cpu",
            "model_name": f"Animación FFmpeg ({motion_style})",
            "prompt": prompt or f"Movimiento {motion_style} sobre {image_path.name}",
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
            "Elige un modelo arriba y pulsa «Descargar y Cargar» (o usa el botón de Animación FFmpeg 0 GB)."
        )

    import torch  # type: ignore

    handle = slot["handle"]
    pipe = handle["pipe"]
    engine = handle.get("engine", "generic_diffusers_video")
    has_img = bool(image_path and image_path.exists())

    # Conmutar limpiamente entre variante I2V y T2V en modelos híbridos (Wan 2.2 TI2V, LTX, LTX-2, Sana)
    if has_img and handle.get("active_mode") == "t2v":
        if handle.get("i2v_cls") is None:
            # Si el usuario cargó un modelo puro T2V (ej. Wan 2.1 T2V 1.3B) pero dejó una imagen marcada, avisar
            if job:
                job.log(
                    f"Aviso: «{slot['name']}» es un modelo Texto→Vídeo puro; generará guiado por tu prompt.",
                    level="warning",
                )
            has_img = False
        else:
            if hasattr(pipe, "remove_all_hooks"):
                pipe.remove_all_hooks()
            pipe = handle["i2v_cls"](**pipe.components)
            vram_total = float(query_gpu_info().get("vram_total_gb", 22.0))
            _apply_best_offload(pipe, vram_total, float(handle.get("min_vram_gb", 14.0)))
            handle["pipe"] = pipe
            handle["active_mode"] = "i2v"
    elif not has_img and handle.get("active_mode") == "i2v":
        if handle.get("t2v_cls") is None:
            raise ValueError(
                f"El modelo activo («{slot['name']}») requiere una imagen inicial (modo Imagen → Vídeo). "
                "Selecciona una de tus fotos arriba o elige un modelo compatible con Texto → Vídeo (ej. Wan 2.2 TI2V, LTX-Video o SANA-Video)."
            )
        if hasattr(pipe, "remove_all_hooks"):
            pipe.remove_all_hooks()
        pipe = handle["t2v_cls"](**pipe.components)
        vram_total = float(query_gpu_info().get("vram_total_gb", 22.0))
        _apply_best_offload(pipe, vram_total, float(handle.get("min_vram_gb", 14.0)))
        handle["pipe"] = pipe
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
    if has_img and image_path is not None:
        raw_img = Image.open(image_path).convert("RGB")
        # Si la foto es vertical pero las dimensiones quedaron horizontales por defecto, invertir orientación para no deformar
        if raw_img.height > raw_img.width * 1.12 and width > height:
            width, height = height, width
        elif raw_img.width > raw_img.height * 1.12 and height > width:
            width, height = height, width
        img_pil = raw_img.resize((width, height), Image.Resampling.LANCZOS)

    # Alinear fotogramas a 8*k + 1 (cumple simultáneamente 8k+1 de LTX/LTX-2 y 4k+1 de Wan/Sana/CogVideoX/SkyReels)
    raw_frames = max(17, int(round(duration_sec * fps)))
    num_frames = ((raw_frames - 1) // 8) * 8 + 1

    flush_ram_and_vram()
    generator = torch.Generator(device="cuda").manual_seed(actual_seed)
    step_ts = {"t": time.time()}

    def _step_cb(_p: Any, idx: int, _ts: Any, kw: Dict[str, Any]) -> Dict[str, Any]:
        if job:
            job.check_cancelled()
            now = time.time()
            dt = max(0.01, now - step_ts["t"])
            step_ts["t"] = now
            cur = idx + 1
            pct = 12.0 + (cur / steps) * 76.0
            job.log(
                f"Vídeo IA paso {cur}/{steps} ({dt:.2f} s/paso)",
                progress=pct,
                step=f"Generando fotogramas: paso {cur}/{steps}",
                speed_text=f"{dt:.2f} s/paso",
            )
        return kw

    if job:
        job.log(
            f"Generando vídeo con «{slot['name']}» ({width}×{height}, {num_frames} fotogramas @ {fps} fps, {steps} pasos, semilla {actual_seed})...",
            progress=10.0,
            step=f"Inferencia vídeo 0/{steps}...",
        )

    call_sig = inspect.signature(pipe.__call__).parameters
    clean_prompt = (prompt or "").strip() or "cinematic smooth camera motion, natural subtle movement, high detail"
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

    if img_pil is not None:
        if "reference_images" in call_sig and "image" not in call_sig:
            kwargs["reference_images"] = [img_pil]
        elif "image" in call_sig:
            kwargs["image"] = img_pil

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
                sr = int(getattr(getattr(getattr(pipe, "vocoder", None), "config", None), "output_sampling_rate", 24000))
                generated_audio_wav = VIDEOS_OUT_DIR / f"_tmp_ltx2_audio_{ts_slug}.wav"
                _write_float_audio_wav(audio_out[0], sr, generated_audio_wav)
        else:
            try:
                output = pipe(**kwargs)
            except TypeError:
                kwargs.pop("callback_on_step_end", None)
                output = pipe(**kwargs)
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
