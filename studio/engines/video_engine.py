import json
import os
import random
import shutil
import subprocess
import tempfile
import time
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
from studio.memory_manager import flush_ram_and_vram, memory_manager


def load_video_model_into_memory(
    model_id: str = "wan22-ti2v-5b",
    auto_download: bool = False,
    job: Optional[JobRecord] = None,
) -> Dict[str, Any]:
    """Carga un modelo de vídeo del catálogo (Wan 2.2 TI2V, Wan 2.2 S2V, LTX-Video) en RAM/VRAM."""
    defn = catalog_manager.get_definition(model_id)
    if not defn:
        raise ValueError(f"Modelo de vídeo desconocido: {model_id}")

    if not catalog_manager.is_downloaded(model_id):
        if auto_download and job is not None:
            job.log("El modelo de vídeo no está en disco. Descargando primero...", step="Descargando modelo de vídeo...")
            catalog_manager.download_model(model_id, job)
        else:
            raise RuntimeError(
                f"«{defn['name']}» no está descargado todavía. Descárgalo desde el catálogo o pulsa «Descargar + Cargar»."
            )

    try:
        import torch  # type: ignore
        from diffusers import DiffusionPipeline  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            f"Faltan librerías de vídeo/GPU ({exc}). En Colab se instalan automáticamente."
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No hay GPU activa en este entorno (modo CPU). "
            "Para animar sin GPU puedes usar «Clip con Movimiento FFmpeg (CPU)», o abrir en Google Colab con GPU."
        )

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Liberar modelo de imagen previo de VRAM para dejar sitio limpio al modelo de vídeo
    img_slot = memory_manager.get_slot("image")
    if img_slot:
        if job:
            job.log(
                f"Liberando «{img_slot['name']}» de RAM/VRAM (sus archivos siguen en disco)...",
                progress=10.0,
                step="Liberando VRAM de imagen...",
            )
        memory_manager.unload_slot("image")

    flush_ram_and_vram()

    art = defn["artifacts"][0]
    base_dir = catalog_manager.model_dir(model_id) / art["subdir"]

    if job:
        job.log(
            f"Cargando pipeline de vídeo «{defn['name']}» en bfloat16...",
            progress=25.0,
            step="Cargando pesos de vídeo...",
        )

    pipe = DiffusionPipeline.from_pretrained(
        str(base_dir),
        torch_dtype=torch.bfloat16,
    )

    gpu_info = query_gpu_info()
    vram_total = float(gpu_info.get("vram_total_gb", 0.0))
    min_vram = float(defn.get("min_vram_gb", 16.0))

    if vram_total < min_vram + 2.0 and hasattr(pipe, "enable_sequential_cpu_offload"):
        if job:
            job.log(
                f"Activando Sequential CPU Offload para encajar en {vram_total:.1f} GB VRAM...",
                progress=75.0,
                step="Activando Sequential Offload...",
            )
        pipe.enable_sequential_cpu_offload()
    elif hasattr(pipe, "enable_model_cpu_offload"):
        if job:
            job.log("Activando Model CPU Offload...", progress=75.0, step="Activando CPU Offload...")
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")

    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()

    vram_gb = round(torch.cuda.memory_allocated(0) / (1024**3), 2)
    memory_manager.register_loaded(
        slot="video",
        model_id=model_id,
        name=defn["name"],
        handle={"pipe": pipe, "engine": defn.get("engine", "wan_ti2v")},
    )

    if job:
        job.log(
            f"Modelo de vídeo «{defn['name']}» listo en RAM/VRAM ({vram_gb} GB VRAM activa).",
            level="success",
            progress=100.0,
            step="Modelo de vídeo listo",
        )
    return {"model_id": model_id, "vram_allocated_gb": vram_gb}


def _frames_to_mp4(
    frames: List[Any],
    output_mp4: Path,
    fps: int = 16,
    audio_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Codifica una secuencia de fotogramas PIL/numpy a MP4 H.264 BT.709 con FFmpeg."""
    ffmpeg_bin = get_ffmpeg_bin()
    tmp_dir = Path(tempfile.mkdtemp(prefix="vframes_"))
    try:
        for i, frame in enumerate(frames):
            if isinstance(frame, Image.Image):
                pil_f = frame.convert("RGB")
            else:
                import numpy as np  # type: ignore

                arr = np.asarray(frame)
                if arr.dtype != np.uint8:
                    arr = (arr * 255).clip(0, 255).astype("uint8")
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
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
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
    steps: int = 20,
    guidance_scale: float = 5.0,
    seed: int = 0,
    motion_style: str = "zoom_in",
    job: Optional[JobRecord] = None,
) -> Dict[str, Any]:
    """Genera un clip de vídeo con modelo de IA cargado o con animación FFmpeg (CPU, 0 VRAM)."""
    t0 = time.time()
    VIDEOS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts_slug = datetime.now().strftime("%Y%m%d_%H%M%S")

    if mode == "ffmpeg_motion":
        if not image_path or not image_path.exists():
            raise ValueError("Selecciona o sube una imagen para crear el clip animado con FFmpeg.")
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

    # Modo IA con modelo de vídeo en GPU
    slot = memory_manager.get_slot("video")
    if not slot:
        raise RuntimeError(
            "No hay ningún modelo de vídeo cargado en RAM/VRAM. "
            "Carga Wan 2.2 o LTX-Video arriba, o elige «Clip Rápido FFmpeg (CPU)» si no quieres usar GPU."
        )

    import torch  # type: ignore

    pipe = slot["handle"]["pipe"]
    actual_seed = int(seed) if int(seed) > 0 else random.randint(1, 2_147_483_647)

    # Si hay audio guía (ej. Wan S2V), ajustar duración a la pista de audio
    if audio_path and audio_path.exists():
        a_info = probe_media_info(audio_path)
        if a_info["duration_seconds"] > 0.3:
            duration_sec = a_info["duration_seconds"]

    width = max(256, (int(width) // 16) * 16)
    height = max(256, (int(height) // 16) * 16)
    fps = max(8, min(30, int(fps)))
    # Wan / LTX suelen requerir 4k + 1 fotogramas
    raw_frames = max(9, int(round(duration_sec * fps)))
    num_frames = ((raw_frames - 1) // 4) * 4 + 1

    img_pil = None
    if image_path and image_path.exists():
        img_pil = Image.open(image_path).convert("RGB").resize((width, height), Image.Resampling.LANCZOS)

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
            f"Generando clip IA con «{slot['name']}» ({width}x{height}, {num_frames} frames @ {fps}fps, semilla {actual_seed})...",
            progress=10.0,
            step=f"Inferencia vídeo 0/{steps}...",
        )

    kwargs: Dict[str, Any] = {
        "prompt": prompt or "cinematic smooth camera motion, high detail",
        "negative_prompt": negative_prompt.strip() or None,
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "num_inference_steps": int(steps),
        "guidance_scale": float(guidance_scale),
        "generator": generator,
        "callback_on_step_end": _step_cb,
    }
    if img_pil is not None:
        kwargs["image"] = img_pil

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

    info = _frames_to_mp4(frames=frames, output_mp4=out_path, fps=fps, audio_path=audio_path)
    flush_ram_and_vram()

    meta = {
        "id": out_path.stem,
        "filename": out_name,
        "media_url": f"/media/outputs/videos/{out_name}",
        "type": "video",
        "model_id": slot["model_id"],
        "model_name": slot["name"],
        "prompt": prompt,
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
