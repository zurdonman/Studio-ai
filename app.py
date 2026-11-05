import base64
import io
import json
import os
import shutil
import time
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from studio.catalog import (
    COLAB_GPU_PROFILES,
    catalog_manager,
    resolve_gpu_profile,
    search_huggingface_live,
)
from studio.config import (
    AUDIO_OUT_DIR,
    DATA_ROOT,
    IMAGES_OUT_DIR,
    MONTAGES_OUT_DIR,
    TEMP_DIR,
    UPLOADS_DIR,
    VIDEOS_OUT_DIR,
    ensure_directories,
)
from studio.engines.ffmpeg_utils import (
    probe_media_info,
    render_montage_mp4,
    verify_image_file,
)
from studio.engines.image_engine import (
    generate_or_edit_image,
    load_image_model_into_memory,
    process_image_cpu_ffmpeg_only,
)
from studio.engines.tts_engine import (
    EDGE_TTS_VOICES,
    load_tts_model_into_memory,
    synthesize_voice,
)
from studio.engines.video_engine import (
    generate_video_clip,
    load_video_model_into_memory,
)
from studio.hardware import (
    colab_console_log,
    hardware_tracker,
    query_gpu_info,
    request_colab_shutdown,
)
from studio.jobs import job_manager
from studio.memory_manager import memory_manager

ensure_directories()

STATIC_DIR = Path(__file__).resolve().parent / "studio" / "static"

app = FastAPI(
    title="Studio IA",
    description="Estudio generativo personal de Imagen, Vídeo, Voz y Montaje para Google Colab",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/media", StaticFiles(directory=str(DATA_ROOT)), name="media")

_export_tracker = {"last_export_ts": 0.0}


# ==================== ESQUEMAS DE PETICIÓN ====================


class CreditsConfigReq(BaseModel):
    initial_credits: Optional[float] = None
    hourly_rate: Optional[float] = None
    use_auto_gpu_rate: bool = False
    reset_session_spent: bool = False
    idle_alert_minutes: Optional[int] = None


class HfTokenReq(BaseModel):
    token: str = ""


class CustomModelReq(BaseModel):
    name: str
    repo_id: str
    category: str = "lora"
    filename: Optional[str] = None
    estimated_gb: float = 1.0
    min_vram_gb: Optional[float] = None
    adapter_name: Optional[str] = None


class MemoryLoadReq(BaseModel):
    model_id: str
    auto_download: bool = False
    extra_lora_ids: List[str] = Field(default_factory=list)


class MemoryUnloadReq(BaseModel):
    target: str = "all"  # "all" o un model_id concreto


class MaskUploadReq(BaseModel):
    data_url: str


class ImageGenReq(BaseModel):
    prompt: str = ""
    negative_prompt: str = ""
    base_image_rel: Optional[str] = None
    reference_rels: List[str] = Field(default_factory=list)
    mask_image_rel: Optional[str] = None
    width: int = 1024
    height: int = 1024
    steps: int = 4
    guidance_scale: float = 4.0
    seed: int = 0
    use_lightning_lora: bool = True
    use_nsfw_lora: bool = False
    nsfw_lora_weight: float = 1.0
    custom_lora_ids: List[str] = Field(default_factory=list)
    use_ffmpeg_bt709: bool = True
    mask_feather_px: int = 8
    cpu_bt709_only: bool = False


class VideoGenReq(BaseModel):
    mode: str = "ai"  # "ai" | "ffmpeg_motion"
    video_submode: str = "auto"  # "auto" | "t2v" | "i2v" | "flf2v"
    model_id: Optional[str] = None
    prompt: str = ""
    negative_prompt: str = ""
    image_rel: Optional[str] = None
    end_image_rel: Optional[str] = None
    audio_rel: Optional[str] = None
    width: int = 1280
    height: int = 704
    duration_sec: float = 5.0
    fps: int = 24
    steps: int = 45
    guidance_scale: Optional[float] = None
    seed: int = 0
    motion_style: str = "zoom_in"


class TTSGenReq(BaseModel):
    text: str
    engine_id: str = "edge-tts-cloud"
    voice_id: str = "es-ES-AlvaroNeural"
    rate_pct: int = 0
    pitch_hz: int = 0


class MontageShotItem(BaseModel):
    media_rel: str
    duration: float = 3.5
    motion: str = "none"
    audio_rel: Optional[str] = None
    fit_to_audio: bool = True


class MontageRenderReq(BaseModel):
    title: str = "montaje_final"
    items: List[MontageShotItem]
    width: int = 1280
    height: int = 720
    fps: int = 24
    global_audio_rel: Optional[str] = None
    global_audio_volume: float = 0.30
    fade_seconds: float = 0.20


# ==================== UTILIDADES INTERNAS ====================


def _resolve_media_rel(rel: Optional[str]) -> Optional[Path]:
    if not rel:
        return None
    cleaned = rel.strip()
    if cleaned.startswith("/media/"):
        cleaned = cleaned[len("/media/") :]
    cleaned = cleaned.lstrip("/")
    candidate = (DATA_ROOT / cleaned).resolve()
    root_resolved = DATA_ROOT.resolve()
    if root_resolved not in candidate.parents and candidate != root_resolved:
        raise HTTPException(status_code=400, detail="Ruta de archivo fuera del directorio de trabajo.")
    if not candidate.exists():
        raise HTTPException(status_code=404, detail=f"Archivo no encontrado: {cleaned}")
    return candidate


def _count_unexported_outputs() -> int:
    last_ts = _export_tracker["last_export_ts"]
    count = 0
    for folder in (IMAGES_OUT_DIR, VIDEOS_OUT_DIR, AUDIO_OUT_DIR, MONTAGES_OUT_DIR):
        if not folder.exists():
            continue
        for fp in folder.iterdir():
            if fp.is_file() and fp.suffix.lower() != ".json":
                if fp.stat().st_mtime > last_ts:
                    count += 1
    return count


def _collect_assets() -> Dict[str, List[Dict[str, Any]]]:
    result: Dict[str, List[Dict[str, Any]]] = {
        "uploads": [],
        "images": [],
        "videos": [],
        "audio": [],
        "montages": [],
    }

    if UPLOADS_DIR.exists():
        files = sorted(
            [f for f in UPLOADS_DIR.iterdir() if f.is_file() and not f.name.startswith("mask_")],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for fp in files:
            ext = fp.suffix.lower()
            atype = (
                "image"
                if ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp")
                else "video"
                if ext in (".mp4", ".webm", ".mov", ".mkv")
                else "audio"
            )
            result["uploads"].append(
                {
                    "id": fp.stem,
                    "filename": fp.name,
                    "rel_path": f"uploads/{fp.name}",
                    "media_url": f"/media/uploads/{fp.name}",
                    "type": atype,
                    "size_kb": round(fp.stat().st_size / 1024, 1),
                    "created_at": datetime.fromtimestamp(fp.stat().st_mtime).isoformat(timespec="seconds"),
                }
            )

    mapping = [
        ("images", IMAGES_OUT_DIR, "outputs/images", "image"),
        ("videos", VIDEOS_OUT_DIR, "outputs/videos", "video"),
        ("audio", AUDIO_OUT_DIR, "outputs/audio", "audio"),
        ("montages", MONTAGES_OUT_DIR, "outputs/montages", "video"),
    ]
    for key, folder, rel_prefix, default_type in mapping:
        if not folder.exists():
            continue
        media_files = sorted(
            [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() != ".json"],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for fp in media_files:
            meta_file = folder / f"{fp.stem}.json"
            meta: Dict[str, Any] = {}
            if meta_file.exists():
                try:
                    meta = json.loads(meta_file.read_text(encoding="utf-8"))
                except Exception:
                    meta = {}
            result[key].append(
                {
                    **meta,
                    "id": fp.stem,
                    "filename": fp.name,
                    "rel_path": f"{rel_prefix}/{fp.name}",
                    "media_url": f"/media/{rel_prefix}/{fp.name}",
                    "type": meta.get("type", default_type),
                    "size_kb": round(fp.stat().st_size / 1024, 1),
                }
            )
    return result


# ==================== RUTAS PRINCIPALES ====================


@app.get("/")
def serve_index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/status")
def api_status() -> Dict[str, Any]:
    active_cnt = job_manager.active_count()
    hw = hardware_tracker.get_snapshot(active_jobs_count=active_cnt)
    jobs = job_manager.list_jobs(limit=25, include_logs=True)
    return {
        "hardware": hw,
        "memory": {
            "loaded_slots": memory_manager.status_summary(),
            "loaded_ids": memory_manager.loaded_model_ids(),
        },
        "jobs": jobs,
        "active_jobs_count": active_cnt,
        "unexported_outputs_count": _count_unexported_outputs(),
        "hf_auth": catalog_manager.hf_auth_status(),
    }


@app.get("/api/hf/auth")
def api_get_hf_auth() -> Dict[str, Any]:
    return catalog_manager.hf_auth_status()


@app.post("/api/hf/token")
def api_set_hf_token(req: HfTokenReq) -> Dict[str, Any]:
    try:
        auth = catalog_manager.set_hf_token(req.token)
        if auth["has_token"]:
            who_txt = f" (@{auth['username']})" if auth.get("username") else ""
            colab_console_log(f"Token de Hugging Face activado en la sesión{who_txt}.", icon="🔑")
        else:
            colab_console_log("Token de Hugging Face eliminado de la sesión.", icon="🔑")
        return {"ok": True, "hf_auth": auth}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve)) from ve


@app.delete("/api/hf/token")
def api_clear_hf_token() -> Dict[str, Any]:
    auth = catalog_manager.clear_hf_token()
    colab_console_log("Token de Hugging Face eliminado de la sesión.", icon="🔑")
    return {"ok": True, "hf_auth": auth}


@app.post("/api/credits/config")
def api_update_credits(req: CreditsConfigReq) -> Dict[str, Any]:
    snap = hardware_tracker.update_credits_config(
        initial_credits=req.initial_credits,
        hourly_rate=req.hourly_rate,
        use_auto_gpu_rate=req.use_auto_gpu_rate,
        reset_session_spent=req.reset_session_spent,
        idle_alert_minutes=req.idle_alert_minutes,
    )
    return {"ok": True, "hardware": snap}


# ==================== CATÁLOGO Y MEMORIA ====================


@app.get("/api/catalog")
def api_get_catalog(gpu_profile: str = "auto", only_compatible: bool = False) -> Dict[str, Any]:
    loaded_ids = memory_manager.loaded_model_ids()
    hw_gpu = query_gpu_info()
    resolved_prof = resolve_gpu_profile(gpu_profile, detected_gpu=hw_gpu)
    return {
        "models": catalog_manager.list_catalog(
            loaded_ids=loaded_ids,
            gpu_profile=gpu_profile,
            only_compatible=only_compatible,
            detected_gpu=hw_gpu,
        ),
        "gpu_profiles": COLAB_GPU_PROFILES,
        "active_gpu_profile": resolved_prof,
        "loaded_slots": memory_manager.status_summary(),
    }


@app.get("/api/catalog/hf-search")
def api_hf_live_search(
    query: str = "",
    category: str = "video",
    gpu_profile: str = "auto",
    only_compatible: bool = False,
    limit: int = 12,
) -> Dict[str, Any]:
    hw_gpu = query_gpu_info()
    return search_huggingface_live(
        query=query,
        category=category,
        gpu_profile=gpu_profile,
        only_compatible=only_compatible,
        detected_gpu=hw_gpu,
        limit=max(1, min(24, limit)),
    )


@app.post("/api/catalog/custom")
def api_add_custom_model(req: CustomModelReq) -> Dict[str, Any]:
    try:
        entry = catalog_manager.add_custom_hf_model(
            name=req.name,
            repo_id=req.repo_id,
            category=req.category,
            filename=req.filename,
            estimated_gb=req.estimated_gb,
            min_vram_gb=req.min_vram_gb,
            adapter_name=req.adapter_name,
        )
        colab_console_log(f"Añadido al catálogo desde HF: {entry['name']} ({req.repo_id})", icon="➕")
        return {"ok": True, "model": entry}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve)) from ve


@app.post("/api/catalog/{model_id}/check-space")
def api_check_model_space(model_id: str) -> Dict[str, Any]:
    try:
        return catalog_manager.check_disk_space(model_id)
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve)) from ve


@app.post("/api/catalog/{model_id}/download")
def api_download_model(model_id: str) -> Dict[str, Any]:
    defn = catalog_manager.get_definition(model_id)
    if not defn:
        raise HTTPException(status_code=404, detail=f"Modelo desconocido: {model_id}")

    space = catalog_manager.check_disk_space(model_id)
    if not space["ok"]:
        raise HTTPException(status_code=400, detail=space)

    job = job_manager.submit(
        kind="download",
        title=f"Descargar: {defn['name']}",
        fn=lambda j: catalog_manager.download_model(model_id, j),
    )
    return {"ok": True, "job": job.to_dict()}


@app.delete("/api/catalog/{model_id}")
def api_delete_model(model_id: str) -> Dict[str, Any]:
    try:
        res = catalog_manager.delete_model(
            model_id,
            unload_callback=lambda mid: memory_manager.unload_by_model_id(mid),
        )
        return res
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve)) from ve


@app.post("/api/memory/load")
def api_load_model(req: MemoryLoadReq) -> Dict[str, Any]:
    defn = catalog_manager.get_definition(req.model_id)
    if not defn:
        raise HTTPException(status_code=404, detail=f"Modelo desconocido: {req.model_id}")

    if req.auto_download and not catalog_manager.is_downloaded(req.model_id):
        space = catalog_manager.check_disk_space(req.model_id)
        if not space["ok"]:
            raise HTTPException(status_code=400, detail=space)

    cat = defn["category"]
    if cat in ("image", "lora"):
        target_id = defn.get("parent_model") or req.model_id
        extra = list(req.extra_lora_ids)
        if cat == "lora" and req.model_id not in extra:
            extra.append(req.model_id)

        def _load_img(j: Any) -> Dict[str, Any]:
            if cat == "lora" and req.auto_download and not catalog_manager.is_downloaded(req.model_id):
                catalog_manager.download_model(req.model_id, j)
            return load_image_model_into_memory(
                model_id=target_id,
                extra_lora_ids=extra,
                auto_download=req.auto_download,
                job=j,
            )

        job = job_manager.submit(
            kind="load_model",
            title=f"Cargar en RAM/VRAM: {defn['name']}",
            fn=_load_img,
        )
    elif cat == "video":
        job = job_manager.submit(
            kind="load_model",
            title=f"Cargar en RAM/VRAM: {defn['name']}",
            fn=lambda j: load_video_model_into_memory(
                model_id=req.model_id,
                auto_download=req.auto_download,
                job=j,
            ),
        )
    elif cat == "tts":
        job = job_manager.submit(
            kind="load_model",
            title=f"Activar motor TTS: {defn['name']}",
            fn=lambda j: load_tts_model_into_memory(
                model_id=req.model_id,
                auto_download=req.auto_download,
                job=j,
            ),
        )
    else:
        raise HTTPException(status_code=400, detail=f"Categoría no soportada: {cat}")

    return {"ok": True, "job": job.to_dict()}


@app.post("/api/memory/unload")
def api_unload_memory(req: MemoryUnloadReq) -> Dict[str, Any]:
    if req.target == "all":
        info = memory_manager.unload_all()
    elif req.target in ("image", "video", "tts"):
        info = memory_manager.unload_slot(req.target)
    else:
        info = memory_manager.unload_by_model_id(req.target)
    return {
        "ok": True,
        **info,
        "message": (
            f"Pesos liberados de RAM y VRAM ({info['vram_freed_gb']} GB VRAM liberados). "
            "Los modelos descargados siguen guardados en disco."
        ),
    }


# ==================== TRABAJOS (COLA Y LOGS) ====================


@app.get("/api/jobs")
def api_list_jobs() -> Dict[str, Any]:
    return {"jobs": job_manager.list_jobs(limit=30, include_logs=True)}


@app.get("/api/jobs/{job_id}")
def api_get_job(job_id: str) -> Dict[str, Any]:
    job = job_manager.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Trabajo no encontrado.")
    return job.to_dict(include_logs=True)


@app.post("/api/jobs/{job_id}/cancel")
def api_cancel_job(job_id: str) -> Dict[str, Any]:
    job = job_manager.cancel(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Trabajo no encontrado.")
    return {"ok": True, "job": job.to_dict(include_logs=True)}


# ==================== ARCHIVOS Y LIENZO ====================


@app.get("/api/assets")
def api_get_assets() -> Dict[str, Any]:
    return _collect_assets()


@app.post("/api/assets/upload")
async def api_upload_asset(file: UploadFile = File(...)) -> Dict[str, Any]:
    ensure_directories()
    orig_name = Path(file.filename or "archivo").name
    safe_stem = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in Path(orig_name).stem)[:40]
    ext = Path(orig_name).suffix.lower() or ".png"
    ts = datetime.now().strftime("%H%M%S")
    out_name = f"{safe_stem}_{ts}_{uuid.uuid4().hex[:4]}{ext}"
    dest = UPLOADS_DIR / out_name

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="El archivo subido está vacío.")
    dest.write_bytes(content)

    # Verificar integridad del archivo subido
    if ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        try:
            w, h = verify_image_file(dest)
        except Exception as exc:
            dest.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=f"Imagen no válida: {exc}") from exc
        extra = {"width": w, "height": h, "type": "image"}
    else:
        try:
            info = probe_media_info(dest)
        except Exception as exc:
            dest.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=f"Archivo multimedia no válido: {exc}") from exc
        extra = {
            "duration_seconds": info["duration_seconds"],
            "width": info["width"],
            "height": info["height"],
            "type": "video" if info["has_video"] else "audio",
        }

    hardware_tracker.mark_activity()
    colab_console_log(
        f"Archivo subido desde la app: {out_name} ({round(len(content) / 1024, 1)} KB, tipo={extra['type']})",
        icon="📥",
    )
    return {
        "ok": True,
        "asset": {
            "filename": out_name,
            "rel_path": f"uploads/{out_name}",
            "media_url": f"/media/uploads/{out_name}",
            "size_kb": round(len(content) / 1024, 1),
            **extra,
        },
    }


@app.post("/api/assets/upload-mask")
def api_upload_mask(req: MaskUploadReq) -> Dict[str, Any]:
    ensure_directories()
    raw = req.data_url
    if "," in raw:
        raw = raw.split(",", 1)[1]
    try:
        binary = base64.b64decode(raw)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Máscara base64 inválida: {exc}") from exc

    mask_name = f"mask_{datetime.now().strftime('%H%M%S')}_{uuid.uuid4().hex[:6]}.png"
    dest = UPLOADS_DIR / mask_name
    dest.write_bytes(binary)
    verify_image_file(dest)
    return {
        "ok": True,
        "rel_path": f"uploads/{mask_name}",
        "media_url": f"/media/uploads/{mask_name}",
    }


@app.delete("/api/assets/{bucket}/{filename}")
def api_delete_asset(bucket: str, filename: str) -> Dict[str, Any]:
    buckets = {
        "uploads": UPLOADS_DIR,
        "images": IMAGES_OUT_DIR,
        "videos": VIDEOS_OUT_DIR,
        "audio": AUDIO_OUT_DIR,
        "montages": MONTAGES_OUT_DIR,
    }
    folder = buckets.get(bucket)
    if not folder:
        raise HTTPException(status_code=400, detail="Categoría inválida.")
    safe_name = Path(filename).name
    target = folder / safe_name
    if target.exists():
        target.unlink()
    meta_json = folder / f"{Path(safe_name).stem}.json"
    if meta_json.exists():
        meta_json.unlink()
    return {"ok": True}


# ==================== GENERACIÓN: IMAGEN, VÍDEO, VOZ, MONTAJE ====================


@app.post("/api/image/generate")
def api_generate_image(req: ImageGenReq) -> Dict[str, Any]:
    base_p = _resolve_media_rel(req.base_image_rel)
    refs_p = [p for p in (_resolve_media_rel(r) for r in req.reference_rels) if p is not None]
    mask_p = _resolve_media_rel(req.mask_image_rel)

    if req.cpu_bt709_only:
        if not base_p:
            raise HTTPException(status_code=400, detail="Sube o elige una imagen primero para procesarla con FFmpeg.")
        job = job_manager.submit(
            kind="image",
            title=f"Normalizar BT.709 (FFmpeg): {base_p.name}",
            fn=lambda j: process_image_cpu_ffmpeg_only(
                base_image_path=base_p,
                width=req.width if req.width else None,
                height=req.height if req.height else None,
                job=j,
            ),
        )
        return {"ok": True, "job": job.to_dict()}

    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Escribe un prompt para indicar qué quieres hacer en la imagen.")

    job = job_manager.submit(
        kind="image",
        title=f"Imagen Qwen: {req.prompt.strip()[:42]}",
        fn=lambda j: generate_or_edit_image(
            prompt=req.prompt,
            base_image_path=base_p,
            reference_paths=refs_p,
            mask_image_path=mask_p,
            negative_prompt=req.negative_prompt,
            width=req.width,
            height=req.height,
            steps=req.steps,
            guidance_scale=req.guidance_scale,
            seed=req.seed,
            use_lightning_lora=req.use_lightning_lora,
            use_nsfw_lora=req.use_nsfw_lora,
            nsfw_lora_weight=req.nsfw_lora_weight,
            custom_lora_ids=req.custom_lora_ids,
            use_ffmpeg_bt709=req.use_ffmpeg_bt709,
            mask_feather_px=req.mask_feather_px,
            job=j,
        ),
    )
    return {"ok": True, "job": job.to_dict()}


@app.post("/api/video/generate")
def api_generate_video(req: VideoGenReq) -> Dict[str, Any]:
    img_p = _resolve_media_rel(req.image_rel)
    end_img_p = _resolve_media_rel(req.end_image_rel)
    aud_p = _resolve_media_rel(req.audio_rel)

    sub = (req.video_submode or "auto").strip().lower()
    if sub == "i2v" and img_p is None:
        raise HTTPException(
            status_code=400,
            detail="Modo «Imagen a Vídeo»: sube o elige una imagen inicial (o cambia arriba a «Texto a Vídeo»).",
        )
    if sub == "flf2v" and (img_p is None or end_img_p is None):
        raise HTTPException(
            status_code=400,
            detail="Modo «Frame a Frame (Inicio → Fin)»: sube o elige tanto el Fotograma Inicial como el Fotograma Final.",
        )
    if req.mode == "ffmpeg_motion" and img_p is None:
        raise HTTPException(
            status_code=400,
            detail="La animación FFmpeg (0 GB) necesita al menos una imagen inicial.",
        )

    if req.mode == "ai" and req.model_id:
        m_defn = catalog_manager.get_definition(req.model_id)
        if m_defn and sub in ("t2v", "i2v", "flf2v"):
            allowed_modes = m_defn.get("video_modes") or ["i2v", "t2v"]
            if sub not in allowed_modes:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"El modelo «{m_defn['name']}» no soporta este modo ({sub}). "
                        "Para Frame a Frame elige «Wan 2.1 VACE 1.3B», «SkyReels V2 1.3B 540P» o «LTX-Video 0.9.5 2B»."
                    ),
                )

    sub_labels = {
        "t2v": "Texto→Vídeo",
        "i2v": "Imagen→Vídeo",
        "flf2v": "Frame a Frame",
    }
    sub_tag = sub_labels.get(sub, "Vídeo IA")
    label = (
        f"Clip FFmpeg ({'Frame→Frame' if end_img_p else req.motion_style})"
        if req.mode == "ffmpeg_motion"
        else f"{sub_tag}: {(req.prompt or 'Animación IA')[:36]}"
    )
    job = job_manager.submit(
        kind="video",
        title=label,
        fn=lambda j: generate_video_clip(
            mode=req.mode,
            video_submode=sub,
            model_id=req.model_id,
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            image_path=img_p,
            end_image_path=end_img_p,
            audio_path=aud_p,
            width=req.width,
            height=req.height,
            duration_sec=req.duration_sec,
            fps=req.fps,
            steps=req.steps,
            guidance_scale=req.guidance_scale,
            seed=req.seed,
            motion_style=req.motion_style,
            job=j,
        ),
    )
    return {"ok": True, "job": job.to_dict()}


@app.get("/api/tts/voices")
def api_tts_voices() -> Dict[str, Any]:
    return {"voices": EDGE_TTS_VOICES}


@app.post("/api/tts/generate")
def api_generate_tts(req: TTSGenReq) -> Dict[str, Any]:
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Escribe el texto para generar la voz.")
    job = job_manager.submit(
        kind="tts",
        title=f"Voz ({req.voice_id}): {req.text.strip()[:36]}",
        fn=lambda j: synthesize_voice(
            text=req.text,
            engine_id=req.engine_id,
            voice_id=req.voice_id,
            rate_pct=req.rate_pct,
            pitch_hz=req.pitch_hz,
            job=j,
        ),
    )
    return {"ok": True, "job": job.to_dict()}


@app.post("/api/montage/render")
def api_render_montage(req: MontageRenderReq) -> Dict[str, Any]:
    if not req.items:
        raise HTTPException(status_code=400, detail="Añade al menos un plano (imagen o vídeo) al montaje.")

    resolved_items: List[Dict[str, Any]] = []
    for idx, it in enumerate(req.items, start=1):
        p = _resolve_media_rel(it.media_rel)
        if not p:
            raise HTTPException(status_code=400, detail=f"Falta el archivo del plano {idx}.")
        aud_p = _resolve_media_rel(it.audio_rel) if it.audio_rel else None
        resolved_items.append(
            {
                "path": str(p),
                "duration": it.duration,
                "motion": it.motion,
                "audio_path": str(aud_p) if aud_p else None,
                "fit_to_audio": it.fit_to_audio,
            }
        )

    global_aud = _resolve_media_rel(req.global_audio_rel) if req.global_audio_rel else None

    def _do_montage(j: Any) -> Dict[str, Any]:
        t0 = time.time()
        MONTAGES_OUT_DIR.mkdir(parents=True, exist_ok=True)
        safe_title = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in req.title)[:32] or "montaje"
        ts_slug = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_name = f"{safe_title}_{ts_slug}.mp4"
        out_path = MONTAGES_OUT_DIR / out_name

        info = render_montage_mp4(
            items=resolved_items,
            output_mp4=out_path,
            width=req.width,
            height=req.height,
            fps=req.fps,
            global_audio_path=global_aud,
            global_audio_volume=req.global_audio_volume,
            fade_seconds=req.fade_seconds,
            job=j,
        )
        meta = {
            "id": out_path.stem,
            "filename": out_name,
            "rel_path": f"outputs/montages/{out_name}",
            "media_url": f"/media/outputs/montages/{out_name}",
            "type": "video",
            "shots_count": len(resolved_items),
            "duration_seconds": info["duration_seconds"],
            "width": info["width"] or req.width,
            "height": info["height"] or req.height,
            "fps": req.fps,
            "size_mb": round(info["size_bytes"] / (1024**2), 2),
            "elapsed_seconds": round(time.time() - t0, 2),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        (MONTAGES_OUT_DIR / f"{out_path.stem}.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return meta

    job = job_manager.submit(
        kind="montage",
        title=f"Montaje MP4 ({len(resolved_items)} planos · {req.width}x{req.height})",
        fn=_do_montage,
    )
    return {"ok": True, "job": job.to_dict()}


# ==================== EXPORTAR ZIP Y APAGAR GPU ====================


@app.get("/api/export/zip")
def api_export_zip() -> StreamingResponse:
    """Empaqueta todas las imágenes, vídeos, voces, montajes y metadatos en un .zip descargable."""
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for sub_name, folder in (
            ("imagenes", IMAGES_OUT_DIR),
            ("videos", VIDEOS_OUT_DIR),
            ("voces_audio", AUDIO_OUT_DIR),
            ("montajes_mp4", MONTAGES_OUT_DIR),
        ):
            if not folder.exists():
                continue
            for fp in sorted(folder.iterdir()):
                if fp.is_file():
                    zf.write(fp, arcname=f"{sub_name}/{fp.name}")
    buf.seek(0)
    _export_tracker["last_export_ts"] = time.time()
    colab_console_log("Archivo .zip de resultados empaquetado y enviado al navegador.", icon="📦")
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    headers = {"Content-Disposition": f'attachment; filename="studio_ia_resultados_{stamp}.zip"'}
    return StreamingResponse(buf, media_type="application/zip", headers=headers)


@app.post("/api/system/shutdown")
def api_shutdown_colab() -> Dict[str, Any]:
    memory_manager.unload_all()
    res = request_colab_shutdown()
    return res


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "7860"))
    colab_console_log(f"Servidor FastAPI Studio IA escuchando en puerto {port}.", icon="🟢")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning", access_log=False)
