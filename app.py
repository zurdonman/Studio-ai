import base64
import io
import json
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import shutil
import time
import uuid
import zipfile
from datetime import datetime
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from studio.catalog import (
    COLAB_GPU_PROFILES,
    catalog_manager,
    resolve_gpu_profile,
    search_huggingface_live,
)
from studio.compat_patches import apply_runtime_compatibility_patches
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
from studio.engines.audio_studio_engine import (
    AMBIENCE_PRESETS,
    audio_studio_engine,
    mix_and_master_media,
)
from studio.engines.coherence_engine import (
    load_coherence_vlm_into_memory,
    run_coherence_analysis,
)
from studio.engines.ffmpeg_utils import (
    probe_media_info,
    render_montage_mp4,
    verify_image_file,
)
from studio.engines.higgsfield_engine import (
    HIGGSFIELD_MOTIONS,
    build_character_appearance_from_traits,
    character_vault,
    compile_cinema_prompt,
    get_higgsfield_catalog_payload,
    render_higgsfield_cpu_motion_clip,
    run_higgsfield_scene_director,
    smart_clip_and_reframe_video,
)
from studio.engines.image_engine import (
    generate_or_edit_image,
    load_image_model_into_memory,
    process_image_cpu_ffmpeg_only,
)
from studio.engines.lipsync_engine import lipsync_engine
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
    enforce_idle_shutdown,
    build_full_session_log_document,
    colab_console_log,
    hardware_tracker,
    query_gpu_info,
    request_colab_shutdown,
)
from studio import runtime_flags
from studio.mcp_preferences import get_preferences, pending_questions, set_preferences
from studio.jobs import job_manager, model_prepare_job_key
from studio.mcp_server import mcp_router
from studio.memory_manager import memory_manager
from studio.model_discovery import model_discovery
from studio.model_compatibility import assessment_context, assess_definition
from studio.model_recipes import image_defaults
from studio.model_oracle import recommend_pipeline

apply_runtime_compatibility_patches()
ensure_directories()

STATIC_DIR = Path(__file__).resolve().parent / "studio" / "static"

@asynccontextmanager
async def studio_lifespan(_app: FastAPI):
    model_discovery.start()
    memory_manager.start()
    try:
        yield
    finally:
        model_discovery.stop()
        memory_manager.stop()


app = FastAPI(
    lifespan=studio_lifespan,
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

app.include_router(mcp_router)

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
    idle_autoshutdown_minutes: Optional[int] = None


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


class DiscoverySettingsReq(BaseModel):
    auto_refresh: bool
    interval_minutes: int = Field(default=360, ge=15, le=1440)


class DiscoveryRefreshReq(BaseModel):
    query: str = Field(default="", max_length=160)
    category: str = "all"
    sort: str = "downloads"


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
    model_id: Optional[str] = None
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
    lora_scales: Dict[str, float] = Field(default_factory=dict)
    use_ffmpeg_bt709: bool = True
    mask_feather_px: int = 8
    cpu_bt709_only: bool = False
    use_official_defaults: bool = False


class VideoGenReq(BaseModel):
    mode: str = "ai"  # "ai" | "ffmpeg_motion"
    video_submode: str = "auto"  # "auto" | "t2v" | "i2v" | "flf2v"
    model_id: Optional[str] = None
    video_lora_id: Optional[str] = None
    video_lora_scale: float = 1.0
    extra_video_lora_ids: List[str] = Field(default_factory=list)
    extra_video_lora_scales: Dict[str, float] = Field(default_factory=dict)
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
    payload = {
        "hardware": hw,
        "memory": {
            "loaded_slots": memory_manager.status_summary(),
            "loaded_ids": memory_manager.loaded_model_ids(),
            "guardian": memory_manager.guardian_status(),
        },
        "jobs": jobs,
        "active_jobs_count": active_cnt,
        "unexported_outputs_count": _count_unexported_outputs(),
        "hf_auth": catalog_manager.hf_auth_status(),
        "runtime": runtime_flags.snapshot(),
    }
    # Red de seguridad: si nadie usa la GPU mucho tiempo, se apaga sola (configurable).
    try:
        enforce_idle_shutdown(hardware_tracker, active_cnt)
    except Exception:
        pass
    return payload


@app.get("/api/runtime/status")
def api_runtime_status() -> Dict[str, Any]:
    """Whether this process reloaded GPU frameworks for the compatibility patches."""
    runtime_flags.set_server_pid(os.getpid())
    return {"ok": True, **runtime_flags.snapshot()}


@app.get("/api/preferences")
def api_get_preferences() -> Dict[str, Any]:
    return {"ok": True, "preferences": get_preferences(), "pending_questions": pending_questions()}


@app.post("/api/preferences")
def api_set_preferences(req: Dict[str, Any]) -> Dict[str, Any]:
    saved = set_preferences(req if isinstance(req, dict) else {})
    return {"ok": True, "preferences": saved, "pending_questions": pending_questions()}


@app.get("/api/memory/status")
def api_memory_status() -> Dict[str, Any]:
    return {"guardian": memory_manager.guardian_status(), "loaded_slots": memory_manager.status_summary()}


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
        idle_autoshutdown_minutes=req.idle_autoshutdown_minutes,
    )
    return {"ok": True, "hardware": snap}


# ==================== CATÁLOGO Y MEMORIA ====================


@app.get("/api/catalog")
def api_get_catalog(gpu_profile: str = "auto", only_compatible: bool = False) -> Dict[str, Any]:
    loaded_ids = memory_manager.loaded_model_ids()
    hw = hardware_tracker.get_snapshot(active_jobs_count=job_manager.active_count())
    hw_gpu = hw["gpu"]
    resolved_prof = resolve_gpu_profile(gpu_profile, detected_gpu=hw_gpu)
    try:
        context = assessment_context(snapshot=hw, gpu_profile=gpu_profile)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    models = catalog_manager.list_catalog(loaded_ids=loaded_ids, gpu_profile=gpu_profile,
        only_compatible=only_compatible, detected_gpu=hw_gpu, assessment_context_value=context)
    for model in models:
        model["hardware_assessment"] = model["compatibility"]
        if model["category"] == "image":
            model["image_defaults"] = image_defaults(model)
    if only_compatible:
        models = [m for m in models if m["hardware_assessment"]["compatible"] or m["downloaded"] or m["loaded_in_memory"]]
    return {"models": models, "gpu_profiles": COLAB_GPU_PROFILES, "active_gpu_profile": resolved_prof,
            "loaded_slots": memory_manager.status_summary(), "discovery_status": model_discovery.status()}


@app.get("/api/catalog/hf-search")
def api_hf_live_search(
    query: str = Query(default="", max_length=160), category: str = "video", gpu_profile: str = "auto",
    only_compatible: bool = False, limit: int = Query(default=12, ge=1, le=50),
    offset: int = Query(default=0, ge=0, le=5000), refresh: bool = False, sort: str = "downloads",
) -> Dict[str, Any]:
    try:
        return model_discovery.search(query=query, category=category, gpu_profile=gpu_profile,
            only_compatible=only_compatible, limit=limit, offset=offset, refresh=refresh, sort=sort)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/models/discovery/status")
def api_model_discovery_status() -> Dict[str, Any]:
    return model_discovery.status()


@app.post("/api/models/discovery/settings")
def api_model_discovery_settings(req: DiscoverySettingsReq) -> Dict[str, Any]:
    return model_discovery.configure(auto_refresh=req.auto_refresh, interval_minutes=req.interval_minutes)


@app.post("/api/models/discovery/refresh")
def api_refresh_model_discovery(req: DiscoveryRefreshReq) -> Dict[str, Any]:
    try:
        return model_discovery.queue_refresh(query=req.query, category=req.category, sort=req.sort)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/models/discovery/inspect")
def api_inspect_discovered_model(repo_id: str, gpu_profile: str = "auto", refresh: bool = False) -> Dict[str, Any]:
    try:
        return model_discovery.inspect(repo_id, gpu_profile=gpu_profile, refresh=refresh)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/models/manual")
def api_model_manual(guide: str = "operations") -> PlainTextResponse:
    documents = {"operations": "MCP_MODEL_ORACLE.md", "research": "MODEL_RESEARCH.md"}
    if guide not in documents:
        raise HTTPException(status_code=400, detail="Guía permitida: operations o research.")
    text = (Path(__file__).resolve().parent / "docs" / documents[guide]).read_text(encoding="utf-8")
    return PlainTextResponse(text, media_type="text/markdown; charset=utf-8")


@app.get("/api/models/recommend")
def api_recommend_model_recipe(task: str = "image_create", priority: str = "balanced_fast") -> Dict[str, Any]:
    try:
        return {"ok": True, **recommend_pipeline(task, priority, catalog=catalog_manager,
            snapshot=hardware_tracker.get_snapshot(active_jobs_count=job_manager.active_count()))}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/models/{model_id}/assessment")
def api_assess_catalog_model(model_id: str, gpu_profile: str = "auto") -> Dict[str, Any]:
    definition = catalog_manager.get_definition(model_id)
    if not definition:
        raise HTTPException(status_code=404, detail="ID de catálogo no encontrado. Un repo HF no es un model_id.")
    try:
        assessment = assess_definition(definition, assessment_context(gpu_profile=gpu_profile), catalog=catalog_manager)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "model_id": model_id, "assessment": assessment}


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
        candidate = entry.get("validation_state") == "candidate"
        return {"ok": True, "model": entry, "candidate_only": candidate, "download_allowed": not candidate,
                "message": "Candidato guardado; falta receta revisada, no se descargarán pesos." if candidate else "Receta existente reutilizada."}
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
    if defn.get("validation_state") == "candidate":
        raise HTTPException(status_code=400, detail="MODEL_RECIPE_UNVALIDATED: candidato guardado sin receta de ejecución revisada. Inspecciona metadatos, no descargues pesos.")


    if not catalog_manager.is_downloaded(model_id):
        space = catalog_manager.check_disk_space(model_id)
        if not space["ok"]:
            raise HTTPException(status_code=400, detail=space)

    job = job_manager.submit(
        kind="download",
        title=f"Descargar: {defn['name']}",
        fn=lambda j: catalog_manager.download_model(model_id, j),
        dedup_key=f"download:{model_id}",
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
    if defn.get("validation_state") == "candidate":
        raise HTTPException(status_code=400, detail="MODEL_RECIPE_UNVALIDATED: candidato guardado sin receta de ejecución revisada. Inspecciona metadatos, no descargues pesos.")


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
            dedup_key=model_prepare_job_key(target_id, extra),
        )
    elif cat == "video":
        job = job_manager.submit(
            kind="load_model",
            title=f"Cargar en RAM/VRAM: {defn['name']}",
            fn=lambda j: load_video_model_into_memory(
                model_id=req.model_id,
                auto_download=req.auto_download,
                job=j,
                extra_lora_ids=list(req.extra_lora_ids),
            ),
            dedup_key=model_prepare_job_key(req.model_id, req.extra_lora_ids),
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
            dedup_key=f"prepare:{req.model_id}",
        )
    elif cat == "vision":
        if defn.get("builtin_no_download"):
            return {"ok": True, "builtin": True, "message": "OpenCV integrado en CPU (0 GB, sin carga previa)."}
        job = job_manager.submit(
            kind="load_model",
            title=f"Cargar inspector multimodal: {defn['name']}",
            fn=lambda j: load_coherence_vlm_into_memory(
                model_id=req.model_id,
                auto_download=req.auto_download,
                job=j,
            ),
            dedup_key=f"prepare:{req.model_id}",
        )
    elif cat == "lipsync":
        def _load_ls(j: Any) -> Dict[str, Any]:
            if req.auto_download and not catalog_manager.is_downloaded(req.model_id):
                catalog_manager.download_model(req.model_id, job=j)
            lipsync_engine._ensure_loaded(req.model_id, job=j)
            return {"model_id": req.model_id, "loaded": True}

        job = job_manager.submit(
            kind="load_model",
            title=f"Cargar motor LipSync HD: {defn['name']}",
            fn=_load_ls,
            dedup_key=f"prepare:{req.model_id}",
        )
    elif cat == "audio":
        if defn.get("builtin_no_download"):
            return {"ok": True, "builtin": True, "message": "Sintetizador DSP integrado en CPU (0 GB, sin carga previa)."}

        def _load_aud(j: Any) -> Dict[str, Any]:
            if req.auto_download and not catalog_manager.is_downloaded(req.model_id):
                catalog_manager.download_model(req.model_id, job=j)
            return {"model_id": req.model_id, "ready": True}

        job = job_manager.submit(
            kind="load_model",
            title=f"Preparar motor musical/sonoro: {defn['name']}",
            fn=_load_aud,
            dedup_key=f"prepare:{req.model_id}",
        )
    else:
        raise HTTPException(status_code=400, detail=f"Categoría no soportada: {cat}")

    return {"ok": True, "job": job.to_dict()}


@app.post("/api/memory/unload")
def api_unload_memory(req: MemoryUnloadReq) -> Dict[str, Any]:
    if req.target == "all":
        info = memory_manager.unload_all()
    elif req.target in ("image", "video", "tts", "vision", "lipsync", "audio_bgm"):
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
async def api_upload_asset(request: Request, file: UploadFile = File(...)) -> Dict[str, Any]:
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
    from_mcp = bool(request.headers.get("x-mcp-client"))
    origin_tag = "[MCP → Colab] Archivo subido desde tu terminal" if from_mcp else "Archivo subido desde la app"
    colab_console_log(
        f"{origin_tag}: {out_name} ({round(len(content) / 1024, 1)} KB, tipo={extra['type']})",
        icon="🤖" if from_mcp else "📥",
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

    definition = catalog_manager.get_definition(req.model_id or "qwen-image-edit-2511-pack") or {}
    if not definition or definition.get("category") != "image" or definition.get("validation_state") == "candidate":
        raise HTTPException(status_code=400, detail="MODEL_RECIPE_UNVALIDATED: se requiere un ID de receta revisada de imagen.")
    if (base_p or refs_p) and "i2i" not in definition.get("capabilities", []):
        raise HTTPException(status_code=400, detail="IMAGE_MODEL_NO_REFERENCE_CONDITIONING: la receta solo admite texto, no edición/referencias.")
    defaults = image_defaults(definition, req.use_lightning_lora) if req.use_official_defaults else {}
    job = job_manager.submit(
        kind="image",
        title=f"Imagen: {req.prompt.strip()[:42]}",
        fn=lambda j: generate_or_edit_image(
            prompt=req.prompt,
            model_id=req.model_id,
            base_image_path=base_p,
            reference_paths=refs_p,
            mask_image_path=mask_p,
            negative_prompt=req.negative_prompt,
            width=req.width,
            height=req.height,
            steps=defaults.get("steps", req.steps),
            guidance_scale=defaults.get("guidance_scale", req.guidance_scale),
            seed=req.seed,
            use_lightning_lora=req.use_lightning_lora,
            use_nsfw_lora=req.use_nsfw_lora,
            nsfw_lora_weight=req.nsfw_lora_weight,
            custom_lora_ids=req.custom_lora_ids,
            lora_scales=req.lora_scales,
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
        if not m_defn or m_defn.get("category") != "video" or m_defn.get("validation_state") == "candidate":
            raise HTTPException(status_code=400, detail="MODEL_RECIPE_UNVALIDATED: se requiere un ID de receta revisada de vídeo.")
        if m_defn and sub in ("t2v", "i2v", "flf2v"):
            allowed_modes = m_defn.get("video_modes") or ["i2v", "t2v"]
            if sub not in allowed_modes:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"El modelo «{m_defn['name']}» no soporta este modo ({sub}). "
                        "Para Frame a Frame elige «Wan 2.1 VACE 14B Quantized FP8», «Wan 2.1 FLF2V 14B 720P Quantized FP8», "
                        "«LTX-Video 0.9.8 13B Distilled» o «Wan 2.1 VACE 1.3B»."
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
            video_lora_id=req.video_lora_id,
            video_lora_scale=req.video_lora_scale,
            extra_video_lora_ids=req.extra_video_lora_ids,
            extra_video_lora_scales=req.extra_video_lora_scales,
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


class AudioBgmReq(BaseModel):
    prompt: str = "warm smooth acoustic jazz piano in a cozy Italian restaurant"
    model_id: str = "auto"
    preset: str = "auto"
    duration_seconds: float = 10.0
    seed: int = 42


class VoiceTrackMixItem(BaseModel):
    audio_rel: str
    start_sec: float = 0.0
    volume: float = 1.0


class AudioMixMasterReq(BaseModel):
    video_rel: Optional[str] = None
    voice_tracks: List[VoiceTrackMixItem] = Field(default_factory=list)
    music_rel: Optional[str] = None
    music_volume: float = 0.24
    ambient_rel: Optional[str] = None
    ambient_volume: float = 0.18
    keep_original_audio: bool = False
    original_audio_volume: float = 0.35
    auto_duck_bg: bool = True
    extend_video_to_audio: bool = True
    fade_out_seconds: float = 0.80


class DialogueTurnReqItem(BaseModel):
    voice_audio_rel: str
    target_face: str = "largest"
    start_sec: float = 0.0
    volume: float = 1.0


class LipSyncReq(BaseModel):
    media_rel: str
    voice_audio_rel: Optional[str] = None
    target_face: str = "largest"
    dialogue_turns: List[DialogueTurnReqItem] = Field(default_factory=list)
    model_id: str = "edtalk-256-lipsync-hd"
    lip_weight: float = 0.85
    enhance_face: bool = True
    restore_strength: float = 0.50
    music_rel: Optional[str] = None
    music_volume: float = 0.22
    ambient_rel: Optional[str] = None
    ambient_volume: float = 0.18
    extend_video_to_audio: bool = True


@app.get("/api/audio/presets")
def api_audio_presets() -> Dict[str, Any]:
    return {"ok": True, "presets": AMBIENCE_PRESETS}


@app.post("/api/audio/generate-bgm")
def api_generate_bgm(req: AudioBgmReq) -> Dict[str, Any]:
    job = job_manager.submit(
        kind="audio_bgm",
        title=f"Música / Ambiente ({req.model_id}): {(req.prompt or req.preset)[:38]}",
        fn=lambda j: audio_studio_engine.generate_music_or_ambience(
            job=j,
            prompt=req.prompt,
            model_id=req.model_id,
            preset=req.preset,
            duration_seconds=req.duration_seconds,
            seed=req.seed,
        ),
    )
    return {"ok": True, "job": job.to_dict()}


@app.post("/api/audio/mix-master")
def api_mix_master(req: AudioMixMasterReq) -> Dict[str, Any]:
    vid_p = _resolve_media_rel(req.video_rel) if req.video_rel else None
    if req.video_rel and not vid_p:
        raise HTTPException(status_code=400, detail=f"No se encontró el vídeo: {req.video_rel}")

    mus_p = _resolve_media_rel(req.music_rel) if req.music_rel else None
    if req.music_rel and not mus_p:
        raise HTTPException(status_code=400, detail=f"No se encontró la música: {req.music_rel}")

    amb_p = _resolve_media_rel(req.ambient_rel) if req.ambient_rel else None
    if req.ambient_rel and not amb_p:
        raise HTTPException(status_code=400, detail=f"No se encontró el ambiente: {req.ambient_rel}")

    resolved_voices: List[Dict[str, Any]] = []
    for idx_v, vt in enumerate(req.voice_tracks, start=1):
        vp = _resolve_media_rel(vt.audio_rel)
        if not vp:
            raise HTTPException(status_code=400, detail=f"No se encontró la voz #{idx_v}: {vt.audio_rel}")
        resolved_voices.append({"path": vp, "start_sec": vt.start_sec, "volume": vt.volume})

    if vid_p is None and not resolved_voices and mus_p is None and amb_p is None:
        raise HTTPException(status_code=400, detail="Indica al menos un vídeo o una pista de audio para mezclar.")

    def _do_mix(j: Any) -> Dict[str, Any]:
        ts_slug = datetime.now().strftime("%Y%m%d_%H%M%S")
        if vid_p is not None:
            VIDEOS_OUT_DIR.mkdir(parents=True, exist_ok=True)
            out_name = f"master_{ts_slug}_{j.id[:6]}.mp4"
            out_path = VIDEOS_OUT_DIR / out_name
            rel_path = f"outputs/videos/{out_name}"
            m_type = "video"
            sidecar_dir = VIDEOS_OUT_DIR
        else:
            AUDIO_OUT_DIR.mkdir(parents=True, exist_ok=True)
            out_name = f"mix_{ts_slug}_{j.id[:6]}.wav"
            out_path = AUDIO_OUT_DIR / out_name
            rel_path = f"outputs/audio/{out_name}"
            m_type = "audio"
            sidecar_dir = AUDIO_OUT_DIR

        res = mix_and_master_media(
            output_path=out_path,
            video_path=vid_p,
            voice_tracks=resolved_voices,
            music_path=mus_p,
            music_volume=req.music_volume,
            ambient_path=amb_p,
            ambient_volume=req.ambient_volume,
            keep_original_audio=req.keep_original_audio,
            original_audio_volume=req.original_audio_volume,
            auto_duck_bg=req.auto_duck_bg,
            extend_video_to_audio=req.extend_video_to_audio,
            fade_out_seconds=req.fade_out_seconds,
            job=j,
        )
        meta = {
            "id": out_path.stem,
            "filename": out_name,
            "rel_path": rel_path,
            "media_url": f"/media/{rel_path}",
            "type": m_type,
            **res,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        (sidecar_dir / f"{out_path.stem}.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        j.log(f"Mezcla multipista lista: {out_name} ({res['duration_seconds']:.2f}s).", level="success", progress=100.0)
        return meta

    job = job_manager.submit(
        kind="audio_mix",
        title=f"Mezcla Multipista + Auto-Ducking ({len(resolved_voices)} voces)",
        fn=_do_mix,
    )
    return {"ok": True, "job": job.to_dict()}


@app.post("/api/video/lipsync")
def api_lipsync_video(req: LipSyncReq) -> Dict[str, Any]:
    med_p = _resolve_media_rel(req.media_rel)
    if not med_p:
        raise HTTPException(status_code=400, detail=f"No se encontró la imagen o vídeo: {req.media_rel}")

    v_single: Optional[Path] = None
    if req.voice_audio_rel:
        v_single = _resolve_media_rel(req.voice_audio_rel)
        if not v_single:
            raise HTTPException(status_code=400, detail=f"No se encontró el audio de voz: {req.voice_audio_rel}")

    turns_resolved: List[Dict[str, Any]] = []
    for idx_t, tr in enumerate(req.dialogue_turns, start=1):
        vp = _resolve_media_rel(tr.voice_audio_rel)
        if not vp:
            raise HTTPException(status_code=400, detail=f"No se encontró el audio del turno #{idx_t}: {tr.voice_audio_rel}")
        turns_resolved.append(
            {
                "voice_path": vp,
                "target_face": tr.target_face,
                "start_sec": tr.start_sec,
                "volume": tr.volume,
            }
        )

    if v_single is None and not turns_resolved:
        raise HTTPException(status_code=400, detail="Indica `voice_audio_rel` o al menos un turno en `dialogue_turns`.")

    mus_p = _resolve_media_rel(req.music_rel) if req.music_rel else None
    amb_p = _resolve_media_rel(req.ambient_rel) if req.ambient_rel else None

    job = job_manager.submit(
        kind="lipsync",
        title=f"LipSync HD ({req.target_face if not turns_resolved else f'{len(turns_resolved)} turnos'} · {med_p.name})",
        fn=lambda j: lipsync_engine.render_lipsync(
            job=j,
            media_path=med_p,
            voice_audio_path=v_single,
            target_face=req.target_face,
            dialogue_turns=turns_resolved if turns_resolved else None,
            model_id=req.model_id,
            lip_weight=req.lip_weight,
            enhance_face=req.enhance_face,
            restore_strength=req.restore_strength,
            music_path=mus_p,
            music_volume=req.music_volume,
            ambient_path=amb_p,
            ambient_volume=req.ambient_volume,
            extend_video_to_audio=req.extend_video_to_audio,
        ),
    )
    return {"ok": True, "job": job.to_dict()}


# ==================== MODO HIGGSFIELD (100% LOCAL / COLAB · 0€ APIs) ====================


class CharacterSaveReq(BaseModel):
    char_id: Optional[str] = None
    name: str
    appearance_prompt: str = ""
    traits: Dict[str, str] = Field(default_factory=dict)
    anchor_image_rel: str = ""
    voice_id: str = "es-ES-AlvaroNeural"
    default_face_side: str = "largest"


class CinemaShotReq(BaseModel):
    prompt: str
    camera_id: str = "arri_full_frame_8k"
    lens_id: str = "classic_anamorphic"
    focal_mm: str = "35"
    aperture: str = "f/2.8"
    lighting_id: str = "natural_cinema"
    aspect_ratio: str = "16:9"
    character_ids: List[str] = Field(default_factory=list)
    image_lora_ids: List[str] = Field(default_factory=list)
    base_image_rel: Optional[str] = None
    seed: int = 0


class HiggsfieldMotionReq(BaseModel):
    image_rel: str
    end_image_rel: Optional[str] = None
    prompt: str = ""
    motion_id: str = "dolly_in_dramatic"
    engine_mode: str = "auto"  # "auto" | "ai_14b" | "ai_fast" | "ffmpeg_instant"
    video_lora_ids: List[str] = Field(default_factory=list)
    aspect_ratio: str = "16:9"
    duration_seconds: float = 4.5
    audio_rel: Optional[str] = None
    seed: int = 0


class SmartClipReq(BaseModel):
    video_rel: str
    start_sec: float = 0.0
    end_sec: float = 0.0
    target_aspect: str = "9:16"
    fade_seconds: float = 0.25


class SceneDialogueLineReq(BaseModel):
    character_id: Optional[str] = None
    voice_id: Optional[str] = None
    target_face: Optional[str] = None
    start_sec: Optional[float] = None
    text: str


class HiggsfieldSceneDirectorReq(BaseModel):
    scene_prompt: str
    camera_id: str = "arri_full_frame_8k"
    lens_id: str = "classic_anamorphic"
    focal_mm: str = "35"
    aperture: str = "f/2.8"
    lighting_id: str = "natural_cinema"
    motion_id: str = "dolly_in_dramatic"
    video_engine_mode: str = "auto"
    aspect_ratio: str = "16:9"
    duration_seconds: float = 4.5
    base_image_rel: Optional[str] = None
    character_ids: List[str] = Field(default_factory=list)
    dialogue_lines: List[SceneDialogueLineReq] = Field(default_factory=list)
    ambience_preset: str = "none"
    apply_lipsync: bool = True
    image_lora_ids: List[str] = Field(default_factory=list)
    video_lora_ids: List[str] = Field(default_factory=list)
    seed: int = 42


@app.get("/api/loras")
def api_list_loras(model_id: Optional[str] = None) -> Dict[str, Any]:
    loaded_ids = memory_manager.loaded_model_ids()
    all_defs = catalog_manager.list_catalog(loaded_ids=loaded_ids)
    loras = [d for d in all_defs if d.get("category") == "lora"]
    if model_id:
        filtered = []
        for d in loras:
            compat = d.get("compatible_models") or []
            if not compat or model_id in compat:
                filtered.append(d)
        loras = filtered
    return {
        "ok": True,
        "image_loras": [d for d in loras if d.get("lora_target", "image") == "image"],
        "video_loras": [d for d in loras if d.get("lora_target") == "video"],
    }


@app.get("/api/higgsfield/catalog")
def api_higgsfield_catalog() -> Dict[str, Any]:
    return {"ok": True, **get_higgsfield_catalog_payload()}


@app.post("/api/higgsfield/characters")
def api_save_higgsfield_character(req: CharacterSaveReq) -> Dict[str, Any]:
    app_prompt = req.appearance_prompt.strip()
    if not app_prompt and req.traits:
        app_prompt = build_character_appearance_from_traits(req.traits)
    try:
        saved = character_vault.save_character(
            name=req.name,
            appearance_prompt=app_prompt,
            anchor_image_rel=req.anchor_image_rel,
            voice_id=req.voice_id,
            default_face_side=req.default_face_side,
            traits=req.traits,
            char_id=req.char_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "character": saved, "characters": character_vault.list_characters()}


@app.delete("/api/higgsfield/characters/{char_id}")
def api_delete_higgsfield_character(char_id: str) -> Dict[str, Any]:
    character_vault.delete_character(char_id)
    return {"ok": True, "characters": character_vault.list_characters()}


@app.post("/api/higgsfield/cinema-shot")
def api_higgsfield_cinema_shot(req: CinemaShotReq) -> Dict[str, Any]:
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Describe la escena cinematográfica a fotografiar.")
    chars = [c for cid in req.character_ids if (c := character_vault.get_character(cid)) is not None]
    compiled = compile_cinema_prompt(
        user_prompt=req.prompt,
        camera_id=req.camera_id,
        lens_id=req.lens_id,
        focal_mm=req.focal_mm,
        aperture=req.aperture,
        lighting_id=req.lighting_id,
        characters=chars,
    )
    anchor_paths: List[Path] = []
    if req.base_image_rel:
        bp = _resolve_media_rel(req.base_image_rel)
        if bp:
            anchor_paths.append(bp)
    for ch in chars:
        ap = _resolve_media_rel(ch.get("anchor_image_rel")) if ch.get("anchor_image_rel") else None
        if ap and ap not in anchor_paths:
            anchor_paths.append(ap)

    if req.aspect_ratio == "9:16":
        w, h = 704, 1280
    elif req.aspect_ratio == "1:1":
        w, h = 1024, 1024
    else:
        w, h = 1280, 704

    job = job_manager.submit(
        kind="image",
        title=f"🎬 Cinema Shot ({req.camera_id} · {req.focal_mm}mm {req.aperture})",
        fn=lambda j: generate_or_edit_image(
            prompt=compiled,
            model_id="qwen-image-edit-2511-pack",
            base_image_path=anchor_paths[0] if anchor_paths else None,
            reference_paths=anchor_paths[:3] if len(anchor_paths) > 1 else None,
            width=w,
            height=h,
            steps=4,
            guidance_scale=1.0,
            seed=req.seed,
            use_lightning_lora=True,
            custom_lora_ids=req.image_lora_ids,
            job=j,
        ),
    )
    return {"ok": True, "compiled_prompt": compiled, "job": job.to_dict()}


@app.post("/api/higgsfield/motion-clip")
def api_higgsfield_motion_clip(req: HiggsfieldMotionReq) -> Dict[str, Any]:
    img_p = _resolve_media_rel(req.image_rel)
    if not img_p:
        raise HTTPException(status_code=400, detail=f"No se encontró la imagen inicial: {req.image_rel}")
    end_p = _resolve_media_rel(req.end_image_rel) if req.end_image_rel else None
    aud_p = _resolve_media_rel(req.audio_rel) if req.audio_rel else None

    m_spec = HIGGSFIELD_MOTIONS.get(req.motion_id) or HIGGSFIELD_MOTIONS["dolly_in_dramatic"]
    full_prompt = f"{req.prompt.strip()}, {m_spec['video_prompt']}".strip(", ")

    if req.aspect_ratio == "9:16":
        vw, vh = 704, 1280
    elif req.aspect_ratio == "1:1":
        vw, vh = 960, 960
    else:
        vw, vh = 832, 480

    def _run_hf_motion(j: Any) -> Dict[str, Any]:
        eff = (req.engine_mode or "auto").lower()
        if eff == "auto":
            if catalog_manager.is_downloaded("wan21-vace-14b-fp8"):
                eff = "ai_14b"
            elif catalog_manager.is_downloaded("wan21-vace-1.3b"):
                eff = "ai_fast"
            else:
                eff = "ffmpeg_instant"

        if eff in ("ai_14b", "ai_fast"):
            mid = "wan21-vace-14b-fp8" if eff == "ai_14b" else "wan21-vace-1.3b"
            lid = "lora-wan21-14b-fusionx-phantom" if eff == "ai_14b" else "lora-wan21-1.3b-bidirect2"
            if not catalog_manager.is_downloaded(lid):
                lid = ""
            submode = "flf2v" if end_p is not None else "i2v"
            return generate_video_clip(
                mode="ai",
                video_submode=submode,
                model_id=mid,
                video_lora_id=lid or None,
                extra_video_lora_ids=req.video_lora_ids,
                prompt=full_prompt,
                image_path=img_p,
                end_image_path=end_p,
                width=vw,
                height=vh,
                duration_seconds=req.duration_seconds,
                fps=16,
                steps=6 if lid else 24,
                guidance_scale=1.0 if lid else 5.0,
                seed=req.seed,
                audio_path=aud_p,
                job=j,
            )

        t0 = time.time()
        VIDEOS_OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_name = f"hf_motion_{req.motion_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        out_p = VIDEOS_OUT_DIR / out_name
        if end_p is not None:
            info = generate_video_clip(
                mode="ffmpeg_motion",
                video_submode="flf2v",
                model_id="ffmpeg-motion-cpu",
                prompt=full_prompt,
                image_path=img_p,
                end_image_path=end_p,
                width=vw,
                height=vh,
                duration_seconds=req.duration_seconds,
                fps=24,
                audio_path=aud_p,
                job=j,
            )
            return info

        info = render_higgsfield_cpu_motion_clip(
            image_path=img_p,
            output_mp4=out_p,
            motion_id=req.motion_id,
            width=vw,
            height=vh,
            fps=24,
            duration_sec=req.duration_seconds,
            audio_path=aud_p,
            job=j,
        )
        meta = {
            "id": out_p.stem,
            "filename": out_name,
            "rel_path": f"outputs/videos/{out_name}",
            "media_url": f"/media/outputs/videos/{out_name}",
            "type": "video",
            "model_id": "higgsfield-ffmpeg-cpu",
            "model_name": f"Higgsfield Motion ({m_spec['label']})",
            "motion_id": req.motion_id,
            "duration_seconds": info["duration_seconds"],
            "width": vw,
            "height": vh,
            "fps": 24,
            "elapsed_seconds": round(time.time() - t0, 2),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        (VIDEOS_OUT_DIR / f"{out_p.stem}.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        j.log(f"Clip Higgsfield listo: {out_name} ({info['duration_seconds']}s).", level="success", progress=100.0)
        return meta

    job = job_manager.submit(
        kind="video",
        title=f"🌪️ Higgsfield Motion: {m_spec['label']}",
        fn=_run_hf_motion,
    )
    return {"ok": True, "job": job.to_dict()}


@app.post("/api/higgsfield/smart-clip")
def api_higgsfield_smart_clip(req: SmartClipReq) -> Dict[str, Any]:
    vid_p = _resolve_media_rel(req.video_rel)
    if not vid_p:
        raise HTTPException(status_code=400, detail=f"No se encontró el vídeo: {req.video_rel}")
    job = job_manager.submit(
        kind="video",
        title=f"✂️ Smart Clipper ({req.target_aspect} · {vid_p.name})",
        fn=lambda j: smart_clip_and_reframe_video(
            video_path=vid_p,
            start_sec=req.start_sec,
            end_sec=req.end_sec,
            target_aspect=req.target_aspect,
            fade_seconds=req.fade_seconds,
            job=j,
        ),
    )
    return {"ok": True, "job": job.to_dict()}


@app.post("/api/higgsfield/scene-director")
def api_higgsfield_scene_director(req: HiggsfieldSceneDirectorReq) -> Dict[str, Any]:
    if not req.scene_prompt.strip():
        raise HTTPException(status_code=400, detail="Escribe la descripción de la escena para el Auto-Director.")
    d_lines = [dl.model_dump() for dl in req.dialogue_lines]
    job = job_manager.submit(
        kind="lipsync",
        title=f"🎬 Auto-Director Higgsfield: {req.scene_prompt.strip()[:36]}",
        fn=lambda j: run_higgsfield_scene_director(
            job=j,
            scene_prompt=req.scene_prompt,
            resolve_rel_fn=_resolve_media_rel,
            camera_id=req.camera_id,
            lens_id=req.lens_id,
            focal_mm=req.focal_mm,
            aperture=req.aperture,
            lighting_id=req.lighting_id,
            motion_id=req.motion_id,
            video_engine_mode=req.video_engine_mode,
            aspect_ratio=req.aspect_ratio,
            duration_seconds=req.duration_seconds,
            base_image_rel=req.base_image_rel,
            character_ids=req.character_ids,
            dialogue_lines=d_lines,
            ambience_preset=req.ambience_preset,
            apply_lipsync=req.apply_lipsync,
            image_lora_ids=req.image_lora_ids,
            video_lora_ids=req.video_lora_ids,
            seed=req.seed,
        ),
    )
    return {"ok": True, "job": job.to_dict()}


# ==================== COHERENCIA VISUAL, EXPORTAR LOG DE SESIÓN Y ZIP ====================


class CoherenceAnalyzeReq(BaseModel):
    anchor_image_rel: str
    candidate_image_rels: List[str] = Field(min_length=1)
    scene_goal: str = ""
    mode: str = "hybrid_auto"
    strictness_threshold: float = 78.0
    expect_spatial_progression: bool = False


@app.post("/api/coherence/analyze")
def api_analyze_coherence(req: CoherenceAnalyzeReq) -> Dict[str, Any]:
    anchor_p = _resolve_media_rel(req.anchor_image_rel)
    if not anchor_p:
        raise HTTPException(status_code=400, detail=f"No se encontró la imagen ancla: {req.anchor_image_rel}")
    cand_ps: List[Path] = []
    for r in req.candidate_image_rels:
        cp = _resolve_media_rel(r)
        if not cp:
            raise HTTPException(status_code=400, detail=f"No se encontró el fotograma candidato: {r}")
        cand_ps.append(cp)

    job = job_manager.submit(
        kind="coherence",
        title=f"Coherencia ({req.mode} · {len(cand_ps)} fotos vs {anchor_p.name})",
        fn=lambda j: run_coherence_analysis(
            anchor_path=anchor_p,
            candidate_paths=cand_ps,
            scene_goal=req.scene_goal,
            mode=req.mode,
            strictness_threshold=req.strictness_threshold,
            expect_spatial_progression=req.expect_spatial_progression,
            job=j,
        ),
    )
    return {"ok": True, "job": job.to_dict()}


def _generate_current_session_report(fmt: str = "md") -> tuple[str, str]:
    downloaded = [
        {
            "id": m["id"],
            "name": m["name"],
            "category": m["category"],
            "disk_gb": catalog_manager.disk_usage_gb(m["id"]),
        }
        for m in catalog_manager.get_all_definitions()
        if not m.get("builtin_no_download") and catalog_manager.is_downloaded(m["id"])
    ]
    return build_full_session_log_document(
        jobs_list=job_manager.list_jobs(limit=500, include_logs=True),
        loaded_summary=memory_manager.status_summary(),
        downloaded_models=downloaded,
        fmt=fmt,
    )


@app.get("/api/logs/diagnostics")
def api_export_diagnostics() -> FileResponse:
    """Bounded, redacted raw process/tunnel/memory logs; no weights or output media."""
    from studio.diagnostics import export_diagnostic_bundle
    path = export_diagnostic_bundle(DATA_ROOT, TEMP_DIR / f"studio_diagnostics_{uuid.uuid4().hex[:10]}.zip")
    return FileResponse(path, media_type="application/zip", filename=path.name)


@app.get("/api/logs/export")
def api_export_session_log(format: str = "md") -> Response:
    """
    Descarga un documento `.md` o `.txt` con TODO el registro de la sesión desde que se encendió
    Studio IA hasta el instante actual (ideal en móvil para compartir con una IA sin límite de caracteres).
    """
    fmt = "txt" if str(format).lower() == "txt" else "md"
    colab_console_log(f"Generando e imprimiendo documento completo de registro de sesión (.{fmt})...", icon="📄")
    filename, content = _generate_current_session_report(fmt=fmt)
    media_type = "text/plain; charset=utf-8" if fmt == "txt" else "text/markdown; charset=utf-8"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return Response(content=content.encode("utf-8"), media_type=media_type, headers=headers)


@app.get("/api/export/zip")
def api_export_zip() -> StreamingResponse:
    """Empaqueta todas las imágenes, vídeos, voces, montajes y el log completo de sesión en un .zip descargable."""
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
        try:
            log_fn, log_body = _generate_current_session_report(fmt="md")
            zf.writestr(f"registro_sesion/{log_fn}", log_body.encode("utf-8"))
        except Exception:
            pass
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
