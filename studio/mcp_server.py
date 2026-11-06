"""
Servidor MCP (Model Context Protocol) integrado en Studio IA.

Permite controlar Studio IA ejecutándose en Google Colab desde cualquier agente de terminal
(Claude Code, Codex CLI, Gemini CLI, OpenCode, Cursor, Goose, etc.) tanto mediante:
  1. Puente local `stdio` (`studio_mcp.py`, cero timeouts de Cloudflare + subida/bajada de archivos locales).
  2. Conexión remota directa Streamable HTTP (`POST /mcp` con JSON-RPC 2.0).

Todas las acciones invocadas por el agente quedan registradas EN VIVO en la celda de Google Colab
y en la barra de actividad de la interfaz web.
"""

from __future__ import annotations

import base64
import json
import os
import secrets
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from fastapi import APIRouter, Header, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse

from studio.catalog import catalog_manager, evaluate_gpu_tier, resolve_gpu_profile
from studio.config import (
    AUDIO_OUT_DIR,
    DATA_ROOT,
    IMAGES_OUT_DIR,
    MONTAGES_OUT_DIR,
    UPLOADS_DIR,
    VIDEOS_OUT_DIR,
    ensure_directories,
)
from studio.engines.ffmpeg_utils import probe_media_info, render_montage_mp4, verify_image_file
from studio.engines.image_engine import (
    generate_or_edit_image,
    load_image_model_into_memory,
    process_image_cpu_ffmpeg_only,
)
from studio.engines.tts_engine import EDGE_TTS_VOICES, load_tts_model_into_memory, synthesize_voice
from studio.engines.video_engine import generate_video_clip, load_video_model_into_memory
from studio.hardware import colab_console_log, hardware_tracker, query_gpu_info
from studio.jobs import job_manager
from studio.memory_manager import memory_manager

ROOT_REPO_DIR = Path(__file__).resolve().parent.parent
BRIDGE_SCRIPT_PATH = ROOT_REPO_DIR / "studio_mcp.py"

_mcp_runtime_state: Dict[str, Any] = {
    "public_url": os.environ.get("STUDIO_PUBLIC_URL", "").strip(),
    "session_id": f"studio-mcp-{uuid.uuid4().hex[:12]}",
    "connected_clients": 0,
}


def get_expected_mcp_token() -> str:
    return os.environ.get("STUDIO_MCP_TOKEN", "").strip()


def set_mcp_public_url(url: str) -> None:
    _mcp_runtime_state["public_url"] = (url or "").strip().rstrip("/")


def get_mcp_connection_details(request: Optional[Request] = None) -> Dict[str, Any]:
    pub = _mcp_runtime_state.get("public_url") or ""
    if not pub and request is not None:
        scheme = request.headers.get("x-forwarded-proto") or request.url.scheme or "http"
        host = request.headers.get("x-forwarded-host") or request.headers.get("host") or "127.0.0.1:7860"
        pub = f"{scheme}://{host}".rstrip("/")
    if not pub:
        port = os.environ.get("PORT", "7860")
        pub = f"http://127.0.0.1:{port}"

    tok = get_expected_mcp_token()
    mcp_http_url = f"{pub}/mcp"
    mcp_url_with_token = f"{pub}/mcp?token={tok}" if tok else mcp_http_url
    bridge_url = f"{pub}/mcp/bridge.py"
    bridge_preconfigured_url = f"{pub}/mcp/bridge.py?token={tok}" if tok else bridge_url
    agent_guide_url = f"{pub}/mcp/guia?token={tok}" if tok else f"{pub}/mcp/guia"

    tok_arg = f" --token {tok}" if tok else ""
    auth_hdr = f' --header "Authorization: Bearer {tok}"' if tok else ""

    curl_bridge = f'curl -fsSL "{bridge_preconfigured_url}" -o studio_mcp.py'
    termux_one_liner = f'{curl_bridge} && python3 studio_mcp.py --cli studio_get_state_and_manual'
    agent_chat_prompt = (
        f"Conéctate a mi servidor Studio IA en Google Colab leyendo primero con curl este enlace oficial "
        f"(incluye URL, token de sesión y el manual obligatorio sin inventar parámetros): {agent_guide_url}"
    )
    claude_stdio = (
        f'{curl_bridge} && (claude mcp remove studio-ia >/dev/null 2>&1 || true) && '
        f'claude mcp add studio-ia -- python3 "$(pwd)/studio_mcp.py" --url {pub}{tok_arg} --out-dir "$(pwd)/salidas_studio"'
    )
    claude_powershell = (
        f'curl.exe -fsSL "{bridge_preconfigured_url}" -o studio_mcp.py; claude mcp remove studio-ia 2>$null; '
        f'claude mcp add studio-ia -- python "$PWD\\studio_mcp.py" --url {pub}{tok_arg} --out-dir "$PWD\\salidas_studio"'
    )
    codex_stdio = (
        f'{curl_bridge} && (codex mcp remove studio-ia >/dev/null 2>&1 || true) && '
        f'codex mcp add studio-ia -- python3 "$(pwd)/studio_mcp.py" --url {pub}{tok_arg} --out-dir "$(pwd)/salidas_studio"'
    )
    claude_http = (
        f'(claude mcp remove studio-ia >/dev/null 2>&1 || true) && '
        f'claude mcp add --transport http studio-ia {mcp_http_url}{auth_hdr}'
    )

    json_args = ["studio_mcp.py", "--url", pub]
    if tok:
        json_args += ["--token", tok]
    json_args += ["--out-dir", "./salidas_studio"]

    mcp_json_snippet = json.dumps(
        {
            "mcpServers": {
                "studio-ia": {
                    "command": "python3",
                    "args": json_args,
                }
            }
        },
        indent=2,
        ensure_ascii=False,
    )

    return {
        "base_url": pub,
        "mcp_endpoint": mcp_http_url,
        "mcp_url_with_token": mcp_url_with_token,
        "agent_guide_url": agent_guide_url,
        "bridge_script_url": bridge_url,
        "bridge_preconfigured_url": bridge_preconfigured_url,
        "token_required": bool(tok),
        "token": tok,
        "commands": {
            "download_bridge": curl_bridge,
            "termux_one_liner": termux_one_liner,
            "agent_chat_prompt": agent_chat_prompt,
            "claude_code_stdio": claude_stdio,
            "claude_code_powershell": claude_powershell,
            "codex_cli_stdio": codex_stdio,
            "claude_code_http": claude_http,
            "mcp_servers_json": mcp_json_snippet,
        },
    }


# ==================== MANUALES E INSTRUCCIONES ESTRICTAS ANTI-ALUCINACIÓN ====================

MCP_SERVER_INSTRUCTIONS = """\
ESTÁS CONECTADO A «STUDIO IA» (SERVIDOR GENERATIVO GPU EN GOOGLE COLAB).
SIGUE OBLIGATORIAMENTE ESTE MANUAL OPERATIVO Y NO INVENTES NINGÚN ID NI PARÁMETRO:

1. DIAGNÓSTICO INICIAL OBLIGATORIO:
   - En tu primera acción de la sesión, llama SIEMPRE a `studio_get_state_and_manual`.
   - Comprueba la GPU activa (ej. T4 15 GB, L4 22.5 GB, A100 40/80 GB), la VRAM libre, el disco libre (`free_disk_gb`) y los archivos existentes.

2. CERO INVENCIÓN DE MODELOS O PARÁMETROS:
   - Antes de lanzar una tarea nueva si dudas qué modelo o LoRA elegir, llama a `studio_recommend_pipeline(task=..., priority=...)`. Te devolverá la combinación exacta verificada para la GPU actual.
   - En `studio_generate_video` y `studio_generate_or_edit_image` mantén `use_official_defaults=true` (valor por defecto): Studio IA aplica automáticamente `steps`, `guidance_scale`, `video_lora_scale`, `width`, `height` y `fps` óptimos del fabricante.

3. REGLAS ESTRICTAS DE VÍDEO Y «FRAME A FRAME» (INICIO → FIN):
   - `video_submode="t2v"` → Texto a Vídeo (no pases imágenes).
   - `video_submode="i2v"` → Imagen a Vídeo (requiere `image_rel` o `local_image_path`).
   - `video_submode="flf2v"` → Frame a Frame Inicio → Fin (requiere OBLIGATORIAMENTE imagen inicial Y final: `image_rel` + `end_image_rel`, o `local_image_path` + `local_end_image_path`).
   - En `flf2v` ESTÁ PROHIBIDO usar modelos de 1 sola imagen (`wan22-ti2v-5b`, `wan22-ti2v-5b-turbo`, `sana-video-2b-480p`, `cogvideox-5b-i2v`) y PROHIBIDO usar LoRAs causales (`lora-wan22-ti2v-5b-turbo`).
   - Recetas oficiales `flf2v` en GPU L4 (22.5 GB):
     a) Máxima textura y anatomía (14B FP8 + LoRA 6 pasos, ~1.6 min):
        `model_id="wan21-vace-14b-fp8"` + `video_lora_id="lora-wan21-14b-fusionx-phantom"`
     b) Especialista puro 14B HD 720P Frame a Frame (~1.8 min):
        `model_id="wan21-flf2v-14b-fp8"` + `video_lora_id="lora-wan21-14b-fusionx-i2v"` (o `"lora-wan21-14b-lightx2v-i2v"`)
     c) 13B destilado de fábrica en 8 pasos sin LoRA (~45 s, prompt en inglés):
        `model_id="ltx-video-13b-distilled"`
     d) Prototipo rápido o GPU T4 (~25 s):
        `model_id="wan21-vace-1.3b"` + `video_lora_id="lora-wan21-1.3b-bidirect2"`

4. GESTIÓN DE ESPACIO EN DISCO Y MEMORIA EN COLAB:
   - Si una descarga avisa `DISK_SPACE_INSUFFICIENT`, consulta las sugerencias devueltas y usa `studio_free_memory_or_disk(action="delete_from_disk", model_id=...)` (preguntando antes al usuario si procede).
   - Todo lo que ejecutes se muestra EN TIEMPO REAL en la celda de Google Colab del usuario.
"""

MANUAL_RESOURCES: Dict[str, Dict[str, str]] = {
    "studio://manual/quickstart": {
        "name": "Manual Rápido Oficial de Studio IA (Checklist Paso a Paso)",
        "mimeType": "text/markdown",
        "content": (
            "# Manual Operativo Oficial — Studio IA MCP\n\n"
            "## Flujo de trabajo obligatorio sin errores\n"
            "1. **Inspeccionar entorno:** Llama a `studio_get_state_and_manual` para conocer GPU, VRAM, disco libre y archivos en galería.\n"
            "2. **Pedir receta verificada:** Llama a `studio_recommend_pipeline(task=..., priority=...)` para recibir el JSON exacto recomendado.\n"
            "3. **Subir o localizar entradas:**\n"
            "   - Si estás conectado mediante `studio_mcp.py` (puente local), pasa directamente rutas de tu PC (`local_image_path`, `local_end_image_path`, `save_to_local_path`).\n"
            "   - Si estás conectado por HTTP directo, usa `studio_list_assets` o `studio_upload_file_base64` para obtener `rel_path` (ej. `uploads/foto.png`).\n"
            "4. **Generar con `use_official_defaults=true`:** No inventes pasos, CFG ni escalas LoRA; deja que el motor aplique los valores verificados.\n"
            "5. **Entregar resultado:** Devuelve siempre al usuario el `rel_path`, el tiempo empleado y, si usaste `save_to_local_path`, la ruta guardada en su PC.\n"
        ),
    },
    "studio://manual/video-recipes": {
        "name": "Tabla Oficial de Recetas de Vídeo (Frame a Frame, I2V, T2V y LoRAs)",
        "mimeType": "text/markdown",
        "content": (
            "# Recetas Verificadas de Vídeo por Modo y GPU\n\n"
            "## A. Modo `flf2v` — Frame a Frame (Fotograma Inicial → Fotograma Final)\n"
            "| Prioridad | GPU mínima | `model_id` | `video_lora_id` | Pasos / CFG / Escala (auto) | Tiempo L4 |\n"
            "|---|---|---|---|---|---|\n"
            "| ★ Textura Pro 14B | L4 22.5 GB | `wan21-vace-14b-fp8` | `lora-wan21-14b-fusionx-phantom` | 6 pasos · CFG 1.0 · escala 0.90 | ~1.6 min |\n"
            "| ★ 14B FLF2V Puro HD | L4 22.5 GB | `wan21-flf2v-14b-fp8` | `lora-wan21-14b-fusionx-i2v` | 8 pasos · CFG 1.0 · escala 0.85 | ~1.8 min |\n"
            "| 14B Fun-InP | L4 22.5 GB | `wan21-fun-inp-14b-fp8` | `lora-wan21-14b-lightx2v-i2v` | 6 pasos · CFG 1.0 · escala 0.75 | ~1.6 min |\n"
            "| ⚡ 13B Destilado Nativo | L4 22.5 GB | `ltx-video-13b-distilled` | *(ninguno, ya destilado)* | 8 pasos · CFG 1.0 (prompt EN) | ~45 s |\n"
            "| Ligero / T4 Rápido | T4 15 GB | `wan21-vace-1.3b` | `lora-wan21-1.3b-bidirect2` | 8 pasos · CFG 1.5 · escala 0.65 | ~25 s |\n\n"
            "**Regla crítica:** NUNCA uses LoRAs causales (`lora-wan22-ti2v-5b-turbo`) en `flf2v`; solo atención bidireccional.\n\n"
            "## B. Modo `i2v` (1 Imagen a Vídeo) y `t2v` (Texto a Vídeo)\n"
            "- **HD 720P Rápido en L4 (~1.2 min):** `model_id='wan22-ti2v-5b'` + `video_lora_id='lora-wan22-ti2v-5b-turbo'` (6 pasos, CFG 1.0).\n"
            "- **4 Pasos de Fábrica (~1.0 min):** `model_id='wan22-ti2v-5b-turbo'` (4 pasos, CFG 1.0, sin LoRA extra).\n"
            "- **14B Versátil en L4 (~1.6 min):** `model_id='wan21-vace-14b-fp8'` + `video_lora_id='lora-wan21-14b-fusionx-phantom'`.\n"
            "- **0 GB Instantáneo CPU (2 s):** `mode='ffmpeg_motion'`, `model_id='ffmpeg-motion-cpu'`.\n"
        ),
    },
    "studio://manual/image-tts-montage": {
        "name": "Manual Oficial de Imagen Qwen, Voz Neural TTS y Montaje MP4",
        "mimeType": "text/markdown",
        "content": (
            "# Imagen, Voz y Montaje Multipista\n\n"
            "## 1. Imagen (`studio_generate_or_edit_image`)\n"
            "- Motor por defecto: `qwen-image-edit-2511-pack` (4-bit NF4 + Lightning LoRA 4 pasos, ~18–25 s en L4).\n"
            "- Soporta creación desde cero (`base_image_rel` vacío), edición de 1 foto + `reference_rels`, e Inpainting con máscara (`mask_image_rel`, blanco=editar, negro=intacto).\n\n"
            "## 2. Voz TTS (`studio_generate_speech`)\n"
            "- Motor instantáneo 0 GB: `engine_id='edge-tts-cloud'` (ej. `es-ES-AlvaroNeural`, `es-ES-ElviraNeural`, `es-MX-DaliaNeural`).\n"
            "- Motores GPU locales: `kokoro-82m-tts`, `mms-tts-es`, `bark-voice-Cloning`.\n\n"
            "## 3. Montaje Final (`studio_render_montage`)\n"
            "- Combina planos (`images` y/o `videos`) con su `audio_rel` individual (`fit_to_audio=true` ajusta la duración del plano a la locución), `global_audio_rel` (música de fondo) y `fade_seconds`.\n"
        ),
    },
}


# ==================== UTILIDADES INTERNAS MCP ====================


def _verify_mcp_auth(
    request: Request,
    authorization: Optional[str] = None,
    x_studio_token: Optional[str] = None,
    token_query: Optional[str] = None,
) -> None:
    expected = get_expected_mcp_token()
    if not expected:
        return

    provided = ""
    auth_hdr = authorization or request.headers.get("authorization") or ""
    if auth_hdr.lower().startswith("bearer "):
        provided = auth_hdr[7:].strip()
    elif x_studio_token or request.headers.get("x-studio-token"):
        provided = (x_studio_token or request.headers.get("x-studio-token") or "").strip()
    elif token_query or request.query_params.get("token"):
        provided = (token_query or request.query_params.get("token") or "").strip()

    if not provided or not secrets.compare_digest(provided, expected):
        colab_console_log("Intento de conexión MCP bloqueado: token ausente o incorrecto.", icon="🛡️")
        raise HTTPException(
            status_code=401,
            detail="Token MCP inválido o ausente. Pasa Authorization: Bearer <STUDIO_MCP_TOKEN>.",
        )


def _resolve_rel_safe(rel: Optional[str]) -> Optional[Path]:
    if not rel:
        return None
    cleaned = rel.strip()
    if cleaned.startswith("/media/"):
        cleaned = cleaned[len("/media/") :]
    cleaned = cleaned.lstrip("/")
    cand = (DATA_ROOT / cleaned).resolve()
    root_res = DATA_ROOT.resolve()
    if root_res not in cand.parents and cand != root_res:
        raise ValueError(f"Ruta fuera del directorio permitido de Studio IA: {rel}")
    if not cand.exists():
        raise FileNotFoundError(
            f"No existe `{cleaned}` en Studio IA. Llama a `studio_list_assets` para ver los `rel_path` exactos o sube el archivo primero."
        )
    return cand


def _list_assets_summary(bucket_filter: str = "all", limit_per_bucket: int = 25) -> Dict[str, List[Dict[str, Any]]]:
    buckets_spec = [
        ("uploads", UPLOADS_DIR, "uploads", "upload"),
        ("images", IMAGES_OUT_DIR, "outputs/images", "image"),
        ("videos", VIDEOS_OUT_DIR, "outputs/videos", "video"),
        ("audio", AUDIO_OUT_DIR, "outputs/audio", "audio"),
        ("montages", MONTAGES_OUT_DIR, "outputs/montages", "video"),
    ]
    out: Dict[str, List[Dict[str, Any]]] = {}
    for key, folder, prefix, def_type in buckets_spec:
        if bucket_filter != "all" and bucket_filter != key:
            continue
        items: List[Dict[str, Any]] = []
        if folder.exists():
            files = sorted(
                [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() != ".json"],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )[: max(1, min(100, limit_per_bucket))]
            for fp in files:
                meta_f = folder / f"{fp.stem}.json"
                meta: Dict[str, Any] = {}
                if meta_f.exists():
                    try:
                        meta = json.loads(meta_f.read_text(encoding="utf-8"))
                    except Exception:
                        meta = {}
                items.append(
                    {
                        "filename": fp.name,
                        "rel_path": f"{prefix}/{fp.name}",
                        "media_url": f"/media/{prefix}/{fp.name}",
                        "type": meta.get("type", def_type),
                        "width": meta.get("width"),
                        "height": meta.get("height"),
                        "duration_seconds": meta.get("duration_seconds"),
                        "model_id": meta.get("model_id"),
                        "size_kb": round(fp.stat().st_size / 1024, 1),
                        "created_at": datetime.fromtimestamp(fp.stat().st_mtime).isoformat(timespec="seconds"),
                    }
                )
        out[key] = items
    return out


def _wait_for_job_bounded(job_id: str, max_wait_seconds: float = 82.0) -> Dict[str, Any]:
    deadline = time.time() + max(1.0, min(88.0, float(max_wait_seconds)))
    while True:
        job = job_manager.get(job_id)
        if not job:
            return {
                "ok": False,
                "error_code": "JOB_NOT_FOUND",
                "message": f"No existe el trabajo {job_id}.",
            }
        snap = job.to_dict(include_logs=True)
        st = snap["status"]
        if st == "completed":
            res = dict(snap.get("result") or {})
            if not res.get("rel_path") and str(res.get("media_url", "")).startswith("/media/"):
                res["rel_path"] = str(res["media_url"])[len("/media/") :]
            return {
                "ok": True,
                "job_id": job_id,
                "status": "completed",
                "elapsed_seconds": snap.get("elapsed_seconds"),
                "result": res,
            }
        if st in ("failed", "cancelled"):
            return {
                "ok": False,
                "error_code": f"JOB_{st.upper()}",
                "job_id": job_id,
                "status": st,
                "error": snap.get("error") or "Trabajo interrumpido.",
                "recent_logs": (snap.get("logs") or [])[-8:],
            }
        if time.time() >= deadline:
            return {
                "ok": True,
                "job_id": job_id,
                "status": st,
                "progress_pct": snap.get("progress"),
                "current_step": snap.get("current_step"),
                "speed_text": snap.get("speed_text"),
                "next_action_required": (
                    f"El trabajo sigue ejecutándose en Google Colab ({snap.get('progress')}% · {snap.get('current_step')}). "
                    f"Llama inmediatamente a `studio_wait_for_job(job_id='{job_id}')` para continuar esperando hasta el 100%."
                ),
            }
        time.sleep(1.0)


def _guided_error(error_code: str, message: str, exact_fix_action: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    colab_console_log(f"[MCP Guardia] {error_code}: {message} → Guía enviada al agente.", icon="⚠️")
    payload: Dict[str, Any] = {
        "ok": False,
        "error_code": error_code,
        "why_it_failed": message,
        "exact_fix_action": exact_fix_action,
    }
    if extra:
        payload.update(extra)
    return {
        "content": [{"type": "text", "text": json.dumps(payload, indent=2, ensure_ascii=False)}],
        "isError": True,
    }


def _ok_result(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "content": [{"type": "text", "text": json.dumps(payload, indent=2, ensure_ascii=False)}],
        "isError": False,
    }


# ==================== DEFINICIÓN DINÁMICA DE HERRAMIENTAS CON ENUMS CERRADOS ====================


def _build_mcp_tools_schema() -> List[Dict[str, Any]]:
    all_defs = catalog_manager.get_all_definitions()
    all_model_ids = [m["id"] for m in all_defs]
    video_model_ids = [m["id"] for m in all_defs if m["category"] == "video"]
    video_lora_ids = [""] + [m["id"] for m in all_defs if m["category"] == "lora" and m.get("lora_target") == "video"]
    image_lora_ids = [m["id"] for m in all_defs if m["category"] == "lora" and (m.get("lora_target") or "image") == "image"]
    tts_engine_ids = [m["id"] for m in all_defs if m["category"] == "tts"]
    voice_ids = [v["id"] for v in EDGE_TTS_VOICES]

    return [
        {
            "name": "studio_get_state_and_manual",
            "description": (
                "PASO 1 OBLIGATORIO AL INICIAR SESIÓN. Devuelve el estado real de Google Colab "
                "(GPU exacta, VRAM total/usada, RAM, GB libres en disco, créditos restantes, modelos ya descargados, "
                "modelo cargado actualmente en VRAM y últimos archivos) junto con las reglas operativas."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "include_recent_assets": {
                        "type": "boolean",
                        "default": True,
                        "description": "Incluir los últimos archivos subidos y generados con su `rel_path`.",
                    }
                },
            },
        },
        {
            "name": "studio_recommend_pipeline",
            "description": (
                "ORÁCULO ANTI-INVENCIÓN. Indícale la tarea (`video_flf2v`, `video_i2v`, `video_t2v`, `image_create`, "
                "`image_edit`, `tts`, `montage`) y la prioridad (`max_quality_14b`, `balanced_fast`, `ultra_fast_draft`, "
                "`zero_vram_cpu`), y te devuelve los IDs y parámetros oficiales exactos para la GPU y disco actuales."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "enum": [
                            "video_flf2v",
                            "video_i2v",
                            "video_t2v",
                            "image_create",
                            "image_edit",
                            "tts",
                            "montage",
                        ],
                        "description": "Tipo exacto de tarea multimedia que quieres ejecutar.",
                    },
                    "priority": {
                        "type": "string",
                        "enum": [
                            "max_quality_14b",
                            "balanced_fast",
                            "ultra_fast_draft",
                            "zero_vram_cpu",
                        ],
                        "default": "max_quality_14b",
                        "description": "Prioridad entre máxima textura/anatomía (14B FP8 + LoRA), velocidad media o borrador rápido (1.3B).",
                    },
                },
                "required": ["task"],
            },
        },
        {
            "name": "studio_list_models",
            "description": (
                "Lista los modelos y LoRAs verificados del catálogo filtrando por categoría, modo de vídeo "
                "(`t2v`, `i2v`, `flf2v`) y compatibilidad real con la GPU de Colab."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ["all", "video", "image", "lora", "tts"],
                        "default": "video",
                    },
                    "required_video_submode": {
                        "type": "string",
                        "enum": ["any", "flf2v", "i2v", "t2v"],
                        "default": "any",
                        "description": "Filtra solo modelos/LoRAs compatibles con un submodo concreto (ej. 'flf2v' para Frame a Frame).",
                    },
                    "only_gpu_compatible": {
                        "type": "boolean",
                        "default": True,
                        "description": "Oculta modelos que excedan la VRAM de la GPU actual.",
                    },
                },
            },
        },
        {
            "name": "studio_prepare_model",
            "description": (
                "Descarga de Hugging Face (si aún no está en disco, verificando espacio antes) y/o carga en RAM/VRAM "
                "cualquier modelo o LoRA del catálogo."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "enum": all_model_ids,
                        "description": "ID exacto del modelo o LoRA en el catálogo.",
                    },
                    "load_into_vram": {
                        "type": "boolean",
                        "default": True,
                        "description": "Si es true, además de descargarlo lo deja cargado y listo en RAM/VRAM (para LoRAs solo descarga a disco).",
                    },
                    "wait": {
                        "type": "boolean",
                        "default": True,
                        "description": "Esperar a que termine la preparación antes de devolver.",
                    },
                },
                "required": ["model_id"],
            },
        },
        {
            "name": "studio_free_memory_or_disk",
            "description": (
                "Libera pesos de RAM/VRAM (`unload_vram`, sin borrar del disco) o elimina un modelo descargado "
                "del disco de Colab (`delete_from_disk`) para recuperar gigabytes libres."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["unload_vram", "delete_from_disk"],
                        "default": "unload_vram",
                    },
                    "target": {
                        "type": "string",
                        "default": "all",
                        "description": "Para `unload_vram`: 'all', 'video', 'image', 'tts' o un `model_id`. Para `delete_from_disk`: el `model_id` exacto a borrar.",
                    },
                },
                "required": ["action"],
            },
        },
        {
            "name": "studio_list_assets",
            "description": (
                "Lista los archivos disponibles en Google Colab (`uploads`, `images`, `videos`, `audio`, `montages`) "
                "con su `rel_path` exacto para encadenarlos en imagen, vídeo o montaje."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "bucket": {
                        "type": "string",
                        "enum": ["all", "uploads", "images", "videos", "audio", "montages"],
                        "default": "all",
                    },
                    "limit": {"type": "integer", "default": 25, "minimum": 1, "maximum": 100},
                },
            },
        },
        {
            "name": "studio_upload_file_base64",
            "description": (
                "Sube una imagen, máscara PNG, vídeo o audio a la carpeta `uploads/` de Studio IA en Colab y devuelve su `rel_path`. "
                "(Si usas el puente local `studio_mcp.py`, pasa `local_file_path` en vez de base64)."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Nombre del archivo con extensión (ej. 'pose_inicio.png').",
                    },
                    "data_base64": {
                        "type": "string",
                        "description": "Contenido del archivo codificado en Base64 (o data URL).",
                    },
                },
                "required": ["filename", "data_base64"],
            },
        },
        {
            "name": "studio_generate_or_edit_image",
            "description": (
                "Genera una imagen desde cero, edita una foto existente con referencias múltiples o aplica Inpainting "
                "con máscara usando Qwen Image Edit 2511 + LoRA Lightning 4 pasos + normalización de color BT.709."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Instrucción clara de qué generar o qué cambiar en la imagen.",
                    },
                    "base_image_rel": {
                        "type": "string",
                        "description": "Opcional: `rel_path` en Colab de la imagen base a editar (vacío para crear desde cero).",
                    },
                    "reference_rels": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Opcional: lista de `rel_path` de imágenes de referencia adicionales (ropa, cara, fondo).",
                    },
                    "mask_image_rel": {
                        "type": "string",
                        "description": "Opcional: `rel_path` de máscara PNG binaria (blanco=zona a regenerar, negro=intacto al 100%).",
                    },
                    "aspect_ratio": {
                        "type": "string",
                        "enum": ["1:1", "16:9", "9:16", "4:3", "3:4", "custom"],
                        "default": "1:1",
                        "description": "Formato oficial Qwen Image (1:1=1024×1024, 16:9=1280×720, 9:16=720×1280).",
                    },
                    "width": {"type": "integer", "default": 1024, "minimum": 256, "maximum": 1536},
                    "height": {"type": "integer", "default": 1024, "minimum": 256, "maximum": 1536},
                    "use_official_defaults": {
                        "type": "boolean",
                        "default": True,
                        "description": "Mantenlo en `true` para aplicar automáticamente 4 pasos con LoRA Lightning y CFG 4.0.",
                    },
                    "steps": {"type": "integer", "default": 4, "minimum": 1, "maximum": 50},
                    "guidance_scale": {"type": "number", "default": 4.0},
                    "use_lightning_lora": {"type": "boolean", "default": True},
                    "custom_lora_ids": {
                        "type": "array",
                        "items": {"type": "string", "enum": image_lora_ids if image_lora_ids else [""]},
                        "description": "LoRAs de imagen opcionales del catálogo.",
                    },
                    "seed": {"type": "integer", "default": 0, "description": "0 = semilla aleatoria."},
                    "wait": {"type": "boolean", "default": True},
                },
                "required": ["prompt"],
            },
        },
        {
            "name": "studio_generate_video",
            "description": (
                "Genera un clip de vídeo MP4 en cualquiera de los 4 modos verificados: "
                "`flf2v` (Frame a Frame: Inicio → Fin con 2 imágenes), `i2v` (1 Imagen → Vídeo), "
                "`t2v` (Texto → Vídeo) o `ffmpeg_motion` (cámara/transición CPU en 2 s). "
                "Con `use_official_defaults=true` (por defecto) calcula automáticamente pasos, CFG, escala LoRA y resolución oficial."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "video_submode": {
                        "type": "string",
                        "enum": ["flf2v", "i2v", "t2v"],
                        "description": "Modo obligatorio: 'flf2v' (Inicio→Fin, exige 2 fotos), 'i2v' (1 foto inicial) o 't2v' (solo texto).",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["ai", "ffmpeg_motion"],
                        "default": "ai",
                        "description": "'ai' usa modelos GPU Diffusers; 'ffmpeg_motion' crea movimiento/transición CPU instantánea (0 GB).",
                    },
                    "model_id": {
                        "type": "string",
                        "enum": video_model_ids,
                        "description": (
                            "ID exacto del modelo de vídeo. Si se omite, Studio IA elige automáticamente el mejor compatible "
                            "(ej. `wan21-vace-14b-fp8` en L4 para `flf2v`, `wan21-vace-1.3b` en T4, `wan22-ti2v-5b` para `i2v`/`t2v`)."
                        ),
                    },
                    "video_lora_id": {
                        "type": "string",
                        "enum": video_lora_ids,
                        "default": "",
                        "description": (
                            "LoRA acelerador opcional (cadena vacía '' = sin LoRA). En `flf2v` usa solo LoRAs bidireccionales "
                            "como `lora-wan21-14b-fusionx-phantom` (VACE 14B), `lora-wan21-14b-fusionx-i2v` (FLF2V 14B) o `lora-wan21-1.3b-bidirect2` (1.3B)."
                        ),
                    },
                    "prompt": {
                        "type": "string",
                        "description": "Descripción clara de la acción o de cómo ocurre la transición del fotograma inicial al final.",
                    },
                    "negative_prompt": {
                        "type": "string",
                        "default": "",
                        "description": "Déjalo vacío '' para usar el negative_prompt oficial del fabricante.",
                    },
                    "image_rel": {
                        "type": "string",
                        "description": "`rel_path` en Colab del Fotograma Inicial (obligatorio en `i2v`, `flf2v` y `ffmpeg_motion`).",
                    },
                    "end_image_rel": {
                        "type": "string",
                        "description": "`rel_path` en Colab del Fotograma Final (obligatorio en `flf2v`).",
                    },
                    "audio_rel": {
                        "type": "string",
                        "description": "Opcional: `rel_path` de un audio para ajustar duración y muxear sonido en el MP4.",
                    },
                    "orientation": {
                        "type": "string",
                        "enum": ["auto_from_image", "landscape", "portrait", "square", "custom"],
                        "default": "auto_from_image",
                        "description": "Con 'auto_from_image' respeta automáticamente la proporción de tus fotos en el bucket oficial del modelo.",
                    },
                    "use_official_defaults": {
                        "type": "boolean",
                        "default": True,
                        "description": "RECOMENDADO `true`: aplica automáticamente pasos, CFG, escala LoRA y FPS verificados para el modelo y LoRA elegidos.",
                    },
                    "duration_sec": {"type": "number", "default": 5.0, "minimum": 1.5, "maximum": 8.0},
                    "width": {"type": "integer", "default": 832, "minimum": 256, "maximum": 1280},
                    "height": {"type": "integer", "default": 480, "minimum": 256, "maximum": 1280},
                    "fps": {"type": "integer", "default": 16, "minimum": 8, "maximum": 30},
                    "steps": {"type": "integer", "default": 28, "minimum": 4, "maximum": 60},
                    "guidance_scale": {"type": "number", "minimum": 1.0, "maximum": 12.0},
                    "video_lora_scale": {"type": "number", "default": 0.9, "minimum": 0.1, "maximum": 1.5},
                    "seed": {"type": "integer", "default": 0},
                    "motion_style": {
                        "type": "string",
                        "enum": ["zoom_in", "zoom_out", "pan_right", "pan_left", "tilt_up", "static"],
                        "default": "zoom_in",
                        "description": "Solo usado cuando `mode='ffmpeg_motion'`.",
                    },
                    "wait": {"type": "boolean", "default": True},
                },
                "required": ["video_submode", "prompt"],
            },
        },
        {
            "name": "studio_list_tts_voices",
            "description": "Lista todas las voces neuronales TTS disponibles con su `voice_id`, idioma y género.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "studio_generate_speech",
            "description": "Genera locución de voz neuronal limpia a partir de texto en español o inglés.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Texto exacto a locutar."},
                    "engine_id": {
                        "type": "string",
                        "enum": tts_engine_ids,
                        "default": "edge-tts-cloud",
                    },
                    "voice_id": {
                        "type": "string",
                        "enum": voice_ids,
                        "default": "es-ES-AlvaroNeural",
                    },
                    "rate_pct": {"type": "integer", "default": 0, "minimum": -40, "maximum": 40},
                    "pitch_hz": {"type": "integer", "default": 0, "minimum": -30, "maximum": 30},
                    "wait": {"type": "boolean", "default": True},
                },
                "required": ["text"],
            },
        },
        {
            "name": "studio_render_montage",
            "description": (
                "Une varios planos (`images` y/o `videos`) con voces por plano, música de fondo opcional y "
                "transiciones suaves en un vídeo MP4 H.264 BT.709 final."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "default": "montaje_final"},
                    "items": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "properties": {
                                "media_rel": {
                                    "type": "string",
                                    "description": "`rel_path` de la imagen o clip de vídeo del plano.",
                                },
                                "duration": {"type": "number", "default": 4.0},
                                "motion": {
                                    "type": "string",
                                    "enum": ["none", "zoom_in", "zoom_out", "pan_right", "pan_left"],
                                    "default": "none",
                                },
                                "audio_rel": {
                                    "type": "string",
                                    "description": "Opcional: `rel_path` de la locución de este plano.",
                                },
                                "fit_to_audio": {"type": "boolean", "default": True},
                            },
                            "required": ["media_rel"],
                        },
                    },
                    "width": {"type": "integer", "default": 1280},
                    "height": {"type": "integer", "default": 720},
                    "fps": {"type": "integer", "default": 24},
                    "global_audio_rel": {
                        "type": "string",
                        "description": "Opcional: `rel_path` de música ambiental global.",
                    },
                    "global_audio_volume": {"type": "number", "default": 0.25, "minimum": 0.0, "maximum": 1.0},
                    "fade_seconds": {"type": "number", "default": 0.20, "minimum": 0.0, "maximum": 2.0},
                    "wait": {"type": "boolean", "default": True},
                },
                "required": ["items"],
            },
        },
        {
            "name": "studio_wait_for_job",
            "description": (
                "Espera o consulta el progreso en vivo (0-100%, logs y resultado final) de un `job_id`, "
                "o lo cancela limpiamente si `cancel=true`."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "job_id": {"type": "string", "description": "ID del trabajo devuelto por una herramienta."},
                    "wait_seconds": {
                        "type": "number",
                        "default": 80.0,
                        "minimum": 1.0,
                        "maximum": 88.0,
                    },
                    "cancel": {"type": "boolean", "default": False},
                },
                "required": ["job_id"],
            },
        },
    ]


# ==================== LÓGICA DEL ORÁCULO Y EJECUTORES DE HERRAMIENTAS ====================


def _compute_recommendation(task: str, priority: str) -> Dict[str, Any]:
    gpu_info = query_gpu_info()
    vram = float(gpu_info.get("vram_total_gb") or 0.0)
    has_gpu = bool(gpu_info.get("has_gpu"))
    snap = hardware_tracker.get_snapshot(active_jobs_count=job_manager.active_count())
    free_disk = float((snap.get("disk") or {}).get("free_gb") or 0.0)

    if priority == "zero_vram_cpu" or (not has_gpu and task.startswith("video_")):
        return {
            "task": task,
            "detected_gpu": gpu_info.get("gpu_name", "CPU"),
            "free_disk_gb": free_disk,
            "recommended_tool": "studio_generate_video",
            "recommended_arguments": {
                "mode": "ffmpeg_motion",
                "video_submode": "flf2v" if task == "video_flf2v" else "i2v",
                "model_id": "ffmpeg-motion-cpu",
                "motion_style": "zoom_in",
                "duration_sec": 4.5,
            },
            "why_this_choice": "Ejecución instantánea CPU con FFmpeg (0 GB VRAM, ~2 segundos).",
        }

    if task == "video_flf2v":
        if vram >= 18.0 and priority == "max_quality_14b":
            return {
                "task": task,
                "detected_gpu": gpu_info.get("gpu_name"),
                "vram_total_gb": vram,
                "free_disk_gb": free_disk,
                "recommended_tool": "studio_generate_video",
                "recommended_arguments": {
                    "mode": "ai",
                    "video_submode": "flf2v",
                    "model_id": "wan21-vace-14b-fp8",
                    "video_lora_id": "lora-wan21-14b-fusionx-phantom",
                    "use_official_defaults": True,
                },
                "alternative_hd_specialist": {
                    "model_id": "wan21-flf2v-14b-fp8",
                    "video_lora_id": "lora-wan21-14b-fusionx-i2v",
                    "note": "Especialista puro 14B 720P con doble CLIP 514 tokens (~30.7 GB disco).",
                },
                "why_this_choice": (
                    "GPU >=18 GB detectada: Wan 2.1 VACE 14B Quantized FP8 (29.1 GB) + LoRA FusionX Phantom (6 pasos @ CFG 1.0) "
                    "ofrece calidad de textura y anatomía 14B en ~1.6 min en L4 con atención bidireccional completa."
                ),
            }
        if vram >= 18.0 and priority == "balanced_fast":
            return {
                "task": task,
                "detected_gpu": gpu_info.get("gpu_name"),
                "vram_total_gb": vram,
                "free_disk_gb": free_disk,
                "recommended_tool": "studio_generate_video",
                "recommended_arguments": {
                    "mode": "ai",
                    "video_submode": "flf2v",
                    "model_id": "ltx-video-13b-distilled" if free_disk >= 50.0 else "wan21-vace-14b-fp8",
                    "video_lora_id": "" if free_disk >= 50.0 else "lora-wan21-14b-fusionx-phantom",
                    "use_official_defaults": True,
                },
                "why_this_choice": "Velocidad rápida en 13B/14B destilado (6–8 pasos @ CFG 1.0).",
            }
        return {
            "task": task,
            "detected_gpu": gpu_info.get("gpu_name"),
            "vram_total_gb": vram,
            "free_disk_gb": free_disk,
            "recommended_tool": "studio_generate_video",
            "recommended_arguments": {
                "mode": "ai",
                "video_submode": "flf2v",
                "model_id": "wan21-vace-1.3b",
                "video_lora_id": "lora-wan21-1.3b-bidirect2",
                "use_official_defaults": True,
            },
            "why_this_choice": (
                "Wan 2.1 VACE 1.3B (17.7 GB) + LoRA Bidireccional v2 (91 MB, 8 pasos, escala 0.65, CFG 1.5): "
                "cabe incluso en T4 (15 GB) y genera transiciones Frame a Frame en ~25 s en L4."
            ),
        }

    if task in ("video_i2v", "video_t2v"):
        sub = "i2v" if task == "video_i2v" else "t2v"
        if vram >= 18.0 and priority == "max_quality_14b":
            return {
                "task": task,
                "detected_gpu": gpu_info.get("gpu_name"),
                "recommended_tool": "studio_generate_video",
                "recommended_arguments": {
                    "mode": "ai",
                    "video_submode": sub,
                    "model_id": "wan22-ti2v-5b",
                    "video_lora_id": "lora-wan22-ti2v-5b-turbo",
                    "use_official_defaults": True,
                },
                "alternative_14b": {
                    "model_id": "wan21-vace-14b-fp8",
                    "video_lora_id": "lora-wan21-14b-fusionx-phantom",
                },
                "why_this_choice": "Wan 2.2 TI2V-5B HD 720P @ 24fps con LoRA Turbo (6 pasos @ CFG 1.0, ~1.2 min en L4).",
            }
        return {
            "task": task,
            "detected_gpu": gpu_info.get("gpu_name"),
            "recommended_tool": "studio_generate_video",
            "recommended_arguments": {
                "mode": "ai",
                "video_submode": sub,
                "model_id": "wan21-vace-1.3b" if sub == "i2v" else "wan21-t2v-1.3b",
                "video_lora_id": "lora-wan21-1.3b-bidirect2",
                "use_official_defaults": True,
            },
            "why_this_choice": "Modelo ligero 1.3B con LoRA rápido (6 pasos @ CFG 1.0, ~22 s en L4, compatible con T4).",
        }

    if task in ("image_create", "image_edit"):
        return {
            "task": task,
            "recommended_tool": "studio_generate_or_edit_image",
            "recommended_arguments": {
                "aspect_ratio": "1:1",
                "use_official_defaults": True,
                "use_lightning_lora": True,
            },
            "why_this_choice": "Qwen Image Edit 2511 4-bit NF4 + Lightning 4 pasos + normalización BT.709 (~20 s en L4).",
        }

    if task == "tts":
        return {
            "task": task,
            "recommended_tool": "studio_generate_speech",
            "recommended_arguments": {
                "engine_id": "edge-tts-cloud",
                "voice_id": "es-ES-AlvaroNeural",
                "rate_pct": 0,
            },
            "why_this_choice": "Síntesis neuronal instantánea en español sin consumir VRAM ni disco (0 GB).",
        }

    return {
        "task": "montage",
        "recommended_tool": "studio_render_montage",
        "recommended_arguments": {
            "width": 1280,
            "height": 720,
            "fps": 24,
            "fade_seconds": 0.2,
            "global_audio_volume": 0.25,
        },
        "why_this_choice": "Renderizado multipista H.264 BT.709 con ajuste automático de duración a la locución.",
    }


def _execute_mcp_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    compact_args = {
        k: (f"<base64:{len(str(v))} chars>" if k == "data_base64" else v)
        for k, v in args.items()
    }
    colab_console_log(
        f"[MCP → Studio IA] Agente ejecutó «{name}» · args={json.dumps(compact_args, ensure_ascii=False)[:240]}",
        icon="🤖",
    )

    # 1. Diagnóstico y manual
    if name == "studio_get_state_and_manual":
        hw = hardware_tracker.get_snapshot(active_jobs_count=job_manager.active_count())
        loaded = memory_manager.status_summary()
        downloaded_models = [
            {
                "id": m["id"],
                "name": m["name"],
                "category": m["category"],
                "disk_gb": catalog_manager.disk_usage_gb(m["id"]),
            }
            for m in catalog_manager.get_all_definitions()
            if not m.get("builtin_no_download") and catalog_manager.is_downloaded(m["id"])
        ]
        payload: Dict[str, Any] = {
            "ok": True,
            "hardware": hw,
            "loaded_in_vram": loaded,
            "downloaded_on_disk": downloaded_models,
            "active_jobs": job_manager.list_jobs(limit=5, include_logs=False),
            "golden_rules": [
                "Llama a `studio_recommend_pipeline` si dudas qué modelo o LoRA usar.",
                "En `studio_generate_video` deja `use_official_defaults=true` para no inventar pasos/CFG.",
                "En `flf2v` pasa siempre `image_rel` + `end_image_rel` (o rutas locales en el puente) y usa solo modelos/LoRAs bidireccionales.",
            ],
        }
        if bool(args.get("include_recent_assets", True)):
            payload["recent_assets"] = _list_assets_summary(bucket_filter="all", limit_per_bucket=8)
        return _ok_result(payload)

    # 2. Oráculo de recomendación
    if name == "studio_recommend_pipeline":
        task = str(args.get("task") or "video_flf2v")
        priority = str(args.get("priority") or "max_quality_14b")
        return _ok_result({"ok": True, **_compute_recommendation(task=task, priority=priority)})

    # 3. Listar modelos del catálogo
    if name == "studio_list_models":
        cat = str(args.get("category") or "video")
        req_sub = str(args.get("required_video_submode") or "any")
        only_compat = bool(args.get("only_gpu_compatible", True))
        items = catalog_manager.list_catalog(
            loaded_ids=memory_manager.loaded_model_ids(),
            gpu_profile="auto",
            only_compatible=only_compat,
            detected_gpu=query_gpu_info(),
        )
        filtered: List[Dict[str, Any]] = []
        for m in items:
            if cat != "all" and m["category"] != cat:
                continue
            if req_sub != "any":
                if m["category"] == "video":
                    vmodes = m.get("video_modes") or []
                    if req_sub not in vmodes:
                        continue
                elif m["category"] == "lora" and m.get("lora_target") == "video":
                    incompat = m.get("incompatible_submodes") or []
                    if req_sub in incompat:
                        continue
            filtered.append(
                {
                    "id": m["id"],
                    "name": m["name"],
                    "category": m["category"],
                    "lora_target": m.get("lora_target"),
                    "video_modes": m.get("video_modes"),
                    "compatible_models": m.get("compatible_models"),
                    "incompatible_submodes": m.get("incompatible_submodes"),
                    "estimated_gb": m.get("estimated_gb"),
                    "downloaded": m.get("downloaded"),
                    "loaded_in_memory": m.get("loaded_in_memory"),
                    "gpu_compatibility": m.get("compatibility"),
                    "l4_seconds_hint": m.get("l4_seconds_hint"),
                    "official_spec": m.get("official_spec"),
                }
            )
        return _ok_result({"ok": True, "count": len(filtered), "models": filtered})

    # 4. Preparar (descargar y/o cargar) modelo
    if name == "studio_prepare_model":
        mid = str(args.get("model_id") or "").strip()
        load_vram = bool(args.get("load_into_vram", True))
        wait_flag = bool(args.get("wait", True))

        defn = catalog_manager.get_definition(mid)
        if not defn:
            return _guided_error(
                "UNKNOWN_MODEL_ID",
                f"El modelo `{mid}` no existe en el catálogo.",
                "Llama a `studio_list_models` para obtener los `id` exactos permitidos.",
            )

        if not catalog_manager.is_downloaded(mid):
            space = catalog_manager.check_disk_space(mid)
            if not space["ok"]:
                return _guided_error(
                    "DISK_SPACE_INSUFFICIENT",
                    space["message"],
                    "Libera espacio llamando a `studio_free_memory_or_disk(action='delete_from_disk', target='<model_id>')` con alguno de los modelos sugeridos.",
                    extra={"disk_check": space},
                )

        cat = defn["category"]
        if not load_vram or cat == "lora":
            job = job_manager.submit(
                kind="download",
                title=f"[MCP] Descargar: {defn['name']}",
                fn=lambda j: catalog_manager.download_model(mid, j),
            )
        elif cat == "video":
            gpu_info = query_gpu_info()
            tier = evaluate_gpu_tier(defn, resolve_gpu_profile("auto", detected_gpu=gpu_info), gpu_info=gpu_info)
            if gpu_info.get("has_gpu") and not tier["compatible"]:
                return _guided_error(
                    "GPU_VRAM_INCOMPATIBLE",
                    f"«{defn['name']}» requiere ~{defn.get('min_vram_gb')} GB VRAM y la GPU actual tiene {gpu_info.get('vram_total_gb')} GB.",
                    "Llama a `studio_recommend_pipeline` para obtener el modelo adecuado a esta GPU.",
                )
            job = job_manager.submit(
                kind="load_model",
                title=f"[MCP] Preparar vídeo: {defn['name']}",
                fn=lambda j: load_video_model_into_memory(model_id=mid, auto_download=True, job=j),
            )
        elif cat == "image":
            job = job_manager.submit(
                kind="load_model",
                title=f"[MCP] Preparar imagen: {defn['name']}",
                fn=lambda j: load_image_model_into_memory(model_id=mid, auto_download=True, job=j),
            )
        else:
            job = job_manager.submit(
                kind="load_model",
                title=f"[MCP] Preparar TTS: {defn['name']}",
                fn=lambda j: load_tts_model_into_memory(model_id=mid, auto_download=True, job=j),
            )

        if not wait_flag:
            return _ok_result({"ok": True, "job_id": job.id, "status": job.status})
        return _ok_result(_wait_for_job_bounded(job.id))

    # 5. Liberar VRAM o borrar modelo del disco
    if name == "studio_free_memory_or_disk":
        action = str(args.get("action") or "unload_vram")
        target = str(args.get("target") or "all").strip()
        if action == "unload_vram":
            if target == "all":
                info = memory_manager.unload_all()
            elif target in ("image", "video", "tts"):
                info = memory_manager.unload_slot(target)
            else:
                info = memory_manager.unload_by_model_id(target)
            colab_console_log(
                f"[MCP] Memoria RAM/VRAM liberada (`{target}` → {info.get('vram_freed_gb', 0)} GB VRAM libres).",
                icon="🧹",
            )
            return _ok_result({"ok": True, "action": action, **info})

        if target in ("", "all", "video", "image", "tts"):
            return _guided_error(
                "MISSING_MODEL_ID_FOR_DELETE",
                "Para `delete_from_disk` debes indicar en `target` el `model_id` concreto a borrar del disco.",
                "Consulta `downloaded_on_disk` en `studio_get_state_and_manual` y pasa ese `model_id`.",
            )
        try:
            del_res = catalog_manager.delete_model(
                target,
                unload_callback=lambda mid: memory_manager.unload_by_model_id(mid),
            )
            return _ok_result(del_res)
        except ValueError as ve:
            return _guided_error("DELETE_MODEL_FAILED", str(ve), "Verifica el `model_id` con `studio_list_models`.")

    # 6. Listar archivos
    if name == "studio_list_assets":
        bucket = str(args.get("bucket") or "all")
        lim = int(args.get("limit") or 25)
        return _ok_result({"ok": True, "assets": _list_assets_summary(bucket_filter=bucket, limit_per_bucket=lim)})

    # 7. Subir archivo Base64
    if name == "studio_upload_file_base64":
        ensure_directories()
        orig_name = Path(str(args.get("filename") or "archivo.png")).name
        raw_b64 = str(args.get("data_base64") or "").strip()
        if not raw_b64:
            return _guided_error(
                "EMPTY_UPLOAD",
                "No se recibió `data_base64` (ni `local_file_path` en el puente local).",
                "Pasa `local_file_path` si usas `studio_mcp.py` o `data_base64` válido.",
            )
        if "," in raw_b64 and raw_b64.startswith("data:"):
            raw_b64 = raw_b64.split(",", 1)[1]
        try:
            binary = base64.b64decode(raw_b64)
        except Exception as exc:
            return _guided_error("INVALID_BASE64", f"Base64 inválido: {exc}", "Codifica el archivo en Base64 estándar.")

        safe_stem = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in Path(orig_name).stem)[:40] or "mcp"
        ext = Path(orig_name).suffix.lower() or ".png"
        out_name = f"{safe_stem}_{datetime.now().strftime('%H%M%S')}_{uuid.uuid4().hex[:4]}{ext}"
        dest = UPLOADS_DIR / out_name
        dest.write_bytes(binary)

        try:
            if ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
                w, h = verify_image_file(dest)
                extra = {"width": w, "height": h, "type": "image"}
            else:
                pinfo = probe_media_info(dest)
                extra = {
                    "duration_seconds": pinfo["duration_seconds"],
                    "width": pinfo["width"],
                    "height": pinfo["height"],
                    "type": "video" if pinfo["has_video"] else "audio",
                }
        except Exception as exc:
            dest.unlink(missing_ok=True)
            return _guided_error("CORRUPT_MEDIA_FILE", f"Archivo inválido: {exc}", "Sube un PNG/JPG/MP4/WAV/MP3 íntegro.")

        colab_console_log(
            f"[MCP → Colab] Archivo subido: {out_name} ({round(len(binary) / 1024, 1)} KB, {extra['type']})",
            icon="📥",
        )
        return _ok_result(
            {
                "ok": True,
                "asset": {
                    "filename": out_name,
                    "rel_path": f"uploads/{out_name}",
                    "media_url": f"/media/uploads/{out_name}",
                    "size_kb": round(len(binary) / 1024, 1),
                    **extra,
                },
            }
        )

    # 8. Generar o editar imagen Qwen
    if name == "studio_generate_or_edit_image":
        prompt = str(args.get("prompt") or "").strip()
        if not prompt:
            return _guided_error(
                "EMPTY_IMAGE_PROMPT",
                "El campo `prompt` está vacío.",
                "Describe claramente la escena a crear o la edición que deseas aplicar sobre `base_image_rel`.",
            )
        try:
            base_p = _resolve_rel_safe(args.get("base_image_rel"))
            refs_p = [p for p in (_resolve_rel_safe(r) for r in (args.get("reference_rels") or [])) if p is not None]
            mask_p = _resolve_rel_safe(args.get("mask_image_rel"))
        except (ValueError, FileNotFoundError) as exc:
            return _guided_error(
                "ASSET_PATH_NOT_FOUND",
                str(exc),
                "Llama a `studio_list_assets` para verificar el `rel_path` o usa `local_base_image_path` en el puente local.",
            )

        ar_map = {
            "1:1": (1024, 1024),
            "16:9": (1280, 720),
            "9:16": (720, 1280),
            "4:3": (1152, 864),
            "3:4": (864, 1152),
        }
        ar = str(args.get("aspect_ratio") or "1:1")
        if ar in ar_map:
            width, height = ar_map[ar]
        else:
            width = int(args.get("width") or 1024)
            height = int(args.get("height") or 1024)

        use_defaults = bool(args.get("use_official_defaults", True))
        use_l_lora = bool(args.get("use_lightning_lora", True))
        steps = (4 if use_l_lora else 28) if use_defaults else int(args.get("steps") or 4)
        cfg = 4.0 if use_defaults else float(args.get("guidance_scale") or 4.0)
        seed = int(args.get("seed") or 0)
        custom_loras = list(args.get("custom_lora_ids") or [])
        wait_flag = bool(args.get("wait", True))

        job = job_manager.submit(
            kind="image",
            title=f"[MCP] Imagen Qwen: {prompt[:40]}",
            fn=lambda j: generate_or_edit_image(
                prompt=prompt,
                base_image_path=base_p,
                reference_paths=refs_p,
                mask_image_path=mask_p,
                width=width,
                height=height,
                steps=steps,
                guidance_scale=cfg,
                seed=seed,
                use_lightning_lora=use_l_lora,
                custom_lora_ids=custom_loras,
                use_ffmpeg_bt709=True,
                job=j,
            ),
        )
        if not wait_flag:
            return _ok_result({"ok": True, "job_id": job.id, "status": job.status})
        return _ok_result(_wait_for_job_bounded(job.id))

    # 9. Generar vídeo (T2V, I2V, Frame a Frame FLF2V o FFmpeg)
    if name == "studio_generate_video":
        sub = str(args.get("video_submode") or "").strip().lower()
        mode = str(args.get("mode") or "ai").strip().lower()
        prompt = str(args.get("prompt") or "").strip()
        wait_flag = bool(args.get("wait", True))

        if sub not in ("flf2v", "i2v", "t2v"):
            return _guided_error(
                "INVALID_VIDEO_SUBMODE",
                f"`video_submode='{sub}'` no es válido.",
                "Usa exactamente `'flf2v'` (Frame a Frame 2 fotos), `'i2v'` (1 foto inicial) o `'t2v'` (solo texto).",
            )

        try:
            img_p = _resolve_rel_safe(args.get("image_rel"))
            end_img_p = _resolve_rel_safe(args.get("end_image_rel"))
            aud_p = _resolve_rel_safe(args.get("audio_rel"))
        except (ValueError, FileNotFoundError) as exc:
            return _guided_error(
                "VIDEO_INPUT_FILE_NOT_FOUND",
                str(exc),
                "Verifica los `rel_path` con `studio_list_assets` o pasa `local_image_path` / `local_end_image_path` en `studio_mcp.py`.",
            )

        if sub == "i2v" and img_p is None:
            return _guided_error(
                "MISSING_START_IMAGE",
                "El modo `i2v` (Imagen a Vídeo) exige `image_rel` (o `local_image_path`).",
                "Pasa la imagen inicial o cambia `video_submode='t2v'` si quieres generar solo desde texto.",
            )
        if sub == "flf2v" and (img_p is None or end_img_p is None):
            return _guided_error(
                "MISSING_FLF2V_IMAGES",
                "El modo `flf2v` (Frame a Frame Inicio → Fin) exige TANTO `image_rel` COMO `end_image_rel`.",
                "Proporciona ambos fotogramas (`image_rel` + `end_image_rel` o `local_image_path` + `local_end_image_path`).",
            )
        if mode == "ffmpeg_motion" and img_p is None:
            return _guided_error(
                "MISSING_FFMPEG_IMAGE",
                "`mode='ffmpeg_motion'` requiere al menos `image_rel`.",
                "Pasa `image_rel` o cambia `mode='ai'` con `video_submode='t2v'`.",
            )

        model_id = str(args.get("model_id") or "").strip()
        if mode == "ffmpeg_motion":
            model_id = "ffmpeg-motion-cpu"
        elif not model_id:
            rec = _compute_recommendation(task=f"video_{sub}", priority="max_quality_14b")
            model_id = str((rec.get("recommended_arguments") or {}).get("model_id") or "wan21-vace-1.3b")

        m_defn = catalog_manager.get_definition(model_id)
        if not m_defn:
            return _guided_error(
                "UNKNOWN_VIDEO_MODEL",
                f"El modelo `{model_id}` no existe en el catálogo.",
                "Usa `studio_list_models(category='video', required_video_submode=...)` para ver IDs válidos.",
            )

        if mode == "ai":
            allowed_modes = m_defn.get("video_modes") or ["i2v", "t2v"]
            if sub not in allowed_modes:
                compat_ids = [
                    x["id"]
                    for x in catalog_manager.get_all_definitions()
                    if x["category"] == "video" and x.get("engine") != "ffmpeg_motion" and sub in (x.get("video_modes") or [])
                ]
                return _guided_error(
                    "SUBMODE_NOT_SUPPORTED_BY_MODEL",
                    f"El modelo «{m_defn['name']}» (`{model_id}`) solo admite {allowed_modes} y NO soporta `{sub}`.",
                    f"Elige uno de estos `model_id` compatibles con `{sub}`: {compat_ids[:5]}.",
                    extra={"compatible_model_ids": compat_ids},
                )

            gpu_info = query_gpu_info()
            tier = evaluate_gpu_tier(m_defn, resolve_gpu_profile("auto", detected_gpu=gpu_info), gpu_info=gpu_info)
            if gpu_info.get("has_gpu") and not tier["compatible"]:
                return _guided_error(
                    "GPU_VRAM_TOO_SMALL",
                    f"«{m_defn['name']}» pide ~{m_defn.get('min_vram_gb')} GB VRAM y tu GPU actual tiene {gpu_info.get('vram_total_gb')} GB.",
                    "Usa `wan21-vace-1.3b` (cabe en T4 15 GB) o llama a `studio_recommend_pipeline`.",
                )

            if not catalog_manager.is_downloaded(model_id):
                space = catalog_manager.check_disk_space(model_id)
                if not space["ok"]:
                    return _guided_error(
                        "DISK_SPACE_INSUFFICIENT",
                        space["message"],
                        "Libera disco con `studio_free_memory_or_disk(action='delete_from_disk', target=...)` o elige un modelo ya descargado (`studio_get_state_and_manual`).",
                        extra={"disk_check": space},
                    )

        lora_id = str(args.get("video_lora_id") or "").strip() or None
        lora_defn = catalog_manager.get_definition(lora_id) if lora_id else None
        if lora_id and not lora_defn:
            return _guided_error(
                "UNKNOWN_VIDEO_LORA",
                f"El LoRA `{lora_id}` no existe.",
                "Pasa `video_lora_id=''` o consulta `studio_list_models(category='lora')`.",
            )
        if lora_defn and mode == "ai":
            compat_m = lora_defn.get("compatible_models") or []
            if compat_m and model_id not in compat_m:
                valid_loras = [
                    x["id"]
                    for x in catalog_manager.get_all_definitions()
                    if x["category"] == "lora"
                    and x.get("lora_target") == "video"
                    and model_id in (x.get("compatible_models") or [])
                    and sub not in (x.get("incompatible_submodes") or [])
                ]
                return _guided_error(
                    "LORA_MODEL_MISMATCH",
                    f"El LoRA «{lora_defn['name']}» solo sirve para {compat_m} y NO para `{model_id}`.",
                    f"Para `{model_id}` en modo `{sub}` usa `video_lora_id` en {valid_loras or ['']} (o vacío '').",
                )
            incompat_s = lora_defn.get("incompatible_submodes") or []
            if sub in incompat_s or (sub == "flf2v" and bool(lora_defn.get("causal_attention"))):
                return _guided_error(
                    "CAUSAL_LORA_FORBIDDEN_IN_FLF2V",
                    f"El LoRA «{lora_defn['name']}» usa atención causal y rompe el fotograma final en `flf2v`.",
                    "Usa un LoRA bidireccional (`lora-wan21-14b-fusionx-phantom`, `lora-wan21-14b-fusionx-i2v`, `lora-wan21-1.3b-bidirect2`) o déjalo vacío ''.",
                )

        # Calcular parámetros oficiales exactos si `use_official_defaults=True`
        spec = m_defn.get("official_spec") or {}
        use_defaults = bool(args.get("use_official_defaults", True))
        orient = str(args.get("orientation") or "auto_from_image")
        if orient == "portrait" and isinstance(spec.get("portrait"), list):
            width, height = int(spec["portrait"][0]), int(spec["portrait"][1])
        elif orient == "square" and isinstance(spec.get("square"), list):
            width, height = int(spec["square"][0]), int(spec["square"][1])
        elif orient in ("landscape", "auto_from_image") and isinstance(spec.get("landscape"), list):
            width, height = int(spec["landscape"][0]), int(spec["landscape"][1])
        else:
            width = int(args.get("width") or 832)
            height = int(args.get("height") or 480)

        if use_defaults:
            fps = int(spec.get("default_fps") or m_defn.get("default_fps") or 16)
            duration_sec = float(args.get("duration_sec") or spec.get("default_duration_sec") or 5.0)
            if lora_defn:
                steps = int(
                    (lora_defn.get("flf2v_recommended_steps") if sub == "flf2v" else None)
                    or lora_defn.get("recommended_steps")
                    or 6
                )
                cfg = float(
                    lora_defn.get("flf2v_recommended_cfg")
                    if sub == "flf2v" and lora_defn.get("flf2v_recommended_cfg") is not None
                    else lora_defn.get("recommended_cfg", 1.0)
                )
                lora_scale = float(
                    (lora_defn.get("flf2v_recommended_scale") if sub == "flf2v" else None)
                    or lora_defn.get("default_lora_scale")
                    or 1.0
                )
            else:
                steps = int(spec.get("default_steps") or m_defn.get("default_steps") or 30)
                cfg = float(spec.get("default_cfg") or m_defn.get("default_cfg") or 5.0)
                lora_scale = 1.0
        else:
            fps = int(args.get("fps") or spec.get("default_fps") or 16)
            duration_sec = float(args.get("duration_sec") or 5.0)
            steps = int(args.get("steps") or spec.get("default_steps") or 30)
            cfg = (
                float(args["guidance_scale"])
                if args.get("guidance_scale") is not None
                else float(spec.get("default_cfg") or 5.0)
            )
            lora_scale = float(args.get("video_lora_scale") or 1.0)

        seed = int(args.get("seed") or 0)
        motion_style = str(args.get("motion_style") or "zoom_in")
        neg_prompt = str(args.get("negative_prompt") or "")

        job = job_manager.submit(
            kind="video",
            title=f"[MCP] Vídeo ({sub} · {model_id}): {(prompt or 'animación')[:34]}",
            fn=lambda j: generate_video_clip(
                mode=mode,
                video_submode=sub,
                model_id=model_id,
                video_lora_id=lora_id,
                video_lora_scale=lora_scale,
                prompt=prompt,
                negative_prompt=neg_prompt,
                image_path=img_p,
                end_image_path=end_img_p,
                audio_path=aud_p,
                width=width,
                height=height,
                duration_sec=duration_sec,
                fps=fps,
                steps=steps,
                guidance_scale=cfg,
                seed=seed,
                motion_style=motion_style,
                job=j,
            ),
        )
        if not wait_flag:
            return _ok_result({"ok": True, "job_id": job.id, "status": job.status})
        return _ok_result(_wait_for_job_bounded(job.id))

    # 10. Listar voces TTS
    if name == "studio_list_tts_voices":
        return _ok_result({"ok": True, "voices": EDGE_TTS_VOICES})

    # 11. Generar voz TTS
    if name == "studio_generate_speech":
        txt = str(args.get("text") or "").strip()
        if not txt:
            return _guided_error("EMPTY_TTS_TEXT", "El texto a locutar está vacío.", "Pasa el guion en `text`.")
        eng_id = str(args.get("engine_id") or "edge-tts-cloud")
        v_id = str(args.get("voice_id") or "es-ES-AlvaroNeural")
        rate_pct = int(args.get("rate_pct") or 0)
        pitch_hz = int(args.get("pitch_hz") or 0)
        wait_flag = bool(args.get("wait", True))

        job = job_manager.submit(
            kind="tts",
            title=f"[MCP] Voz ({v_id}): {txt[:36]}",
            fn=lambda j: synthesize_voice(
                text=txt,
                engine_id=eng_id,
                voice_id=v_id,
                rate_pct=rate_pct,
                pitch_hz=pitch_hz,
                job=j,
            ),
        )
        if not wait_flag:
            return _ok_result({"ok": True, "job_id": job.id, "status": job.status})
        return _ok_result(_wait_for_job_bounded(job.id))

    # 12. Renderizar montaje MP4
    if name == "studio_render_montage":
        raw_items = args.get("items") or []
        if not isinstance(raw_items, list) or not raw_items:
            return _guided_error(
                "EMPTY_MONTAGE_ITEMS",
                "La lista `items` está vacía.",
                "Pasa al menos un plano con `media_rel` de una imagen o vídeo existente.",
            )
        resolved_items: List[Dict[str, Any]] = []
        try:
            for idx, it in enumerate(raw_items, start=1):
                mp = _resolve_rel_safe(it.get("media_rel"))
                if not mp:
                    return _guided_error("MISSING_SHOT_MEDIA", f"Falta `media_rel` en el plano {idx}.", "Consulta `studio_list_assets`.")
                ap = _resolve_rel_safe(it.get("audio_rel")) if it.get("audio_rel") else None
                resolved_items.append(
                    {
                        "path": str(mp),
                        "duration": float(it.get("duration") or 4.0),
                        "motion": str(it.get("motion") or "none"),
                        "audio_path": str(ap) if ap else None,
                        "fit_to_audio": bool(it.get("fit_to_audio", True)),
                    }
                )
            global_aud = _resolve_rel_safe(args.get("global_audio_rel")) if args.get("global_audio_rel") else None
        except (ValueError, FileNotFoundError) as exc:
            return _guided_error("MONTAGE_FILE_NOT_FOUND", str(exc), "Verifica todos los `rel_path` con `studio_list_assets`.")

        title = str(args.get("title") or "montaje_mcp")
        width = int(args.get("width") or 1280)
        height = int(args.get("height") or 720)
        fps = int(args.get("fps") or 24)
        g_vol = float(args.get("global_audio_volume") or 0.25)
        fade_s = float(args.get("fade_seconds") or 0.20)
        wait_flag = bool(args.get("wait", True))

        def _run_montage(j: Any) -> Dict[str, Any]:
            t0 = time.time()
            MONTAGES_OUT_DIR.mkdir(parents=True, exist_ok=True)
            safe_title = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in title)[:32] or "montaje"
            out_name = f"{safe_title}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            out_path = MONTAGES_OUT_DIR / out_name
            info = render_montage_mp4(
                items=resolved_items,
                output_mp4=out_path,
                width=width,
                height=height,
                fps=fps,
                global_audio_path=global_aud,
                global_audio_volume=g_vol,
                fade_seconds=fade_s,
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
                "width": info["width"] or width,
                "height": info["height"] or height,
                "fps": fps,
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
            title=f"[MCP] Montaje MP4 ({len(resolved_items)} planos)",
            fn=_run_montage,
        )
        if not wait_flag:
            return _ok_result({"ok": True, "job_id": job.id, "status": job.status})
        return _ok_result(_wait_for_job_bounded(job.id))

    # 13. Esperar o cancelar job
    if name == "studio_wait_for_job":
        jid = str(args.get("job_id") or "").strip()
        if bool(args.get("cancel", False)):
            cancelled = job_manager.cancel(jid)
            if not cancelled:
                return _guided_error("JOB_NOT_FOUND", f"No existe `job_id='{jid}'`.", "Verifica el ID devuelto anteriormente.")
            colab_console_log(f"[MCP] Trabajo {jid} cancelado por orden del agente.", icon="🛑")
            return _ok_result({"ok": True, "job": cancelled.to_dict(include_logs=True)})
        w_sec = float(args.get("wait_seconds") or 80.0)
        return _ok_result(_wait_for_job_bounded(jid, max_wait_seconds=w_sec))

    return _guided_error(
        "UNKNOWN_TOOL",
        f"La herramienta `{name}` no existe en Studio IA.",
        "Consulta `tools/list` para usar únicamente nombres de herramientas registrados.",
    )


# ==================== DESPACHADOR JSON-RPC 2.0 MCP ====================


def _dispatch_single_jsonrpc(msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    req_id = msg.get("id")
    method = str(msg.get("method") or "")
    params = msg.get("params") or {}

    # Notificaciones sin `id` (ej. `notifications/initialized`)
    if req_id is None:
        if method == "notifications/initialized":
            colab_console_log("[MCP] Handshake completado: el agente de terminal ya tiene el manual y las herramientas activas.", icon="✅")
        return None

    if method == "initialize":
        client_info = params.get("clientInfo") or {}
        c_name = client_info.get("name", "mcp-client")
        c_ver = client_info.get("version", "")
        req_proto = str(params.get("protocolVersion") or "2024-11-05")
        supported_protos = ("2025-03-26", "2024-11-05")
        negotiated_proto = req_proto if req_proto in supported_protos else "2024-11-05"
        _mcp_runtime_state["connected_clients"] = int(_mcp_runtime_state.get("connected_clients", 0)) + 1
        colab_console_log(
            f"[MCP] Agente conectado desde terminal: «{c_name} {c_ver}» (protocolo {negotiated_proto}).",
            icon="🔌",
        )
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": negotiated_proto,
                "capabilities": {
                    "tools": {"listChanged": False},
                    "resources": {"subscribe": False, "listChanged": False},
                    "prompts": {"listChanged": False},
                },
                "serverInfo": {
                    "name": "studio-ia-colab",
                    "version": "1.2.0",
                },
                "instructions": MCP_SERVER_INSTRUCTIONS,
            },
        }

    if method == "ping":
        return {"jsonrpc": "2.0", "id": req_id, "result": {}}

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": _build_mcp_tools_schema()},
        }

    if method == "tools/call":
        t_name = str(params.get("name") or "")
        t_args = dict(params.get("arguments") or {})
        try:
            call_res = _execute_mcp_tool(t_name, t_args)
            return {"jsonrpc": "2.0", "id": req_id, "result": call_res}
        except Exception as exc:
            colab_console_log(f"[MCP Error] Excepción en `{t_name}`: {exc}", icon="❌")
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": _guided_error(
                    "INTERNAL_TOOL_ERROR",
                    str(exc),
                    "Revisa el estado con `studio_get_state_and_manual` antes de reintentar.",
                ),
            }

    if method == "resources/list":
        resources = [
            {
                "uri": uri,
                "name": meta["name"],
                "mimeType": meta["mimeType"],
                "description": meta["name"],
            }
            for uri, meta in MANUAL_RESOURCES.items()
        ]
        resources.append(
            {
                "uri": "studio://state/live",
                "name": "Estado en Vivo de GPU, VRAM, Disco y Galería en Colab",
                "mimeType": "application/json",
                "description": "Telemetría instantánea del entorno Google Colab.",
            }
        )
        return {"jsonrpc": "2.0", "id": req_id, "result": {"resources": resources}}

    if method == "resources/read":
        uri = str(params.get("uri") or "")
        colab_console_log(f"[MCP] Agente leyó el recurso «{uri}».", icon="📖")
        if uri in MANUAL_RESOURCES:
            meta = MANUAL_RESOURCES[uri]
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "contents": [
                        {
                            "uri": uri,
                            "mimeType": meta["mimeType"],
                            "text": meta["content"],
                        }
                    ]
                },
            }
        if uri == "studio://state/live":
            live_data = {
                "hardware": hardware_tracker.get_snapshot(active_jobs_count=job_manager.active_count()),
                "loaded_in_vram": memory_manager.status_summary(),
                "assets": _list_assets_summary(bucket_filter="all", limit_per_bucket=10),
            }
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "contents": [
                        {
                            "uri": uri,
                            "mimeType": "application/json",
                            "text": json.dumps(live_data, indent=2, ensure_ascii=False),
                        }
                    ]
                },
            }
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32602, "message": f"Recurso desconocido: {uri}"},
        }

    if method == "prompts/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "prompts": [
                    {
                        "name": "flf2v_production_workflow",
                        "description": "Flujo guiado paso a paso para generar un vídeo Frame a Frame (Inicio → Fin) perfecto sin saltos.",
                        "arguments": [
                            {"name": "start_image", "description": "Ruta local o rel_path del fotograma inicial", "required": True},
                            {"name": "end_image", "description": "Ruta local o rel_path del fotograma final", "required": True},
                            {"name": "action_prompt", "description": "Cómo ocurre la transición entre ambas poses", "required": True},
                        ],
                    },
                    {
                        "name": "full_short_film_pipeline",
                        "description": "Flujo completo: generar/editar fotogramas Qwen → animar clips → locutar TTS → renderizar montaje MP4.",
                        "arguments": [
                            {"name": "story_idea", "description": "Idea o guion breve de la escena", "required": True},
                        ],
                    },
                ]
            },
        }

    if method == "prompts/get":
        p_name = str(params.get("name") or "")
        p_args = params.get("arguments") or {}
        colab_console_log(f"[MCP] Agente solicitó plantilla guiada «{p_name}».", icon="🧭")
        if p_name == "flf2v_production_workflow":
            text = (
                "Ejecuta exactamente estos pasos en orden sin inventar ningún parámetro:\n"
                "1. Llama a `studio_get_state_and_manual` para verificar GPU y disco libre en Colab.\n"
                "2. Llama a `studio_recommend_pipeline(task='video_flf2v', priority='max_quality_14b')`.\n"
                f"3. Llama a `studio_generate_video` en modo `video_submode='flf2v'` con fotograma inicial `{p_args.get('start_image', '')}`, "
                f"fotograma final `{p_args.get('end_image', '')}`, prompt `{p_args.get('action_prompt', '')}` y `use_official_defaults=true`.\n"
                "4. Entrega al usuario la ruta final del MP4 generado."
            )
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "description": "Flujo oficial Frame a Frame en Studio IA",
                    "messages": [{"role": "user", "content": {"type": "text", "text": text}}],
                },
            }
        if p_name == "full_short_film_pipeline":
            text = (
                f"Produce un micro-cortometraje completo para: «{p_args.get('story_idea', '')}» siguiendo el manual oficial:\n"
                "1. Llama a `studio_get_state_and_manual`.\n"
                "2. Genera los fotogramas clave con `studio_generate_or_edit_image` (`use_official_defaults=true`).\n"
                "3. Anima cada transición o plano con `studio_generate_video` usando la receta devuelta por `studio_recommend_pipeline`.\n"
                "4. Genera la narración en español con `studio_generate_speech`.\n"
                "5. Une todos los planos y voces en un MP4 final con `studio_render_montage`."
            )
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "description": "Flujo oficial de cortometraje completo en Studio IA",
                    "messages": [{"role": "user", "content": {"type": "text", "text": text}}],
                },
            }
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32602, "message": f"Prompt desconocido: {p_name}"},
        }

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Método MCP no soportado: {method}"},
    }


# ==================== ROUTER FASTAPI (`/mcp`, `/mcp/bridge.py`, `/api/mcp/*`) ====================

mcp_router = APIRouter()


@mcp_router.post("/mcp")
async def mcp_streamable_http_post(
    request: Request,
    authorization: Optional[str] = Header(default=None),
    x_studio_token: Optional[str] = Header(default=None),
    token: Optional[str] = Query(default=None),
) -> Response:
    _verify_mcp_auth(request, authorization=authorization, x_studio_token=x_studio_token, token_query=token)
    try:
        payload = await request.json()
    except Exception as exc:
        return JSONResponse(
            status_code=400,
            content={"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": f"JSON inválido: {exc}"}},
        )

    headers = {"Mcp-Session-Id": _mcp_runtime_state["session_id"]}

    if isinstance(payload, list):
        responses = [r for r in (_dispatch_single_jsonrpc(item) for item in payload if isinstance(item, dict)) if r is not None]
        if not responses:
            return Response(status_code=202, headers=headers)
        return JSONResponse(content=responses, headers=headers)

    if not isinstance(payload, dict):
        return JSONResponse(
            status_code=400,
            content={"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "Petición JSON-RPC inválida."}},
        )

    resp = _dispatch_single_jsonrpc(payload)
    if resp is None:
        return Response(status_code=202, headers=headers)
    return JSONResponse(content=resp, headers=headers)


@mcp_router.get("/mcp")
async def mcp_streamable_http_get(
    request: Request,
    authorization: Optional[str] = Header(default=None),
    x_studio_token: Optional[str] = Header(default=None),
    token: Optional[str] = Query(default=None),
) -> Response:
    """
    En Streamable HTTP (`2025-03-26`), `GET /mcp` con `Accept: text/event-stream` puede devolver 405
    cuando el servidor responde en JSON directo por `POST /mcp`, o devolver información de estado si se abre en navegador.
    """
    accept = (request.headers.get("accept") or "").lower()
    if "text/event-stream" in accept:
        return Response(status_code=405, headers={"Allow": "POST, DELETE"})
    _verify_mcp_auth(request, authorization=authorization, x_studio_token=x_studio_token, token_query=token)
    return JSONResponse(
        content={
            "ok": True,
            "service": "Studio IA MCP Server",
            "protocol": "MCP Streamable HTTP (JSON-RPC 2.0 via POST /mcp) + Local Bridge (/mcp/bridge.py)",
            "tools_count": len(_build_mcp_tools_schema()),
        }
    )


@mcp_router.delete("/mcp")
async def mcp_streamable_http_delete() -> Response:
    return Response(status_code=200)


@mcp_router.get("/mcp/bridge.py")
def mcp_download_bridge_script(
    request: Request,
    token: Optional[str] = Query(default=None),
) -> PlainTextResponse:
    """
    Sirve `studio_mcp.py` directamente desde Colab.
    Si incluye `?token=st_mcp_...` válido, incrusta automáticamente `EMBEDDED_DEFAULT_URL` y `EMBEDDED_DEFAULT_TOKEN`
    para que en Termux / Android / harnesses ligeros no haya que repetir `--url` ni `--token` en cada comando.
    """
    if not BRIDGE_SCRIPT_PATH.exists():
        raise HTTPException(status_code=404, detail="No se encontró studio_mcp.py en el servidor.")
    code = BRIDGE_SCRIPT_PATH.read_text(encoding="utf-8")
    expected = get_expected_mcp_token()
    details = get_mcp_connection_details(request)
    pub = details["base_url"]
    if not expected or (token and secrets.compare_digest(token.strip(), expected)):
        code = code.replace('EMBEDDED_DEFAULT_URL = ""', f"EMBEDDED_DEFAULT_URL = {json.dumps(pub)}", 1)
        if expected:
            code = code.replace('EMBEDDED_DEFAULT_TOKEN = ""', f"EMBEDDED_DEFAULT_TOKEN = {json.dumps(expected)}", 1)
    return PlainTextResponse(
        content=code,
        media_type="text/x-python; charset=utf-8",
    )


@mcp_router.get("/mcp/guia")
@mcp_router.get("/mcp/skill.md")
def mcp_agent_guide_markdown(
    request: Request,
    authorization: Optional[str] = Header(default=None),
    x_studio_token: Optional[str] = Header(default=None),
    token: Optional[str] = Query(default=None),
) -> PlainTextResponse:
    """
    Manual auto-contenido con credenciales integradas pensado para Termux Android, DeepSeek Harness,
    Qwen Flash, OpenCode o cualquier agente al que el usuario le pase este único enlace en el chat.
    """
    _verify_mcp_auth(request, authorization=authorization, x_studio_token=x_studio_token, token_query=token)
    details = get_mcp_connection_details(request)
    hw = hardware_tracker.get_snapshot(active_jobs_count=job_manager.active_count())
    gpu = hw["gpu"]
    disk = hw["disk"]

    colab_console_log(
        f"[MCP] Guía y credenciales leídas por tu agente mediante enlace directo ({request.headers.get('user-agent', 'terminal')[:48]}).",
        icon="📱",
    )

    video_recipes_md = MANUAL_RESOURCES["studio://manual/video-recipes"]["content"]
    image_tts_md = MANUAL_RESOURCES["studio://manual/image-tts-montage"]["content"]

    guide = f"""# MANUAL OFICIAL DE CONEXIÓN Y CONTROL — STUDIO IA (GOOGLE COLAB)

Estás conectado a una instancia real de **Studio IA** corriendo sobre GPU en Google Colab:
- **GPU detectada ahora mismo:** `{gpu.get('gpu_name', 'CPU')}` ({gpu.get('total_vram_gb', 0)} GB VRAM total · {gpu.get('free_vram_gb', 0)} GB libres)
- **Disco libre en Colab:** `{disk.get('free_gb', '?')} GB`
- **URL Base:** `{details['base_url']}`
- **Endpoint MCP HTTP directo (con token incluido):** `{details['mcp_url_with_token']}`

---

## PASO 1 OBLIGATORIO EN TERMUX / BASH (EJECUTA ESTO PRIMERO)

Descarga el cliente-puente oficial `studio_mcp.py` (cero dependencias externas, solo librería estándar de Python; **ya viene preconfigurado con tu URL y tu Token** y evita cortes por timeout de Cloudflare de 100 s haciendo *polling* cada 2 s):

```bash
curl -fsSL "{details['bridge_preconfigured_url']}" -o studio_mcp.py
python3 studio_mcp.py --cli studio_get_state_and_manual '{{"include_recent_assets": true}}'
```

*(Si tu harness usa protocolo MCP nativo en vez de Bash CLI, puedes registrar `python3 studio_mcp.py` por `stdio` o conectar por HTTP Streamable directamente a `{details['mcp_url_with_token']}`).*

---

## REGLAS ESTRICTAS ANTI-INVENCIÓN (OBLIGATORIAS)

{MCP_SERVER_INSTRUCTIONS}

---

## COMANDOS EXACTOS PARA USAR DESDE TERMUX / BASH (`--cli`)

Todos los archivos generados (`.mp4`, `.png`, `.mp3`, `.wav`, `.zip`) se descargan **automáticamente** a tu carpeta local `./salidas_studio` en Termux (o a la ruta que pongas en `"save_to_local_path"`, ej. `"/sdcard/Download/mi_video.mp4"`).

### 1) Ver todas las herramientas disponibles y sus `enum` cerrados permitidos
```bash
python3 studio_mcp.py --list-tools
```

### 2) Pedir la receta exacta verificada antes de generar (Cero Alucinaciones)
- `task` permitidos: `"video_flf2v"`, `"video_i2v"`, `"video_t2v"`, `"image_create"`, `"image_edit"`, `"tts"`, `"montage"`.
- `priority` permitidos: `"max_quality_14b"`, `"balanced_fast"`, `"ultra_fast_draft"`, `"zero_vram_cpu"`.
```bash
python3 studio_mcp.py --cli studio_recommend_pipeline '{{"task": "video_flf2v", "priority": "max_quality_14b"}}'
```

### 3) Generar Vídeo «Frame a Frame» (Inicio → Fin) subiendo 2 fotos desde Termux/Android y bajando el MP4
```bash
python3 studio_mcp.py --cli studio_generate_video '{{
  "video_submode": "flf2v",
  "mode": "ai",
  "model_id": "wan21-vace-14b-fp8",
  "video_lora_id": "lora-wan21-14b-fusionx-phantom",
  "use_official_defaults": true,
  "prompt": "el personaje gira suavemente la cabeza y sonríe con iluminación cinematográfica realista",
  "local_image_path": "./inicio.png",
  "local_end_image_path": "./final.png",
  "duration_sec": 3.5
}}'
```

### 4) Generar Imagen o Editar Imagen local
```bash
python3 studio_mcp.py --cli studio_generate_or_edit_image '{{
  "model_id": "qwen-image-2512",
  "use_official_defaults": true,
  "prompt": "retrato fotográfico hiperrealista en estudio con luz lateral cálida"
}}'
```

### 5) Generar Voz / Narración TTS (`studio_generate_speech`)
```bash
python3 studio_mcp.py --cli studio_generate_speech '{{
  "engine_id": "edge-tts-cloud",
  "voice_id": "es-ES-AlvaroNeural",
  "rate_pct": 0,
  "text": "Hola, este audio ha sido generado en Google Colab controlado desde Termux."
}}'
```

### 6) Montaje Final Multipista (`studio_render_montage`)
```bash
python3 studio_mcp.py --cli studio_render_montage '{{
  "title": "corto_final",
  "items": [
    {{"media_rel": "outputs/videos/clip_1.mp4", "audio_rel": "outputs/audio/voz_1.mp3", "fit_to_audio": true}},
    {{"media_rel": "outputs/videos/clip_2.mp4", "duration": 3.5}}
  ],
  "fade_seconds": 0.2
}}'
```

### 7) Liberar VRAM o borrar un modelo del disco de Colab si falta espacio
```bash
python3 studio_mcp.py --cli studio_free_memory_or_disk '{{"action": "unload_vram", "target": "all"}}'
```

---

{video_recipes_md}

---

{image_tts_md}
"""
    return PlainTextResponse(content=guide, media_type="text/markdown; charset=utf-8")


@mcp_router.get("/api/mcp/info")
def api_get_mcp_info(request: Request) -> Dict[str, Any]:
    return {"ok": True, **get_mcp_connection_details(request)}


@mcp_router.post("/api/mcp/tunnel-info")
async def api_set_mcp_tunnel_info(request: Request) -> Dict[str, Any]:
    client_host = (request.client.host if request.client else "") or ""
    if client_host not in ("127.0.0.1", "::1", "localhost", "testclient"):
        _verify_mcp_auth(request)
    body = await request.json()
    pub_url = str(body.get("public_url") or "").strip()
    if pub_url:
        set_mcp_public_url(pub_url)
    return {"ok": True, "public_url": _mcp_runtime_state["public_url"]}
