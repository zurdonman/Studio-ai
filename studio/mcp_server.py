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
    OUTPUTS_DIR,
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
from studio.engines.ffmpeg_utils import probe_media_info, render_montage_mp4, verify_image_file
from studio.engines.higgsfield_engine import (
    APERTURES,
    CINEMA_CAMERAS,
    CINEMA_LENSES,
    FOCAL_LENGTHS,
    HIGGSFIELD_MOTIONS,
    LIGHTING_MOODS,
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
from studio.engines.tts_engine import EDGE_TTS_VOICES, load_tts_model_into_memory, synthesize_voice
from studio.engines.video_engine import generate_video_clip, load_video_model_into_memory
from studio.hardware import (
    build_full_session_log_document,
    colab_console_log,
    hardware_tracker,
    query_gpu_info,
)
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

4. CONSISTENCIA ESTRICTA DE PERSONAJES EN SECUENCIAS / VARIOS FOTOGRAMAS (OBLIGATORIO):
   - NUNCA generes el Fotograma 2, 3, 4... desde cero (`Text-to-Image` sin referencia) cuando deban salir las mismas personas o el mismo lugar (ej. personas comiendo en un restaurante), porque cada llamada suelta inventará caras y ropa distintas.
   - Sigue SIEMPRE este protocolo de 3 pasos para secuencias coherentes:
     1) **Fotograma 1 = ANCLA MAESTRA (`anchor_image_rel`):** Genera únicamente el Fotograma 1 desde texto (o usa la foto subida por el usuario).
     2) **Fotogramas 2..N = EDICIÓN ANCLADA (`qwen-image-edit-2511-pack`):** Llama a `studio_generate_or_edit_image` pasando SIEMPRE `model_id="qwen-image-edit-2511-pack"`, `base_image_rel=<anchor_image_rel>` y `reference_rels=[<anchor_image_rel>]`, pidiendo conservar exactamente los mismos rostros, peinados, ropa, mesa e iluminación y cambiar SOLO la pose/acción de ese instante.
     3) **AUDITORÍA DE COHERENCIA (`studio_analyze_coherence`):** Antes de renderizar vídeo pesado, llama a `studio_analyze_coherence(anchor_image_rel=<anchor>, candidate_image_rels=[...], mode="hybrid_auto")` (combina OpenCV fotométrico 0.1s + Qwen2.5-VL-3B 4-bit semántico). Si algún fotograma sale `verdict="NEEDS_REPAIR"`, ejecuta directamente el `recommended_repair_tool_call` que te devuelve la herramienta para corregirlo.

5. LIPSYNC HD MULTI-PERSONAJE, MÚSICA, SONIDO DE FONDO Y MEZCLA FFMPEG CON AUTO-DUCKING:
   - Para que los personajes MUEVAN LA BOCA en sincronía real con las voces (sobre una foto o sobre un vídeo generado con Wan 14B):
     1) Si `edtalk-256-lipsync-hd` (solo 0.65 GB) aún no está descargado, prepáralo con `studio_prepare_model(model_id="edtalk-256-lipsync-hd")`.
     2) Llama a `studio_lipsync_character`:
        - 1 personaje: pasa `media_rel` + `voice_audio_rel` + `target_face="largest"` (o `"leftmost"` / `"rightmost"`).
        - Varias personas conversando (ej. en un restaurante): pasa `dialogue_turns=[{"voice_audio_rel": "voz_1.mp3", "target_face": "leftmost", "start_sec": 0.2}, {"voice_audio_rel": "voz_2.mp3", "target_face": "rightmost", "start_sec": 3.5}]` para que cada comensal mueva los labios solo en su turno (XSeg protege copas/cubiertos delante de la boca y GFPGAN 1.4 restaura labios a 512×512).
   - Para crear MÚSICA y SONIDO DE FONDO (ambiente restaurante, lluvia, pad cálido):
     - Llama a `studio_generate_music_or_sfx` (`model_id="procedural-ambience-cpu"` con `preset="restaurant_warm_murmur"` en 0.2s y 0 GB VRAM, o `model_id="musicgen-small-bgm"` para música instrumental IA, o `model_id="audioldm2-sfx"` para efectos Foley).
   - Para MEZCLAR VOCES + MÚSICA + AMBIENTE sin bajar el volumen de las voces ni congelar el vídeo:
     - Usa `studio_mix_audio_tracks` (o pasa `music_rel` y `ambient_rel` directamente a `studio_lipsync_character` / `studio_render_montage`). Aplica Auto-Ducking (`sidechaincompress`) atenuando música/ambiente automáticamente mientras alguien habla y extiende el vídeo con fluidez (`setpts` / ping-pong) si el diálogo dura más que el clip.

6. GESTIÓN DE ESPACIO EN DISCO, MEMORIA Y EXPORTACIÓN DE LOGS EN COLAB:
   - Si una descarga avisa `DISK_SPACE_INSUFFICIENT`, consulta las sugerencias devueltas y usa `studio_free_memory_or_disk(action="delete_from_disk", target=...)` (preguntando antes al usuario si procede).
   - Si el usuario pide el registro completo de lo ocurrido en la sesión, llama a `studio_export_session_log(format="md")`.
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
    "studio://manual/character-consistency": {
        "name": "Protocolo Oficial de Coherencia y Consistencia de Personajes (Ancla + Qwen2.5-VL + OpenCV)",
        "mimeType": "text/markdown",
        "content": (
            "# Protocolo Anti-Deriva de Personajes para Storyboards y Frame a Frame\n\n"
            "## ¿Por qué cambian las caras si generas fotogramas sueltos?\n"
            "Si llamas a `studio_generate_or_edit_image` sin `base_image_rel` varias veces, el modelo crea actores distintos cada vez.\n\n"
            "## Flujo obligatorio en 3 pasos\n"
            "1. **Crear Fotograma 1 (Ancla):** `studio_generate_or_edit_image(model_id='qwen-image-edit-2511-pack', prompt='...')` → guarda su `rel_path` como `ANCHOR`.\n"
            "2. **Derivar Fotogramas 2, 3, 4 desde `ANCHOR`:** `studio_generate_or_edit_image(model_id='qwen-image-edit-2511-pack', base_image_rel=ANCHOR, reference_rels=[ANCHOR], prompt='Mantén exactamente las mismas personas, rostros, ropa e iluminación; cambia solo la acción a: ...')`.\n"
            "3. **Auditar con el Inspector Híbrido (`studio_analyze_coherence`):**\n"
            "   - Etapa 1 (`OpenCV`, 0.1s, 0 GB VRAM): Verifica proporción, paleta HSV/LAB, iluminación y nitidez.\n"
            "   - Etapa 2 (`Qwen2.5-VL-3B-Instruct` 4-bit, ~2.8 GB VRAM): Inspecciona semánticamente rostros, pelo, ropa, número de personas y viabilidad de transición.\n"
            "   - Si devuelve `NEEDS_REPAIR`, ejecuta el `recommended_repair_tool_call` incluido en la respuesta antes de animar en `14B`.\n"
        ),
    },
    "studio://manual/lipsync-and-sound": {
        "name": "Manual Oficial de LipSync HD Multi-Persona, Música, Ambiente y Mezcla FFmpeg con Auto-Ducking",
        "mimeType": "text/markdown",
        "content": (
            "# LipSync HD (`EDTalk-256 + GFPGAN 1.4 + XSeg`) y Diseño Sonoro Multipista\n\n"
            "## 1. LipSync Fonético Realista (`studio_lipsync_character`)\n"
            "- Paquete recomendado: `edtalk-256-lipsync-hd` (0.65 GB en disco, ~1.5 GB VRAM, 4–9 s en Colab L4).\n"
            "- Incluye detector multi-rostro `YuNet`, animación labial `EDTalk 256`, máscara de oclusión `XSeg` (respeta copas/cubiertos/manos) y restauración `GFPGAN 1.4` a 512×512.\n"
            "- En escenas con dos personas (p.ej. comiendo en un restaurante), usa `dialogue_turns` indicando `target_face='leftmost'` para la persona de la izquierda y `target_face='rightmost'` para la de la derecha con sus `start_sec`.\n\n"
            "## 2. Música y Ambiente Sonoro (`studio_generate_music_or_sfx`)\n"
            "- `procedural-ambience-cpu` (0 GB, 0.2 s): presets estéreo `restaurant_warm_murmur`, `rain_cozy`, `cinematic_warm_pad`, `wind_nature`, `room_tone_clean`.\n"
            "- `musicgen-small-bgm` (2.0 GB): música instrumental IA original (jazz suave, piano, cine).\n"
            "- `audioldm2-sfx` (3.8 GB): efectos Foley y paisajes acústicos desde texto.\n\n"
            "## 3. Mezcla Multipista sin Pérdida de Volumen (`studio_mix_audio_tracks`)\n"
            "- Posiciona voces con `start_sec` al milisegundo (`normalize=0`, sin perder -6 dB), aplica Auto-Ducking (`sidechaincompress`) sobre música/ambiente cuando alguien habla y extiende el vídeo sin congelar fotogramas.\n"
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
    image_model_ids = [m["id"] for m in all_defs if m["category"] == "image"]
    video_model_ids = [m["id"] for m in all_defs if m["category"] == "video"]
    video_lora_ids = [""] + [m["id"] for m in all_defs if m["category"] == "lora" and m.get("lora_target") == "video"]
    image_lora_ids = [m["id"] for m in all_defs if m["category"] == "lora" and (m.get("lora_target") or "image") == "image"]
    tts_engine_ids = [m["id"] for m in all_defs if m["category"] == "tts"]
    lipsync_model_ids = [m["id"] for m in all_defs if m["category"] == "lipsync"]
    audio_gen_ids = ["auto"] + [m["id"] for m in all_defs if m["category"] == "audio"]
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
                            "consistent_keyframes",
                            "coherence_check",
                            "lipsync_talking_scene",
                            "music_and_ambience",
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
                        "enum": ["all", "video", "image", "vision", "lipsync", "audio", "lora", "tts"],
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
                        "description": "Para `unload_vram`: 'all', 'video', 'image', 'vision', 'tts' o un `model_id`. Para `delete_from_disk`: el `model_id` exacto a borrar.",
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
                    "model_id": {
                        "type": "string",
                        "enum": image_model_ids,
                        "default": "qwen-image-edit-2511-pack",
                        "description": (
                            "Modelo de imagen verificado. Usa SIEMPRE `'qwen-image-edit-2511-pack'` cuando pases `base_image_rel` "
                            "o `reference_rels` para mantener los mismos personajes y escenario entre fotogramas."
                        ),
                    },
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
                        "description": "Lista de LoRAs apilables de imagen (coherencia de personaje/ángulos, relight de entorno, piel realista, flatlog cine, anime, etc.).",
                    },
                    "lora_scales": {
                        "type": "object",
                        "description": "Opcional: pesos por LoRA (ej. `{'qwen-lora-skin-realism': 0.85, 'qwen-lora-anylight-env': 0.8}`).",
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
                            "LoRA principal / acelerador opcional (cadena vacía '' = sin LoRA). En `flf2v` usa solo LoRAs bidireccionales "
                            "como `lora-wan21-14b-fusionx-phantom` (VACE 14B), `lora-wan21-14b-fusionx-i2v` (FLF2V 14B) o `lora-wan21-1.3b-bidirect2` (1.3B)."
                        ),
                    },
                    "extra_video_lora_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Opcional: LoRAs creativos apilables de coherencia de personaje, iluminación/entorno, detalle o cámara "
                            "(ej. `['lora-wan21-14b-unianimate-char', 'lora-wan21-14b-moviigen-cinema', 'lora-wan21-14b-detailz']`, "
                            "`['lora-ltx13b-detailer-ic']`, `['lora-ltx2-cam-dolly-in', 'lora-ltx2-detailer-ic']`)."
                        ),
                    },
                    "extra_video_lora_scales": {
                        "type": "object",
                        "description": "Opcional: escala de cada LoRA extra por su ID (ej. `{'lora-wan21-14b-detailz': 0.55}`).",
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
        {
            "name": "studio_analyze_coherence",
            "description": (
                "Inspector híbrido de coherencia visual y consistencia de personajes (`OpenCV` fotométrico 0.1s + "
                "`Qwen2.5-VL-3B-Instruct` 4-bit NF4 semántico). Compara el Fotograma Ancla (`anchor_image_rel`) frente a "
                "uno o varios fotogramas candidatos (`candidate_image_rels`) verificando si las personas, rostros, pelo, "
                "ropa, escenario e iluminación siguen siendo idénticos, y devuelve `recommended_repair_tool_call` listo "
                "para regenerar automáticamente cualquier fotograma con deriva."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "anchor_image_rel": {
                        "type": "string",
                        "description": "`rel_path` en Colab del Fotograma Ancla #1 (la imagen maestra cuya identidad de personajes debe respetarse).",
                    },
                    "candidate_image_rels": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"type": "string"},
                        "description": "Lista de `rel_path` de los fotogramas 2..N que quieres auditar frente al ancla.",
                    },
                    "scene_goal": {
                        "type": "string",
                        "default": "",
                        "description": "Contexto breve de la escena (ej. 'dos amigos comiendo en el mismo restaurante manteniendo rostros y ropa').",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["hybrid_auto", "vlm_qwen", "fast_cv"],
                        "default": "hybrid_auto",
                        "description": (
                            "`'hybrid_auto'` (recomendado): ejecuta OpenCV (color, luz, aspecto, estructura) + Qwen2.5-VL-3B 4-bit en GPU (~2.8 GB VRAM). "
                            "`'fast_cv'`: solo métricas físicas OpenCV instantáneas en CPU (0 GB VRAM)."
                        ),
                    },
                    "strictness_threshold": {
                        "type": "number",
                        "default": 78.0,
                        "minimum": 45.0,
                        "maximum": 95.0,
                        "description": "Puntuación mínima (0-100) para dar por válido (`PASS`) un fotograma.",
                    },
                    "wait": {"type": "boolean", "default": True},
                },
                "required": ["anchor_image_rel", "candidate_image_rels"],
            },
        },
        {
            "name": "studio_generate_music_or_sfx",
            "description": (
                "Genera música instrumental de fondo (`musicgen-small-bgm`), efectos Foley/paisajes sonoros IA (`audioldm2-sfx`) "
                "o ambiente estéreo instantáneo en 0.2s sin VRAM (`procedural-ambience-cpu` con presets como `restaurant_warm_murmur`, "
                "`rain_cozy`, `cinematic_warm_pad`, `wind_nature`, `room_tone_clean`)."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "default": "warm smooth acoustic jazz piano in a cozy restaurant",
                        "description": "Descripción de la música o ambiente deseado (en inglés para MusicGen/AudioLDM2, o cualquier idioma para DSP).",
                    },
                    "model_id": {
                        "type": "string",
                        "enum": audio_gen_ids,
                        "default": "auto",
                        "description": "`'auto'`, `'procedural-ambience-cpu'` (0 GB instantáneo), `'musicgen-small-bgm'` (música IA) o `'audioldm2-sfx'` (Foley IA).",
                    },
                    "preset": {
                        "type": "string",
                        "enum": ["auto"] + list(AMBIENCE_PRESETS.keys()),
                        "default": "auto",
                        "description": "Preset acústico cuando se usa `procedural-ambience-cpu` (ej. `'restaurant_warm_murmur'`).",
                    },
                    "duration_seconds": {
                        "type": "number",
                        "default": 10.0,
                        "minimum": 2.0,
                        "maximum": 30.0,
                    },
                    "seed": {"type": "integer", "default": 42},
                    "wait": {"type": "boolean", "default": True},
                },
            },
        },
        {
            "name": "studio_lipsync_character",
            "description": (
                "Sincronización labial fonética HD (`edtalk-256-lipsync-hd`: EDTalk 256 + YuNet + oclusión XSeg para copas/cubiertos/manos "
                "+ restauración GFPGAN 1.4 a 512×512). Anima la boca de 1 o varios personajes sobre una imagen fija o un vídeo MP4 "
                "(como los generados con Wan 14B) y mezcla automáticamente voces, música y ambiente con Auto-Ducking."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "media_rel": {
                        "type": "string",
                        "description": "`rel_path` de la imagen retrato o del clip de vídeo MP4 donde aparecen los personajes.",
                    },
                    "voice_audio_rel": {
                        "type": "string",
                        "description": "`rel_path` del audio de voz cuando habla un único personaje (opcional si usas `dialogue_turns`).",
                    },
                    "target_face": {
                        "type": "string",
                        "enum": ["largest", "leftmost", "rightmost", "center", "index_0", "index_1"],
                        "default": "largest",
                        "description": "Qué rostro animar (`'leftmost'` = persona de la izquierda, `'rightmost'` = persona de la derecha, `'largest'` = rostro principal).",
                    },
                    "dialogue_turns": {
                        "type": "array",
                        "description": (
                            "Lista opcional de turnos de diálogo para escenas con varios personajes (ej. dos amigos en restaurante): "
                            "cada elemento lleva `voice_audio_rel`, `target_face` (`'leftmost'`/`'rightmost'`), `start_sec` y `volume`."
                        ),
                        "items": {
                            "type": "object",
                            "properties": {
                                "voice_audio_rel": {"type": "string"},
                                "target_face": {
                                    "type": "string",
                                    "enum": ["largest", "leftmost", "rightmost", "center", "index_0", "index_1"],
                                    "default": "largest",
                                },
                                "start_sec": {"type": "number", "default": 0.0, "minimum": 0.0},
                                "volume": {"type": "number", "default": 1.0, "minimum": 0.1, "maximum": 2.0},
                            },
                            "required": ["voice_audio_rel"],
                        },
                    },
                    "model_id": {
                        "type": "string",
                        "enum": lipsync_model_ids or ["edtalk-256-lipsync-hd"],
                        "default": "edtalk-256-lipsync-hd",
                    },
                    "lip_weight": {
                        "type": "number",
                        "default": 0.85,
                        "minimum": 0.3,
                        "maximum": 1.15,
                        "description": "Intensidad de apertura articular labial (0.85 recomendado).",
                    },
                    "enhance_face": {
                        "type": "boolean",
                        "default": True,
                        "description": "Restaura labios y dientes en alta definición 512×512 con GFPGAN 1.4.",
                    },
                    "restore_strength": {
                        "type": "number",
                        "default": 0.50,
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                    "music_rel": {
                        "type": "string",
                        "description": "`rel_path` opcional de música de fondo (se atenúa sola con Auto-Ducking al hablar).",
                    },
                    "music_volume": {"type": "number", "default": 0.22, "minimum": 0.02, "maximum": 1.5},
                    "ambient_rel": {
                        "type": "string",
                        "description": "`rel_path` opcional de sonido ambiente de fondo (ej. murmullo cálido de restaurante).",
                    },
                    "ambient_volume": {"type": "number", "default": 0.18, "minimum": 0.02, "maximum": 1.5},
                    "extend_video_to_audio": {
                        "type": "boolean",
                        "default": True,
                        "description": "Si el diálogo dura más que el vídeo, extiende el vídeo suavemente sin congelar fotogramas.",
                    },
                    "wait": {"type": "boolean", "default": True},
                },
                "required": ["media_rel"],
            },
        },
        {
            "name": "studio_mix_audio_tracks",
            "description": (
                "Mezclador y masterizador multipista FFmpeg profesional para vídeo o audio: posiciona 1 o varias voces con "
                "`start_sec` exacto sin perder volumen (`normalize=0`), añade música (`music_rel`) y sonido ambiente (`ambient_rel`) "
                "con **Auto-Ducking dinámico (`sidechaincompress`)** cuando hablan los personajes, y extiende el vídeo sin congelarlo si hace falta."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "video_rel": {
                        "type": "string",
                        "description": "`rel_path` opcional del vídeo MP4 al que mezclar/reemplazar la banda sonora.",
                    },
                    "voice_tracks": {
                        "type": "array",
                        "description": "Pistas de diálogo con su instante de entrada `start_sec` en segundos.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "audio_rel": {"type": "string"},
                                "start_sec": {"type": "number", "default": 0.0, "minimum": 0.0},
                                "volume": {"type": "number", "default": 1.0, "minimum": 0.05, "maximum": 2.5},
                            },
                            "required": ["audio_rel"],
                        },
                    },
                    "music_rel": {"type": "string", "description": "`rel_path` opcional de música de fondo."},
                    "music_volume": {"type": "number", "default": 0.24, "minimum": 0.02, "maximum": 2.0},
                    "ambient_rel": {"type": "string", "description": "`rel_path` opcional de ambiente sonoro (restaurante, lluvia, sala)."},
                    "ambient_volume": {"type": "number", "default": 0.18, "minimum": 0.02, "maximum": 2.0},
                    "keep_original_audio": {
                        "type": "boolean",
                        "default": False,
                        "description": "Conservar también el audio que ya traía el `video_rel` (p.ej. `true` si el vídeo ya salió de `studio_lipsync_character`).",
                    },
                    "original_audio_volume": {"type": "number", "default": 1.0, "minimum": 0.05, "maximum": 2.0},
                    "auto_duck_bg": {
                        "type": "boolean",
                        "default": True,
                        "description": "Atenuar automáticamente música y ambiente mientras suenan las voces.",
                    },
                    "extend_video_to_audio": {
                        "type": "boolean",
                        "default": True,
                        "description": "Extender vídeo suavemente (setpts / ping-pong) si las voces duran más que el clip.",
                    },
                    "fade_out_seconds": {"type": "number", "default": 0.80, "minimum": 0.0, "maximum": 4.0},
                    "wait": {"type": "boolean", "default": True},
                },
            },
        },
        {
            "name": "studio_higgsfield_presets_and_cast",
            "description": (
                "MODO HIGGSFIELD (100% gratis en Google Colab). Lista todas las cámaras de cine (ARRI 8K, IMAX 70mm, 16mm...), "
                "lentes ópticas (Anamórfica 2x, 70s Prime, Halation, Swirl Bokeh...), focales (14-85mm), aperturas (f/1.4-f/11), "
                "los 12 movimientos de cámara Higgsfield (Crash Zoom, Dolly Zoom, Bullet Time, FPV Drone, SnorriCam, Handheld...) "
                "y las fichas de personajes guardados en el Casting Vault."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {},
            },
        },
        {
            "name": "studio_higgsfield_save_character",
            "description": (
                "Guarda, actualiza o elimina un personaje persistente en el Casting Vault de Modo Higgsfield "
                "(nombre, foto ancla `anchor_image_rel`, rasgos físicos y voz TTS asignada)."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["save", "delete"],
                        "default": "save",
                    },
                    "char_id": {"type": "string", "description": "ID existente para actualizar o borrar (opcional al crear)."},
                    "name": {"type": "string", "description": "Nombre del personaje (ej. 'Marco', 'Lucía')."},
                    "appearance_prompt": {"type": "string", "description": "Descripción física del rostro, pelo, edad y vestimenta."},
                    "anchor_image_rel": {"type": "string", "description": "`rel_path` de su retrato/foto ancla en Colab."},
                    "voice_id": {"type": "string", "enum": voice_ids, "default": "es-ES-AlvaroNeural"},
                    "default_face_side": {
                        "type": "string",
                        "enum": ["largest", "leftmost", "rightmost", "center"],
                        "default": "largest",
                    },
                },
            },
        },
        {
            "name": "studio_higgsfield_director",
            "description": (
                "MODO HIGGSFIELD TODO-EN-UNO (100% local en Colab, 0€ APIs): ejecuta cualquiera de los 4 estudios Higgsfield:\n"
                "- `action='scene_director'`: crea plano Cinema Studio + anima movimiento Higgsfield + locuta diálogos + LipSync HD + Ambiente Auto-Ducking.\n"
                "- `action='cinema_shot'`: fotografía un fotograma con Cámara + Lente + Focal + Apertura + Personajes del Vault.\n"
                "- `action='motion_clip'`: anima una imagen con un preset Higgsfield Motion (`crash_zoom`, `dolly_zoom_vertigo`, `bullet_time_orbit`, etc.).\n"
                "- `action='smart_clip_9_16'`: recorta y reencuadra inteligentemente al rostro un vídeo 16:9 a formato vertical 9:16 Reels/Shorts."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["scene_director", "cinema_shot", "motion_clip", "smart_clip_9_16"],
                        "default": "scene_director",
                    },
                    "prompt": {"type": "string", "description": "Acción o escena cinematográfica."},
                    "camera_id": {"type": "string", "enum": list(CINEMA_CAMERAS.keys()), "default": "arri_full_frame_8k"},
                    "lens_id": {"type": "string", "enum": list(CINEMA_LENSES.keys()), "default": "classic_anamorphic"},
                    "focal_mm": {"type": "string", "enum": list(FOCAL_LENGTHS.keys()), "default": "35"},
                    "aperture": {"type": "string", "enum": list(APERTURES.keys()), "default": "f/2.8"},
                    "lighting_id": {"type": "string", "enum": list(LIGHTING_MOODS.keys()), "default": "natural_cinema"},
                    "motion_id": {"type": "string", "enum": list(HIGGSFIELD_MOTIONS.keys()), "default": "dolly_in_dramatic"},
                    "video_engine_mode": {
                        "type": "string",
                        "enum": ["auto", "ai_14b", "ai_fast", "ffmpeg_instant"],
                        "default": "auto",
                    },
                    "aspect_ratio": {"type": "string", "enum": ["16:9", "9:16", "1:1"], "default": "16:9"},
                    "duration_seconds": {"type": "number", "default": 4.5, "minimum": 1.5, "maximum": 15.0},
                    "base_image_rel": {"type": "string", "description": "Imagen inicial opcional (para `cinema_shot`, `motion_clip` o `scene_director`)."},
                    "video_rel": {"type": "string", "description": "Vídeo de entrada cuando `action='smart_clip_9_16'`."},
                    "start_sec": {"type": "number", "default": 0.0},
                    "end_sec": {"type": "number", "default": 0.0},
                    "character_ids": {"type": "array", "items": {"type": "string"}},
                    "dialogue_lines": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "character_id": {"type": "string"},
                                "voice_id": {"type": "string"},
                                "target_face": {"type": "string", "enum": ["largest", "leftmost", "rightmost", "center"]},
                                "start_sec": {"type": "number"},
                                "text": {"type": "string"},
                            },
                            "required": ["text"],
                        },
                    },
                    "ambience_preset": {
                        "type": "string",
                        "enum": ["none"] + list(AMBIENCE_PRESETS.keys()),
                        "default": "none",
                    },
                    "apply_lipsync": {"type": "boolean", "default": True},
                    "image_lora_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Opcional: LoRAs de imagen de coherencia/calidad/iluminación para el plano Cinema Studio.",
                    },
                    "video_lora_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Opcional: LoRAs de vídeo creativos apilables para la animación Higgsfield.",
                    },
                    "seed": {"type": "integer", "default": 42},
                    "wait": {"type": "boolean", "default": True},
                },
            },
        },
        {
            "name": "studio_export_session_log",
            "description": (
                "Genera y guarda el registro completo de todo lo sucedido desde que se encendió Studio IA hasta este instante "
                "(estado de GPU/disco, tabla de trabajos, logs internos, errores y consola cronológica) en `.md` o `.txt`."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "format": {
                        "type": "string",
                        "enum": ["md", "txt"],
                        "default": "md",
                        "description": "Formato del documento exportado (`'md'` Markdown estructurado o `'txt'` texto plano).",
                    }
                },
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

    if task == "consistent_keyframes":
        return {
            "task": task,
            "detected_gpu": gpu_info.get("gpu_name", "CPU"),
            "free_disk_gb": free_disk,
            "workflow_steps": [
                {
                    "step": 1,
                    "action": "Generar Fotograma 1 (ANCLA) con `studio_generate_or_edit_image` (`model_id='qwen-image-edit-2511-pack'`).",
                },
                {
                    "step": 2,
                    "action": (
                        "Generar Fotogramas 2..N llamando a `studio_generate_or_edit_image` con `model_id='qwen-image-edit-2511-pack'`, "
                        "`base_image_rel=<rel_del_ancla>` y `reference_rels=[<rel_del_ancla>]`, cambiando SOLO la acción en el `prompt`."
                    ),
                },
                {
                    "step": 3,
                    "action": "Auditar todos los fotogramas con `studio_analyze_coherence(mode='hybrid_auto')` y ejecutar `recommended_repair_tool_call` si alguno falla.",
                },
            ],
            "recommended_tool": "studio_generate_or_edit_image",
            "recommended_arguments": {
                "model_id": "qwen-image-edit-2511-pack",
                "use_official_defaults": True,
                "use_lightning_lora": True,
            },
            "why_this_choice": "Anclar cada fotograma a la Imagen #1 en Qwen-Image-Edit 2511 + auditar con Qwen2.5-VL-3B evita que cambien caras o ropa.",
        }

    if task == "lipsync_talking_scene":
        return {
            "task": task,
            "detected_gpu": gpu_info.get("gpu_name", "CPU"),
            "free_disk_gb": free_disk,
            "workflow_steps": [
                {"step": 1, "action": "Descargar una vez `edtalk-256-lipsync-hd` (0.65 GB) con `studio_prepare_model(model_id='edtalk-256-lipsync-hd')`."},
                {"step": 2, "action": "Generar las voces en español con `studio_generate_speech` y el ambiente con `studio_generate_music_or_sfx(model_id='procedural-ambience-cpu', preset='restaurant_warm_murmur')`."},
                {"step": 3, "action": "Aplicar LipSync por personaje (`target_face='leftmost'` / `'rightmost'` en `dialogue_turns`) + `ambient_rel` + `music_rel` llamando a `studio_lipsync_character`."},
            ],
            "recommended_tool": "studio_lipsync_character",
            "recommended_arguments": {
                "model_id": "edtalk-256-lipsync-hd",
                "target_face": "largest",
                "lip_weight": 0.85,
                "enhance_face": True,
                "extend_video_to_audio": True,
            },
            "why_this_choice": "EDTalk-256 + YuNet + XSeg + GFPGAN 1.4 ocupa solo 0.65 GB (~1.5 GB VRAM) y sincroniza labios de 1 o varios personajes con Auto-Ducking integrado.",
        }

    if task == "music_and_ambience":
        use_ai_bgm = has_gpu and priority != "zero_vram_cpu" and catalog_manager.is_downloaded("musicgen-small-bgm")
        return {
            "task": task,
            "detected_gpu": gpu_info.get("gpu_name", "CPU"),
            "recommended_tool": "studio_generate_music_or_sfx",
            "recommended_arguments": {
                "model_id": "musicgen-small-bgm" if use_ai_bgm else "procedural-ambience-cpu",
                "preset": "restaurant_warm_murmur",
                "duration_seconds": 12.0,
            },
            "why_this_choice": (
                "Usa `procedural-ambience-cpu` (0 GB, 0.2s) para ambiente inmediato o descarga `musicgen-small-bgm` (2 GB) para música instrumental IA original."
            ),
        }

    if task == "coherence_check":
        return {
            "task": task,
            "detected_gpu": gpu_info.get("gpu_name", "CPU"),
            "free_disk_gb": free_disk,
            "recommended_tool": "studio_analyze_coherence",
            "recommended_arguments": {
                "mode": "hybrid_auto" if has_gpu else "fast_cv",
                "strictness_threshold": 78.0,
            },
            "why_this_choice": (
                "Combina filtro fotométrico instantáneo OpenCV (0.12s, color/luz/aspecto) + auditoría semántica de identidad "
                "con Qwen2.5-VL-3B-Instruct en 4-bit (~2.8 GB VRAM, puede convivir en L4 junto a Qwen-Image-Edit)."
            ),
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
        elif cat == "vision":
            if defn.get("builtin_no_download"):
                return _ok_result({"ok": True, "model_id": mid, "builtin": True})
            job = job_manager.submit(
                kind="load_model",
                title=f"[MCP] Preparar inspector multimodal: {defn['name']}",
                fn=lambda j: load_coherence_vlm_into_memory(model_id=mid, auto_download=True, job=j),
            )
        elif cat == "lipsync":
            def _prep_ls(j: Any) -> Dict[str, Any]:
                if not catalog_manager.is_downloaded(mid):
                    catalog_manager.download_model(mid, j)
                if load_vram:
                    lipsync_engine._ensure_loaded(mid, job=j)
                return {"model_id": mid, "loaded": load_vram}

            job = job_manager.submit(
                kind="load_model",
                title=f"[MCP] Preparar LipSync HD: {defn['name']}",
                fn=_prep_ls,
            )
        elif cat == "audio":
            if defn.get("builtin_no_download"):
                return _ok_result({"ok": True, "model_id": mid, "builtin": True})
            job = job_manager.submit(
                kind="download",
                title=f"[MCP] Preparar audio/música: {defn['name']}",
                fn=lambda j: catalog_manager.download_model(mid, j),
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
            elif target in ("image", "video", "tts", "vision", "lipsync", "audio_bgm"):
                info = memory_manager.unload_slot(target)
            else:
                info = memory_manager.unload_by_model_id(target)
            colab_console_log(
                f"[MCP] Memoria RAM/VRAM liberada (`{target}` → {info.get('vram_freed_gb', 0)} GB VRAM libres).",
                icon="🧹",
            )
            return _ok_result({"ok": True, "action": action, **info})

        if target in ("", "all", "video", "image", "tts", "vision"):
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
        raw_lora_scales = args.get("lora_scales") if isinstance(args.get("lora_scales"), dict) else {}
        lora_scales_map = {str(k): float(v) for k, v in raw_lora_scales.items()}
        wait_flag = bool(args.get("wait", True))

        req_img_model = str(args.get("model_id") or "qwen-image-edit-2511-pack").strip()
        # Protección anti-deriva: si hay imagen base o referencias ancla, forzar un modelo capaz de condicionar por imagen
        if (base_p is not None or refs_p) and req_img_model == "qwen-image-2512-pack":
            colab_console_log(
                "[MCP Guardia] Se detectaron imágenes de referencia/ancla con `qwen-image-2512-pack` (solo T2I); "
                "cambiando automáticamente a `qwen-image-edit-2511-pack` para preservar identidad de personajes.",
                icon="🛡️",
            )
            req_img_model = "qwen-image-edit-2511-pack"

        def _run_mcp_image(j: Any) -> Dict[str, Any]:
            return generate_or_edit_image(
                prompt=prompt,
                model_id=req_img_model,
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
                lora_scales=lora_scales_map,
                auto_download_loras=True,
                use_ffmpeg_bt709=True,
                job=j,
            )

        job = job_manager.submit(
            kind="image",
            title=f"[MCP] Imagen ({req_img_model}): {prompt[:38]}",
            fn=_run_mcp_image,
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
            lora_has_distill_steps = bool(
                lora_defn
                and (lora_defn.get("recommended_steps") is not None or lora_defn.get("flf2v_recommended_steps") is not None)
            )
            if lora_defn and lora_has_distill_steps:
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
                lora_scale = float(lora_defn.get("default_lora_scale", 1.0)) if lora_defn else 1.0
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
        extra_v_loras = [str(x).strip() for x in (args.get("extra_video_lora_ids") or []) if str(x).strip()]
        raw_ev_scales = args.get("extra_video_lora_scales") if isinstance(args.get("extra_video_lora_scales"), dict) else {}
        extra_v_scales = {str(k): float(v) for k, v in raw_ev_scales.items()}

        job = job_manager.submit(
            kind="video",
            title=f"[MCP] Vídeo ({sub} · {model_id}): {(prompt or 'animación')[:34]}",
            fn=lambda j: generate_video_clip(
                mode=mode,
                video_submode=sub,
                model_id=model_id,
                video_lora_id=lora_id,
                video_lora_scale=lora_scale,
                extra_video_lora_ids=extra_v_loras,
                extra_video_lora_scales=extra_v_scales,
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

    # 14. Inspector híbrido de coherencia visual y consistencia de personajes (OpenCV + Qwen2.5-VL-3B)
    if name == "studio_analyze_coherence":
        cand_rels = list(args.get("candidate_image_rels") or [])
        if not cand_rels:
            return _guided_error(
                "EMPTY_CANDIDATE_IMAGES",
                "`candidate_image_rels` está vacío.",
                "Pasa al menos un `rel_path` (o `local_candidate_image_paths` en `studio_mcp.py`) para compararlo contra `anchor_image_rel`.",
            )
        try:
            anchor_p = _resolve_rel_safe(args.get("anchor_image_rel"))
            if anchor_p is None:
                return _guided_error(
                    "MISSING_ANCHOR_IMAGE",
                    "Falta `anchor_image_rel` (la imagen maestra del Fotograma #1).",
                    "Pasa el `rel_path` del Fotograma #1 en `anchor_image_rel` (o `local_anchor_image_path` en el puente local).",
                )
            cand_ps: List[Path] = []
            for r in cand_rels:
                cp = _resolve_rel_safe(r)
                if cp is not None:
                    cand_ps.append(cp)
        except (ValueError, FileNotFoundError) as exc:
            return _guided_error(
                "COHERENCE_IMAGE_NOT_FOUND",
                str(exc),
                "Verifica los `rel_path` con `studio_list_assets` o pasa rutas locales si usas `studio_mcp.py`.",
            )

        coh_mode = str(args.get("mode") or "hybrid_auto").strip().lower()
        if coh_mode not in ("hybrid_auto", "vlm_qwen", "fast_cv"):
            return _guided_error(
                "INVALID_COHERENCE_MODE",
                f"`mode='{coh_mode}'` no es válido.",
                "Usa `'hybrid_auto'` (recomendado: OpenCV + Qwen2.5-VL-3B), `'vlm_qwen'` o `'fast_cv'`.",
            )
        scene_goal = str(args.get("scene_goal") or "")
        thresh = float(args.get("strictness_threshold") or 78.0)
        wait_flag = bool(args.get("wait", True))

        job = job_manager.submit(
            kind="coherence",
            title=f"[MCP] Coherencia ({coh_mode} · {len(cand_ps)} fotos vs {anchor_p.name})",
            fn=lambda j: run_coherence_analysis(
                anchor_path=anchor_p,
                candidate_paths=cand_ps,
                scene_goal=scene_goal,
                mode=coh_mode,
                strictness_threshold=thresh,
                job=j,
            ),
        )
        if not wait_flag:
            return _ok_result({"ok": True, "job_id": job.id, "status": job.status})
        return _ok_result(_wait_for_job_bounded(job.id))

    # 15. Generar música instrumental o ambiente/efectos sonoros
    if name == "studio_generate_music_or_sfx":
        prm = str(args.get("prompt") or "warm smooth acoustic jazz piano in a cozy restaurant")
        mid = str(args.get("model_id") or "auto").strip()
        prs = str(args.get("preset") or "auto").strip()
        dur_s = float(args.get("duration_seconds") or 10.0)
        sd = int(args.get("seed") if args.get("seed") is not None else 42)
        wait_flag = bool(args.get("wait", True))

        if mid not in ("auto", "procedural-ambience-cpu") and not catalog_manager.is_downloaded(mid):
            return _guided_error(
                "AUDIO_MODEL_NOT_DOWNLOADED",
                f"El modelo `{mid}` aún no está descargado en disco.",
                f"Descárgalo primero con `studio_prepare_model(model_id='{mid}')` o usa `model_id='procedural-ambience-cpu'` (0 GB en 0.2s).",
            )

        job = job_manager.submit(
            kind="audio_bgm",
            title=f"[MCP] Música/Ambiente ({mid}): {(prm or prs)[:38]}",
            fn=lambda j: audio_studio_engine.generate_music_or_ambience(
                job=j,
                prompt=prm,
                model_id=mid,
                preset=prs,
                duration_seconds=dur_s,
                seed=sd,
            ),
        )
        if not wait_flag:
            return _ok_result({"ok": True, "job_id": job.id, "status": job.status})
        return _ok_result(_wait_for_job_bounded(job.id))

    # 16. Sincronización labial HD (1 personaje o diálogo por turnos multi-personaje)
    if name == "studio_lipsync_character":
        try:
            med_p = _resolve_rel_safe(args.get("media_rel"))
            if med_p is None:
                return _guided_error(
                    "MISSING_MEDIA_REL",
                    "Falta `media_rel` (la imagen o clip MP4 sobre el que animar los labios).",
                    "Pasa el `rel_path` devuelto por `studio_generate_video` o `studio_generate_or_edit_image`.",
                )
            v_single = _resolve_rel_safe(args.get("voice_audio_rel")) if args.get("voice_audio_rel") else None
            raw_turns = list(args.get("dialogue_turns") or [])
            turns_resolved: List[Dict[str, Any]] = []
            for idx_t, tr in enumerate(raw_turns, start=1):
                vp = _resolve_rel_safe(tr.get("voice_audio_rel"))
                if vp is None:
                    return _guided_error(
                        "MISSING_TURN_VOICE",
                        f"Falta `voice_audio_rel` en el turno #{idx_t} de `dialogue_turns`.",
                        "Genera antes la voz con `studio_generate_speech` y pasa su `rel_path`.",
                    )
                turns_resolved.append(
                    {
                        "voice_path": vp,
                        "target_face": str(tr.get("target_face") or args.get("target_face") or "largest"),
                        "start_sec": float(tr.get("start_sec") or 0.0),
                        "volume": float(tr.get("volume") if tr.get("volume") is not None else 1.0),
                    }
                )
            mus_p = _resolve_rel_safe(args.get("music_rel")) if args.get("music_rel") else None
            amb_p = _resolve_rel_safe(args.get("ambient_rel")) if args.get("ambient_rel") else None
        except (ValueError, FileNotFoundError) as exc:
            return _guided_error(
                "LIPSYNC_FILE_NOT_FOUND",
                str(exc),
                "Verifica los `rel_path` con `studio_list_assets`.",
            )

        if v_single is None and not turns_resolved:
            return _guided_error(
                "MISSING_VOICE_FOR_LIPSYNC",
                "No se indicó `voice_audio_rel` ni `dialogue_turns`.",
                "Pasa `voice_audio_rel` para 1 hablante o `dialogue_turns` para varios personajes.",
            )

        ls_mid = str(args.get("model_id") or "edtalk-256-lipsync-hd").strip()
        if not catalog_manager.is_downloaded(ls_mid):
            return _guided_error(
                "LIPSYNC_MODEL_NOT_DOWNLOADED",
                f"El paquete `{ls_mid}` (0.65 GB) aún no está descargado en Colab.",
                f"Ejecuta primero `studio_prepare_model(model_id='{ls_mid}')` (tarda ~4s en Colab) y repite la llamada.",
            )

        t_face = str(args.get("target_face") or "largest")
        l_weight = float(args.get("lip_weight") if args.get("lip_weight") is not None else 0.85)
        enh = bool(args.get("enhance_face", True))
        r_str = float(args.get("restore_strength") if args.get("restore_strength") is not None else 0.50)
        m_vol = float(args.get("music_volume") if args.get("music_volume") is not None else 0.22)
        a_vol = float(args.get("ambient_volume") if args.get("ambient_volume") is not None else 0.18)
        ext_vid = bool(args.get("extend_video_to_audio", True))
        wait_flag = bool(args.get("wait", True))

        job = job_manager.submit(
            kind="lipsync",
            title=f"[MCP] LipSync HD ({f'{len(turns_resolved)} turnos' if turns_resolved else t_face} · {med_p.name})",
            fn=lambda j: lipsync_engine.render_lipsync(
                job=j,
                media_path=med_p,
                voice_audio_path=v_single,
                target_face=t_face,
                dialogue_turns=turns_resolved if turns_resolved else None,
                model_id=ls_mid,
                lip_weight=l_weight,
                enhance_face=enh,
                restore_strength=r_str,
                music_path=mus_p,
                music_volume=m_vol,
                ambient_path=amb_p,
                ambient_volume=a_vol,
                extend_video_to_audio=ext_vid,
            ),
        )
        if not wait_flag:
            return _ok_result({"ok": True, "job_id": job.id, "status": job.status})
        return _ok_result(_wait_for_job_bounded(job.id))

    # 17. Mezclador multipista FFmpeg con Auto-Ducking sobre diálogo
    if name == "studio_mix_audio_tracks":
        try:
            vid_p = _resolve_rel_safe(args.get("video_rel")) if args.get("video_rel") else None
            mus_p = _resolve_rel_safe(args.get("music_rel")) if args.get("music_rel") else None
            amb_p = _resolve_rel_safe(args.get("ambient_rel")) if args.get("ambient_rel") else None
            raw_v = list(args.get("voice_tracks") or [])
            v_resolved: List[Dict[str, Any]] = []
            for idx_v, vt in enumerate(raw_v, start=1):
                vp = _resolve_rel_safe(vt.get("audio_rel"))
                if vp is None:
                    return _guided_error("MISSING_VOICE_TRACK", f"Falta `audio_rel` en `voice_tracks[{idx_v - 1}]`.", "Verifica con `studio_list_assets`.")
                v_resolved.append(
                    {
                        "path": vp,
                        "start_sec": float(vt.get("start_sec") or 0.0),
                        "volume": float(vt.get("volume") if vt.get("volume") is not None else 1.0),
                    }
                )
        except (ValueError, FileNotFoundError) as exc:
            return _guided_error("MIX_FILE_NOT_FOUND", str(exc), "Verifica los `rel_path` con `studio_list_assets`.")

        if vid_p is None and not v_resolved and mus_p is None and amb_p is None:
            return _guided_error(
                "EMPTY_MIX_REQUEST",
                "No indicaste ningún vídeo ni pista de voz, música o ambiente.",
                "Pasa `video_rel` y al menos `voice_tracks`, `music_rel` o `ambient_rel`.",
            )

        m_vol = float(args.get("music_volume") if args.get("music_volume") is not None else 0.24)
        a_vol = float(args.get("ambient_volume") if args.get("ambient_volume") is not None else 0.18)
        keep_orig = bool(args.get("keep_original_audio", False))
        orig_vol = float(args.get("original_audio_volume") if args.get("original_audio_volume") is not None else 1.0)
        duck = bool(args.get("auto_duck_bg", True))
        ext_vid = bool(args.get("extend_video_to_audio", True))
        fade_out = float(args.get("fade_out_seconds") if args.get("fade_out_seconds") is not None else 0.80)
        wait_flag = bool(args.get("wait", True))

        def _run_mix_mcp(j: Any) -> Dict[str, Any]:
            ts_slug = datetime.now().strftime("%Y%m%d_%H%M%S")
            if vid_p is not None:
                VIDEOS_OUT_DIR.mkdir(parents=True, exist_ok=True)
                out_name = f"master_{ts_slug}_{j.id[:6]}.mp4"
                out_path = VIDEOS_OUT_DIR / out_name
                rel_p = f"outputs/videos/{out_name}"
                m_type = "video"
                s_dir = VIDEOS_OUT_DIR
            else:
                AUDIO_OUT_DIR.mkdir(parents=True, exist_ok=True)
                out_name = f"mix_{ts_slug}_{j.id[:6]}.wav"
                out_path = AUDIO_OUT_DIR / out_name
                rel_p = f"outputs/audio/{out_name}"
                m_type = "audio"
                s_dir = AUDIO_OUT_DIR

            res = mix_and_master_media(
                output_path=out_path,
                video_path=vid_p,
                voice_tracks=v_resolved,
                music_path=mus_p,
                music_volume=m_vol,
                ambient_path=amb_p,
                ambient_volume=a_vol,
                keep_original_audio=keep_orig,
                original_audio_volume=orig_vol,
                auto_duck_bg=duck,
                extend_video_to_audio=ext_vid,
                fade_out_seconds=fade_out,
                job=j,
            )
            meta = {
                "id": out_path.stem,
                "filename": out_name,
                "rel_path": rel_p,
                "media_url": f"/media/{rel_p}",
                "type": m_type,
                **res,
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }
            (s_dir / f"{out_path.stem}.json").write_text(
                json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            return meta

        job = job_manager.submit(
            kind="audio_mix",
            title=f"[MCP] Mezcla Multipista + Auto-Ducking ({len(v_resolved)} voces)",
            fn=_run_mix_mcp,
        )
        if not wait_flag:
            return _ok_result({"ok": True, "job_id": job.id, "status": job.status})
        return _ok_result(_wait_for_job_bounded(job.id))

    # 18. Exportar documento completo de log de sesión (.md o .txt)
    if name == "studio_export_session_log":
        fmt = "txt" if str(args.get("format") or "md").lower() == "txt" else "md"
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
        log_filename, log_content = build_full_session_log_document(
            jobs_list=job_manager.list_jobs(limit=500, include_logs=True),
            loaded_summary=memory_manager.status_summary(),
            downloaded_models=downloaded_models,
            fmt=fmt,
        )
        logs_dir = OUTPUTS_DIR / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        out_fp = logs_dir / log_filename
        out_fp.write_text(log_content, encoding="utf-8")
        rel_log = f"outputs/logs/{log_filename}"
        colab_console_log(
            f"[MCP] Registro completo de sesión exportado: {rel_log} ({round(len(log_content.encode('utf-8')) / 1024, 1)} KB)",
            icon="📄",
        )
        return _ok_result(
            {
                "ok": True,
                "filename": log_filename,
                "rel_path": rel_log,
                "media_url": f"/media/{rel_log}",
                "size_kb": round(len(log_content.encode("utf-8")) / 1024, 1),
                "lines_count": log_content.count("\n"),
            }
        )

    # 19. Modo Higgsfield: listar cámaras, lentes, movimientos y Casting Vault
    if name == "studio_higgsfield_presets_and_cast":
        return _ok_result({"ok": True, **get_higgsfield_catalog_payload()})

    # 20. Modo Higgsfield: gestionar fichas del Casting Vault
    if name == "studio_higgsfield_save_character":
        act = str(args.get("action") or "save").strip().lower()
        if act == "delete":
            cid = str(args.get("char_id") or "").strip()
            if not cid:
                return _guided_error("MISSING_CHAR_ID", "Indica `char_id` para borrar.", "Consulta `studio_higgsfield_presets_and_cast`.")
            deleted = character_vault.delete_character(cid)
            return _ok_result({"ok": True, "deleted": deleted, "characters": character_vault.list_characters()})

        cname = str(args.get("name") or "").strip()
        if not cname:
            return _guided_error("MISSING_CHARACTER_NAME", "Falta `name` para guardar el personaje.", "Ejemplo: `name='Marco'`.")
        saved = character_vault.save_character(
            name=cname,
            appearance_prompt=str(args.get("appearance_prompt") or ""),
            anchor_image_rel=str(args.get("anchor_image_rel") or ""),
            voice_id=str(args.get("voice_id") or "es-ES-AlvaroNeural"),
            default_face_side=str(args.get("default_face_side") or "largest"),
            char_id=str(args.get("char_id") or "") or None,
        )
        return _ok_result({"ok": True, "character": saved, "characters": character_vault.list_characters()})

    # 21. Modo Higgsfield: Auto-Director, Cinema Shot, Higgsfield Motion y Smart Clipper 9:16
    if name == "studio_higgsfield_director":
        act = str(args.get("action") or "scene_director").strip().lower()
        wait_flag = bool(args.get("wait", True))

        if act == "smart_clip_9_16":
            try:
                vp = _resolve_rel_safe(args.get("video_rel"))
            except (ValueError, FileNotFoundError) as exc:
                return _guided_error("CLIP_VIDEO_NOT_FOUND", str(exc), "Verifica `video_rel` con `studio_list_assets`.")
            if vp is None:
                return _guided_error("MISSING_VIDEO_REL", "Falta `video_rel` para recortar a 9:16.", "Pasa el `rel_path` de un MP4.")
            t_asp = str(args.get("aspect_ratio") or "9:16")
            st_s = float(args.get("start_sec") or 0.0)
            en_s = float(args.get("end_sec") or 0.0)
            job = job_manager.submit(
                kind="video",
                title=f"[MCP] ✂️ Smart Clipper ({t_asp} · {vp.name})",
                fn=lambda j: smart_clip_and_reframe_video(vp, start_sec=st_s, end_sec=en_s, target_aspect=t_asp, job=j),
            )
            return _ok_result({"ok": True, "job_id": job.id, "status": job.status} if not wait_flag else _wait_for_job_bounded(job.id))

        if act == "cinema_shot":
            prm = str(args.get("prompt") or "").strip()
            if not prm:
                return _guided_error("MISSING_CINEMA_PROMPT", "Falta `prompt` para `cinema_shot`.", "Describe el plano deseado.")
            cids = list(args.get("character_ids") or [])
            chars = [c for cid in cids if (c := character_vault.get_character(cid)) is not None]
            compiled = compile_cinema_prompt(
                user_prompt=prm,
                camera_id=str(args.get("camera_id") or "arri_full_frame_8k"),
                lens_id=str(args.get("lens_id") or "classic_anamorphic"),
                focal_mm=str(args.get("focal_mm") or "35"),
                aperture=str(args.get("aperture") or "f/2.8"),
                lighting_id=str(args.get("lighting_id") or "natural_cinema"),
                characters=chars,
            )
            anchors: List[Path] = []
            if args.get("base_image_rel"):
                bp = _resolve_rel_safe(args.get("base_image_rel"))
                if bp:
                    anchors.append(bp)
            for ch in chars:
                ap = _resolve_rel_safe(ch.get("anchor_image_rel")) if ch.get("anchor_image_rel") else None
                if ap and ap not in anchors:
                    anchors.append(ap)
            asp = str(args.get("aspect_ratio") or "16:9")
            w, h = (704, 1280) if asp == "9:16" else ((1024, 1024) if asp == "1:1" else (1280, 704))
            sd = int(args.get("seed") if args.get("seed") is not None else 42)
            job = job_manager.submit(
                kind="image",
                title=f"[MCP] 🎬 Cinema Shot ({args.get('camera_id', 'arri')} · {args.get('focal_mm', '35')}mm)",
                fn=lambda j: generate_or_edit_image(
                    prompt=compiled,
                    model_id="qwen-image-edit-2511-pack",
                    base_image_path=anchors[0] if anchors else None,
                    reference_paths=anchors[:3] if len(anchors) > 1 else None,
                    width=w,
                    height=h,
                    steps=4,
                    guidance_scale=1.0,
                    seed=sd,
                    use_lightning_lora=True,
                    custom_lora_ids=list(args.get("image_lora_ids") or []),
                    job=j,
                ),
            )
            return _ok_result({"ok": True, "job_id": job.id, "status": job.status} if not wait_flag else _wait_for_job_bounded(job.id))

        if act == "motion_clip":
            img_p = _resolve_rel_safe(args.get("base_image_rel"))
            if img_p is None:
                return _guided_error("MISSING_BASE_IMAGE", "Indica `base_image_rel` para `motion_clip`.", "Pasa el `rel_path` de una imagen.")
            mot_id = str(args.get("motion_id") or "dolly_in_dramatic")
            m_spec = HIGGSFIELD_MOTIONS.get(mot_id) or HIGGSFIELD_MOTIONS["dolly_in_dramatic"]
            asp = str(args.get("aspect_ratio") or "16:9")
            vw, vh = (704, 1280) if asp == "9:16" else ((960, 960) if asp == "1:1" else (832, 480))
            dur_s = float(args.get("duration_seconds") or 4.5)
            eng_m = str(args.get("video_engine_mode") or "auto").lower()

            def _do_mcp_hf_mot(j: Any) -> Dict[str, Any]:
                eff = eng_m
                if eff == "auto":
                    eff = "ai_14b" if catalog_manager.is_downloaded("wan21-vace-14b-fp8") else (
                        "ai_fast" if catalog_manager.is_downloaded("wan21-vace-1.3b") else "ffmpeg_instant"
                    )
                if eff in ("ai_14b", "ai_fast"):
                    mid = "wan21-vace-14b-fp8" if eff == "ai_14b" else "wan21-vace-1.3b"
                    lid = "lora-wan21-14b-fusionx-phantom" if eff == "ai_14b" else "lora-wan21-1.3b-bidirect2"
                    if not catalog_manager.is_downloaded(lid):
                        lid = ""
                    return generate_video_clip(
                        mode="ai",
                        video_submode="i2v",
                        model_id=mid,
                        video_lora_id=lid or None,
                        extra_video_lora_ids=list(args.get("video_lora_ids") or []),
                        prompt=f"{args.get('prompt', '')}, {m_spec['video_prompt']}".strip(", "),
                        image_path=img_p,
                        width=vw,
                        height=vh,
                        duration_seconds=dur_s,
                        fps=16,
                        steps=6 if lid else 24,
                        guidance_scale=1.0 if lid else 5.0,
                        seed=int(args.get("seed") or 42),
                        job=j,
                    )
                VIDEOS_OUT_DIR.mkdir(parents=True, exist_ok=True)
                oname = f"hf_motion_{mot_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                op = VIDEOS_OUT_DIR / oname
                info = render_higgsfield_cpu_motion_clip(img_p, op, motion_id=mot_id, width=vw, height=vh, fps=24, duration_sec=dur_s, job=j)
                meta = {
                    "id": op.stem,
                    "filename": oname,
                    "rel_path": f"outputs/videos/{oname}",
                    "media_url": f"/media/outputs/videos/{oname}",
                    "type": "video",
                    "motion_id": mot_id,
                    "duration_seconds": info["duration_seconds"],
                    "width": vw,
                    "height": vh,
                }
                (VIDEOS_OUT_DIR / f"{op.stem}.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
                return meta

            job = job_manager.submit(kind="video", title=f"[MCP] 🌪️ Higgsfield Motion: {m_spec['label']}", fn=_do_mcp_hf_mot)
            return _ok_result({"ok": True, "job_id": job.id, "status": job.status} if not wait_flag else _wait_for_job_bounded(job.id))

        # Default: `scene_director`
        s_prompt = str(args.get("prompt") or "").strip()
        if not s_prompt:
            return _guided_error("MISSING_SCENE_PROMPT", "Falta `prompt` para `scene_director`.", "Describe la escena que quieres dirigir.")
        job = job_manager.submit(
            kind="lipsync",
            title=f"[MCP] 🎬 Auto-Director Higgsfield: {s_prompt[:36]}",
            fn=lambda j: run_higgsfield_scene_director(
                job=j,
                scene_prompt=s_prompt,
                resolve_rel_fn=lambda r: _resolve_rel_safe(r) if r else None,
                camera_id=str(args.get("camera_id") or "arri_full_frame_8k"),
                lens_id=str(args.get("lens_id") or "classic_anamorphic"),
                focal_mm=str(args.get("focal_mm") or "35"),
                aperture=str(args.get("aperture") or "f/2.8"),
                lighting_id=str(args.get("lighting_id") or "natural_cinema"),
                motion_id=str(args.get("motion_id") or "dolly_in_dramatic"),
                video_engine_mode=str(args.get("video_engine_mode") or "auto"),
                aspect_ratio=str(args.get("aspect_ratio") or "16:9"),
                duration_seconds=float(args.get("duration_seconds") or 4.5),
                base_image_rel=args.get("base_image_rel"),
                character_ids=list(args.get("character_ids") or []),
                dialogue_lines=list(args.get("dialogue_lines") or []),
                ambience_preset=str(args.get("ambience_preset") or "none"),
                apply_lipsync=bool(args.get("apply_lipsync", True)),
                image_lora_ids=list(args.get("image_lora_ids") or []),
                video_lora_ids=list(args.get("video_lora_ids") or []),
                seed=int(args.get("seed") if args.get("seed") is not None else 42),
            ),
        )
        return _ok_result({"ok": True, "job_id": job.id, "status": job.status} if not wait_flag else _wait_for_job_bounded(job.id))

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

### 4) Crear Secuencia de Fotogramas con PERSONAJES 100% CONSISTENTES + Auditar con `studio_analyze_coherence`
**¡REGLA CRÍTICA!** Nunca generes el Fotograma 2, 3, 4 desde cero si deben aparecer las mismas personas:
1. Genera el Fotograma 1 (Ancla):
```bash
python3 studio_mcp.py --cli studio_generate_or_edit_image '{{
  "model_id": "qwen-image-edit-2511-pack",
  "use_official_defaults": true,
  "prompt": "dos amigos comiendo pasta en una mesa de restaurante cálido, fotografía cinematográfica realista"
}}'
```
2. Genera el Fotograma 2 anclado al Fotograma 1 (`base_image_rel` + `reference_rels`):
```bash
python3 studio_mcp.py --cli studio_generate_or_edit_image '{{
  "model_id": "qwen-image-edit-2511-pack",
  "base_image_rel": "outputs/images/FOTOGRAMA_1.png",
  "reference_rels": ["outputs/images/FOTOGRAMA_1.png"],
  "use_official_defaults": true,
  "prompt": "Mantén exactamente las mismas dos personas, sus rostros, peinados, ropa, mesa y restaurante de la imagen original; cambia únicamente su acción para que ahora choquen sus copas brindando y sonriendo"
}}'
```
3. Verifica la coherencia con `studio_analyze_coherence` (OpenCV + Qwen2.5-VL-3B 4-bit) antes de animar en vídeo:
```bash
python3 studio_mcp.py --cli studio_analyze_coherence '{{
  "anchor_image_rel": "outputs/images/FOTOGRAMA_1.png",
  "candidate_image_rels": ["outputs/images/FOTOGRAMA_2.png"],
  "scene_goal": "mismas dos personas comiendo en el mismo restaurante",
  "mode": "hybrid_auto"
}}'
```
*(Si devuelve `"verdict": "NEEDS_REPAIR"`, ejecuta directamente el `"recommended_repair_tool_call"` devuelto).*

### 5) Generar Voces + Ambiente de Restaurante / Música + LipSync HD Multi-Personaje (`studio_lipsync_character`)
1. Descargar una sola vez el motor LipSync HD (`0.65 GB`, tarda ~4 s en Colab):
```bash
python3 studio_mcp.py --cli studio_prepare_model '{{"model_id": "edtalk-256-lipsync-hd"}}'
```
2. Generar las voces de cada personaje (`studio_generate_speech`) y el sonido de fondo (`studio_generate_music_or_sfx`):
```bash
python3 studio_mcp.py --cli studio_generate_speech '{{"voice_id": "es-ES-AlvaroNeural", "text": "Esta pasta está increíble, brindemos por esta noche."}}'
python3 studio_mcp.py --cli studio_generate_speech '{{"voice_id": "es-ES-ElviraNeural", "text": "¡Salud! Ha sido una idea genial venir a este restaurante."}}'
python3 studio_mcp.py --cli studio_generate_music_or_sfx '{{"model_id": "procedural-ambience-cpu", "preset": "restaurant_warm_murmur", "duration_seconds": 12}}'
```
*(O descarga `"musicgen-small-bgm"` para componer jazz/piano original por IA).*
3. Aplicar **LipSync HD por turnos** (`leftmost` = persona izquierda, `rightmost` = persona derecha) + ambiente con **Auto-Ducking automático**:
```bash
python3 studio_mcp.py --cli studio_lipsync_character '{{
  "media_rel": "outputs/videos/clip_restaurante.mp4",
  "dialogue_turns": [
    {{"voice_audio_rel": "outputs/audio/VOZ_ALVARO.mp3", "target_face": "leftmost", "start_sec": 0.2}},
    {{"voice_audio_rel": "outputs/audio/VOZ_ELVIRA.mp3", "target_face": "rightmost", "start_sec": 3.4}}
  ],
  "ambient_rel": "outputs/audio/bgm_AMBIENTE.wav",
  "ambient_volume": 0.18
}}'
```

### 6) Mezclador Multipista FFmpeg Independiente (`studio_mix_audio_tracks`) y Montaje Final (`studio_render_montage`)
Si ya tienes un vídeo y quieres añadirle voces posicionadas al segundo + música + ambiente con Auto-Ducking (`sidechaincompress`) sin bajar el volumen de las voces ni congelar el vídeo:
```bash
python3 studio_mcp.py --cli studio_mix_audio_tracks '{{
  "video_rel": "outputs/videos/clip_1.mp4",
  "voice_tracks": [{{"audio_rel": "outputs/audio/voz_1.mp3", "start_sec": 0.3}}],
  "music_rel": "outputs/audio/musica.wav",
  "music_volume": 0.22,
  "ambient_rel": "outputs/audio/ambiente.wav",
  "ambient_volume": 0.16,
  "auto_duck_bg": true
}}'
```
Y para unir varios planos en un cortometraje completo (incluye Auto-Ducking sobre `global_audio_rel` y extensión fluida sin fotogramas congelados):
```bash
python3 studio_mcp.py --cli studio_render_montage '{{
  "title": "corto_final",
  "items": [
    {{"media_rel": "outputs/videos/lipsync_plano1.mp4"}},
    {{"media_rel": "outputs/videos/lipsync_plano2.mp4"}}
  ],
  "global_audio_rel": "outputs/audio/musica.wav",
  "global_audio_volume": 0.20,
  "fade_seconds": 0.2
}}'
```

### 7) Liberar VRAM, borrar modelo del disco o descargar el Log Completo de Sesión (`.md`)
```bash
python3 studio_mcp.py --cli studio_free_memory_or_disk '{{"action": "unload_vram", "target": "all"}}'
python3 studio_mcp.py --cli studio_export_session_log '{{"format": "md"}}'
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
