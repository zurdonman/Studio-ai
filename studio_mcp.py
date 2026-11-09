#!/usr/bin/env python3
"""
Puente MCP Local (stdio) ↔ Studio IA en Google Colab.

Características clave:
- CERO dependencias externas (usa únicamente la librería estándar de Python 3.9+).
- Evita cortes por timeout de Cloudflare (100 s): lanza trabajos pesados en modo asíncrono
  y consulta `/api/jobs/{job_id}` cada 2 segundos hasta completar (incluso si descarga 30 GB).
- Subida transparente de archivos locales de tu PC (`local_image_path`, `local_end_image_path`,
  `local_mask_path`, `local_reference_paths`, `local_audio_path`, `local_file_path`) hacia Colab.
- Descarga automática a tu PC (`save_to_local_path` o `--out-dir`) de imágenes, vídeos, audios y ZIPs
  sin meter megabytes binarios en la ventana de contexto del agente.

Uso en Claude Code / Codex / Gemini CLI / Cursor:
  python3 studio_mcp.py --url https://xxxx.trycloudflare.com --token st_mcp_xxxx --out-dir ./salidas_studio
"""

from __future__ import annotations

import argparse
import copy
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Cuando se descarga desde `/mcp/bridge.py?token=...`, el servidor rellena automáticamente estos valores
# para que en Termux / Android / cualquier harness baste con ejecutar `python3 studio_mcp.py ...` sin repetir URL ni token.
EMBEDDED_DEFAULT_URL = ""
EMBEDDED_DEFAULT_TOKEN = ""
EMBEDDED_DEFAULT_OUT_DIR = "./salidas_studio"


BRIDGE_ONLY_LOCAL_PROPS: Dict[str, Dict[str, Any]] = {
    "studio_upload_file_base64": {
        "local_file_path": {
            "type": "string",
            "description": (
                "Ruta local en el PC del usuario (ej. './foto.png' o '/ruta/audio.mp3'). "
                "El puente local sube el archivo directamente a Colab sin usar base64."
            ),
        },
    },
    "studio_generate_or_edit_image": {
        "local_base_image_path": {
            "type": "string",
            "description": "Opcional: ruta de una imagen en tu PC para editarla en Colab (se sube sola).",
        },
        "local_reference_paths": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Opcional: lista de rutas de imágenes de referencia en tu PC (se suben solas).",
        },
        "local_mask_path": {
            "type": "string",
            "description": "Opcional: ruta local de máscara PNG (blanco=editar, negro=conservar).",
        },
        "save_to_local_path": {
            "type": "string",
            "description": "Opcional: ruta local en tu PC donde guardar el PNG resultante (ej. './salida.png').",
        },
    },
    "studio_generate_video": {
        "local_image_path": {
            "type": "string",
            "description": "Opcional: ruta en tu PC de la Imagen Inicial (para 'i2v' o 'flf2v'). Se sube automáticamente.",
        },
        "local_end_image_path": {
            "type": "string",
            "description": "Opcional: ruta en tu PC del Fotograma Final (para 'flf2v' Frame a Frame). Se sube automáticamente.",
        },
        "local_audio_path": {
            "type": "string",
            "description": "Opcional: ruta en tu PC de un audio (.wav/.mp3) para ajustar duración y sonorizar el vídeo.",
        },
        "save_to_local_path": {
            "type": "string",
            "description": "Opcional: ruta local en tu PC donde guardar el vídeo MP4 terminado (ej. './clip.mp4').",
        },
    },
    "studio_generate_speech": {
        "save_to_local_path": {
            "type": "string",
            "description": "Opcional: ruta local en tu PC donde guardar la locución generada (ej. './voz.mp3').",
        },
    },
    "studio_render_montage": {
        "save_to_local_path": {
            "type": "string",
            "description": "Opcional: ruta local en tu PC donde guardar el montaje MP4 final (ej. './montaje.mp4').",
        },
    },
    "studio_analyze_coherence": {
        "local_anchor_image_path": {
            "type": "string",
            "description": "Opcional: ruta local en tu PC/Termux del Fotograma Ancla #1 (se sube automáticamente a Colab).",
        },
        "local_candidate_image_paths": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Opcional: lista de rutas locales en tu PC/Termux de los fotogramas 2..N a auditar (se suben solos).",
        },
    },
    "studio_export_session_log": {
        "save_to_local_path": {
            "type": "string",
            "description": "Opcional: ruta local en tu PC/Termux donde guardar el archivo .md o .txt del log completo.",
        },
    },
    "studio_generate_music_or_sfx": {
        "save_to_local_path": {
            "type": "string",
            "description": "Opcional: ruta local en tu PC/Termux donde guardar el audio musical o ambiental (.wav).",
        },
    },
    "studio_lipsync_character": {
        "local_media_path": {
            "type": "string",
            "description": "Opcional: ruta local de la imagen o vídeo MP4 base en tu PC/Termux (se sube automáticamente).",
        },
        "local_voice_audio_path": {
            "type": "string",
            "description": "Opcional: ruta local del audio de voz (.mp3/.wav) en tu PC/Termux.",
        },
        "local_music_path": {
            "type": "string",
            "description": "Opcional: ruta local de la música de fondo en tu PC/Termux.",
        },
        "local_ambient_path": {
            "type": "string",
            "description": "Opcional: ruta local del sonido ambiente en tu PC/Termux.",
        },
        "save_to_local_path": {
            "type": "string",
            "description": "Opcional: ruta local en tu PC/Termux donde guardar el MP4 final con LipSync HD.",
        },
    },
    "studio_mix_audio_tracks": {
        "local_video_path": {
            "type": "string",
            "description": "Opcional: ruta local del vídeo MP4 en tu PC/Termux.",
        },
        "local_music_path": {
            "type": "string",
            "description": "Opcional: ruta local de la música en tu PC/Termux.",
        },
        "local_ambient_path": {
            "type": "string",
            "description": "Opcional: ruta local del sonido ambiente en tu PC/Termux.",
        },
        "save_to_local_path": {
            "type": "string",
            "description": "Opcional: ruta local en tu PC/Termux donde guardar el archivo masterizado.",
        },
    },
    "studio_higgsfield_director": {
        "local_base_image_path": {
            "type": "string",
            "description": "Opcional: ruta local de imagen base en tu PC/Termux.",
        },
        "local_video_path": {
            "type": "string",
            "description": "Opcional: ruta local de vídeo MP4 para `smart_clip_9_16`.",
        },
        "save_to_local_path": {
            "type": "string",
            "description": "Opcional: ruta local donde guardar la imagen o vídeo Higgsfield resultante.",
        },
    },
}

DOWNLOAD_ASSET_TOOL_DEF: Dict[str, Any] = {
    "name": "studio_download_asset_to_pc",
    "description": (
        "Descarga cualquier archivo generado en Google Colab (`rel_path` como 'outputs/videos/vid_...mp4', "
        "'outputs/images/img_...png' o `export_all_zip=true`) directamente al disco local de tu PC."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "rel_path": {
                "type": "string",
                "description": "Ruta relativa en Studio IA (ej. 'outputs/videos/vid_2026...mp4'). Ignorado si export_all_zip=true.",
            },
            "export_all_zip": {
                "type": "boolean",
                "default": False,
                "description": "Si es true, descarga el paquete completo `.zip` con todas las imágenes, vídeos, audios y montajes.",
            },
            "save_to_local_path": {
                "type": "string",
                "description": "Ruta de archivo o carpeta local en tu PC donde guardar la descarga (ej. './resultados/video.mp4').",
            },
        },
        "required": ["save_to_local_path"],
    },
}

ASYNC_JOB_TOOLS = {
    "studio_prepare_model",
    "studio_generate_or_edit_image",
    "studio_generate_video",
    "studio_generate_speech",
    "studio_render_montage",
    "studio_analyze_coherence",
    "studio_generate_music_or_sfx",
    "studio_lipsync_character",
    "studio_mix_audio_tracks",
    "studio_higgsfield_director",
}


class BridgeTransportError(RuntimeError):
    def __init__(self, message: str, *, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retryable = status_code is None or status_code in (408, 425, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524, 525, 526, 530)


class StudioColabBridge:
    def __init__(self, base_url: str, token: str = "", out_dir: Optional[str] = None) -> None:
        cleaned = base_url.strip().rstrip("/")
        if cleaned.endswith("/mcp"):
            cleaned = cleaned[: -len("/mcp")]
        self.base_url = cleaned
        self.token = token.strip()
        self.out_dir = Path(out_dir).expanduser().resolve() if out_dir else None
        if self.out_dir:
            self.out_dir.mkdir(parents=True, exist_ok=True)
        self._framing_mode = "line"  # "line" (estándar MCP) o "lsp"

    def _headers(self, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        hdrs: Dict[str, str] = {
            "User-Agent": "StudioIA-MCP-Bridge/1.0",
            "Accept": "application/json",
        }
        if self.token:
            hdrs["Authorization"] = f"Bearer {self.token}"
            hdrs["X-Studio-Token"] = self.token
        if extra:
            hdrs.update(extra)
        return hdrs

    def _http_json(
        self,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
        timeout: float = 65.0,
    ) -> Any:
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        hdrs = self._headers({"Content-Type": "application/json"} if data is not None else None)
        req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw) if raw.strip() else {}
        except urllib.error.HTTPError as he:
            body = ""
            try:
                body = he.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            detail = (body or str(he.reason))[:2000].replace(self.token, "[REDACTED]") if self.token else (body or str(he.reason))[:2000]
            raise BridgeTransportError(f"HTTP {he.code} desde Studio IA ({path}): {detail}", status_code=he.code) from he
        except (urllib.error.URLError, TimeoutError, ConnectionError) as ue:
            raise BridgeTransportError(
                f"No se pudo conectar con Google Colab en {self.base_url} ({getattr(ue, 'reason', type(ue).__name__)}). "
                "Verifica que la celda de Colab siga en ejecución y la URL de cloudflared sea la actual."
            ) from ue

    def upload_local_file(self, local_path_str: str) -> Dict[str, Any]:
        p = Path(local_path_str).expanduser().resolve()
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(f"No existe el archivo local en tu PC: {p}")

        boundary = f"----StudioIAMCPBoundary{uuid.uuid4().hex}"
        mime, _ = mimetypes.guess_type(p.name)
        mime = mime or "application/octet-stream"
        file_bytes = p.read_bytes()

        head = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{p.name}"\r\n'
            f"Content-Type: {mime}\r\n\r\n"
        ).encode("utf-8")
        tail = f"\r\n--{boundary}--\r\n".encode("utf-8")
        body = head + file_bytes + tail

        req = urllib.request.Request(
            f"{self.base_url}/api/assets/upload",
            data=body,
            headers=self._headers(
                {
                    "Content-Type": f"multipart/form-data; boundary={boundary}",
                    "X-MCP-Client": "studio-mcp-local-bridge",
                }
            ),
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120.0) as resp:
            out = json.loads(resp.read().decode("utf-8"))
        return out["asset"]

    def download_to_local(
        self,
        rel_path: Optional[str] = None,
        save_to_local_path: Optional[str] = None,
        export_all_zip: bool = False,
    ) -> str:
        if export_all_zip:
            remote_path = "/api/export/zip"
            default_name = f"studio_ia_resultados_{int(time.time())}.zip"
        else:
            if not rel_path:
                raise ValueError("Se requiere `rel_path` para descargar un archivo de Colab.")
            cleaned = rel_path.lstrip("/")
            if cleaned.startswith("media/"):
                cleaned = cleaned[len("media/") :]
            quoted = "/".join(urllib.parse.quote(seg) for seg in cleaned.split("/"))
            remote_path = f"/media/{quoted}"
            default_name = Path(cleaned).name

        if save_to_local_path:
            dest = Path(save_to_local_path).expanduser().resolve()
            if dest.exists() and dest.is_dir():
                dest = dest / default_name
            elif str(save_to_local_path).endswith(("/", "\\")):
                dest.mkdir(parents=True, exist_ok=True)
                dest = dest / default_name
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
        elif self.out_dir:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            dest = self.out_dir / default_name
        else:
            dest = Path.cwd().resolve() / default_name

        req = urllib.request.Request(
            f"{self.base_url}{remote_path}",
            headers=self._headers(),
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=300.0) as resp, open(dest, "wb") as fh:
            while True:
                chunk = resp.read(256 * 1024)
                if not chunk:
                    break
                fh.write(chunk)
        return str(dest)

    def send_stdout_message(self, obj: Dict[str, Any]) -> None:
        if getattr(self, "_cli_mode", False):
            if obj.get("method") == "notifications/progress":
                p = obj.get("params") or {}
                sys.stderr.write(f"[Progreso Colab {float(p.get('progress', 0)):.1f}%] {p.get('message', '')}\n")
                sys.stderr.flush()
            return
        raw = json.dumps(obj, ensure_ascii=False)
        if self._framing_mode == "lsp":
            b = raw.encode("utf-8")
            sys.stdout.buffer.write(f"Content-Length: {len(b)}\r\n\r\n".encode("ascii") + b)
            sys.stdout.buffer.flush()
        else:
            sys.stdout.write(raw + "\n")
            sys.stdout.flush()

    def _poll_job_until_done(
        self,
        job_id: str,
        progress_token: Optional[Any] = None,
    ) -> Dict[str, Any]:
        last_pct = -1.0
        transport_failures = 0
        while True:
            try:
                job = self._http_json("GET", f"/api/jobs/{urllib.parse.quote(job_id)}", timeout=30.0)
                transport_failures = 0
            except BridgeTransportError as exc:
                if not exc.retryable or transport_failures >= 3:
                    raise
                # Only retry this read, NEVER replay the generation POST.
                transport_failures += 1
                time.sleep(2 ** transport_failures)
                continue
            status = job.get("status")
            pct = float(job.get("progress") or 0.0)
            step_txt = str(job.get("current_step") or "")
            spd_txt = str(job.get("speed_text") or "")

            if progress_token is not None and abs(pct - last_pct) >= 1.0:
                last_pct = pct
                msg_note = f"{step_txt} ({spd_txt})" if spd_txt else step_txt
                self.send_stdout_message(
                    {
                        "jsonrpc": "2.0",
                        "method": "notifications/progress",
                        "params": {
                            "progressToken": progress_token,
                            "progress": round(pct, 1),
                            "total": 100.0,
                            "message": msg_note,
                        },
                    }
                )

            if status in ("completed", "failed", "cancelled"):
                return job
            time.sleep(2.0)

    def _augment_tools_list(self, remote_resp: Dict[str, Any]) -> Dict[str, Any]:
        tools = copy.deepcopy((remote_resp.get("result") or {}).get("tools") or [])
        for t in tools:
            tname = t.get("name", "")
            extra_props = BRIDGE_ONLY_LOCAL_PROPS.get(tname)
            if extra_props:
                schema = t.setdefault("inputSchema", {"type": "object", "properties": {}})
                props = schema.setdefault("properties", {})
                props.update(extra_props)
                if tname == "studio_upload_file_base64" and "required" in schema:
                    schema["required"] = [r for r in schema["required"] if r not in ("filename", "data_base64")]
        if not any(t.get("name") == DOWNLOAD_ASSET_TOOL_DEF["name"] for t in tools):
            tools.append(copy.deepcopy(DOWNLOAD_ASSET_TOOL_DEF))
        remote_resp["result"]["tools"] = tools
        return remote_resp

    def _handle_tool_call(self, req_msg: Dict[str, Any]) -> Dict[str, Any]:
        req_id = req_msg.get("id")
        params = dict(req_msg.get("params") or {})
        tool_name = str(params.get("name") or "")
        args = dict(params.get("arguments") or {})
        meta = params.get("_meta") or {}
        progress_token = meta.get("progressToken")
        pending_job_id: Optional[str] = None
        submission_started = False

        try:
            if tool_name == DOWNLOAD_ASSET_TOOL_DEF["name"]:
                saved_path = self.download_to_local(
                    rel_path=args.get("rel_path"),
                    save_to_local_path=args.get("save_to_local_path"),
                    export_all_zip=bool(args.get("export_all_zip", False)),
                )
                payload = {
                    "ok": True,
                    "saved_local_path": saved_path,
                    "size_kb": round(Path(saved_path).stat().st_size / 1024, 1),
                }
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": json.dumps(payload, indent=2, ensure_ascii=False)}],
                        "isError": False,
                    },
                }

            # 1) Subida directa sin base64 si el agente usa `studio_upload_file_base64` con `local_file_path`
            if tool_name == "studio_upload_file_base64" and args.get("local_file_path"):
                asset = self.upload_local_file(str(args["local_file_path"]))
                payload = {
                    "ok": True,
                    "uploaded_from_local": str(args["local_file_path"]),
                    "asset": asset,
                    "next_step_hint": f"Usa rel_path='{asset['rel_path']}' en las herramientas de imagen, vídeo o montaje.",
                }
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": json.dumps(payload, indent=2, ensure_ascii=False)}],
                        "isError": False,
                    },
                }

            # 2) Subir automáticamente rutas locales de imagen/máscara/audio en herramientas generativas
            upload_single_map = [
                ("local_base_image_path", "base_image_rel"),
                ("local_mask_path", "mask_image_rel"),
                ("local_image_path", "image_rel"),
                ("local_end_image_path", "end_image_rel"),
                ("local_audio_path", "audio_rel"),
                ("local_anchor_image_path", "anchor_image_rel"),
                ("local_media_path", "media_rel"),
                ("local_voice_audio_path", "voice_audio_rel"),
                ("local_video_path", "video_rel"),
                ("local_music_path", "music_rel"),
                ("local_ambient_path", "ambient_rel"),
            ]
            for local_key, remote_key in upload_single_map:
                val = args.pop(local_key, None)
                if val:
                    up = self.upload_local_file(str(val))
                    args[remote_key] = up["rel_path"]

            local_refs = args.pop("local_reference_paths", None)
            if isinstance(local_refs, list) and local_refs:
                cur_refs = list(args.get("reference_rels") or [])
                for ref_p in local_refs:
                    up = self.upload_local_file(str(ref_p))
                    cur_refs.append(up["rel_path"])
                args["reference_rels"] = cur_refs

            local_cands = args.pop("local_candidate_image_paths", None)
            if isinstance(local_cands, list) and local_cands:
                cur_cands = list(args.get("candidate_image_rels") or [])
                for cand_p in local_cands:
                    up = self.upload_local_file(str(cand_p))
                    cur_cands.append(up["rel_path"])
                args["candidate_image_rels"] = cur_cands

            save_local = args.pop("save_to_local_path", None)

            # 3) Para tareas largas, lanzar en modo no bloqueante (`wait=False`) y hacer polling corto cada 2s
            caller_wanted_wait = bool(args.get("wait", True))
            if tool_name in ASYNC_JOB_TOOLS and caller_wanted_wait:
                args["wait"] = False

            forward_msg = {
                "jsonrpc": "2.0",
                "id": req_id,
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": args,
                },
            }
            submission_started = tool_name in ASYNC_JOB_TOOLS
            remote_resp = self._http_json("POST", "/mcp", payload=forward_msg, timeout=65.0)
            if "error" in remote_resp:
                return remote_resp

            res_obj = remote_resp.get("result") or {}
            if res_obj.get("isError"):
                return remote_resp

            text_blocks = res_obj.get("content") or []
            parsed_data: Optional[Dict[str, Any]] = None
            if text_blocks and text_blocks[0].get("type") == "text":
                try:
                    parsed_data = json.loads(text_blocks[0].get("text") or "{}")
                except Exception:
                    parsed_data = None

            if (
                tool_name in ASYNC_JOB_TOOLS
                and caller_wanted_wait
                and isinstance(parsed_data, dict)
                and parsed_data.get("job_id")
                and parsed_data.get("status") in ("queued", "running")
            ):
                pending_job_id = str(parsed_data["job_id"])
                final_job = self._poll_job_until_done(pending_job_id, progress_token=progress_token)
                if final_job.get("status") != "completed":
                    err_payload = {
                        "ok": False,
                        "job_id": final_job.get("id"),
                        "status": final_job.get("status"),
                        "error": final_job.get("error") or "El trabajo no se completó.",
                        "recent_logs": (final_job.get("logs") or [])[-8:],
                    }
                    return {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {
                            "content": [{"type": "text", "text": json.dumps(err_payload, indent=2, ensure_ascii=False)}],
                            "isError": True,
                        },
                    }
                job_result = dict(final_job.get("result") or {})
                rel_out = job_result.get("rel_path")
                if not rel_out and job_result.get("media_url", "").startswith("/media/"):
                    rel_out = job_result["media_url"][len("/media/") :]
                    job_result["rel_path"] = rel_out

                if rel_out and (save_local or self.out_dir):
                    local_saved = self.download_to_local(
                        rel_path=str(rel_out),
                        save_to_local_path=str(save_local) if save_local else None,
                    )
                    job_result["saved_local_path"] = local_saved

                done_payload = {
                    "ok": True,
                    "job_id": final_job.get("id"),
                    "status": "completed",
                    "elapsed_seconds": final_job.get("elapsed_seconds"),
                    "result": job_result,
                }
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": json.dumps(done_payload, indent=2, ensure_ascii=False)}],
                        "isError": False,
                    },
                }

            if isinstance(parsed_data, dict) and parsed_data.get("rel_path") and (save_local or self.out_dir):
                local_saved = self.download_to_local(
                    rel_path=str(parsed_data["rel_path"]),
                    save_to_local_path=str(save_local) if save_local else None,
                )
                parsed_data["saved_local_path"] = local_saved
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": json.dumps(parsed_data, indent=2, ensure_ascii=False)}],
                        "isError": False,
                    },
                }

            return remote_resp

        except Exception as exc:
            err_data = {
                "ok": False,
                "error_code": "BRIDGE_EXECUTION_ERROR",
                "message": str(exc),
                "exact_fix_action": (
                    "Verifica que la ruta local del archivo exista en tu PC y que la sesión de Google Colab siga activa."
                ),
            }
            if pending_job_id or (submission_started and isinstance(exc, BridgeTransportError)):
                err_data.update({
                    "error_code": "JOB_CONNECTION_LOST" if pending_job_id else "SUBMISSION_OUTCOME_UNKNOWN",
                    "job_id": pending_job_id,
                    "status": "unknown_not_failed",
                    "do_not_resubmit": True,
                    "retryable": bool(getattr(exc, "retryable", False)),
                    "exact_fix_action": (
                        "No repetir la generación ni descargar otra receta. Revisar la celda y diagnóstico de Colab. "
                        "502/530 solo describen la conexión, no prueban OOM ni pérdida de VM. "
                        + (f"Al recuperar el enlace, consultar studio_get_job(job_id='{pending_job_id}')." if pending_job_id else
                           "No se recibió job_id; consultar studio_get_state_and_manual para identificar trabajos antes de decidir un nuevo envío.")
                    ),
                })
                if pending_job_id:
                    err_data["resume_tool_call"] = {"tool": "studio_get_job", "arguments": {"job_id": pending_job_id}}
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(err_data, indent=2, ensure_ascii=False)}],
                    "isError": True,
                },
            }

    def handle_jsonrpc_message(self, msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        method = str(msg.get("method") or "")
        msg_id = msg.get("id")

        # Notificaciones sin `id` (ej. `notifications/initialized`, `notifications/cancelled`)
        if msg_id is None:
            try:
                self._http_json("POST", "/mcp", payload=msg, timeout=15.0)
            except Exception:
                pass
            return None

        if method == "ping":
            return {"jsonrpc": "2.0", "id": msg_id, "result": {}}

        if method == "tools/call":
            return self._handle_tool_call(msg)

        try:
            remote_resp = self._http_json("POST", "/mcp", payload=msg, timeout=45.0)
            if method == "tools/list" and "result" in remote_resp:
                return self._augment_tools_list(remote_resp)
            return remote_resp
        except Exception as exc:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {
                    "code": -32000,
                    "message": str(exc),
                },
            }

    def _read_next_stdin_message(self) -> Tuple[Optional[Dict[str, Any]], bool]:
        """Lee mensajes MCP tanto en formato estándar línea-JSON como en formato Content-Length."""
        while True:
            line = sys.stdin.buffer.readline()
            if not line:
                return None, True
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.lower().startswith(b"content-length:"):
                self._framing_mode = "lsp"
                length = int(stripped.split(b":", 1)[1].strip())
                while True:
                    hdr_line = sys.stdin.buffer.readline()
                    if not hdr_line or hdr_line in (b"\r\n", b"\n"):
                        break
                body = sys.stdin.buffer.read(length)
                return json.loads(body.decode("utf-8")), False
            self._framing_mode = "line"
            return json.loads(stripped.decode("utf-8")), False

    def serve_stdio(self) -> None:
        while True:
            try:
                msg, eof = self._read_next_stdin_message()
            except Exception as exc:
                sys.stderr.write(f"[studio_mcp] Error leyendo JSON-RPC en stdin: {exc}\n")
                sys.stderr.flush()
                continue
            if eof or msg is None:
                break
            resp = self.handle_jsonrpc_message(msg)
            if resp is not None:
                self.send_stdout_message(resp)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Puente MCP Local (stdio) y CLI directo (Termux / DeepSeek / Qwen) para Studio IA en Google Colab"
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("STUDIO_IA_URL") or EMBEDDED_DEFAULT_URL or "http://127.0.0.1:7860",
        help="URL pública de Studio IA en Colab (ej. https://xxxx.trycloudflare.com)",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("STUDIO_MCP_TOKEN") or EMBEDDED_DEFAULT_TOKEN or "",
        help="Token de seguridad MCP mostrado al arrancar Colab",
    )
    parser.add_argument(
        "--out-dir",
        default=os.environ.get("STUDIO_OUT_DIR") or EMBEDDED_DEFAULT_OUT_DIR or "",
        help="Carpeta local donde descargar automáticamente cada imagen/vídeo/audio generado",
    )
    parser.add_argument(
        "--list-tools",
        action="store_true",
        help="Modo CLI (Termux/Bash): imprime todas las herramientas disponibles y sus esquemas cerrados",
    )
    parser.add_argument(
        "--cli",
        metavar="TOOL_NAME",
        default="",
        help="Modo CLI directo para harnesses en Termux/Bash: ejecuta una herramienta (ej. --cli studio_get_state_and_manual)",
    )
    parser.add_argument(
        "cli_json",
        nargs="?",
        default="{}",
        help="JSON de argumentos cuando se usa --cli (ej. '{\"task\":\"video_flf2v\"}')",
    )
    args = parser.parse_args()
    bridge = StudioColabBridge(
        base_url=args.url,
        token=args.token,
        out_dir=args.out_dir or None,
    )

    if args.list_tools:
        bridge._cli_mode = True
        resp = bridge.handle_jsonrpc_message({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        print(json.dumps(resp.get("result", resp), indent=2, ensure_ascii=False))
        return

    if args.cli:
        bridge._cli_mode = True
        try:
            parsed_args = json.loads(args.cli_json) if args.cli_json.strip() else {}
        except Exception as exc:
            sys.stderr.write(f"JSON inválido en cli_json: {exc}\n")
            sys.exit(2)
        resp = bridge.handle_jsonrpc_message(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": args.cli,
                    "arguments": parsed_args,
                    "_meta": {"progressToken": "cli-termux"},
                },
            }
        )
        if not resp or "error" in resp:
            print(json.dumps(resp, indent=2, ensure_ascii=False))
            sys.exit(1)
        res = resp.get("result") or {}
        texts = [c.get("text", "") for c in (res.get("content") or []) if c.get("type") == "text"]
        print("\n".join(texts) if texts else json.dumps(res, indent=2, ensure_ascii=False))
        if res.get("isError"):
            sys.exit(1)
        return

    bridge.serve_stdio()


if __name__ == "__main__":
    main()
