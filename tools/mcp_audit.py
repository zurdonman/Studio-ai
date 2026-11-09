#!/usr/bin/env python3
"""Auditoría de una instancia MCP de Studio IA, sin dependencias externas.

Uso (en la máquina que sí alcanza el túnel: Termux, tu PC, Colab):

    python3 mcp_audit.py --url https://xxxx.trycloudflare.com --token st_mcp_...
    # o, sin pegar el token en el historial:
    STUDIO_MCP_TOKEN=st_mcp_... python3 mcp_audit.py --url https://xxxx.trycloudflare.com

    # autenticidad OBLIGATORIA (versión desplegada y commit real):
    python3 mcp_audit.py ... --expect-commit 4a68ec1

Qué comprueba (solo lectura: no genera vídeo, no descarga, no borra nada):
  1. Handshake JSON-RPC: initialize / tools/list (contrato del servidor).
  2. Identidad del despliegue: ¿está el commit que crees, o una versión anterior?
  3. Preferencias y preguntas pendientes (/api/preferences).
  4. Estado real de hardware, memoria y compatibilidad del proceso.
  5. Plan del caso de aceptación de 30 s: 6 clips / 7 keyframes / entorno correcto.
  6. Guardas de producción: DURATION_EXCEEDS_SINGLE_CLIP y SPEED_LORA_MISSING
     (se comprueban ANTES de llegar al motor, así que no consumen GPU).
  7. Instrucciones del agente: ¿el servidor publica las reglas nuevas?

El token nunca se imprime ni se escribe: solo se envía en cabeceras.
Salida: informe markdown por stdout (y con --out FILE).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import ssl
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

REDACTED = "***"

ACCEPTANCE_SCENE = (
    "Haz una animación de dos personas andando por un parque hablando de algo polémico "
    "en España y haciendo una parodia, el vídeo debe durar 30 segundos"
)
REQUIRED_INSTRUCTIONS = ("DURATION_EXCEEDS_SINGLE_CLIP", "SPEED_LORA_MISSING", "use_fast_loras")
FORBIDDEN_IN_OLD_BUILD = ("objeto de tu elección",)


class Audit:
    def __init__(self, base_url: str, token: str, timeout: float = 30.0) -> None:
        self.base = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self.checks: List[Tuple[str, bool, str]] = []
        self.session_id: Optional[str] = None
        self.probe_images: List[str] = []
        self._rpc_id = 0

    # ---------------------------------------------------------------- helpers
    def _url(self, path: str) -> str:
        return f"{self.base}{path}"

    def _open(self, req: urllib.request.Request) -> Tuple[int, Dict[str, str], bytes]:
        ctx = ssl.create_default_context()
        with urllib.request.urlopen(req, timeout=self.timeout, context=ctx) as resp:
            return resp.status, dict(resp.headers), resp.read()

    def http_get(self, path: str) -> Tuple[int, Any]:
        req = urllib.request.Request(self._url(path), method="GET")
        req.add_header("Authorization", f"Bearer {self.token}")
        req.add_header("User-Agent", "studio-ia-audit/1.0")
        try:
            status, _headers, body = self._open(req)
        except urllib.error.HTTPError as exc:
            return exc.code, _safe_json(exc.read())
        except Exception as exc:  # red, TLS, DNS…
            return 0, {"transport_error": type(exc).__name__}
        return status, _safe_json(body)

    def http_text(self, path: str) -> Tuple[int, str]:
        req = urllib.request.Request(self._url(path), method="GET")
        req.add_header("Authorization", f"Bearer {self.token}")
        req.add_header("User-Agent", "studio-ia-audit/1.0")
        try:
            status, _headers, body = self._open(req)
        except urllib.error.HTTPError as exc:
            return exc.code, (exc.read() or b"").decode("utf-8", "replace")
        except Exception as exc:
            return 0, f"transport_error: {type(exc).__name__}"
        return status, (body or b"").decode("utf-8", "replace")

    def rpc(self, method: str, params: Optional[Dict[str, Any]] = None) -> Tuple[Any, Optional[Dict[str, Any]]]:
        self._rpc_id += 1
        payload = {"jsonrpc": "2.0", "id": self._rpc_id, "method": method}
        if params is not None:
            payload["params"] = params
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self._url("/mcp"), data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Accept", "application/json, text/event-stream")
        req.add_header("Authorization", f"Bearer {self.token}")
        req.add_header("User-Agent", "studio-ia-audit/1.0")
        if self.session_id:
            req.add_header("Mcp-Session-Id", self.session_id)
        try:
            status, headers, body = self._open(req)
        except urllib.error.HTTPError as exc:
            return None, {"status": exc.code, "body": _safe_json(exc.read())}
        except Exception as exc:
            return None, {"transport_error": type(exc).__name__}
        self.session_id = headers.get("Mcp-Session-Id") or headers.get("mcp-session-id") or self.session_id
        parsed = _safe_json(body)
        if isinstance(parsed, dict) and "error" in parsed:
            return None, {"status": status, "error": parsed["error"]}
        return (parsed or {}).get("result"), {"status": status}

    def tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        result, transport = self.rpc("tools/call", {"name": name, "arguments": arguments or {}})
        if result is None:
            return {"ok": False, "transport": transport}
        texts = [c.get("text", "") for c in (result.get("content") or []) if c.get("type") == "text"]
        raw = "\n".join(t for t in texts if t)
        payload = _safe_json(raw.encode("utf-8")) if raw else {}
        if not isinstance(payload, dict):
            payload = {"raw": raw}
        if result.get("isError"):
            payload.setdefault("ok", False)
            payload["isError"] = True
        return payload

    def check(self, name: str, ok: bool, detail: str) -> bool:
        self.checks.append((name, bool(ok), detail))
        return bool(ok)


def _safe_json(body: bytes) -> Any:
    text = (body or b"").decode("utf-8", "replace").strip()
    if not text:
        return {}
    # SSE: quedarse con la última línea `data:`
    if text.startswith("event:") or "\ndata:" in text or text.startswith("data:"):
        chunks = [ln[5:].strip() for ln in text.splitlines() if ln.startswith("data:")]
        text = chunks[-1] if chunks else text
    try:
        return json.loads(text)
    except Exception:
        return {"unparsed": text[:400]}


def _redact(obj: Any) -> Any:
    """Nunca dejar escapar un token, ni en los ecos del servidor."""
    if isinstance(obj, dict):
        return {
            k: (REDACTED if "token" in k.lower() and isinstance(v, str) and v else _redact(v))
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_redact(v) for v in obj]
    if isinstance(obj, str):
        return re.sub(r"st_mcp_[A-Za-z0-9]+", REDACTED, re.sub(r"token=[A-Za-z0-9_\-]+", f"token={REDACTED}", obj))
    return obj


# ------------------------------------------------------------------ checks
def run_audit(a: Audit, expect_commit: Optional[str]) -> List[Tuple[str, bool, str]]:
    version_hint = ""

    # 1) ¿Responde y exige token?
    status, body = a.http_get("/mcp")
    a.check("El túnel responde", status in (200, 401, 403, 405), f"GET /mcp → HTTP {status}")
    if status == 0:
        a.check("Diagnóstico", False, f"sin transporte: {body}")
        return a.checks

    # 2) initialize + tools/list
    init, tinfo = a.rpc(
        "initialize",
        {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "studio-ia-audit", "version": "1.0"},
        },
    )
    a.check("Handshake initialize", isinstance(init, dict) and "serverInfo" in (init or {}), json.dumps(_redact(init or tinfo))[:220])
    tools_result, _ = a.rpc("tools/list")
    tools = (tools_result or {}).get("tools") or []
    names = [t.get("name") for t in tools]
    a.check("tools/list devuelve herramientas", len(names) >= 25, f"{len(names)} herramientas")
    a.check(
        "Contrato de 30 herramientas (build nuevo)",
        len(names) == 30,
        f"{len(names)} herramientas — {'nuevo' if len(names) == 30 else 'build anterior'}; faltan: "
        f"{sorted(set(['studio_ask_user','studio_answer_preference','studio_get_work_state']) - set(names)) or 'nada'}",
    )

    # 3) Identidad del despliegue
    state = a.tool("studio_get_state_and_manual", {"include_recent_assets": False})
    blob = json.dumps(_redact(state), ensure_ascii=False)
    build = state.get("build") if isinstance(state.get("build"), dict) else {}
    commit = _find_commit(state) or (build.get("commit_short") or build.get("commit"))
    version_hint = commit or _find_version(state) or "desconocida"
    a.check(
        "Identidad del despliegue publicada",
        bool(commit),
        f"commit={commit or 'NO PUBLICADO (build anterior a 4a68ec1)'}"
        + (f" · rama={build.get('branch')} · sin commitear={build.get('dirty')}" if build else "")
        + (f" · esperado={expect_commit}" if expect_commit else ""),
    )
    if expect_commit:
        a.check(
            "El despliegue ES el commit esperado",
            bool(commit) and commit.startswith(expect_commit[:7]),
            f"desplegado={commit or '?'} esperado={expect_commit[:7]}",
        )
    gpu = (state.get("hardware") or {}).get("gpu") or {}
    a.check(
        "GPU real detectada",
        bool(gpu.get("has_gpu")),
        f"{gpu.get('gpu_name', 'CPU')} · {gpu.get('vram_total_gb', 0)} GB VRAM · libres {gpu.get('vram_free_gb', 0)} GB",
    )

    # 4) Preferencias y compatibilidad
    pstatus, prefs = a.http_get("/api/preferences")
    a.check("GET /api/preferences", pstatus == 200, f"HTTP {pstatus} · {json.dumps(_redact(prefs))[:200]}")
    rstatus, runtime = a.http_get("/api/runtime/status")
    rebuilt = (runtime or {}).get("runtime_rebuilt_for_compatibility") if isinstance(runtime, dict) else None
    a.check(
        "Estado de compatibilidad del proceso",
        rstatus == 200 and rebuilt is False,
        f"HTTP {rstatus} · necesita reinicio={rebuilt}",
    )
    pending = [q.get("key") for q in (prefs or {}).get("pending_questions", [])] if isinstance(prefs, dict) else []

    # 5) Caso de aceptación 30 s
    plan = a.tool(
        "studio_get_workflow_preset",
        {"preset_id": "coherencia_personajes_entorno_avance", "scene_description": ACCEPTANCE_SCENE},
    )
    summary = plan.get("planned_duration_summary") or {}
    a.check(
        "30 s ⇒ 6 clips / 7 keyframes",
        summary.get("num_clips") == 6 and summary.get("num_keyframes") == 7,
        json.dumps(_redact(summary)),
    )
    plan_text = json.dumps(_redact(plan), ensure_ascii=False)
    a.check(
        "El plan habla del entorno pedido (parque)",
        "parque" in plan_text.lower() or "park" in plan_text.lower(),
        "parque presente en el plan" if "parque" in plan_text.lower() else "no se menciona el parque",
    )
    a.check(
        "El plan NO fuerza calle/escaparates en una escena de parque",
        not re.search(r"sidewalk|storefront|café awning", plan_text, re.I),
        "sin restos de «acera/escaparates»" if not re.search(r"sidewalk|storefront", plan_text, re.I) else "quedan restos de escena urbana",
    )
    stack = ((plan.get("detected_gpu") or {}).get("selected_video_stack") or {})
    gpu_present = bool((state.get("hardware") or {}).get("gpu", {}).get("has_gpu"))
    a.check(
        "LoRAs rápidos activos en el plan",
        bool(stack.get("video_lora_id")) or not gpu_present,
        f"modelo={stack.get('model_id')} · lora_rápido={stack.get('video_lora_id') or 'NINGUNO'}"
        + ("" if gpu_present else " (sin GPU: el plan usa el motor CPU, no es un fallo)"),
    )

    # 6) Guardas (se resuelven antes de tocar el motor)
    first, second = _probe_images(state)
    if len(a.probe_images) >= 2:
        first, second = a.probe_images[0], a.probe_images[1]
    dur = a.tool(
        "studio_generate_video",
        {
            "video_submode": "flf2v",
            "mode": "ai",
            "model_id": "wan21-vace-14b-fp8",
            "video_lora_id": stack.get("video_lora_id") or "lora-wan21-14b-fusionx-phantom",
            "image_rel": first,
            "end_image_rel": second,
            "prompt": "audit probe",
            "duration_sec": 30,
        },
    )
    a.check(
        "Hay dos imágenes para probar las guardas",
        bool(first and second),
        f"{first or 'falta'} → {second or 'falta'}",
    )
    a.check(
        "Duración imposible se rechaza con plan de tramos",
        dur.get("error_code") == "DURATION_EXCEEDS_SINGLE_CLIP",
        f"error_code={dur.get('error_code')} · máx por clip={dur.get('max_seconds_per_clip')} · clips={dur.get('recommended_clips')}",
    )
    speed = a.tool(
        "studio_generate_video",
        {
            "video_submode": "flf2v",
            "mode": "ai",
            "model_id": "wan21-vace-14b-fp8",
            "video_lora_id": "",
            "extra_video_lora_ids": ["lora-wan21-14b-unianimate-char"],
            "image_rel": first,
            "end_image_rel": second,
            "prompt": "audit probe",
            "duration_sec": 5,
        },
    )
    a.check(
        "«Rápido» sin LoRA rápido se rechaza",
        speed.get("error_code") == "SPEED_LORA_MISSING",
        f"error_code={speed.get('error_code')}",
    )

    # 7) Instrucciones del agente
    gstatus, guide_text = a.http_text("/mcp/guia")
    if gstatus != 200 or not guide_text or len(guide_text) < 2000:
        time.sleep(1.0)
        gstatus, guide_text = a.http_text("/mcp/guia")
    guide_text = _redact(guide_text) if isinstance(guide_text, str) else str(guide_text)
    a.check(
        "La guía se sirve como markdown completo",
        gstatus == 200 and len(guide_text) > 5000,
        f"HTTP {gstatus} · {len(guide_text)} caracteres",
    )
    missing = [s for s in REQUIRED_INSTRUCTIONS if s not in guide_text]
    a.check(
        "La guía publica las reglas nuevas",
        gstatus == 200 and not missing,
        f"HTTP {gstatus}" + (f" · falta: {missing}" if missing else " · todas presentes"),
    )
    a.check(
        "La guía no es la versión antigua",
        not any(bad in guide_text for bad in FORBIDDEN_IN_OLD_BUILD),
        "sin marcas de versión antigua",
    )

    # 8) Higiene: ¿el token se filtra en las respuestas?
    # La guía incrusta el token A PROPÓSITO (es el enlace que el usuario pega en su agente),
    # así que la comprobación de fuga cubre las respuestas de herramientas y estado.
    raw_probe = "\n".join(json.dumps(x, ensure_ascii=False) if not isinstance(x, str) else x for x in (state, plan, dur, speed))
    token_leak = bool(len(a.token) >= 16 and a.token in raw_probe)
    pattern_leak = bool(re.search(r"st_mcp_[0-9a-f]{8,}", raw_probe))
    a.check(
        "Las respuestas de herramientas no reexponen el token",
        not token_leak and not pattern_leak,
        "limpio (el token solo vive en /mcp/guia y /mcp/bridge.py, que lo necesitan por diseño)",
    )

    a.check(
        "Recordatorio: esto es auditoría, no validación GPU",
        True,
        "las guardas y el plan se comprueban en CPU; la calidad final del vídeo sigue sin validar en GPU",
    )
    if pending:
        a.check("Preguntas pendientes (esperado si el usuario no ha respondido)", True, ", ".join(map(str, pending)))
    return a.checks


def _probe_images(state: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """Dos imágenes ya presentes en la instancia: las guardas de vídeo las exigen.

    `recent_assets` es un dict de buckets (uploads, outputs/images…); también se acepta
    una lista plana por si el contrato cambia.
    """
    rels: List[str] = []

    def _collect(items: Any) -> None:
        for item in items or []:
            rel = item.get("rel_path") if isinstance(item, dict) else None
            if isinstance(rel, str) and rel.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                rels.append(rel)

    recent = state.get("recent_assets")
    if isinstance(recent, dict):
        for bucket in recent.values():
            _collect(bucket)
    elif isinstance(recent, list):
        _collect(recent)
    assets = state.get("assets")
    if isinstance(assets, dict):
        for bucket in assets.values():
            _collect(bucket)
    rels = list(dict.fromkeys(rels))
    return (rels[0] if rels else None, rels[1] if len(rels) > 1 else None)


def _find_commit(state: Dict[str, Any]) -> Optional[str]:
    for key in ("commit", "git_commit", "build_commit", "revision", "code_revision"):
        val = state.get(key)
        if isinstance(val, str) and re.fullmatch(r"[0-9a-f]{7,40}", val):
            return val
    server = state.get("server") if isinstance(state.get("server"), dict) else {}
    for key in ("commit", "git_commit", "revision"):
        val = (server or {}).get(key)
        if isinstance(val, str) and re.fullmatch(r"[0-9a-f]{7,40}", val):
            return val
    return None


def _find_version(state: Dict[str, Any]) -> Optional[str]:
    for key in ("version", "server_version", "build", "app_version"):
        val = state.get(key)
        if isinstance(val, str) and val:
            return val
    return None


def render(checks: List[Tuple[str, bool, str]], base: str, expect_commit: Optional[str]) -> str:
    total = len(checks)
    passed = sum(1 for _, ok, _ in checks if ok)
    lines = [
        "# Auditoría MCP — Studio IA",
        "",
        f"- **Instancia:** `{base}`",
        f"- **Fecha:** {time.strftime('%Y-%m-%d %H:%M:%S')} local",
        f"- **Token:** enviado en cabecera `Authorization`, nunca mostrado",
        f"- **Resultado:** {passed}/{total} comprobaciones OK",
        ("- **Commit desplegado:** `" + next((d for n, ok, d in checks if n.startswith("Identidad") or n.startswith("Commit")), "?") + "`"),
        "" if passed == total else "- **Fallos:** " + "; ".join(n for n, ok, _ in checks if not ok),
        "",
        "| # | Comprobación | Estado | Detalle |",
        "|---|---|---|---|",
    ]
    for i, (name, ok, detail) in enumerate(checks, 1):
        icon = "✅" if ok else "❌"
        safe_detail = str(detail).replace("|", "\\|")[:300]
        lines.append(f"| {i} | {name} | {icon} | {safe_detail} |")
    lines += [
        "",
        "## Límites de esta auditoría",
        "",
        "- No genera vídeo, no descarga pesos y no borra nada. Las guardas se comprueban antes del motor.",
        "- Un ✅ en el plan significa que el contrato es correcto en CPU; **no** es validación de calidad GPU.",
        "- Si el commit desplegado no coincide con `--expect-commit`, reinicia la instancia con el comando de "
        "`docs/COLAB_RECOVERY.md`; no hace falta volver a descargar pesos.",
    ]
    return "\n".join(lines)


def _split_link(link: str, token: str) -> Tuple[str, str]:
    """Acepta tanto la URL base como el enlace completo `/mcp?token=...`."""
    url = (link or "").strip()
    if "?" in url:
        base, _, query = url.partition("?")
        for part in query.split("&"):
            key, _, value = part.partition("=")
            if key.strip().lower() in ("token", "st_token") and not token:
                token = value.strip()
        url = base
    if url.endswith("/mcp"):
        url = url[: -len("/mcp")]
    return url.rstrip("/"), token


def main() -> int:
    ap = argparse.ArgumentParser(description="Auditoría de solo lectura del servidor MCP de Studio IA.")
    ap.add_argument("--url", required=True, help="URL base del túnel (sin /mcp)")
    ap.add_argument("--token", default=os.environ.get("STUDIO_MCP_TOKEN", ""), help="Token MCP (mejor por variable de entorno)")
    ap.add_argument("--expect-commit", default="", help="Commit que debería estar desplegado (ej. 4a68ec1)")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--probe-image", action="append", default=[],
                    help="rel_path de una imagen ya subida (repítelo 2 veces) para probar las guardas de vídeo")
    ap.add_argument("--out", default="", help="Guardar el informe markdown en este archivo")
    args = ap.parse_args()

    url, token = _split_link(args.url, (args.token or "").strip())
    args.url = url
    token = (token or "").strip()
    if not token:
        print("Falta el token: usa --token o la variable STUDIO_MCP_TOKEN.", file=sys.stderr)
        return 2

    audit = Audit(args.url, token, timeout=args.timeout)
    audit.probe_images = list(args.probe_image)
    try:
        checks = run_audit(audit, args.expect_commit or None)
    finally:
        pass
    report = render(checks, args.url.rstrip("/"), args.expect_commit or None)
    print(report)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(report + "\n")
        print(f"\nInforme guardado en {args.out}", file=sys.stderr)
    return 0 if all(ok for _, ok, _ in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
