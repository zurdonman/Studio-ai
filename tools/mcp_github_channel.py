#!/usr/bin/env python3
"""Canal de control por GitHub para Studio IA (para agentes sin acceso al túnel).

El agente no puede abrir conexiones hacia tu Colab: su egreso solo permite unos
pocos dominios. `api.github.com` SÍ está permitido, así que este agente corre en
la celda de Colab, sondea un fichero de órdenes en el repositorio y ejecuta las
herramientas MCP en proceso (sin pasar por la red ni por el túnel).

Seguridad:
  - Solo ejecuta VERBOS PERMITIDOS (lista cerrada). Nada de shell, nada de
    descargas de modelos ni borrado de pesos: esas herramientas NO están.
  - Solo lee el fichero de órdenes de la rama configurada y del repositorio
    configurado.
  - Escribe resultados en `state/channel/resultados.json`, sin credenciales.
  - Las órdenes se ejecutan en orden y una sola vez por `id`.

Normalmente NO se lanza a mano: la aplicación lo enciende desde el panel MCP
(POST /api/mcp/github-channel/start) y lo vigila en segundo plano. El comando
manual queda solo para depurar:

    python3 tools/mcp_github_channel.py --once --dry-run

Transporte (`--backend`):
  - `state` (por defecto): un fichero JSON en la rama `studio-channel` hace de buzón.
    El agente escribe `agent.orders`; la app escribe `app.results`. La lectura
    funciona sin credenciales cuando el repo es público.
  - `repo`: ficheros `channel/ordenes.json` y `state/channel/resultados.json` (legado).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

DEFAULT_REPO = "zurdonman/Studio-ai"

DEFAULT_ORDERS = "channel/ordenes.json"
DEFAULT_RESULTS = "state/channel/resultados.json"

# Latido máximo: sin cambios reales no se escribe más de una vez cada 10 minutos.
HEARTBEAT_SECONDS = 600

# Lista CERRADA de verbos. Si una orden pide otra cosa, se rechaza y se reporta.
ALLOWED_VERBS = {
    "get_state",
    "get_workflow_preset",
    "get_work_state",
    "answer_preference",
    "generate_image",
    "generate_video",
    "generate_speech",
    "generate_ambience",
    "analyze_coherence",
    "lipsync",
    "render_montage",
    "wait_for_job",
    "export_session_log",
}

VERB_TO_TOOL = {
    "get_state": "studio_get_state_and_manual",
    "get_workflow_preset": "studio_get_workflow_preset",
    "get_work_state": "studio_get_work_state",
    "answer_preference": "studio_answer_preference",
    "generate_image": "studio_generate_or_edit_image",
    "generate_video": "studio_generate_video",
    "generate_speech": "studio_generate_speech",
    "generate_ambience": "studio_generate_music_or_sfx",
    "analyze_coherence": "studio_analyze_coherence",
    "lipsync": "studio_lipsync_character",
    "render_montage": "studio_render_montage",
    "wait_for_job": "studio_wait_for_job",
    "export_session_log": "studio_export_session_log",
}

# Nunca se reenvían campos de credencial.
FORBIDDEN_ARGS = {"token", "authorization", "hf_token", "password", "api_key"}


def _api_get(repo: str, path: str, token: str, branch: str = "") -> Optional[Dict[str, Any]]:
    """Lee un fichero del repo por la API de contenidos (JSON base64)."""
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    if branch:
        url += f"?ref={branch}"
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    req.add_header("User-Agent", "studio-ia-github-channel/1.0")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise SystemExit(f"GitHub {exc.code} leyendo {path}")


def _read_json_file(repo: str, path: str, token: str, branch: str = "") -> Optional[Dict[str, Any]]:
    import base64

    data = _api_get(repo, path, token, branch)
    if not data or "content" not in data:
        return None
    try:
        raw = base64.b64decode(data["content"]).decode("utf-8")
        return json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        return None


def _gh_api(url: str, token: str, method: str = "GET",
            payload: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Llamada mínima a la API de GitHub (api.github.com es el único dominio que el agente alcanza)."""
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("User-Agent", "studio-ia-github-channel/2.0")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8", "replace")
        return json.loads(body) if body.strip() else {}
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise
    except urllib.error.URLError:
        return None


class StateMailbox:
    """Buzón = un fichero JSON en una rama aparte del repositorio.

    Estructura, pensada para que dos partes escriban sin pisarse:

        {"schema": 1,
         "agent": {"orders": [{"id": "a1", "verb": "get_state", "args": {...}}]},
         "app":   {"executed_ids": ["a1"], "results": [...], "updated": "..."}}

    El agente escribe `agent`; la app escribe `app`. La lectura funciona **sin
    credenciales** cuando el repositorio (y la rama) son públicos, así que la app
    no necesita token para leer órdenes. Escribir resultados sí exige token.
    """

    def __init__(self, repo: str, branch: str, path: str, token: str) -> None:
        self.repo = repo
        self.branch = branch
        self.path = path
        self.token = token

    def _url(self) -> str:
        return f"https://api.github.com/repos/{self.repo}/contents/{self.path}"

    def read(self) -> Dict[str, Any]:
        import base64

        data = _gh_api(f"{self._url()}?ref={self.branch}", self.token)
        if not isinstance(data, dict) or "content" not in data:
            return {}
        try:
            return json.loads(base64.b64decode(data["content"]).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return {}

    def write_app(self, ids: List[str], results: List[Dict[str, Any]]) -> None:
        import base64

        if not self.token:
            raise RuntimeError("sin token de GitHub no se pueden publicar resultados")
        current = self.read() or {"schema": 1, "agent": {"orders": []}}
        previous = dict(current.get("app") or {})
        # Solo se escribe cuando hay algo nuevo: un comité por sondeo ensuciaría el
        # repositorio para siempre y no aporta nada. El latido va aparte y espaciado.
        same_results = (previous.get("executed_ids") == ids
                        and len(previous.get("results") or []) == len(results))
        last_beat = 0.0
        try:
            stamp = str(previous.get("updated") or "")
            if stamp:
                import calendar

                last_beat = calendar.timegm(time.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ"))
        except (ValueError, TypeError):
            last_beat = 0.0
        if same_results and (time.time() - last_beat) < HEARTBEAT_SECONDS:
            return
        current["app"] = {
            "executed_ids": ids,
            "results": results[-200:],
            "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        existing = _gh_api(f"{self._url()}?ref={self.branch}", self.token) or {}
        payload: Dict[str, Any] = {
            "message": f"canal: resultados {time.strftime('%H:%M:%S')}",
            "content": base64.b64encode(json.dumps(current, ensure_ascii=False, indent=2).encode()).decode(),
            "branch": self.branch,
        }
        if isinstance(existing, dict) and existing.get("sha"):
            payload["sha"] = existing["sha"]
        _gh_api(self._url(), self.token, "PUT", payload)


def _rpc(base: str, token: str, payload: Dict[str, Any], session: Dict[str, str]) -> Dict[str, Any]:
    req = urllib.request.Request(
        f"{base}/mcp",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "User-Agent": "studio-ia-github-channel/1.0",
        },
    )
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    if session.get("id"):
        req.add_header("Mcp-Session-Id", session["id"])
    with urllib.request.urlopen(req, timeout=90) as resp:
        session["id"] = resp.headers.get("Mcp-Session-Id") or session.get("id", "")
        body = resp.read().decode("utf-8", "replace").strip()
    if body.startswith("data:"):
        body = [ln[5:].strip() for ln in body.splitlines() if ln.startswith("data:")][-1]
    return json.loads(body) if body else {}


def _dispatch(verb: str, args: Dict[str, Any], base: str, token: str,
              session: Dict[str, str]) -> Dict[str, Any]:
    """Ejecuta la orden contra el SERVIDOR LOCAL (127.0.0.1) por MCP.

    Se hace por HTTP y no importando `studio.mcp_server` a propósito: así el
    trabajo entra en la cola de la aplicación, se ve en la interfaz y respeta su
    gobernanza de memoria. Importarlo en este proceso crearía una segunda cola
    invisible y duplicaría el modelo en RAM.
    """
    clean = {k: v for k, v in args.items() if k.lower() not in FORBIDDEN_ARGS}
    if clean != args:
        raise ValueError("argumento prohibido en la orden")
    if not session.get("ready"):
        _rpc(base, token, {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                           "params": {"protocolVersion": "2025-03-26", "capabilities": {},
                                      "clientInfo": {"name": "github-channel", "version": "1.0"}}}, session)
        session["ready"] = "1"
        session["seq"] = "1"
    session["seq"] = str(int(session.get("seq", "1")) + 1)
    resp = _rpc(base, token, {"jsonrpc": "2.0", "id": int(session["seq"]), "method": "tools/call",
                              "params": {"name": VERB_TO_TOOL[verb], "arguments": clean}}, session)
    if "error" in resp:
        return {"ok": False, "error_code": "RPC_ERROR", "error": resp["error"]}
    result = resp.get("result") or {}
    texts = [c.get("text", "") for c in (result.get("content") or []) if c.get("type") == "text"]
    raw = "\n".join(t for t in texts if t)
    try:
        return json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return {"raw": raw[:2000]}


def _wait_job(base: str, token: str, session: Dict[str, str], job_id: str,
              budget_sec: float = 3000.0) -> Dict[str, Any]:
    started = time.time()
    while True:
        snapshot = _dispatch("wait_for_job", {"job_id": job_id, "wait_seconds": 80}, base, token, session)
        status = snapshot.get("status")
        if status in ("completed", "failed", "cancelled"):
            return snapshot
        if time.time() - started > budget_sec:
            return dict(snapshot, ok=False, error_code="CLIENT_TIMEOUT")


class _Mailbox:
    """Buzón: estado en rama aparte (defecto) o ficheros del repo (legado)."""

    def __init__(self, backend: str, repo: str, branch: str, orders_path: str, results_path: str,
                 token: str = "", state_branch: str = "", state_path: str = "") -> None:
        self.backend = backend
        self.repo = repo
        self.branch = branch
        self.orders_path = orders_path
        self.results_path = results_path
        self.token = token
        self.state = (StateMailbox(repo, state_branch or branch, state_path, token)
                      if backend == "state" else None)

    def read_orders(self) -> Dict[str, Any]:
        if self.state:
            return {"orders": ((self.state.read() or {}).get("agent") or {}).get("orders") or []}
        return _read_json_file(self.repo, self.orders_path, self.token, self.branch) or {}

    def read_results(self) -> Dict[str, Any]:
        if self.state:
            return (self.state.read() or {}).get("app") or {}
        return _read_json_file(self.repo, self.results_path, self.token, self.branch) or {}

    def write_results(self, ids: List[str], results: List[Dict[str, Any]]) -> None:
        if self.state:
            self.state.write_app(ids, results)
            return
        _write_results(self.repo, self.results_path, self.token, ids, results, self.branch)


def run_once(repo: str, orders_path: str, results_path: str, token: str, dry_run: bool,
             base: str = "http://127.0.0.1:7860", mcp_token: str = "", branch: str = "",
             backend: str = "state", state_branch: str = "", state_path: str = "",
             on_result: Optional[Callable[[Dict[str, Any]], None]] = None) -> int:
    """Una pasada: lee órdenes nuevas, las ejecuta y publica resultados.

    `backend="state"` (por defecto) usa una rama aparte con un único JSON de estado:
    lectura anónima si el repo es público, escritura de resultados solo con token.
    `backend="repo"` mantiene el formato antiguo por compatibilidad.
    """
    box = _Mailbox(backend, repo, branch, orders_path, results_path, token,
                   state_branch, state_path)
    orders = box.read_orders()
    queue = orders.get("orders") or []
    if not queue:
        print("Sin órdenes pendientes.")
        return 0

    def _publish() -> None:
        box.write_results(sorted(executed), results)
        if on_result:
            try:
                on_result(results[-1] if results else {})
            except Exception:
                pass

    done = box.read_results()
    executed = set(done.get("executed_ids") or [])
    results: List[Dict[str, Any]] = list(done.get("results") or [])

    session: Dict[str, str] = {}
    for order in queue:
        oid = str(order.get("id") or "").strip()
        verb = str(order.get("verb") or "").strip()
        if not oid or oid in executed:
            continue
        if verb not in ALLOWED_VERBS:
            print(f"\u2717 {oid}: verbo '{verb}' no permitido; se ignora.")
            executed.add(oid)
            results.append({"id": oid, "verb": verb, "ok": False, "error": "VERB_NOT_ALLOWED"})
            _publish()
            continue
        args = order.get("args") or {}
        print(f"\u2192 {oid}: {verb} {json.dumps(args, ensure_ascii=False)[:120]}")
        if dry_run:
            continue
        started = time.time()
        try:
            out = _dispatch(verb, args, base, mcp_token, session)
            job_id = out.get("job_id") or (out.get("job") or {}).get("id")
            if verb not in ("wait_for_job",) and job_id and order.get("wait", True):
                out = _wait_job(base, mcp_token, session, str(job_id),
                                budget_sec=float(order.get("budget_sec") or 3000))
            entry = {
                "id": oid,
                "verb": verb,
                "ok": bool(out.get("ok", True)) and out.get("status") in (None, "completed"),
                "seconds": round(time.time() - started, 1),
                "result": _compact(out, verb),
            }
        except Exception as exc:  # un error se reporta, nunca tumba el canal
            entry = {"id": oid, "verb": verb, "ok": False, "error": f"{type(exc).__name__}: {exc}"[:400]}
        mark = "OK" if entry["ok"] else "FALLO"
        print(f"   [{mark}] {entry.get('seconds', 0)}s {entry.get('error', '')}")
        results.append(entry)
        executed.add(oid)
        _publish()
    if not dry_run:
        _publish()
    return 0


DIAGNOSTIC_KEYS = ("ok", "status", "job_id", "error_code", "error", "progress_pct",
                   "elapsed_seconds", "rel_path", "filename", "media_url", "model_id",
                   "video_submode")

# Límite de tamaño de lo que se publica: el buzón es un JSON en un repo, no un log.
MAX_PUBLISHED_CHARS = 12000


def _summarize_state(out: Dict[str, Any]) -> Dict[str, Any]:
    """Resumen útil del estado de la máquina.

    Antes se descartaba TODO y el agente recibía `{"ok": true}`: sin GPU, sin VRAM,
    sin disco y sin saber qué código corría. Depurar así es imposible.
    """
    hw = out.get("hardware") or {}
    gpu = hw.get("gpu") or {}
    ram = hw.get("ram") or {}
    disk = hw.get("disk") or {}
    credits = hw.get("credits") or {}
    build = out.get("build") or {}
    loaded = out.get("loaded_in_vram") or []
    jobs = out.get("active_jobs") or []

    def _names(value):
        if isinstance(value, list):
            return [str((v or {}).get("name") or (v or {}).get("id") or v)[:60] for v in value][:10]
        return value

    summary: Dict[str, Any] = {
        "gpu": {k: gpu.get(k) for k in ("gpu_name", "has_gpu", "vram_total_gb", "vram_free_gb",
                                        "measurement_source") if k in gpu},
        "ram": {k: ram.get(k) for k in ("total_gb", "available_gb", "used_gb") if k in ram},
        "disk": {k: disk.get(k) for k in ("free_gb", "total_gb", "used_gb") if k in disk},
        "credits": {k: credits.get(k) for k in ("hourly_rate", "is_estimate", "notice",
                                                "source", "autoshutdown_seconds_left") if k in credits},
        "build": {k: build.get(k) for k in ("commit", "short_commit", "branch", "dirty") if k in build},
        "loaded_in_vram": _names(loaded),
        "active_jobs": [{"id": (j or {}).get("id"), "status": (j or {}).get("status"),
                         "progress_pct": (j or {}).get("progress_pct")} for j in jobs][:5],
        "payload_keys": sorted(out.keys()),
    }
    return summary


def _compact(out: Dict[str, Any], verb: str = "") -> Dict[str, Any]:
    """Resultado legible sin volcar prompts enteros ni credenciales... pero sin cegar."""
    result = out.get("result") if isinstance(out.get("result"), dict) else {}
    merged = {**out, **result}
    compact = {k: merged[k] for k in DIAGNOSTIC_KEYS if k in merged}
    if verb == "get_state":
        compact["state"] = _summarize_state(merged)
    elif verb == "get_workflow_preset":
        plan = merged.get("plan") if isinstance(merged.get("plan"), dict) else {}
        compact["plan_keys"] = sorted(plan.keys())[:20]
        for key in ("phases", "steps", "video_lora_id", "mode", "submode"):
            if key in merged:
                compact[key] = merged[key]
    text = json.dumps(compact, ensure_ascii=False)
    if len(text) > MAX_PUBLISHED_CHARS:
        compact = {"ok": compact.get("ok", True), "truncated": True,
                   "note": f"resultado de {len(text)} caracteres recortado",
                   "keys": sorted(merged.keys())}
    return compact


def _write_results(repo: str, path: str, token: str, ids: List[str], results: List[Dict[str, Any]],
                   branch: str = "") -> None:
    """Escribe el estado en el repo (lo lee el agente con GET normal)."""
    import base64
    import subprocess

    payload = {
        "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "executed_ids": ids,
        "results": results[-200:],
    }
    body = json.dumps(payload, ensure_ascii=False, indent=2)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")
    if not token:
        print(f"   (sin GH_TOKEN: resultados solo en {target})")
        return
    existing = _api_get(repo, path, token, branch) or {}
    message = {"message": f"canal: resultados {time.strftime('%H:%M:%S')}", "content": base64.b64encode(body.encode()).decode()}
    if branch:
        message["branch"] = branch
    if existing.get("sha"):
        message["sha"] = existing["sha"]
    req = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/contents/{path}",
        data=json.dumps(message).encode(),
        method="PUT",
        headers={"Accept": "application/vnd.github+json", "Authorization": f"Bearer {token}",
                 "Content-Type": "application/json", "User-Agent": "studio-ia-github-channel/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            resp.read()
    except Exception as exc:
        print(f"   ⚠ no se pudo publicar en GitHub ({type(exc).__name__}); queda en {target}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Canal de control por GitHub para Studio IA.")
    ap.add_argument("--repo", default=os.environ.get("STUDIO_CHANNEL_REPO", DEFAULT_REPO))
    ap.add_argument("--branch", default=os.environ.get("STUDIO_CHANNEL_BRANCH", "arena/f0997b49-studio-ai"))
    ap.add_argument("--backend", default=os.environ.get("STUDIO_CHANNEL_BACKEND", "state"),
                    choices=("state", "repo"),
                    help="buzón: estado en rama aparte (defecto) o ficheros del repo (legado)")
    ap.add_argument("--state-branch", default=os.environ.get("STUDIO_CHANNEL_STATE_BRANCH", "studio-channel"))
    ap.add_argument("--state-path", default=os.environ.get("STUDIO_CHANNEL_STATE_PATH", "channel/agent_state.json"))
    ap.add_argument("--orders", default=DEFAULT_ORDERS)
    ap.add_argument("--results", default=DEFAULT_RESULTS)
    ap.add_argument("--once", action="store_true", help="una pasada")
    ap.add_argument("--poll", type=float, default=0.0, help="segundos entre pasadas (0 = salir)")
    ap.add_argument("--dry-run", action="store_true", help="muestra las órdenes sin ejecutarlas")
    ap.add_argument("--url", default=os.environ.get("STUDIO_URL", "http://127.0.0.1:7860"),
                    help="URL del servidor Studio IA local (por defecto el de la propia celda)")
    ap.add_argument("--mcp-token", default=os.environ.get("STUDIO_MCP_TOKEN", ""),
                    help="token MCP del servidor local (lo usa el propio lanzador)")
    args = ap.parse_args()

    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN") or ""
    if not token:
        print("Aviso: sin GH_TOKEN solo se puede LEER el repo público (no publicar resultados).")

    while True:
        run_once(args.repo, args.orders, args.results, token, args.dry_run,
                 base=args.url.rstrip("/"), mcp_token=args.mcp_token, branch=args.branch,
                 backend=args.backend, state_branch=args.state_branch, state_path=args.state_path)
        if args.poll <= 0:
            return 0
        time.sleep(max(5.0, args.poll))


if __name__ == "__main__":
    raise SystemExit(main())
