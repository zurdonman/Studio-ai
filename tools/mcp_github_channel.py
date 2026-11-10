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
  - `gist` (por defecto): un gist privado del usuario hace de buzón. El repo puede
    ser privado; el agente solo necesita la URL del gist.
  - `repo`: ficheros `channel/ordenes.json` y `state/channel/resultados.json`.
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
DEFAULT_GIST_ID = ""

# El canal NO depende de que el repo sea público: por defecto usa un gist privado,
# que el agente alcanza por la misma API (api.github.com) y que no expone ni el
# código ni los resultados del usuario en internet.
DEFAULT_ORDERS = "channel/ordenes.json"
DEFAULT_RESULTS = "state/channel/resultados.json"

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
    """Llamada mínima a la API de GitHub (solo api.github.com, único dominio permitido)."""
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("User-Agent", "studio-ia-github-channel/1.0")
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


class GistMailbox:
    """Buzón de órdenes en un gist privado: dos ficheros, `ordenes.json` y `resultados.json`."""

    def __init__(self, gist_id: str, token: str) -> None:
        self.gist_id = gist_id
        self.token = token

    @staticmethod
    def create(token: str, description: str = "Studio IA · canal de control") -> str:
        out = _gh_api("https://api.github.com/gists", token, "POST", {
            "description": description,
            "public": False,
            "files": {"ordenes.json": {"content": json.dumps({"orders": []}, indent=2)},
                      "resultados.json": {"content": json.dumps({"results": []}, indent=2)}},
        })
        if not out or not out.get("id"):
            raise RuntimeError("no se pudo crear el gist del canal")
        return str(out["id"])

    def read(self, name: str) -> Dict[str, Any]:
        out = _gh_api(f"https://api.github.com/gists/{self.gist_id}", self.token)
        if not out:
            return {}
        f = (out.get("files") or {}).get(name) or {}
        content = f.get("content")
        if content is None and f.get("raw_url"):
            content = (_gh_api(f["raw_url"], self.token) or {}).get("raw", "")
        try:
            return json.loads(content or "{}")
        except (TypeError, json.JSONDecodeError):
            return {}

    def write(self, name: str, body: Dict[str, Any]) -> None:
        _gh_api(f"https://api.github.com/gists/{self.gist_id}", self.token, "PATCH",
                {"files": {name: {"content": json.dumps(body, indent=2, ensure_ascii=False)}}})


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
    """Abstracción mínima: leer/escribir el buzón, sea gist (privado) o repo (público)."""

    def __init__(self, backend: str, repo: str, branch: str, orders_path: str, results_path: str,
                 gist_id: str, token: str) -> None:
        self.backend = backend
        self.repo = repo
        self.branch = branch
        self.orders_path = orders_path
        self.results_path = results_path
        self.token = token
        self.gist = GistMailbox(gist_id, token) if backend == "gist" else None

    def read_orders(self) -> Dict[str, Any]:
        if self.gist:
            return self.gist.read("ordenes.json")
        return _read_json_file(self.repo, self.orders_path, self.token, self.branch) or {}

    def read_results(self) -> Dict[str, Any]:
        if self.gist:
            return self.gist.read("resultados.json")
        return _read_json_file(self.repo, self.results_path, self.token, self.branch) or {}

    def write_results(self, ids: List[str], results: List[Dict[str, Any]]) -> None:
        if self.gist:
            self.gist.write("resultados.json", {
                "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "executed_ids": ids, "results": results[-200:]})
            return
        _write_results(self.repo, self.results_path, self.token, ids, results, self.branch)


def run_once(repo: str, orders_path: str, results_path: str, token: str, dry_run: bool,
             base: str = "http://127.0.0.1:7860", mcp_token: str = "", branch: str = "",
             backend: str = "gist", gist_id: str = "",
             on_result: Optional[Callable[[Dict[str, Any]], None]] = None) -> int:
    """Una pasada: lee órdenes nuevas, las ejecuta y publica resultados.

    `backend="gist"` (por defecto) no exige que el repo sea público: el buzón es un
    gist privado. `backend="repo"` usa los ficheros del repositorio y solo tiene
    sentido si el repo es público (o si el agente tiene token de lectura).
    """
    box = _Mailbox(backend, repo, branch, orders_path, results_path, gist_id, token)
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
                "result": _compact(out),
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


def _compact(out: Dict[str, Any]) -> Dict[str, Any]:
    """Resultado legible sin volcar prompts enteros ni credenciales."""
    keys = ("ok", "status", "job_id", "error_code", "error", "progress_pct", "elapsed_seconds",
            "rel_path", "filename", "media_url", "model_id", "video_submode")
    result = out.get("result") if isinstance(out.get("result"), dict) else {}
    merged = {**out, **{k: v for k, v in result.items() if k in keys}}
    return {k: merged[k] for k in keys if k in merged}


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
    ap.add_argument("--backend", default=os.environ.get("STUDIO_CHANNEL_BACKEND", "gist"),
                    choices=("gist", "repo"), help="buzón: gist privado (defecto) o ficheros del repo")
    ap.add_argument("--gist-id", default=os.environ.get("STUDIO_CHANNEL_GIST_ID", DEFAULT_GIST_ID))
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
                 backend=args.backend, gist_id=args.gist_id)
        if args.poll <= 0:
            return 0
        time.sleep(max(5.0, args.poll))


if __name__ == "__main__":
    raise SystemExit(main())
