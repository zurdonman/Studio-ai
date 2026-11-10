"""Canal de control por GitHub, gobernado desde la propia aplicación.

Muchos agentes (arena.ai entre ellos) no pueden abrir conexiones hacia el túnel
de Colab: su egreso solo permite unos pocos dominios, y `api.github.com` es uno
de ellos. Este servicio invierte la dirección del control:

    agente  --(API de GitHub)-->  gist privado  <--(sondeo)-->  este servicio  -->  servidor local

Todo se enciende desde el panel MCP con un botón. La UI entrega a la aplicación
las credenciales que el usuario ya proveyó, la aplicación crea el buzón (un gist
privado) y lanza el bucle en un hilo propio. El usuario no pega comandos.

Decisiones de diseño que importan:

- **Buzón en gist privado por defecto.** El repositorio puede ser privado; no hace
  falta publicar órdenes ni resultados, y el agente solo necesita el identificador
  del gist. `backend="repo"` queda como alternativa para repos públicos.
- **Un solo arranque por proceso.** El estado vive en memoria y el hilo se marca
  daemon: si el servidor cae, no queda nada vivo que no se pueda reiniciar.
- **Lista cerrada de verbos.** La ejecución real la hace `tools/mcp_github_channel.py`,
  que ya rechaza cualquier verbo fuera de la lista y cualquier argumento con
  credenciales.
- **Nunca se guarda el token en el repositorio.** Vive en
  `studio_data/state/github_channel.json` con permisos 0600.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

_lock = threading.Lock()
_state: Dict[str, Any] = {
    "running": False,
    "backend": "",
    "gist_id": "",
    "repo": "",
    "branch": "",
    "started_at": 0.0,
    "last_poll": 0.0,
    "polls": 0,
    "executed": 0,
    "last_error": "",
    "last_verb": "",
    "thread": None,
}
_stop = threading.Event()


def _token_path() -> Path:
    data_root = Path(os.environ.get("STUDIO_DATA_DIR") or "/content/studio_data")
    state = data_root / "state"
    state.mkdir(parents=True, exist_ok=True)
    return state / "github_channel.json"


def load_saved_token() -> str:
    """Token de GitHub ya guardado (por el lanzador o por un arranque anterior)."""
    env = (os.environ.get("STUDIO_CHANNEL_GH_TOKEN") or "").strip()
    if env:
        return env
    legacy = _token_path().with_name("github_channel_token.txt")
    for path, key in ((_token_path(), "gh_token"), (legacy, None)):
        try:
            if not path.exists():
                continue
            if key is None:
                value = path.read_text(encoding="utf-8").strip()
            else:
                value = (json.loads(path.read_text(encoding="utf-8")) or {}).get(key, "")
            if value:
                return str(value).strip()
        except (OSError, ValueError):
            continue
    return ""


def save_token(token: str) -> None:
    token = (token or "").strip()
    if not token:
        return
    path = _token_path()
    path.write_text(json.dumps({"gh_token": token}), encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def _saved_gist() -> str:
    try:
        return str((json.loads(_token_path().read_text(encoding="utf-8")) or {}).get("gist_id", ""))
    except (OSError, ValueError):
        return ""


def _save_gist(gist_id: str) -> None:
    try:
        data = json.loads(_token_path().read_text(encoding="utf-8")) or {}
    except (OSError, ValueError):
        data = {}
    data["gist_id"] = gist_id
    path = _token_path()
    path.write_text(json.dumps(data), encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def _channel_module():
    """Carga `tools/mcp_github_channel.py` como módulo (no está en el paquete)."""
    import importlib.util

    root = Path(__file__).resolve().parent.parent
    path = root / "tools" / "mcp_github_channel.py"
    spec = importlib.util.spec_from_file_location("mcp_github_channel", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _loop(interval: float, repo: str, branch: str, backend: str, gist_id: str,
          base: str, mcp_token: str, orders: str, results: str) -> None:
    ch = _channel_module()
    counted: set = set()

    def _on_result(entry: Dict[str, Any]) -> None:
        oid = str(entry.get("id") or "")
        with _lock:
            if not oid or oid in counted:
                return
            counted.add(oid)
            _state["executed"] += 1
            _state["last_verb"] = f"{entry.get('verb')} · {'ok' if entry.get('ok') else 'fallo'}"

    while not _stop.is_set():
        try:
            with _lock:
                _state["last_poll"] = time.time()
            ch.run_once(repo, orders, results, load_saved_token(), False, base=base,
                        mcp_token=mcp_token, branch=branch, backend=backend, gist_id=gist_id,
                        on_result=_on_result)
            with _lock:
                _state["polls"] += 1
                _state["last_error"] = ""
        except Exception as exc:  # el canal nunca debe tumbar el servidor
            with _lock:
                _state["last_error"] = f"{type(exc).__name__}: {exc}"[:300]
        _stop.wait(interval)


def start(gh_token: str = "", gist_id: str = "", backend: str = "gist",
          repo: str = "zurdonman/Studio-ai", branch: str = "arena/f0997b49-studio-ai",
          interval: float = 20.0, base: str = "", mcp_token: str = "",
          orders: str = "channel/ordenes.json",
          results: str = "state/channel/resultados.json") -> Dict[str, Any]:
    """Enciende el canal. Idempotente: si ya está activo, devuelve el estado actual."""
    global _state
    with _lock:
        if _state["running"]:
            return snapshot()
    token = (gh_token or "").strip() or load_saved_token()
    if not token:
        return {"ok": False, "error_code": "NO_GITHUB_TOKEN",
                "error": "Falta el token de GitHub (Contents: read/write o scope gist)."}
    save_token(token)

    ch = _channel_module()
    backend = "repo" if backend == "repo" else "gist"
    gist_id = (gist_id or "").strip() or _saved_gist()
    if backend == "gist" and not gist_id:
        try:
            gist_id = ch.GistMailbox.create(token)
        except Exception as exc:
            return {"ok": False, "error_code": "GIST_CREATE_FAILED",
                    "error": f"{type(exc).__name__}: {exc}"[:300],
                    "hint": "El token necesita permiso para crear gists (o usa backend=repo)."}
        _save_gist(gist_id)

    _stop.clear()
    with _lock:
        _state.update({"running": True, "backend": backend, "gist_id": gist_id, "repo": repo,
                       "branch": branch, "started_at": time.time(), "last_error": "",
                       "polls": 0, "executed": 0})
    thread = threading.Thread(
        target=_loop,
        args=(max(5.0, float(interval)), repo, branch, backend, gist_id,
              base or f"http://127.0.0.1:{os.environ.get('PORT', '7860')}", mcp_token, orders, results),
        name="github-channel",
        daemon=True,
    )
    _state["thread"] = thread
    thread.start()
    return snapshot()


def stop() -> Dict[str, Any]:
    _stop.set()
    with _lock:
        _state["running"] = False
        _state["thread"] = None
    return snapshot()


def snapshot() -> Dict[str, Any]:
    with _lock:
        return {k: v for k, v in _state.items() if k != "thread"}


def agent_handoff(gist_id: str, repo: str, branch: str, backend: str) -> Dict[str, Any]:
    """Lo que hay que darle al agente: solo el identificador del buzón, nunca el token."""
    if backend == "gist":
        return {
            "backend": "gist",
            "gist_id": gist_id,
            "orders_url": f"https://api.github.com/gists/{gist_id}",
            "orders_file": "ordenes.json",
            "results_file": "resultados.json",
            "how": ("PATCH/POST a https://api.github.com/gists/{id} con files.ordenes.json "
                    "para dejar órdenes; GET al mismo endpoint para leer resultados.json."),
        }
    return {
        "backend": "repo",
        "repo": repo,
        "branch": branch,
        "orders_url": f"https://api.github.com/repos/{repo}/contents/channel/ordenes.json?ref={branch}",
        "results_url": f"https://api.github.com/repos/{repo}/contents/state/channel/resultados.json?ref={branch}",
        "how": "GET/PUT en la API de contenidos del repo (requiere repo público o token).",
    }
