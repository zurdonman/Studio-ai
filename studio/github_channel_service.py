"""Canal de control por GitHub, gobernado desde la propia aplicación.

Muchos agentes (arena.ai entre ellos) no pueden abrir conexiones hacia el túnel
de Colab: su egreso solo permite unos pocos dominios, y `api.github.com` es uno
de ellos. Este servicio invierte la dirección del control:

    agente (token del usuario) --> estado privado en el repo <-- app (token del usuario)

Decisiones que definen el comportamiento y conviene no romper:

- **Token opcional.** Si no hay token de GitHub, la app consulta el buzón en
  MODO ANÓNIMO: la rama `studio-channel` y el estado del repo son públicos, así
  que cualquiera puede leer órdenes y resultados, incluido el agente. Con token
  se usa la API normal (repo privado, o rama publicada por el agente).
- **Sin copiar y pegar.** El punto de encuentro es fijo y conocido por las dos
  partes: rama `STUDIO_CHANNEL_STATE_BRANCH` (por defecto `studio-channel`) y
  `channel/agent_state.json`. La app lo enseña para que se lo pases al agente una
  vez; no hay que reconfigurar nada en cada arranque de Colab.
- **Un solo arranque por proceso**, hilo daemon: si el servidor cae, no queda
  nada vivo que no se pueda reiniciar desde el botón.
- **Lista cerrada de verbos**: la ejecución real la hace `tools/mcp_github_channel.py`.
- **El token nunca se escribe en el repo.** Vive en
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
# Nada de lo que hace el canal puede colgar la interfaz: el READ de estado usa el
# token del usuario y, si ese token es inválido, GitHub puede tardar. El sondeo corre
# en su propio hilo, así que el límite es holgura, no supervivencia.
HTTP_TIMEOUT = float(os.environ.get("STUDIO_CHANNEL_HTTP_TIMEOUT", "8"))
STATE_BRANCH_DEFAULT = "studio-channel"
STATE_PATH_DEFAULT = "channel/agent_state.json"

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
    "mode": "",
    "pending": 0,
    "thread": None,
}
_stop = threading.Event()


def _token_path() -> Path:
    data_root = Path(os.environ.get("STUDIO_DATA_DIR") or "/content/studio_data")
    state = data_root / "state"
    state.mkdir(parents=True, exist_ok=True)
    return state / "github_channel.json"


# Nombre del secreto en el panel «Secrets» (🔑) de Colab. Se configura UNA vez por
# cuenta de Google y sobrevive a todas las sesiones: no hay que volver a pegarlo.
COLAB_SECRET_NAME = os.environ.get("STUDIO_CHANNEL_GH_SECRET", "GH_TOKEN")


# `userdata.get()` puede BLOQUEARSE en Colab esperando autorización del usuario.
# Por eso se consulta una sola vez por proceso y nunca en el camino de la interfaz:
# si se llamase en cada GET, el panel se quedaría en «comprobando» para siempre.
_secret_cache: Dict[str, Any] = {"checked": False, "value": ""}


def _from_colab_secrets(force: bool = False) -> str:
    """Lee el token del gestor de secretos de Colab (persistente entre sesiones)."""
    if _secret_cache["checked"] and not force:
        return str(_secret_cache["value"])
    value = ""
    try:
        from google.colab import userdata  # type: ignore

        value = str(userdata.get(COLAB_SECRET_NAME) or "").strip()
    except Exception:
        value = ""
    _secret_cache.update({"checked": True, "value": value})
    return value


def has_token() -> bool:
    """¿Hay credenciales? Sin bloquearse NUNCA: no consulta Colab Secrets si aún no
    se ha leído. Lo usan los endpoints de la interfaz, que deben responder ya."""
    if (os.environ.get("STUDIO_CHANNEL_GH_TOKEN") or "").strip():
        return True
    if str(_secret_cache.get("value") or ""):
        return True
    try:
        return _token_path().exists()
    except OSError:
        return False


def token_source() -> str:
    """De dónde sale el token: para poder decírselo al usuario sin exponerlo."""
    if (os.environ.get("STUDIO_CHANNEL_GH_TOKEN") or "").strip():
        return "entorno"
    cached = str(_secret_cache.get("value") or "")
    if cached:
        return "colab_secrets"
    if _token_path().exists():
        return "disco_local"
    return "ninguna"


def load_saved_token() -> str:
    """Token de GitHub, por orden de preferencia: entorno → Colab Secrets → disco.

    Colab Secrets es lo que hace que esto sea «una vez en la vida»: el secreto vive
    en la cuenta de Google del usuario, no en la máquina efímera de Colab.
    """
    env = (os.environ.get("STUDIO_CHANNEL_GH_TOKEN") or "").strip()
    if env:
        return env
    secret = _from_colab_secrets(force=True)
    if secret:
        return secret
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


def _ledger_path() -> Path:
    return _token_path().with_name("github_channel_executed.json")


def _load_ledger() -> set:
    try:
        return set(json.loads(_ledger_path().read_text(encoding="utf-8")).get("ids") or [])
    except (OSError, ValueError):
        return set()


def _save_ledger(ids: set) -> None:
    try:
        _ledger_path().write_text(json.dumps({"ids": sorted(ids)[-2000:]}), encoding="utf-8")
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


def state_branch() -> str:
    return (os.environ.get("STUDIO_CHANNEL_STATE_BRANCH") or STATE_BRANCH_DEFAULT).strip()


def state_path() -> str:
    return (os.environ.get("STUDIO_CHANNEL_STATE_PATH") or STATE_PATH_DEFAULT).strip()


def repo_slug() -> str:
    return (os.environ.get("STUDIO_CHANNEL_REPO") or "zurdonman/Studio-ai").strip()


def _api(url: str, token: str, method: str = "GET",
         payload: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """GET/PUT/DELETE a api.github.com. Sin token funciona en repos públicos."""
    import urllib.error
    import urllib.request

    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("User-Agent", "studio-ia-channel/2.0")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            body = resp.read().decode("utf-8", "replace")
        return json.loads(body) if body.strip() else {}
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        if exc.code in (401, 403):
            return {"__http_error__": exc.code}
        raise
    except (urllib.error.URLError, OSError):
        return None


def read_state(token: str = "") -> Dict[str, Any]:
    """Lee el estado del canal desde el repo (anónimo si no hay token)."""
    import base64

    url = (f"https://api.github.com/repos/{repo_slug()}/contents/{state_path()}"
           f"?ref={state_branch()}")
    data = _api(url, token)
    if not isinstance(data, dict) or "content" not in data:
        return {}
    try:
        return json.loads(base64.b64decode(data["content"]).decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return {}


def write_state(body: Dict[str, Any], token: str = "") -> Optional[str]:
    """Publica el estado en la rama del canal. Sin token no se puede escribir."""
    import base64

    if not token:
        return None
    url = f"https://api.github.com/repos/{repo_slug()}/contents/{state_path()}"
    existing = _api(f"{url}?ref={state_branch()}", token) or {}
    payload: Dict[str, Any] = {
        "message": f"canal: estado {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        "content": base64.b64encode(json.dumps(body, ensure_ascii=False, indent=2).encode()).decode(),
        "branch": state_branch(),
    }
    if isinstance(existing, dict) and existing.get("sha"):
        payload["sha"] = existing["sha"]
    out = _api(url, token, "PUT", payload)
    return (out or {}).get("commit", {}).get("sha")


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


def _loop(interval: float, repo: str, branch: str, backend: str, _unused: str,
          base: str, mcp_token: str, orders: str, results: str) -> None:
    ch = _channel_module()
    counted: set = set()
    ledger = _load_ledger()
    try:
        publish_arena_link(base=base, token=load_saved_token())
    except Exception:
        pass

    def _on_result(entry: Dict[str, Any]) -> None:
        oid = str(entry.get("id") or "")
        with _lock:
            if not oid or oid in counted:
                return
            counted.add(oid)
            ledger.add(oid)
            _save_ledger(ledger)
            _state["executed"] += 1
            _state["last_verb"] = f"{entry.get('verb')} · {'ok' if entry.get('ok') else 'fallo'}"

    while not _stop.is_set():
        try:
            with _lock:
                _state["last_poll"] = time.time()
            token = load_saved_token()
            if not token:
                # Sin token NO se ejecuta nada: no se podrían publicar los resultados y
                # las órdenes se repetirían en cada sondeo (gastando créditos por nada).
                pending = _pending_orders()
                with _lock:
                    _state["polls"] += 1
                    _state["pending"] = pending
                    _state["last_error"] = (
                        f"{pending} orden(es) esperando. Sin token de GitHub no se publican "
                        "resultados, y por seguridad no se ejecuta nada." if pending
                        else "Solo lectura: pega un token de GitHub para ejecutar órdenes."
                    )
                _stop.wait(interval)
                continue
            ch.run_once(repo, orders, results, token, False, base=base,
                        mcp_token=mcp_token, branch=branch, backend=backend,
                        state_branch=state_branch(), state_path=state_path(),
                        on_result=_on_result)
            with _lock:
                _state["polls"] += 1
                _state["last_error"] = ""
        except Exception as exc:  # el canal nunca debe tumbar el servidor
            with _lock:
                _state["last_error"] = f"{type(exc).__name__}: {exc}"[:300]
        _stop.wait(interval)


ARENA_LINK_PATH = "channel/arena_link.json"


def publish_arena_link(base: str = "", token: str = "") -> Optional[str]:
    """Publica (una vez por arranque) el enlace que se le da a arena.ai.

    Va en la rama del canal, no en la de código. El enlace es estable, así que la
    URL que el usuario copió en la primera sesión sigue sirviendo siempre.
    """
    import base64

    token = token or load_saved_token()
    if not token:
        return None
    body = {
        "schema": 1,
        "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "arena_url": (f"{base.rstrip('/')}/mcp/arena" if base else "/mcp/arena"),
        "note": ("URL estable del canal. La instancia cambia de túnel en cada sesión, pero el buzón "
                 "de GitHub no: el agente siempre empieza por aquí."),
        "channel": agent_handoff("", repo_slug(), "", "state"),
    }
    url = (f"https://api.github.com/repos/{repo_slug()}/contents/{ARENA_LINK_PATH}")
    existing = _api(f"{url}?ref={state_branch()}", token) or {}
    payload: Dict[str, Any] = {
        "message": "canal: enlace de arena",
        "content": base64.b64encode(json.dumps(body, indent=2, ensure_ascii=False).encode()).decode(),
        "branch": state_branch(),
    }
    if isinstance(existing, dict) and existing.get("sha"):
        payload["sha"] = existing["sha"]
    out = _api(url, token, "PUT", payload)
    return (out or {}).get("commit", {}).get("sha")


def _pending_orders() -> int:
    try:
        agent = (read_state(load_saved_token()) or {}).get("agent") or {}
        return len(agent.get("orders") or [])
    except Exception:
        return 0


def start(gh_token: str = "", gist_id: str = "", backend: str = "gist",
          repo: str = "zurdonman/Studio-ai", branch: str = "arena/f0997b49-studio-ai",
          interval: float = 20.0, base: str = "", mcp_token: str = "",
          orders: str = "channel/ordenes.json",
          results: str = "state/channel/resultados.json") -> Dict[str, Any]:
    """Enciende el canal. Idempotente: si ya está activo, devuelve el estado actual."""
    global _state
    with _lock:
        if _state["running"]:
            return {**snapshot(), "ok": True}
    token = (gh_token or "").strip() or load_saved_token()
    if token:
        save_token(token)
    backend = "repo" if backend == "repo" else "state"

    _stop.clear()
    with _lock:
        _state.update({"running": True, "backend": backend, "gist_id": "", "repo": repo,
                       "branch": branch, "started_at": time.time(), "last_error": "",
                       "polls": 0, "executed": 0})
    thread = threading.Thread(
        target=_loop,
        args=(max(5.0, float(interval)), repo, branch, backend, "",
              base or f"http://127.0.0.1:{os.environ.get('PORT', '7860')}", mcp_token, orders, results),
        name="github-channel",
        daemon=True,
    )
    _state["thread"] = thread
    _state["mode"] = "activo" if token else "solo_lectura"
    # El enlace estable se publica desde el hilo del canal, nunca dentro de la petición
    # HTTP: si GitHub tarda, la interfaz no debe esperar.

    thread.start()
    return {**snapshot(), "ok": True}


def probe(gh_token: str = "") -> Dict[str, Any]:
    """Valida el token y el buzón sin encender nada. Respuesta rápida para la UI.

    Devuelve qué puede hacer el token (leer, escribir, crear ramas) para que el
    panel pueda decir «falta permiso X» en vez de quedarse en «comprobando».
    """
    token = (gh_token or "").strip() or load_saved_token()
    if not token:
        return {"ok": False, "error_code": "NO_GITHUB_TOKEN",
                "error": "Sin token: solo se podrían leer órdenes, no publicar resultados."}
    repo = repo_slug()
    result: Dict[str, Any] = {"ok": True, "repo": repo, "branch": state_branch(), "can_write": False}
    me = _api("https://api.github.com/user", token)
    if me is None or (isinstance(me, dict) and me.get("__http_error__") == 401):
        return {"ok": False, "error_code": "TOKEN_INVALIDO",
                "error": "GitHub rechazó el token (401). Caducado, revocado o mal copiado."}
    result["user"] = (me or {}).get("login", "")
    meta = _api(f"https://api.github.com/repos/{repo}", token) or {}
    perms = (meta or {}).get("permissions") or {}
    result["can_write"] = bool(perms.get("push") or perms.get("maintain") or perms.get("admin"))
    if isinstance(meta, dict) and meta.get("__http_error__"):
        return {"ok": False, "error_code": "SIN_ACCESO_AL_REPO",
                "error": f"El token no tiene acceso a {repo} (HTTP {meta['__http_error__']})."}
    state = read_state(token)
    result["state_readable"] = bool(state)
    result["pending_orders"] = len(((state or {}).get("agent") or {}).get("orders") or [])
    if not result["can_write"]:
        result.update({
            "ok": False, "error_code": "SIN_PERMISO_ESCRITURA",
            "error": "El token no puede escribir en el repositorio: revisa Contents: read/write.",
        })
    return result


def autostart_if_possible(mcp_token: str = "") -> Dict[str, Any]:
    """Arranca el canal sin intervención si ya hay credenciales guardadas.

    Pensado para llamarse al iniciar el servidor: si el usuario configuró el secreto
    una vez, cada sesión de Colab queda conectada sin pulsar nada.
    """
    if (os.environ.get("STUDIO_CHANNEL_AUTOSTART") or "1").strip() == "0":
        return {"ok": False, "skipped": "autostart desactivado"}
    if not load_saved_token():
        return {"ok": False, "skipped": "sin credenciales guardadas"}
    return start(mcp_token=mcp_token)


def stop() -> Dict[str, Any]:
    _stop.set()
    with _lock:
        _state["running"] = False
        _state["thread"] = None
    return snapshot()


def snapshot() -> Dict[str, Any]:
    with _lock:
        return {k: v for k, v in _state.items() if k != "thread"}


def agent_handoff(_gist_id: str, repo: str, branch: str, backend: str) -> Dict[str, Any]:
    """Lo que hay que darle al agente: el punto de encuentro, nunca el token."""
    if backend != "repo":
        return {
            "backend": "state",
            "repo": repo,
            "state_branch": state_branch(),
            "state_path": state_path(),
            "read_url": (f"https://api.github.com/repos/{repo}/contents/{state_path()}"
                         f"?ref={state_branch()}"),
            "anonymous_read": True,
            "write_needs": "un token de GitHub con Contents: read/write sobre el repo",
            "how": ("GET a read_url (base64 → JSON) para leer `app.results`; "
                    "PUT a la misma ruta sin ?ref para publicar `agent.orders`."),
        }
    return {
        "backend": "repo",
        "repo": repo,
        "branch": branch,
        "orders_url": f"https://api.github.com/repos/{repo}/contents/channel/ordenes.json?ref={branch}",
        "results_url": f"https://api.github.com/repos/{repo}/contents/state/channel/resultados.json?ref={branch}",
        "how": "GET/PUT en la API de contenidos del repo (requiere repo público o token).",
    }
