"""Process-wide facts shared by the API, the MCP server and the UI.

Only process state lives here (e.g. whether the GPU frameworks were reloaded),
never weights, tokens or user files. Values are advisory diagnostics.
"""

from __future__ import annotations

import pathlib
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional

_LOCK = threading.Lock()
_STATE: Dict[str, Any] = {
    "server_pid": None,
    "rebuilt_modules": (),
    "rebuild_notice": "",
}
_BUILD_CACHE: Dict[str, Any] = {"ts": 0.0, "value": None}
_BUILD_TTL_SEC = 60.0


def build_identity() -> Dict[str, Any]:
    """Which code is actually running: commit, branch and start time.

    Read-only `git` inspection of this checkout, cached for a minute. It never
    fetches, checks out, or touches weights; without `.git` (a copied folder) it
    returns a build timestamp instead of failing.
    """
    now = time.monotonic()
    cached = _BUILD_CACHE.get("value")
    if cached is not None and now - float(_BUILD_CACHE.get("ts", 0.0)) < _BUILD_TTL_SEC:
        return dict(cached)
    root = pathlib.Path(__file__).resolve().parent.parent
    info: Dict[str, Any] = {
        "root": str(root),
        "commit": None,
        "commit_short": None,
        "commit_date": None,
        "branch": None,
        "dirty": None,
        "source": "unknown",
    }

    def _git(*args: str) -> Optional[str]:
        try:
            out = subprocess.run(
                ["git", "-C", str(root), *args],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    commit = _git("rev-parse", "HEAD")
    if commit:
        info["commit"] = commit
        info["commit_short"] = commit[:7]
        info["commit_date"] = _git("show", "-s", "--format=%cI", "HEAD")
        info["branch"] = _git("rev-parse", "--abbrev-ref", "HEAD")
        status = _git("status", "--porcelain")
        info["dirty"] = bool(status)
        info["source"] = "git"
    else:
        try:
            info["mtime"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime((root / "studio" / "mcp_server.py").stat().st_mtime))
        except OSError:
            info["mtime"] = None
        info["source"] = "copy"
    _BUILD_CACHE["value"] = dict(info)
    _BUILD_CACHE["ts"] = now
    return info


def mark_rebuilt(modules: List[str]) -> None:
    notice = (
        "Esta instancia cargó "
        + ", ".join(sorted(set(modules)))
        + " antes de aplicar los parches de compatibilidad de Studio IA; para no mezclar "
        "clases parcheadas y sin parchear debe reiniciarse el proceso del servidor "
        "(no basta con recargar el módulo de memoria). No hace falta volver a descargar pesos."
    )
    with _LOCK:
        _STATE["rebuilt_modules"] = tuple(sorted(set(modules)))
        _STATE["rebuild_notice"] = notice


def rebuilt_modules() -> List[str]:
    with _LOCK:
        return list(_STATE["rebuilt_modules"])


def set_server_pid(pid: int) -> None:
    with _LOCK:
        _STATE["server_pid"] = int(pid)


def server_pid() -> Any:
    with _LOCK:
        return _STATE["server_pid"]


def snapshot() -> Dict[str, Any]:
    with _LOCK:
        return {
            "server_pid": _STATE["server_pid"],
            "runtime_rebuilt_for_compatibility": bool(_STATE["rebuilt_modules"]),
            "runtime_rebuilt_modules": list(_STATE["rebuilt_modules"]),
            "runtime_rebuild_notice": _STATE["rebuild_notice"],
            "build": build_identity(),
        }
