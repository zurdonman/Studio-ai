"""Process-wide facts shared by the API, the MCP server and the UI.

Only process state lives here (e.g. whether the GPU frameworks were reloaded),
never weights, tokens or user files. Values are advisory diagnostics.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List

_LOCK = threading.Lock()
_STATE: Dict[str, Any] = {
    "server_pid": None,
    "rebuilt_modules": (),
    "rebuild_notice": "",
}


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
        }
