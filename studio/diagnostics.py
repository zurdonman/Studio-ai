"""Local, bounded crash diagnostics. Standard library only; never imports CUDA.

The Colab parent writes these even if the app is SIGKILLed. A destroyed VM cannot
be recovered by a local logger: download the bundle or copy it to durable storage.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import re
import signal
import subprocess
import threading
import time
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MAX_LOG_BYTES = 8 * 1024 * 1024
MAX_BUNDLE_BYTES = 32 * 1024 * 1024


def redact_secrets(text: str) -> str:
    text = re.sub(r"\b(?:st_mcp_[A-Za-z0-9_-]+|hf_[A-Za-z0-9]{8,})\b", "[REDACTED]", str(text))
    text = re.sub(r"(?i)([?&](?:token|access_token|api_key)=)[^\s&#\"'<>]+", r"\1[REDACTED]", text)
    text = re.sub(r"(?i)(\bbearer\s+)[A-Za-z0-9._~+/-]+", r"\1[REDACTED]", text)
    # Also cover nonstandard user-supplied token formats in structured reports.
    text = re.sub(r'(?i)([\"\'](?:token|hf_token|mcp_token|authorization|api_key)[\"\']\s*:\s*[\"\'])[^\"\']*', r'\1[REDACTED]', text)
    for name in ("STUDIO_MCP_TOKEN", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        value = os.environ.get(name, "").strip()
        if len(value) >= 8:
            text = text.replace(value, "[REDACTED]")
    return text


def data_root() -> Path:
    if os.environ.get("STUDIO_DATA_DIR"):
        return Path(os.environ["STUDIO_DATA_DIR"]).expanduser().resolve()
    colab = Path("/var/colab").exists() or os.environ.get("COLAB_RELEASE_TAG") or "COLAB_GPU" in os.environ
    if colab and os.environ.get("STUDIO_FORCE_LOCAL") != "1" and Path("/content").exists():
        return Path("/content/studio_data")
    return Path(__file__).resolve().parent.parent / "studio_data"


def _read_tail(path: Path, limit: int = MAX_LOG_BYTES) -> str:
    try:
        with path.open("rb") as fh:
            size = fh.seek(0, 2)
            fh.seek(max(0, size - limit))
            return fh.read(limit).decode("utf-8", errors="replace")
    except OSError:
        return ""


def _proc_memory(pid: int) -> dict[str, Any]:
    values: dict[str, Any] = {"pid": pid}
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            key, _, value = line.partition(":")
            if key in ("VmRSS", "VmHWM", "VmSize"):
                values[key + "_kib"] = int(value.split()[0])
    except (OSError, ValueError, IndexError):
        values["unavailable"] = True
    return values


def cgroup_memory_snapshot() -> dict[str, Any]:
    """Read the process quota and visible ancestors without optional frameworks.

    Containers can be limited by a parent cgroup, not only their leaf. Keep
    clean inactive page-cache facts separate from the hard quota headroom.
    """
    root = Path("/sys/fs/cgroup")
    candidates = [root]
    legacy = [root / "memory"]
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            _, controllers, relative = line.split(":", 2)
            rel = Path(relative.lstrip("/"))
            if ".." in rel.parts:
                continue
            if not controllers:
                leaf = root / rel
                candidates = [leaf, *(p for p in leaf.parents if p != root and root in p.parents), root]
            elif "memory" in controllers.split(","):
                memory_root = root / "memory"
                leaf = memory_root / rel
                legacy = [leaf, *(p for p in leaf.parents if p != memory_root and memory_root in p.parents), memory_root]
    except (OSError, ValueError):
        pass

    def read(folder: Path, version: int) -> dict[str, Any] | None:
        names = {"memory.max": "memory.max", "memory.current": "memory.current", "memory.peak": "memory.peak"}
        if version == 1:
            names = {"memory.max": "memory.limit_in_bytes", "memory.current": "memory.usage_in_bytes", "memory.peak": "memory.max_usage_in_bytes"}
        record: dict[str, Any] = {"path": str(folder), "version": version}
        try:
            for key, filename in names.items():
                try:
                    value = (folder / filename).read_text().strip()
                    record[key] = int(value) if value.isdigit() else value
                except OSError:
                    record[key] = "unknown"
            if record["memory.max"] == "unknown" or record["memory.current"] == "unknown":
                return None
            for filename, key in (("memory.stat", "stat"), ("memory.events", "events")):
                try:
                    record[key] = {k: int(v) for k, v in (line.split() for line in (folder / filename).read_text().splitlines())}
                except (OSError, ValueError):
                    record[key] = {}
            return record
        except (OSError, ValueError):
            return None

    records = [record for folder in dict.fromkeys(candidates) if (record := read(folder, 2))]
    if not records:
        records = [record for folder in dict.fromkeys(legacy) if (record := read(folder, 1))]
    if not records:
        return {}
    # Keep the tightest numeric ceiling as the primary diagnostic cgroup.
    primary = min(records, key=lambda record: record["memory.max"] if isinstance(record["memory.max"], int) else float("inf"))
    return {**primary, "ancestors": [record for record in records if record is not primary]}


def memory_snapshot(pid: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"ts": time.time(), "process": _proc_memory(pid or os.getpid())}
    try:
        info = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, value = line.partition(":")
            if key in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
                info[key + "_kib"] = int(value.split()[0])
        result["host"] = info
    except (OSError, ValueError):
        pass
    result["cgroup"] = cgroup_memory_snapshot()
    return result


def describe_exit(returncode: int, before: dict | None = None, after: dict | None = None) -> dict[str, Any]:
    before_cg = (before or {}).get("cgroup", {})
    after_cg = (after or {}).get("cgroup", {})
    delta = None
    if before_cg.get("path") == after_cg.get("path") and before_cg.get("events") and after_cg.get("events"):
        delta = max(0, after_cg["events"].get("oom_kill", 0) - before_cg["events"].get("oom_kill", 0))
    signame = None
    if returncode < 0:
        try:
            signame = signal.Signals(-returncode).name
        except ValueError:
            signame = f"SIGNAL_{-returncode}"
    if signame == "SIGKILL":
        explanation = ("SIGKILL; el cgroup registró OOM-kill durante la sesión (no identifica por sí solo la víctima)."
                       if delta else "SIGKILL: posible falta de RAM o señal externa; OOM no confirmado.")
    elif returncode == 0:
        explanation = "Proceso terminado con código 0."
    else:
        explanation = f"Salida inesperada: {signame or returncode}; revisar server.log y faulthandler."
    return {"returncode": returncode, "signal": signame, "cgroup_oom_kill_delta": delta, "explanation": explanation}


class DiagnosticSession:
    def __init__(self, root: Path | None = None) -> None:
        root = root or data_root()
        self.folder = root / "state" / "diagnostics" / (datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6])
        self.folder.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.before = memory_snapshot()
        self.status: dict[str, Any] = {"started_ts": time.time(), "status": "starting", "folder": str(self.folder)}
        self.event("launcher_start", memory=self.before)

    def write(self, name: str, text: str) -> None:
        if name not in ("server.log", "tunnel.log", "events.jsonl", "telemetry.jsonl"):
            raise ValueError("Unknown diagnostic stream")
        try:
            path = self.folder / name
            payload = redact_secrets(text).encode("utf-8", errors="replace")[-MAX_LOG_BYTES:]
            with self._lock:
                if path.exists() and path.stat().st_size + len(payload) > MAX_LOG_BYTES:
                    path.replace(path.with_suffix(path.suffix + ".1"))
                with path.open("ab") as fh:
                    fh.write(payload)
        except OSError:
            # A full disk must not take down an otherwise healthy application.
            pass

    def event(self, event: str, **fields: Any) -> None:
        self.write("events.jsonl", json.dumps({"ts": time.time(), "event": event, **fields}, ensure_ascii=False) + "\n")
        self.status.update(last_event=event, updated_ts=time.time(), **fields)
        try:
            temp = self.folder / "status.tmp"
            temp.write_text(redact_secrets(json.dumps(self.status, ensure_ascii=False, indent=2)), encoding="utf-8")
            temp.replace(self.folder / "status.json")
        except OSError:
            pass

    def sample(self, server_pid: int, tunnel_pid: int | None = None) -> None:
        data = memory_snapshot(server_pid)
        if tunnel_pid:
            data["tunnel_process"] = _proc_memory(tunnel_pid)
        self.write("telemetry.jsonl", json.dumps(data) + "\n")


def _command(args: list[str]) -> str:
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=4)
        return redact_secrets(proc.stdout[:32000] or proc.stderr[:32000])
    except (OSError, subprocess.TimeoutExpired) as exc:
        return str(exc)


def export_diagnostic_bundle(root: Path | None = None, destination: Path | None = None) -> Path:
    """Collect without contacting the app, reading model weights or exposing tokens."""
    root = root or data_root()
    destination = destination or root / "temp" / ("studio_diagnostics_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + ".zip")
    destination.parent.mkdir(parents=True, exist_ok=True)
    diagnostics = root / "state" / "diagnostics"
    files = [root / "state" / "session_console.log", root / "state" / "memory_guardian.json"]
    if diagnostics.is_dir():
        files += sorted((p for p in diagnostics.rglob("*") if p.is_file() and not p.is_symlink()
                         and p.name in ("server.log", "server.log.1", "tunnel.log", "tunnel.log.1", "events.jsonl", "events.jsonl.1", "telemetry.jsonl", "telemetry.jsonl.1", "status.json")),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    repo = Path(__file__).resolve().parent.parent
    environment = {"python": os.sys.version, "memory": memory_snapshot(),
                   "git_commit": _command(["git", "-C", str(repo), "rev-parse", "HEAD"]).strip(),
                   "git_branch": _command(["git", "-C", str(repo), "branch", "--show-current"]).strip(),
                   "gpu": _command(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free", "--format=csv"]),
                   "packages": {d.metadata.get("Name", "unknown"): d.version for d in importlib.metadata.distributions()}}
    inventory = []
    outputs = root / "outputs"
    if outputs.is_dir():
        for path in outputs.rglob("*"):
            if len(inventory) >= 5000:
                break
            if path.is_file() and not path.is_symlink():
                try:
                    inventory.append({"rel_path": str(path.relative_to(root)), "bytes": path.stat().st_size})
                except OSError:
                    pass
    remaining = MAX_BUNDLE_BYTES
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("environment.json", redact_secrets(json.dumps(environment, ensure_ascii=False, indent=2)))
        archive.writestr("outputs_inventory.json", json.dumps(inventory, ensure_ascii=False, indent=2))
        archive.writestr("README.txt", "Logs locales con credenciales ocultas. No incluye pesos ni imágenes/vídeos.\n"
                          "502/530 no prueban OOM ni pérdida de VM. Consultar exit code, eventos y memoria.\n"
                          "Si Colab destruye la VM, el disco /content también se pierde. Descargar este ZIP antes.\n")
        for path in files:
            if remaining <= 0 or path.is_symlink() or not path.is_file():
                continue
            try:
                resolved = path.resolve()
                resolved.relative_to(root.resolve())
            except (OSError, ValueError):
                continue
            text = _read_tail(path, min(MAX_LOG_BYTES, remaining))
            payload = redact_secrets(text).encode("utf-8")[:remaining]
            archive.writestr(str(path.relative_to(root)), payload)
            remaining -= len(payload)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Exporta diagnóstico local aunque Studio o el túnel estén caídos.")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(export_diagnostic_bundle(args.data_dir, args.output))


if __name__ == "__main__":
    main()
