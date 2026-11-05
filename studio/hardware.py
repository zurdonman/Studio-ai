import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional

import psutil

from studio.config import DATA_ROOT, STATE_DIR, is_colab_env

CREDITS_STATE_FILE = STATE_DIR / "credits_state.json"
SHUTDOWN_FLAG_FILE = STATE_DIR / "shutdown_requested.flag"

# Tarifas orientativas de unidades de cómputo por hora en Google Colab (editables por el usuario)
GPU_RATES_DEFAULT = {
    "T4": 1.17,
    "L4": 4.82,
    "V100": 5.36,
    "A100": 11.77,
    "H100": 22.50,
    "GPU": 4.50,
    "CPU": 0.0,
}


def _classify_gpu_tier(gpu_name: str) -> str:
    upper = (gpu_name or "").upper()
    if not upper or upper == "CPU":
        return "CPU"
    for key in ("H100", "A100", "L4", "T4", "V100"):
        if key in upper:
            return key
    return "GPU"


def query_gpu_info() -> Dict[str, Any]:
    """Obtiene información real de GPU y VRAM usando torch.cuda o nvidia-smi."""
    # Intento 1: torch si ya está importado o disponible
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            idx = 0
            props = torch.cuda.get_device_properties(idx)
            name = torch.cuda.get_device_name(idx)
            total_gb = props.total_memory / (1024**3)
            alloc_gb = torch.cuda.memory_allocated(idx) / (1024**3)
            reserved_gb = torch.cuda.memory_reserved(idx) / (1024**3)
            # Intentar leer el uso total real del sistema por nvidia-smi (incluye subprocesos)
            smi = _query_nvidia_smi()
            used_gb = smi["vram_used_gb"] if smi else max(alloc_gb, reserved_gb)
            tier = _classify_gpu_tier(name)
            return {
                "has_gpu": True,
                "gpu_name": name,
                "gpu_tier": tier,
                "gpu_count": torch.cuda.device_count(),
                "vram_total_gb": round(total_gb, 2),
                "vram_used_gb": round(used_gb, 2),
                "vram_allocated_gb": round(alloc_gb, 2),
                "vram_reserved_gb": round(reserved_gb, 2),
                "vram_free_gb": round(max(0.0, total_gb - used_gb), 2),
                "vram_percent": round((used_gb / total_gb) * 100, 1) if total_gb > 0 else 0.0,
            }
    except Exception:
        pass

    # Intento 2: nvidia-smi directo
    smi = _query_nvidia_smi()
    if smi:
        return smi

    return {
        "has_gpu": False,
        "gpu_name": "CPU (Sin GPU activa)",
        "gpu_tier": "CPU",
        "gpu_count": 0,
        "vram_total_gb": 0.0,
        "vram_used_gb": 0.0,
        "vram_allocated_gb": 0.0,
        "vram_reserved_gb": 0.0,
        "vram_free_gb": 0.0,
        "vram_percent": 0.0,
    }


def _query_nvidia_smi() -> Optional[Dict[str, Any]]:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        res = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if res.returncode != 0 or not res.stdout.strip():
            return None
        lines = [ln.strip() for ln in res.stdout.strip().splitlines() if ln.strip()]
        parts = [p.strip() for p in lines[0].split(",")]
        if len(parts) < 4:
            return None
        name = parts[0]
        total_gb = float(parts[1]) / 1024.0
        used_gb = float(parts[2]) / 1024.0
        free_gb = float(parts[3]) / 1024.0
        tier = _classify_gpu_tier(name)
        return {
            "has_gpu": True,
            "gpu_name": name,
            "gpu_tier": tier,
            "gpu_count": len(lines),
            "vram_total_gb": round(total_gb, 2),
            "vram_used_gb": round(used_gb, 2),
            "vram_allocated_gb": round(used_gb, 2),
            "vram_reserved_gb": round(used_gb, 2),
            "vram_free_gb": round(free_gb, 2),
            "vram_percent": round((used_gb / total_gb) * 100, 1) if total_gb > 0 else 0.0,
        }
    except Exception:
        return None


class HardwareAndCreditsTracker:
    """Monitoriza GPU/RAM/Disco, créditos estimados de Colab e inactividad."""

    def __init__(self) -> None:
        self.session_start_ts = time.time()
        self.last_activity_ts = time.time()
        self._state = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        gpu = query_gpu_info()
        default_rate = GPU_RATES_DEFAULT.get(gpu["gpu_tier"], 0.0)
        defaults = {
            "initial_credits": 100.0,
            "hourly_rate": default_rate,
            "rate_auto": True,
            "prior_spent_credits": 0.0,
            "idle_alert_minutes": 10,
        }
        if CREDITS_STATE_FILE.exists():
            try:
                loaded = json.loads(CREDITS_STATE_FILE.read_text(encoding="utf-8"))
                defaults.update(loaded)
            except Exception:
                pass
        if defaults.get("rate_auto", True):
            defaults["hourly_rate"] = default_rate
        return defaults

    def _save_state(self) -> None:
        CREDITS_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        CREDITS_STATE_FILE.write_text(
            json.dumps(self._state, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def mark_activity(self) -> None:
        self.last_activity_ts = time.time()

    def update_credits_config(
        self,
        initial_credits: Optional[float] = None,
        hourly_rate: Optional[float] = None,
        reset_session_spent: bool = False,
        idle_alert_minutes: Optional[int] = None,
    ) -> Dict[str, Any]:
        if initial_credits is not None:
            self._state["initial_credits"] = max(0.0, float(initial_credits))
        if hourly_rate is not None:
            self._state["hourly_rate"] = max(0.0, float(hourly_rate))
            self._state["rate_auto"] = False
        if idle_alert_minutes is not None:
            self._state["idle_alert_minutes"] = max(1, int(idle_alert_minutes))
        if reset_session_spent:
            self.session_start_ts = time.time()
            self._state["prior_spent_credits"] = 0.0
        self._save_state()
        return self.get_snapshot()

    def get_snapshot(self, active_jobs_count: int = 0) -> Dict[str, Any]:
        now = time.time()
        if active_jobs_count > 0:
            self.last_activity_ts = now

        gpu = query_gpu_info()
        if self._state.get("rate_auto", True):
            self._state["hourly_rate"] = GPU_RATES_DEFAULT.get(gpu["gpu_tier"], 0.0)

        vm = psutil.virtual_memory()
        DATA_ROOT.mkdir(parents=True, exist_ok=True)
        disk = shutil.disk_usage(DATA_ROOT)

        uptime_sec = max(0.0, now - self.session_start_ts)
        idle_sec = max(0.0, now - self.last_activity_ts)
        idle_min = idle_sec / 60.0

        hourly_rate = float(self._state.get("hourly_rate", 0.0))
        session_spent = (uptime_sec / 3600.0) * hourly_rate
        total_spent = float(self._state.get("prior_spent_credits", 0.0)) + session_spent
        initial_credits = float(self._state.get("initial_credits", 100.0))
        remaining = max(0.0, initial_credits - total_spent)
        hours_left = (remaining / hourly_rate) if hourly_rate > 0 else None

        idle_threshold = int(self._state.get("idle_alert_minutes", 10))
        idle_warning = bool(
            (gpu["has_gpu"] or hourly_rate > 0)
            and active_jobs_count == 0
            and idle_min >= idle_threshold
        )

        return {
            "is_colab": is_colab_env(),
            "ephemeral_disk_notice": (
                "El disco de sesión de Colab se borra al desconectar. "
                "Usa «Descargar Todo (.zip)» antes de apagar la GPU."
            ),
            "gpu": gpu,
            "ram": {
                "total_gb": round(vm.total / (1024**3), 2),
                "used_gb": round((vm.total - vm.available) / (1024**3), 2),
                "available_gb": round(vm.available / (1024**3), 2),
                "percent": round(vm.percent, 1),
            },
            "disk": {
                "path": str(DATA_ROOT),
                "total_gb": round(disk.total / (1024**3), 2),
                "used_gb": round(disk.used / (1024**3), 2),
                "free_gb": round(disk.free / (1024**3), 2),
                "percent": round((disk.used / disk.total) * 100, 1) if disk.total > 0 else 0.0,
            },
            "credits": {
                "initial_credits": round(initial_credits, 2),
                "hourly_rate": round(hourly_rate, 2),
                "rate_auto": bool(self._state.get("rate_auto", True)),
                "session_spent": round(session_spent, 3),
                "remaining_credits": round(remaining, 2),
                "estimated_hours_left": round(hours_left, 2) if hours_left is not None else None,
                "uptime_seconds": int(uptime_sec),
                "idle_seconds": int(idle_sec),
                "idle_minutes": round(idle_min, 1),
                "idle_alert_minutes": idle_threshold,
                "idle_warning": idle_warning,
                "gpu_rates_reference": GPU_RATES_DEFAULT,
            },
        }


def request_colab_shutdown() -> Dict[str, Any]:
    """Solicita liberar y desconectar el entorno de ejecución de Google Colab."""
    SHUTDOWN_FLAG_FILE.parent.mkdir(parents=True, exist_ok=True)
    SHUTDOWN_FLAG_FILE.write_text(str(time.time()), encoding="utf-8")

    # Si estamos dentro del propio proceso de Colab
    try:
        from google.colab import runtime  # type: ignore

        runtime.unassign()
        return {
            "ok": True,
            "colab_disconnected": True,
            "message": "Entorno de Google Colab desconectado. La facturación de GPU se ha detenido.",
        }
    except Exception:
        pass

    if is_colab_env():
        # En Colab como subproceso: launch_colab.py detecta SHUTDOWN_FLAG_FILE y llama a runtime.unassign()
        return {
            "ok": True,
            "colab_disconnected": True,
            "message": "Señal de apagado enviada al lanzador de Colab. La sesión y la GPU se liberarán en unos segundos.",
        }

    return {
        "ok": True,
        "colab_disconnected": False,
        "message": "Memoria RAM/VRAM liberada. (Estás en entorno local/preview, no hay máquina de Colab facturando).",
    }


hardware_tracker = HardwareAndCreditsTracker()
