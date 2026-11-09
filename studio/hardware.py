import json
import shutil
import subprocess
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from studio.config import DATA_ROOT, STATE_DIR, is_colab_env
from studio.diagnostics import redact_secrets
from studio.memory_policy import host_memory

CREDITS_STATE_FILE = STATE_DIR / "credits_state.json"
SHUTDOWN_FLAG_FILE = STATE_DIR / "shutdown_requested.flag"

_SESSION_BOOT_DT = datetime.now()
_SESSION_LOG_LOCK = threading.Lock()
_SESSION_LOG_MAX_LINES = 40000
_SESSION_LOG_LINES: List[str] = [
    f"[{_SESSION_BOOT_DT.strftime('%Y-%m-%d %H:%M:%S')}] 🎬 Servidor Studio IA encendido (inicio de registro de sesión)."
]

# Tarifas reales de unidades de cómputo por hora en Google Colab (editables en 1 clic desde la interfaz)
GPU_RATES_DEFAULT = {
    "T4": 1.17,
    "L4": 1.92,
    "V100": 3.90,
    "A100": 5.37,
    "H100": 11.77,
    "GPU": 1.92,
    "CPU": 0.0,
}

# Antiguos valores por defecto para migrar automáticamente si quedaron guardados
_LEGACY_AUTO_RATES = {4.82, 4.50}


def colab_console_log(message: str, icon: str = "🔹") -> None:
    """Imprime una línea inmediata en Google Colab y la archiva en el registro íntegro de sesión."""
    message = redact_secrets(message)
    now = datetime.now()
    ts_short = now.strftime("%H:%M:%S")
    ts_full = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts_short}] {icon} {message}", flush=True)

    full_line = f"[{ts_full}] {icon} {message}"
    with _SESSION_LOG_LOCK:
        _SESSION_LOG_LINES.append(full_line)
        if len(_SESSION_LOG_LINES) > _SESSION_LOG_MAX_LINES:
            del _SESSION_LOG_LINES[: len(_SESSION_LOG_LINES) - _SESSION_LOG_MAX_LINES]
        try:
            STATE_DIR.mkdir(parents=True, exist_ok=True)
            with open(STATE_DIR / "session_console.log", "a", encoding="utf-8") as fh:
                fh.write(full_line + "\n")
        except Exception:
            pass


def get_session_console_lines() -> List[str]:
    with _SESSION_LOG_LOCK:
        return list(_SESSION_LOG_LINES)


def _classify_gpu_tier(gpu_name: str) -> str:
    upper = (gpu_name or "").upper()
    if not upper or upper == "CPU":
        return "CPU"
    for key in ("H100", "A100", "L4", "T4", "V100"):
        if key in upper:
            return key
    return "GPU"


def query_gpu_info() -> Dict[str, Any]:
    """Obtiene información real e instantánea de GPU y VRAM usando cudaMemGetInfo o nvidia-smi."""
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            idx = 0
            props = torch.cuda.get_device_properties(idx)
            name = torch.cuda.get_device_name(idx)
            total_gb = props.total_memory / (1024**3)
            alloc_gb = torch.cuda.memory_allocated(idx) / (1024**3)
            reserved_gb = torch.cuda.memory_reserved(idx) / (1024**3)

            # mem_get_info consulta directamente al driver CUDA (incluye pesos 4-bit bitsandbytes y contexto)
            try:
                free_b, total_b = torch.cuda.mem_get_info(idx)
                total_gb = total_b / (1024**3)
                free_gb = free_b / (1024**3)
                used_gb = max(0.0, total_gb - free_gb)
                measurement_source = "cuda_driver_mem_get_info"
            except Exception:
                # PyTorch's own allocator excludes ONNX/bnb/external processes.
                # Never mistake its unused share for real device headroom.
                smi = _query_nvidia_smi()
                if smi:
                    return {**smi, "vram_allocated_gb": round(alloc_gb, 2),
                            "vram_reserved_gb": round(reserved_gb, 2)}
                used_gb = max(alloc_gb, reserved_gb)
                free_gb = None
                measurement_source = "allocator_only_device_free_unknown"

            tier = _classify_gpu_tier(name)
            return {
                "has_gpu": True,
                "gpu_name": name,
                "gpu_tier": tier,
                "gpu_count": torch.cuda.device_count(),
                "memory_scope": "device_0_not_sum",
                "measurement_source": measurement_source,
                "compute_capability": f"{props.major}.{props.minor}" if hasattr(props, "major") else None,
                "bf16_supported": props.major >= 8 if hasattr(props, "major") else None,
                "vram_total_gb": round(total_gb, 2),
                "vram_used_gb": round(used_gb, 2),
                "vram_allocated_gb": round(alloc_gb, 2),
                "vram_reserved_gb": round(reserved_gb, 2),
                "vram_free_gb": round(free_gb, 2) if free_gb is not None else None,
                "vram_percent": round((used_gb / total_gb) * 100, 1) if total_gb > 0 else 0.0,
            }
    except Exception:
        pass

    smi = _query_nvidia_smi()
    if smi:
        return smi

    return {
        "has_gpu": False,
        "gpu_name": "CPU (Sin GPU activa)",
        "gpu_tier": "CPU",
        "gpu_count": 0,
        "memory_scope": "device_0_not_sum",
        "compute_capability": None,
        "bf16_supported": False,
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
            "memory_scope": "device_0_not_sum",
            "measurement_source": "nvidia_smi_driver",
            "compute_capability": None,
            "bf16_supported": True if tier in ("L4", "A100", "H100") else False if tier in ("T4", "V100") else None,
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
                # Si tenía guardada la tarifa antigua de L4 (4.82), actualizar a 1.92
                if gpu["gpu_tier"] == "L4" and round(float(defaults.get("hourly_rate", 0)), 2) in _LEGACY_AUTO_RATES:
                    defaults["hourly_rate"] = 1.92
                    defaults["rate_auto"] = True
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
        use_auto_gpu_rate: bool = False,
        reset_session_spent: bool = False,
        idle_alert_minutes: Optional[int] = None,
    ) -> Dict[str, Any]:
        gpu = query_gpu_info()
        if initial_credits is not None:
            self._state["initial_credits"] = max(0.0, float(initial_credits))
        if use_auto_gpu_rate:
            self._state["rate_auto"] = True
            self._state["hourly_rate"] = GPU_RATES_DEFAULT.get(gpu["gpu_tier"], 0.0)
        elif hourly_rate is not None:
            self._state["hourly_rate"] = max(0.0, float(hourly_rate))
            self._state["rate_auto"] = False
        if idle_alert_minutes is not None:
            self._state["idle_alert_minutes"] = max(1, int(idle_alert_minutes))
        if reset_session_spent:
            self.session_start_ts = time.time()
            self._state["prior_spent_credits"] = 0.0
        self._save_state()
        colab_console_log(
            f"Contador de créditos actualizado: saldo inicial={self._state['initial_credits']} u., "
            f"consumo={self._state['hourly_rate']} u/h ({gpu['gpu_tier']}).",
            icon="💳",
        )
        return self.get_snapshot()

    def get_snapshot(self, active_jobs_count: int = 0) -> Dict[str, Any]:
        now = time.time()
        if active_jobs_count > 0:
            self.last_activity_ts = now

        gpu = query_gpu_info()
        detected_default_rate = GPU_RATES_DEFAULT.get(gpu["gpu_tier"], 0.0)
        if self._state.get("rate_auto", True):
            self._state["hourly_rate"] = detected_default_rate

        effective_ram = host_memory()
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
                **{k: round(v, 2) if isinstance(v, float) else v for k, v in effective_ram.items()},
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
                "detected_gpu_default_rate": round(detected_default_rate, 2),
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
    colab_console_log("Solicitud de «Liberar GPU y apagar sesión» recibida desde la interfaz.", icon="🛑")
    SHUTDOWN_FLAG_FILE.parent.mkdir(parents=True, exist_ok=True)
    SHUTDOWN_FLAG_FILE.write_text(str(time.time()), encoding="utf-8")

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


def build_full_session_log_document(
    jobs_list: List[Dict[str, Any]],
    loaded_summary: List[Dict[str, Any]],
    downloaded_models: List[Dict[str, Any]],
    fmt: str = "md",
) -> Tuple[str, str]:
    """
    Construye el informe completo de diagnóstico desde el encendido de Studio IA hasta el instante actual.
    Devuelve `(filename, content_str)` en formato `.md` o `.txt`, ideal para compartir con una IA sin límite de portapapeles.
    """
    now = datetime.now()
    ext = "txt" if str(fmt).lower() == "txt" else "md"
    filename = f"studio_ia_log_{now.strftime('%Y%m%d_%H%M%S')}.{ext}"

    hw = hardware_tracker.get_snapshot(active_jobs_count=sum(1 for j in jobs_list if j.get("status") == "running"))
    gpu = hw["gpu"]
    ram = hw["ram"]
    disk = hw["disk"]
    cred = hw["credits"]
    console_lines = get_session_console_lines()
    uptime_s = max(0, int((now - _SESSION_BOOT_DT).total_seconds()))
    up_h, rem = divmod(uptime_s, 3600)
    up_m, up_sec = divmod(rem, 60)

    # Ordenar trabajos del más antiguo al más reciente para lectura cronológica natural
    jobs_chrono = sorted(jobs_list, key=lambda x: float(x.get("created_ts") or x.get("created_at") or 0.0))

    lines: List[str] = []
    if ext == "md":
        lines.append("# 📄 Registro Completo de Sesión — Studio IA")
        lines.append("")
        lines.append("## 1. Estado del Sistema y Entorno")
        lines.append(f"- **Encendido de la aplicación:** `{_SESSION_BOOT_DT.strftime('%Y-%m-%d %H:%M:%S')}`")
        lines.append(f"- **Exportado en:** `{now.strftime('%Y-%m-%d %H:%M:%S')}` (Tiempo activo: `{up_h}h {up_m}m {up_sec}s`)")
        lines.append(f"- **Entorno:** `{'Google Colab' if hw['is_colab'] else 'Local / Sandbox'}`")
        lines.append(
            f"- **GPU:** `{gpu.get('gpu_name', 'CPU')}` · VRAM usada: `{gpu.get('vram_used_gb', 0)} / {gpu.get('vram_total_gb', 0)} GB` (`{gpu.get('vram_free_gb', 0)} GB` libres)"
        )
        lines.append(f"- **RAM Sistema:** `{ram['used_gb']} / {ram['total_gb']} GB` (`{ram['percent']}%`)")
        lines.append(f"- **Disco (`{disk['path']}`):** Usado `{disk['used_gb']} GB` · Libre `{disk['free_gb']} GB` de `{disk['total_gb']} GB`")
        lines.append(
            f"- **Créditos Colab estimados:** Gastados `{cred['session_spent']} u` · Restantes `{cred['remaining_credits']} u` (tarifa `{cred['hourly_rate']} u/h`)"
        )
        lines.append(
            "- **Modelos cargados ahora en RAM/VRAM:** "
            + (
                ", ".join(f"`{m.get('model_id')}` ({m.get('slot')})" for m in loaded_summary)
                if loaded_summary
                else "_Ninguno cargado actualmente_"
            )
        )
        lines.append(
            "- **Modelos descargados en disco:** "
            + (
                ", ".join(f"`{m.get('id')}` ({m.get('disk_gb')} GB)" for m in downloaded_models)
                if downloaded_models
                else "_Ninguno descargado en disco_"
            )
        )
        lines.append("")
        lines.append(f"## 2. Resumen de Tareas Ejecutadas ({len(jobs_chrono)} en total)")
        if not jobs_chrono:
            lines.append("_Aún no se ha lanzado ninguna tarea pesada en esta sesión._")
        else:
            lines.append("| # | Job ID | Estado | Duración | Tarea | Resultado / Error |")
            lines.append("|---|---|---|---|---|---|")
            for idx, j in enumerate(jobs_chrono, 1):
                st = j.get("status", "?")
                icon = {"completed": "✅", "failed": "❌", "cancelled": "🛑", "running": "⏳"}.get(st, "🔹")
                el = f"{j.get('elapsed_seconds', 0)}s"
                res_obj = j.get("result") or {}
                err_str = j.get("error") or ""
                outcome = err_str[:120] if err_str else (res_obj.get("rel_path") or res_obj.get("verdict") or "OK")
                title_clean = str(j.get("title") or "").replace("|", "/")
                outcome_clean = str(outcome).replace("|", "/").replace("\n", " ")
                lines.append(f"| {idx} | `{j.get('id')}` | {icon} `{st}` | `{el}` | {title_clean} | `{outcome_clean}` |")
        lines.append("")
        lines.append("## 3. Detalle Paso a Paso por Tarea (Logs Internos y Errores)")
        if not jobs_chrono:
            lines.append("_Sin tareas registradas._")
        else:
            for idx, j in enumerate(jobs_chrono, 1):
                lines.append(f"### Tarea {idx}: `{j.get('id')}` — {j.get('title')} (`{j.get('status')}`, {j.get('elapsed_seconds', 0)}s)")
                if j.get("error"):
                    lines.append(f"- **Error detectado:** `{j.get('error')}`")
                if j.get("result"):
                    lines.append(f"- **Resultado JSON:** `{json.dumps(j.get('result'), ensure_ascii=False)[:600]}`")
                j_logs = j.get("logs") or []
                if j_logs:
                    lines.append("```text")
                    for lg in j_logs:
                        lines.append(f"[{lg.get('time', '')}] ({lg.get('level', 'info')}) {lg.get('message', '')}")
                    lines.append("```")
                lines.append("")
        lines.append(f"## 4. Registro Cronológico Íntegro de Consola desde el Arranque ({len(console_lines)} líneas)")
        lines.append("```text")
        lines.extend(console_lines)
        lines.append("```")
    else:
        lines.append("======================================================================")
        lines.append("REGISTRO COMPLETO DE SESIÓN — STUDIO IA")
        lines.append("======================================================================")
        lines.append(f"Encendido: {_SESSION_BOOT_DT.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Exportado: {now.strftime('%Y-%m-%d %H:%M:%S')} (Activo: {up_h}h {up_m}m {up_sec}s)")
        lines.append(f"GPU: {gpu.get('gpu_name', 'CPU')} | VRAM: {gpu.get('vram_used_gb', 0)}/{gpu.get('vram_total_gb', 0)} GB")
        lines.append(f"RAM: {ram['used_gb']}/{ram['total_gb']} GB | Disco libre: {disk['free_gb']} GB")
        lines.append("")
        lines.append(f"--- TAREAS EJECUTADAS ({len(jobs_chrono)}) ---")
        for idx, j in enumerate(jobs_chrono, 1):
            lines.append(f"[{idx}] {j.get('id')} | {j.get('status')} | {j.get('elapsed_seconds', 0)}s | {j.get('title')}")
            if j.get("error"):
                lines.append(f"    ERROR: {j.get('error')}")
            for lg in j.get("logs") or []:
                lines.append(f"    [{lg.get('time', '')}] {lg.get('message', '')}")
        lines.append("")
        lines.append(f"--- CONSOLA CRONOLÓGICA COMPLETA DESDE EL ENCENDIDO ({len(console_lines)} líneas) ---")
        lines.extend(console_lines)

    return filename, redact_secrets("\n".join(lines)) + "\n"
