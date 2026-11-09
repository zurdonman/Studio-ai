import ctypes
import gc
import threading
import time
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

import psutil

from studio.hardware import colab_console_log, query_gpu_info

# Candado reentrante global para serializar cargas/descargas de modelos en GPU y evitar colisiones concurrentes
GPU_MODEL_OP_LOCK = threading.RLock()


def serialized_gpu_operation(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Don't let a prepare/unload job change a pipeline during another job's inference."""
    @wraps(fn)
    def _locked(*args: Any, **kwargs: Any) -> Any:
        job = kwargs.get("job")
        if job is None:
            with GPU_MODEL_OP_LOCK:
                return fn(*args, **kwargs)
        # Long model loads are normal. Don't label another job waiting for the
        # GPU as hung, and allow cancellation before it starts allocating.
        while not GPU_MODEL_OP_LOCK.acquire(timeout=1.0):
            job.check_cancelled()
            job.heartbeat(step="Esperando turno de GPU (otra carga/inferencia en curso)...")
        try:
            job.check_cancelled()
            return fn(*args, **kwargs)
        finally:
            GPU_MODEL_OP_LOCK.release()

    return _locked


def _release_model_handle(handle: Any) -> None:
    # Pipeline bundles are dictionaries; dropping only the slot leaves their GPU
    # objects alive in callers' slot snapshots. Do not recurse into shared `defn`.
    objects = [handle.get(key) for key in ("pipe", "model")] if isinstance(handle, dict) else [handle]
    for obj in objects:
        if obj is None:
            continue
        try:
            if callable(getattr(obj, "unload", None)):
                obj.unload()
            if callable(getattr(obj, "maybe_free_model_hooks", None)):
                obj.maybe_free_model_hooks()
        except Exception:
            pass
    if isinstance(handle, dict):
        handle.clear()


def flush_ram_and_vram() -> None:
    """Fuerza liberación real de objetos en Python, caché CUDA (VRAM) y arenas glibc (RAM Linux)."""
    gc.collect()
    gc.collect()
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        pass

    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass


def require_available_host_ram(required_gb: float, *, stage: str, job: Any = None) -> Dict[str, float]:
    """Re-measure after unloading, before native allocations that SIGKILL cannot catch.

    A planning guard, not a hard memory limit or a guarantee against OOM.
    """
    vm = psutil.virtual_memory()
    available = vm.available / (1024**3)
    total = vm.total / (1024**3)
    message = f"RAM antes de {stage}: {available:.2f}/{total:.2f} GiB libres; presupuesto estimado {required_gb:.2f} GiB."
    if available < required_gb:
        raise RuntimeError(
            f"HOST_RAM_BUDGET_EXCEEDED: {message} Carga detenida antes de reservar tensores. "
            "Libera memoria o usa explícitamente otra receta tras consultar el oráculo; los pesos en disco se conservan."
        )
    if job:
        job.log(message, step="Comprobando RAM disponible antes de cargar")
    return {"available_gb": available, "total_gb": total, "required_gb": required_gb}


def run_with_live_memory_progress(
    job: Any,
    start_pct: float,
    end_pct: float,
    step_label: str,
    fn: Callable[[], Any],
    expected_seconds: float = 45.0,
) -> Any:
    """
    Ejecuta una operación bloqueante de carga de pesos (ej. from_pretrained) emitiendo
    en vivo cada 1.5s el porcentaje gradual y el uso real de RAM y VRAM para que tanto
    la interfaz como la celda de Google Colab muestren actividad continua 0-100%.
    """
    if job is None:
        return fn()

    done = {"flag": False}
    t0 = time.time()

    def _watcher() -> None:
        while not done["flag"] and not job.cancel_requested:
            time.sleep(1.5)
            if done["flag"]:
                break
            elapsed = time.time() - t0
            # Avance asintótico suave entre start_pct y (end_pct - 1.5)
            ratio = 1.0 - (1.0 / (1.0 + elapsed / max(5.0, expected_seconds * 0.6)))
            cur_pct = start_pct + ratio * max(0.0, (end_pct - 1.5) - start_pct)
            vm = psutil.virtual_memory()
            ram_gb = (vm.total - vm.available) / (1024**3)
            gpu = query_gpu_info()
            mem_txt = (
                f"RAM {ram_gb:.1f} GB · VRAM {gpu['vram_used_gb']:.1f}/{gpu['vram_total_gb']:.1f} GB"
                if gpu["has_gpu"]
                else f"RAM {ram_gb:.1f} GB"
            )
            job.heartbeat(
                progress=round(cur_pct, 1),
                step=f"{step_label} ({int(elapsed)}s)",
                speed_text=mem_txt,
            )

    t = threading.Thread(target=_watcher, daemon=True)
    t.start()
    try:
        return fn()
    finally:
        done["flag"] = True


class RuntimeMemoryManager:
    """Gestiona qué modelos están cargados en RAM/VRAM y permite liberarlos sin borrar el disco."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # slot -> {"model_id": str, "name": str, "handle": Any, "adapters": List[str], "loaded_at": float}
        self._slots: Dict[str, Dict[str, Any]] = {}

    @serialized_gpu_operation
    def register_loaded(
        self,
        slot: str,
        model_id: str,
        name: str,
        handle: Any,
        adapters: Optional[List[str]] = None,
    ) -> None:
        with self._lock:
            old = self._slots.pop(slot, None)
        if old is not None:
            _release_model_handle(old.pop("handle", None))
            del old
            flush_ram_and_vram()
        with self._lock:
            self._slots[slot] = {
                "slot": slot,
                "model_id": model_id,
                "name": name,
                "handle": handle,
                "adapters": list(adapters or []),
                "loaded_at": time.time(),
            }

    def get_slot(self, slot: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._slots.get(slot)

    def loaded_model_ids(self) -> List[str]:
        with self._lock:
            ids: List[str] = []
            for entry in self._slots.values():
                ids.append(entry["model_id"])
                for ad in entry.get("adapters", []):
                    ids.append(ad)
            return ids

    def status_summary(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [
                {
                    "slot": v["slot"],
                    "model_id": v["model_id"],
                    "name": v["name"],
                    "adapters": v.get("adapters", []),
                    "loaded_at": round(v["loaded_at"], 2),
                }
                for v in self._slots.values()
            ]

    @serialized_gpu_operation
    def unload_by_model_id(self, model_id: str) -> Dict[str, Any]:
        before_gpu = query_gpu_info()
        before_ram = psutil.virtual_memory().used / (1024**3)

        removed_names: List[str] = []
        to_drop: List[Any] = []
        with self._lock:
            for slot in list(self._slots.keys()):
                entry = self._slots[slot]
                if entry["model_id"] == model_id or model_id in entry.get("adapters", []):
                    removed_names.append(entry["name"])
                    to_drop.append(self._slots.pop(slot))

        for item in to_drop:
            _release_model_handle(item.pop("handle", None))
        del to_drop

        flush_ram_and_vram()

        after_gpu = query_gpu_info()
        after_ram = psutil.virtual_memory().used / (1024**3)
        vram_freed = round(max(0.0, before_gpu["vram_used_gb"] - after_gpu["vram_used_gb"]), 2)
        ram_freed = round(max(0.0, before_ram - after_ram), 2)
        colab_console_log(
            f"Liberados de RAM/VRAM: {', '.join(removed_names) or 'caché'} "
            f"(-{vram_freed} GB VRAM, -{ram_freed} GB RAM). Archivos en disco intactos.",
            icon="🧹",
        )

        return {
            "unloaded": removed_names,
            "vram_freed_gb": vram_freed,
            "ram_freed_gb": ram_freed,
            "vram_used_now_gb": after_gpu["vram_used_gb"],
        }

    @serialized_gpu_operation
    def unload_slot(self, slot: str) -> Dict[str, Any]:
        with self._lock:
            entry = self._slots.get(slot)
        if not entry:
            flush_ram_and_vram()
            return {"unloaded": [], "vram_freed_gb": 0.0, "ram_freed_gb": 0.0}
        return self.unload_by_model_id(entry["model_id"])

    @serialized_gpu_operation
    def unload_all(self) -> Dict[str, Any]:
        before_gpu = query_gpu_info()
        before_ram = psutil.virtual_memory().used / (1024**3)

        names: List[str] = []
        to_drop: List[Any] = []
        with self._lock:
            for slot in list(self._slots.keys()):
                entry = self._slots.pop(slot)
                names.append(entry["name"])
                to_drop.append(entry)

        for item in to_drop:
            _release_model_handle(item.pop("handle", None))
        del to_drop

        flush_ram_and_vram()

        after_gpu = query_gpu_info()
        after_ram = psutil.virtual_memory().used / (1024**3)
        v_freed = round(max(0.0, before_gpu["vram_used_gb"] - after_gpu["vram_used_gb"]), 2)
        r_freed = round(max(0.0, before_ram - after_ram), 2)
        colab_console_log(
            f"Memoria global limpiada ({', '.join(names) or 'caché'}): -{v_freed} GB VRAM, -{r_freed} GB RAM.",
            icon="🧹",
        )
        return {
            "unloaded": names,
            "vram_freed_gb": v_freed,
            "ram_freed_gb": r_freed,
            "vram_used_now_gb": after_gpu["vram_used_gb"],
        }


memory_manager = RuntimeMemoryManager()
