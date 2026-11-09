import ctypes
import gc
import inspect
import json
import os
from contextlib import contextmanager
from copy import deepcopy
import threading
import time
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

import psutil

from studio.hardware import colab_console_log, query_gpu_info
from studio.memory_policy import MemoryPressureError, host_memory, model_budget, number, pressure

# Candado reentrante global para serializar cargas/descargas de modelos en GPU y evitar colisiones concurrentes
GPU_MODEL_OP_LOCK = threading.RLock()


_OPERATION_LOCAL = threading.local()


def checkpoint_current_operation(*, job: Any = None, force: bool = False) -> None:
    stack = getattr(_OPERATION_LOCAL, "stack", [])
    if stack:
        owner_job = job or stack[-1]["job"]
        if owner_job is not None and owner_job.cancel_requested:
            from studio.jobs import JobCancelledError
            raise JobCancelledError("Cancelado manualmente por el usuario.")
        stack[-1]["manager"].checkpoint(job=owner_job, force=force)


@contextmanager
def native_model_load_scope() -> Any:
    """Use the full reserve as a fuse while adding model tensors, not just half."""
    stack = getattr(_OPERATION_LOCAL, "stack", [])
    if not stack:
        yield
        return
    scope = stack[-1]
    scope["native_load_depth"] = scope.get("native_load_depth", 0) + 1
    try:
        yield
    finally:
        scope["native_load_depth"] -= 1


def serialized_gpu_operation(fn: Optional[Callable[..., Any]] = None, *, slots: Any = (), managed: bool = False) -> Any:
    """Serialize GPU owners; managed engine calls lease their slots until return.

    Memory-manager mutations use the same lock but do not recursively start an
    engine scope. A watchdog never unloads a handle another thread is using.
    """
    if fn is None:
        return lambda wrapped: serialized_gpu_operation(wrapped, slots=slots, managed=managed)
    signature = inspect.signature(fn)

    @wraps(fn)
    def _locked(*args: Any, **kwargs: Any) -> Any:
        bound = signature.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        job = bound.arguments.get("job")
        if job is None:
            GPU_MODEL_OP_LOCK.acquire()
        else:
            while not GPU_MODEL_OP_LOCK.acquire(timeout=1.0):
                job.check_cancelled()
                job.heartbeat(step="Esperando turno de GPU (otra carga/inferencia en curso)...")
        try:
            if job is not None:
                job.check_cancelled()
            if not managed:
                return fn(*args, **kwargs)
            manager = fn.__globals__.get("memory_manager", memory_manager)
            try:
                protected = slots(bound.arguments) if callable(slots) else slots
                with manager.operation(slots=protected, job=job, label=fn.__name__):
                    return fn(*args, **kwargs)
            except Exception as exc:
                if isinstance(exc, (MemoryPressureError, MemoryError)) or "out of memory" in str(exc).lower():
                    # A traceback keeps failed loader frames (and their tensors)
                    # alive even after their registry handle has been removed.
                    import traceback
                    pending, visited = [exc], set()
                    while pending:
                        failure = pending.pop()
                        if failure is None or id(failure) in visited:
                            continue
                        visited.add(id(failure))
                        traceback.clear_frames(failure.__traceback__)
                        pending.extend([failure.__cause__, failure.__context__])
                    flush_ram_and_vram()
                raise
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


@serialized_gpu_operation
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
    stack = getattr(_OPERATION_LOCAL, "stack", [])
    manager = stack[-1]["manager"] if stack else memory_manager
    snap = manager.ensure_capacity(ram_gb=required_gb, vram_gb=0, stage=stage, job=job)
    return {"available_gb": snap["ram"]["available_gb"], "total_gb": snap["ram"]["total_gb"], "required_gb": required_gb}


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
    from studio.memory_safety_hooks import guarded_model_load, install_memory_safety_hooks
    install_memory_safety_hooks()
    checkpoint_current_operation(job=job, force=True)
    if job is None:
        result = guarded_model_load(fn)
        checkpoint_current_operation(force=True)
        return result

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
            vm = host_memory()
            ram_gb = vm["used_gb"]
            gpu = query_gpu_info()
            mem_txt = (
                f"RAM {ram_gb:.1f}/{vm['total_gb']:.1f} GiB · VRAM {gpu['vram_used_gb']:.1f}/{gpu['vram_total_gb']:.1f} GB"
                if gpu["has_gpu"]
                else f"RAM {ram_gb:.1f} GiB"
            )
            job.heartbeat(
                progress=round(cur_pct, 1),
                step=f"{step_label} ({int(elapsed)}s)",
                speed_text=mem_txt,
            )

    t = threading.Thread(target=_watcher, daemon=True)
    t.start()
    try:
        result = guarded_model_load(fn)
        checkpoint_current_operation(job=job, force=True)
        return result
    finally:
        done["flag"] = True


class RuntimeMemoryManager:
    """Gestiona qué modelos están cargados en RAM/VRAM y permite liberarlos sin borrar el disco."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # slot -> {"model_id": str, "name": str, "handle": Any, "adapters": List[str], "loaded_at": float}
        self._slots: Dict[str, Dict[str, Any]] = {}
        self._pins: Dict[str, int] = {}
        self._active: Dict[int, Dict[str, Any]] = {}
        self._events: List[Dict[str, Any]] = []
        self._last_observation: Dict[str, Any] = {}
        self._last_action = "starting"
        self._last_cleanup = -1000.0
        self._last_checkpoint = threading.local()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        poll = number(os.environ.get("STUDIO_MEMORY_POLL_SECONDS", "1"))
        self.poll_seconds = max(0.25, min(60.0, poll if poll is not None else 1.0))

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
        if old is not None and old.get("handle") is not handle:
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
                "last_used_at": time.time(),
            }

    def get_slot(self, slot: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            item = self._slots.get(slot)
            if item:
                item["last_used_at"] = time.time()
            return item

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
                    "last_used_at": round(v.get("last_used_at", v["loaded_at"]), 2),
                    "in_use": self._pins.get(v["slot"], 0) > 0,
                }
                for v in self._slots.values()
            ]

    def _unload_slots(self, slots: List[str], *, global_cleanup: bool = False) -> Dict[str, Any]:
        # Telemetry failure must not prevent dropping a failed/idle owner's
        # references. Unknown freed bytes are reported as unknown, not invented.
        def measure() -> tuple:
            try:
                ram = number(host_memory().get("used_gb"))
            except Exception:
                ram = None
            try:
                gpu = query_gpu_info()
            except Exception:
                gpu = {}
            return ram, gpu
        before_ram, before_gpu = measure()
        with self._lock:
            to_drop = [self._slots.pop(slot) for slot in slots if slot in self._slots]
        names = [item["name"] for item in to_drop]
        for item in to_drop:
            _release_model_handle(item.pop("handle", None))
        del to_drop
        flush_ram_and_vram()
        after_ram, after_gpu = measure()
        def freed(before: Any, after: Any) -> Optional[float]:
            a, b = number(before), number(after)
            return round(max(0.0, a - b), 2) if a is not None and b is not None else None
        vram_freed = freed(before_gpu.get("vram_used_gb"), after_gpu.get("vram_used_gb"))
        ram_freed = freed(before_ram, after_ram)
        vram_text = f"{vram_freed:.2f} GiB" if vram_freed is not None else "no medida"
        ram_text = f"{ram_freed:.2f} GiB" if ram_freed is not None else "no medida"
        colab_console_log(
            f"{'Memoria global limpiada' if global_cleanup else 'Liberados de RAM/VRAM'}: {', '.join(names) or 'caché'} "
            f"(VRAM liberada: {vram_text}; RAM liberada: {ram_text}). Archivos en disco intactos.", icon="🧹")
        return {"unloaded": names, "vram_freed_gb": vram_freed, "ram_freed_gb": ram_freed,
                "vram_used_now_gb": after_gpu.get("vram_used_gb")}

    @serialized_gpu_operation
    def unload_by_model_id(self, model_id: str) -> Dict[str, Any]:
        with self._lock:
            slots = [slot for slot, entry in self._slots.items()
                     if entry["model_id"] == model_id or model_id in entry.get("adapters", [])]
        return self._unload_slots(slots)

    @serialized_gpu_operation
    def unload_slot(self, slot: str) -> Dict[str, Any]:
        return self._unload_slots([slot])

    @serialized_gpu_operation
    def unload_all(self) -> Dict[str, Any]:
        with self._lock:
            slots = list(self._slots)
        return self._unload_slots(slots, global_cleanup=True)

    @serialized_gpu_operation
    def update_adapters(self, slot: str, adapter_ids: List[str]) -> None:
        with self._lock:
            if slot in self._slots:
                self._slots[slot]["adapters"] = list(dict.fromkeys(adapter_ids))
                self._slots[slot]["last_used_at"] = time.time()

    def sample(self) -> Dict[str, Any]:
        try:
            snapshot = {"ts": time.time(), "ram": host_memory(), "gpu": query_gpu_info()}
            snapshot["pressure"] = pressure(snapshot)
        except Exception as exc:
            raise MemoryPressureError("MEMORY_MEASUREMENT_UNAVAILABLE: no se puede comprobar la memoria real; carga/inferencia detenida antes de seguir reservando.") from exc
        try:
            snapshot["process_rss_gb"] = psutil.Process().memory_info().rss / (1024**3)
        except (psutil.Error, OSError):
            pass
        with self._lock:
            self._last_observation = snapshot
            operations = list(self._active.values())
            for op in operations:
                op["pressure_pending"] = snapshot["pressure"]["level"] != "normal"
                peaks = op.setdefault("peaks", {})
                for key, value in (("ram_used_gb", snapshot["ram"]["used_gb"]),
                                   ("vram_used_gb", snapshot["gpu"].get("vram_used_gb", 0)),
                                   ("process_rss_gb", snapshot.get("process_rss_gb", 0))):
                    peaks[key] = max(peaks.get(key, 0), value)
        return snapshot

    def _event(self, action: str, message: str, **fields: Any) -> None:
        from studio.diagnostics import redact_secrets
        message = redact_secrets(message)
        event = {"ts": time.time(), "action": action, "message": message, **fields}
        with self._lock:
            self._events.append(event)
            self._events = self._events[-40:]
            self._last_action = action
        colab_console_log(f"[Memoria automática] {message}", icon="🛡️")
        try:
            from studio.config import STATE_DIR
            path = STATE_DIR / "memory_guardian.json"
            tmp = path.with_suffix(".tmp")
            with self._lock:
                payload = json.dumps({"events": self._events[-40:], "last_observation": self._last_observation}, ensure_ascii=False)
                tmp.write_text(payload, encoding="utf-8")
                tmp.replace(path)
        except OSError:
            pass

    @contextmanager
    def operation(self, *, slots: tuple, job: Any = None, label: str = "engine") -> Any:
        GPU_MODEL_OP_LOCK.acquire()
        stack = getattr(_OPERATION_LOCAL, "stack", [])
        if job is None:
            job = next((parent["job"] for parent in reversed(stack) if parent["manager"] is self and parent["job"] is not None), None)
        scope = {"manager": self, "slots": tuple(slots), "job": job, "label": label,
                 "started_ts": time.time(), "peaks": {}}
        with self._lock:
            for slot in slots:
                self._pins[slot] = self._pins.get(slot, 0) + 1
            self._active[id(scope)] = scope
        _OPERATION_LOCAL.stack = stack + [scope]
        try:
            # Install idempotent, thread-scoped checks at tensor transfer sites.
            # Optional packages are only examined after an engine imported them.
            from studio.memory_safety_hooks import install_memory_safety_hooks
            install_memory_safety_hooks()
            self._reclaim(stage=f"inicio de {label}")
            yield
            self.checkpoint(job=job, force=True)
        except Exception as exc:
            if isinstance(exc, (MemoryPressureError, MemoryError)) or "out of memory" in str(exc).lower():
                self._event("safety_stop", f"{label}: {exc}", slots=list(slots))
                if job is not None and hasattr(job, "memory_protection"):
                    job.memory_protection.update(action="safety_stop", reason=str(exc),
                                                  code=getattr(exc, "code", "MEMORY_ALLOCATION_FAILED"))
                # The failing call has unwound. Drop potentially inconsistent
                # caches, never active handles from a different thread/owner.
                for slot in slots:
                    self.unload_slot(slot)
            raise
        finally:
            _OPERATION_LOCAL.stack = stack
            with self._lock:
                self._active.pop(id(scope), None)
                for slot in slots:
                    self._pins[slot] = max(0, self._pins.get(slot, 1) - 1)
            if job is not None and hasattr(job, "memory_protection"):
                job.memory_protection.update(observed_peaks=dict(scope["peaks"]),
                    measurement_scope="process_and_device_not_model_certification")
            GPU_MODEL_OP_LOCK.release()

    def _enough(self, snapshot: Dict[str, Any], ram_gb: float, vram_gb: float) -> bool:
        headroom = snapshot["pressure"]["reserve"]
        if snapshot["pressure"]["level"] == "unknown":
            return False
        ram_ok = snapshot["ram"]["available_gb"] >= ram_gb + headroom["ram_gb"]
        free = snapshot["gpu"].get("vram_free_gb")
        gpu_ok = not vram_gb or (snapshot["gpu"].get("has_gpu") and free is not None
                                and free >= vram_gb + headroom["vram_gb"])
        return bool(ram_ok and gpu_ok)

    def _reclaim(self, *, ram_gb: float = 0, vram_gb: float = 0, stage: str) -> Dict[str, Any]:
        snapshot = self.sample()
        if snapshot["pressure"]["level"] == "unknown":
            return snapshot
        if self._enough(snapshot, ram_gb, vram_gb) and snapshot["pressure"]["level"] == "normal":
            return snapshot
        if time.monotonic() - self._last_cleanup >= 3:
            # Cache is allocator-owned, not live tensor storage. Never pretend
            # empty_cache frees live CPU-offloaded weights or an active model.
            flush_ram_and_vram()
            self._last_cleanup = time.monotonic()
            snapshot = self.sample()
        with self._lock:
            victims = sorted((v["slot"] for v in self._slots.values() if not self._pins.get(v["slot"])),
                             key=lambda slot: self._slots[slot].get("last_used_at", self._slots[slot]["loaded_at"]))
        for slot in victims:
            if self._enough(snapshot, ram_gb, vram_gb) and snapshot["pressure"]["level"] == "normal":
                break
            with self._lock:
                entry = self._slots.get(slot)
                if not entry or self._pins.get(slot):
                    continue
                model_id = entry["model_id"]
            result = self.unload_slot(slot)
            self._event("evicted_idle_model", f"Liberado {model_id} ({slot}) para {stage}; pesos en disco intactos.",
                        model_id=model_id, slot=slot, freed=result)
            snapshot = self.sample()
        return snapshot

    @serialized_gpu_operation
    def ensure_capacity(self, *, ram_gb: float, vram_gb: float = 0, stage: str, job: Any = None) -> Dict[str, Any]:
        if number(ram_gb) is None or number(vram_gb) is None:
            raise ValueError("Memory budgets must be finite and nonnegative")
        snapshot = self._reclaim(ram_gb=ram_gb, vram_gb=vram_gb, stage=stage)
        reserve = snapshot["pressure"]["reserve"]
        if not self._enough(snapshot, ram_gb, vram_gb):
            message = (f"MEMORY_BUDGET_EXCEEDED: {stage}: RAM libre {snapshot['ram']['available_gb']:.2f} GiB, "
                       f"presupuesto {ram_gb:.2f} + reserva {reserve['ram_gb']:.2f}; "
                       f"VRAM libre {snapshot['gpu'].get('vram_free_gb')} GiB, presupuesto {vram_gb:.2f} "
                       f"+ reserva {reserve['vram_gb']:.2f}. No quedan modelos inactivos suficientes que liberar. "
                       "No se borran pesos ni se descarga otra receta.")
            self._event("admission_rejected", message)
            raise MemoryPressureError(message)
        if job is not None:
            job.log(f"Presupuesto admitido para {stage}: RAM {ram_gb:.2f} + margen {reserve['ram_gb']:.2f} GiB; "
                    f"VRAM {vram_gb:.2f} + margen {reserve['vram_gb']:.2f} GiB.", step="Presupuesto de memoria comprobado")
        return snapshot

    def prepare_for_load(self, definition: Dict[str, Any], *, adapters: Optional[List[Dict[str, Any]]] = None,
                         job: Any = None) -> Dict[str, Any]:
        snapshot = self.sample()
        budget = model_budget(definition, has_gpu=bool(snapshot["gpu"].get("has_gpu")), adapters=adapters)
        if budget["ram_gb"] is None or budget["vram_gb"] is None:
            raise MemoryPressureError("MEMORY_BUDGET_UNKNOWN: no se inicia una carga sin presupuesto de base/componentes/adaptadores.")
        return self.ensure_capacity(ram_gb=budget["ram_gb"], vram_gb=budget["vram_gb"],
                                    stage=f"carga de {definition['id']} + {len(budget['adapter_ids'])} adaptador(es)", job=job)

    @serialized_gpu_operation
    def prepare_for_adapters(self, definition: Dict[str, Any], adapters: List[Dict[str, Any]], *, job: Any = None) -> Dict[str, Any]:
        if not definition:
            raise MemoryPressureError("MEMORY_BUDGET_UNKNOWN: falta la base exacta de los adaptadores.")
        snapshot = self.sample()
        has_gpu = bool(snapshot["gpu"].get("has_gpu"))
        budget = model_budget(definition, has_gpu=has_gpu, adapters=adapters)
        reserve = snapshot["pressure"]["reserve"]
        if budget["ram_gb"] is None or budget["vram_gb"] is None:
            raise MemoryPressureError("MEMORY_BUDGET_UNKNOWN: faltan estimaciones de los adaptadores solicitados.")
        if budget["ram_gb"] + reserve["ram_gb"] > snapshot["ram"]["total_gb"] or (
                budget["vram_gb"] and budget["vram_gb"] + reserve["vram_gb"] > (snapshot["gpu"].get("vram_total_gb") or 0)):
            raise MemoryPressureError("MEMORY_BUDGET_EXCEEDED: la receta completa (base + todos los LoRA + reserva) supera la capacidad; no se descargan adaptadores.")
        incremental = model_budget({**definition, "min_ram_gb": 0, "min_vram_gb": 0}, has_gpu=has_gpu, adapters=adapters)
        return self.ensure_capacity(ram_gb=incremental["ram_gb"], vram_gb=incremental["vram_gb"],
                                    stage=f"adaptadores de {definition['id']} (base ya residente)", job=job)

    def checkpoint(self, *, job: Any = None, force: bool = False) -> None:
        if getattr(self._last_checkpoint, "running", False):
            return
        stack = getattr(_OPERATION_LOCAL, "stack", [])
        pending = bool(stack and stack[-1].get("pressure_pending"))
        now = time.monotonic()
        if not force and not pending and now - getattr(self._last_checkpoint, "ts", -1000.0) < 0.25:
            return
        self._last_checkpoint.ts = now
        self._last_checkpoint.running = True
        try:
            self._check_pressure(job=job)
        finally:
            self._last_checkpoint.running = False

    def _check_pressure(self, *, job: Any = None) -> None:
        snapshot = self.sample()
        if snapshot["pressure"]["level"] != "normal":
            snapshot = self._reclaim(stage="punto seguro de la operación activa")
        if job is not None and hasattr(job, "memory_protection"):
            job.memory_protection.update(level=snapshot["pressure"]["level"],
                reserve=snapshot["pressure"]["reserve"], last_check_ts=snapshot["ts"])
        stack = getattr(_OPERATION_LOCAL, "stack", [])
        loading = any(scope.get("native_load_depth", 0) for scope in stack if scope["manager"] is self)
        if snapshot["pressure"]["level"] in ("critical", "unknown") or (loading and snapshot["pressure"]["level"] == "pressure"):
            raise MemoryPressureError(
                "MEMORY_SAFETY_STOP: punto seguro sin margen de memoria suficiente (reserva completa durante carga; mitad durante inferencia) "
                f"(RAM libre {snapshot['ram']['available_gb']:.2f} GiB, VRAM libre {snapshot['gpu'].get('vram_free_gb')} GiB). "
                "Los pesos se conservan; no se reintenta automáticamente."
            )

    def tick(self) -> Dict[str, Any]:
        snapshot = self.sample()
        if snapshot["pressure"]["level"] in ("normal", "unknown"):
            with self._lock:
                self._last_action = "monitoring" if snapshot["pressure"]["level"] == "normal" else "measurement_unavailable"
            return snapshot
        # Never block a telemetry/watchdog thread behind a long inference, and
        # never run gc/unload concurrently with a different GPU owner.
        if not GPU_MODEL_OP_LOCK.acquire(blocking=False):
            with self._lock:
                self._last_action = "pressure_pending_safe_checkpoint"
            return snapshot
        try:
            result = self._reclaim(stage="vigilancia preventiva")
            if result["pressure"]["level"] != "normal":
                with self._lock:
                    self._last_action = "pressure_no_idle_models_reclaimable"
            return result
        finally:
            GPU_MODEL_OP_LOCK.release()

    def guardian_status(self) -> Dict[str, Any]:
        with self._lock:
            age = max(0.0, time.time() - self._last_observation["ts"]) if self._last_observation else None
            healthy = (age is not None and age < max(5, self.poll_seconds * 3)
                       and self._last_action not in ("monitor_error", "measurement_unavailable")
                       and self._last_observation.get("pressure", {}).get("level") != "unknown")
            return deepcopy({"enabled": True, "running": bool(self._thread and self._thread.is_alive()),
                    "measurements_healthy": healthy, "sample_age_seconds": age,
                    "poll_seconds": self.poll_seconds, "last_action": self._last_action,
                    "last_observation": self._last_observation,
                    "protected_slots": sorted(slot for slot, count in self._pins.items() if count),
                    "active_operations": [{"label": op["label"], "job_id": getattr(op["job"], "id", None),
                                           "slots": list(op["slots"]), "native_load": bool(op.get("native_load_depth", 0))} for op in self._active.values()],
                    "recent_events": list(self._events[-12:]),
                    "policy": "LRU idle eviction + additional reserves + safe-point abort; never delete weights",
                    "limits": "Cooperative safety: a single native allocation may outrun sampling; not an OS memory limit or zero-OOM guarantee."})

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        with self._lock:
            self._last_action = "monitoring"
        try:
            self.tick()
        except Exception as exc:
            self._event("monitor_error", f"Error de medición inicial: {exc}; no se elimina un modelo activo.")
        def watch() -> None:
            while not self._stop.wait(self.poll_seconds):
                try:
                    self.tick()
                except Exception as exc:
                    self._event("monitor_error", f"Error de medición: {exc}; no se elimina un modelo activo.")
        self._thread = threading.Thread(target=watch, name="studio-memory-guardian", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=4)


memory_manager = RuntimeMemoryManager()
