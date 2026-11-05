import ctypes
import gc
import threading
import time
from typing import Any, Dict, List, Optional

import psutil

from studio.hardware import query_gpu_info


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


class RuntimeMemoryManager:
    """Gestiona qué modelos están cargados en RAM/VRAM y permite liberarlos sin borrar el disco."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # slot -> {"model_id": str, "name": str, "handle": Any, "adapters": List[str], "loaded_at": float}
        self._slots: Dict[str, Dict[str, Any]] = {}

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
            try:
                handle = item.get("handle")
                if hasattr(handle, "maybe_free_model_hooks"):
                    handle.maybe_free_model_hooks()
            except Exception:
                pass
            del item
        del to_drop

        flush_ram_and_vram()

        after_gpu = query_gpu_info()
        after_ram = psutil.virtual_memory().used / (1024**3)
        vram_freed = round(max(0.0, before_gpu["vram_used_gb"] - after_gpu["vram_used_gb"]), 2)
        ram_freed = round(max(0.0, before_ram - after_ram), 2)

        return {
            "unloaded": removed_names,
            "vram_freed_gb": vram_freed,
            "ram_freed_gb": ram_freed,
            "vram_used_now_gb": after_gpu["vram_used_gb"],
        }

    def unload_slot(self, slot: str) -> Dict[str, Any]:
        with self._lock:
            entry = self._slots.get(slot)
        if not entry:
            flush_ram_and_vram()
            return {"unloaded": [], "vram_freed_gb": 0.0, "ram_freed_gb": 0.0}
        return self.unload_by_model_id(entry["model_id"])

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
            try:
                handle = item.get("handle")
                if hasattr(handle, "maybe_free_model_hooks"):
                    handle.maybe_free_model_hooks()
            except Exception:
                pass
            del item
        del to_drop

        flush_ram_and_vram()

        after_gpu = query_gpu_info()
        after_ram = psutil.virtual_memory().used / (1024**3)
        return {
            "unloaded": names,
            "vram_freed_gb": round(max(0.0, before_gpu["vram_used_gb"] - after_gpu["vram_used_gb"]), 2),
            "ram_freed_gb": round(max(0.0, before_ram - after_ram), 2),
            "vram_used_now_gb": after_gpu["vram_used_gb"],
        }


memory_manager = RuntimeMemoryManager()
