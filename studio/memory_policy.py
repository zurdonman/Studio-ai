"""Framework-free memory budgets shared by admission, watchdog and the oracle.

Budgets are conservative estimates, not GPU measurements. Safety reserves are
additional; offloading GPU weights does not free their host-RAM copies.
"""
from __future__ import annotations

import math
from typing import Any

import psutil

GIB = 1024 ** 3


class MemoryPressureError(RuntimeError):
    """A safe admission/allocation/checkpoint rejected a memory-unsafe operation."""

    code = "MEMORY_SAFETY_STOP"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        prefix = message.partition(":")[0]
        if prefix.startswith("MEMORY_"):
            self.code = prefix


def number(value: Any) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) and result >= 0 else None
    except (ValueError, TypeError):
        return None


def read_cgroup_memory() -> dict[str, Any]:
    from studio.diagnostics import cgroup_memory_snapshot
    return cgroup_memory_snapshot()


def host_memory() -> dict[str, Any]:
    vm = psutil.virtual_memory()
    total, available = vm.total / GIB, vm.available / GIB
    original_total = total
    cgroup = read_cgroup_memory()
    source = "system_memavailable"
    ceilings = []
    for record in [cgroup, *cgroup.get("ancestors", [])]:
        limit, current = number(record.get("memory.max")), number(record.get("memory.current"))
        if limit is None or current is None or not 0 < limit / GIB <= original_total:
            continue
        stat = record.get("stat", {})
        inactive = number(stat.get("inactive_file", stat.get("total_inactive_file"))) or 0.0
        dirty = (number(stat.get("file_dirty")) or 0.0) + (number(stat.get("file_writeback")) or 0.0)
        reclaimable = min(current, max(0.0, inactive - dirty))
        total = min(total, limit / GIB)
        hard_free = max(0.0, (limit - current) / GIB)
        available = min(available, max(0.0, (limit - current + reclaimable) / GIB))
        source = f"cgroup_v{record.get('version', 2)}_clean_inactive_cache_estimate"
        ceilings.append({"limit_gb": limit / GIB, "current_gb": current / GIB,
                         "hard_free_gb": hard_free, "clean_inactive_cache_gb": reclaimable / GIB})
    available = max(0.0, min(total, available))
    return {"total_gb": total, "available_gb": available, "used_gb": total - available,
            "percent": 100 * (total - available) / total if total else 100,
            "host_total_gb": original_total, "source": source, "cgroup_ceilings": ceilings}


def reserves(ram_total: float, vram_total: float = 0) -> dict[str, float]:
    ram = max(1.0, ram_total * 0.10)
    vram = max(0.5, vram_total * 0.08) if vram_total > 0 else 0.0
    return {"ram_gb": ram, "vram_gb": vram,
            "critical_ram_gb": ram / 2, "critical_vram_gb": vram / 2}


def model_budget(definition: dict[str, Any], *, has_gpu: bool,
                 adapters: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    from studio.model_recipes import ENGINE_CONTRACTS
    engine = definition.get("engine")
    contract = ENGINE_CONTRACTS.get(engine, {})
    cpu = bool(contract.get("cpu") or (contract.get("cpu_capable") and not has_gpu))
    ram = number(definition.get("min_ram_gb", contract.get("ram_gb")))
    vram = 0.0 if cpu else number(definition.get("min_vram_gb"))
    if "min_ram_gb" not in definition and engine == "wan_i2v_14b":
        if definition.get("id") == "wan21-fun-1.3b-inp":
            ram = 32.0
        elif definition.get("id") == "wan22-i2v-a14b":
            ram = 128.0
    ids = []
    for adapter in adapters or []:
        if adapter.get("id") in ids or adapter.get("id") == definition.get("id"):
            continue
        ids.append(adapter.get("id"))
        extra = number(adapter.get("adapter_ram_gb"))
        weights = number(adapter.get("estimated_gb"))
        if extra is None and weights is not None:
            extra = weights * 2 + 0.25
        gpu_extra = 0.0 if cpu else number(adapter.get("min_vram_gb"))
        ram = ram + extra if ram is not None and extra is not None else None
        vram = vram + gpu_extra if vram is not None and gpu_extra is not None else None
    return {"ram_gb": ram, "vram_gb": vram, "adapter_ids": ids, "cpu": cpu,
            "confidence": "planning_estimate_not_measured_peak"}


def pressure(snapshot: dict[str, Any]) -> dict[str, Any]:
    ram, gpu = snapshot["ram"], snapshot.get("gpu", {})
    reserve = reserves(float(ram["total_gb"]), float(gpu.get("vram_total_gb") or 0))
    ram_free = number(ram.get("available_gb"))
    vram_free = number(gpu.get("vram_free_gb"))
    unknown = ram_free is None or (bool(gpu.get("has_gpu")) and vram_free is None)
    soft_ram = ram_free is not None and ram_free < reserve["ram_gb"]
    soft_vram = bool(gpu.get("has_gpu")) and vram_free is not None and vram_free < reserve["vram_gb"]
    critical_ram = ram_free is not None and ram_free < reserve["critical_ram_gb"]
    critical_vram = bool(gpu.get("has_gpu")) and vram_free is not None and vram_free < reserve["critical_vram_gb"]
    return {"level": "unknown" if unknown else "critical" if critical_ram or critical_vram else "pressure" if soft_ram or soft_vram else "normal",
            "ram_pressure": soft_ram, "vram_pressure": soft_vram, "reserve": reserve}
