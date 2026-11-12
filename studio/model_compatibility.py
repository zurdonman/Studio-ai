"""Hardware-aware, explainable assessments shared by discovery and the MCP oracle.

Capacity, engine support, dependencies and current availability are separate axes.
`compatible` means a reviewed recipe fits its *estimated* capacity envelope;
`runnable_now` is stricter and is never true for a simulated GPU. Neither means
GPU-tested or guarantees that arbitrary resolution/frame counts will avoid OOM.
"""
from __future__ import annotations

import importlib.metadata
import re
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from types import SimpleNamespace

from packaging.version import InvalidVersion, Version

from studio.model_recipes import ENGINE_CONTRACTS
from studio.memory_policy import model_budget, reserves

_RUNTIME_LOCK = threading.Lock()
_RUNTIME_CACHE: Dict[str, Any] = {}


def get_runtime_capabilities() -> Dict[str, Any]:
    """Inspect installed distributions/exports without importing Torch or using CUDA."""
    with _RUNTIME_LOCK:
        if time.monotonic() - _RUNTIME_CACHE.get("ts", -1000) < 30:
            return _RUNTIME_CACHE["value"]
        packages = {"torch", "diffusers", "transformers", "accelerate", "safetensors", "peft", "bitsandbytes"}
        for contract in ENGINE_CONTRACTS.values():
            packages.update(contract.get("packages", []))
        versions: Dict[str, Optional[str]] = {}
        for package in sorted(packages):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        exported_classes = None
        if versions.get("diffusers"):
            try:
                init = Path(importlib.metadata.distribution("diffusers").locate_file("diffusers/__init__.py"))
                exported_classes = sorted(set(re.findall(r"\b[A-Z]\w*(?:Pipeline|Model)\b", init.read_text(encoding="utf-8"))))
            except (OSError, importlib.metadata.PackageNotFoundError):
                pass
        ffmpeg_available = bool(shutil.which("ffmpeg"))
        if not ffmpeg_available:
            try:
                from studio.engines.ffmpeg_utils import get_ffmpeg_bin
                ffmpeg_available = bool(get_ffmpeg_bin())
            except RuntimeError:
                pass
        executables = {program: bool(shutil.which(program)) for c in ENGINE_CONTRACTS.values() for program in c.get("executables", [])}
        value = {"executables": executables, "python": sys.version.split()[0], "versions": versions,
                 "diffusers_classes": exported_classes, "ffmpeg": ffmpeg_available,
                 "ffprobe": bool(shutil.which("ffprobe"))}
        _RUNTIME_CACHE.update(ts=time.monotonic(), value=value)
        return value


def runtime_for_engine(engine: str, runtime: Dict[str, Any], *, with_adapter: bool = False) -> Dict[str, Any]:
    contract = ENGINE_CONTRACTS.get(engine)
    if not contract:
        return {"ready": False, "status": "unsupported", "missing": [], "required_classes": []}
    required = {p: None for p in contract.get("packages", [])}
    if contract.get("diffusers"):
        required.update({"torch": None, "diffusers": contract["diffusers"], "transformers": contract.get("transformers", "4.48.0"),
                         "accelerate": "1.4.0", "safetensors": "0.4.5"})
    if with_adapter:
        required["peft"] = "0.17.1"
    if "peft" in required:
        required["peft"] = "0.17.1"
    if "bitsandbytes" in required:
        required["bitsandbytes"] = "0.45.0"
    versions = runtime.get("versions", {})
    missing, too_old = [], []
    for package, minimum in required.items():
        installed = versions.get(package)
        if not installed:
            missing.append(package)
        elif minimum:
            try:
                if Version(installed) < Version(minimum):
                    too_old.append(f"{package}>={minimum} (instalado: {installed})")
            except InvalidVersion:
                too_old.append(f"{package}: versión no verificable")
    classes = contract.get("classes", [])
    exports = runtime.get("diffusers_classes")
    missing_classes = [c for c in classes if exports is not None and c not in exports]
    class_check_unknown = bool(classes and versions.get("diffusers") and exports is None)
    if contract.get("ffmpeg"):
        missing.extend(tool for tool in ("ffmpeg",) if not runtime.get(tool))
    missing.extend(program for program in contract.get("executables", []) if not runtime.get("executables", {}).get(program))
    ready = not (missing or too_old or missing_classes or class_check_unknown)
    return {"ready": ready, "status": "ready" if ready else "needs_dependencies",
            "missing": missing, "version_conflicts": too_old, "missing_classes": missing_classes,
            "class_check_unknown": class_check_unknown, "required_classes": classes,
            "required_packages": required, "check_level": "static_package_metadata_and_exports",
            "inference_validated": False, "cuda_abi_verified": False}


def assessment_context(snapshot: Optional[Dict[str, Any]] = None, gpu_profile: str = "auto",
                       detected_gpu: Optional[Dict[str, Any]] = None,
                       runtime: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    from studio.catalog import COLAB_GPU_PROFILES, resolve_gpu_profile
    from studio.hardware import hardware_tracker

    allowed = {p["id"].lower() for p in COLAB_GPU_PROFILES}
    if not isinstance(gpu_profile, str) or gpu_profile.lower() not in allowed:
        raise ValueError("Perfil GPU no válido. Usa auto, CPU, T4, L4, A100_40, A100_80 o H100.")
    snap = snapshot if snapshot is not None else hardware_tracker.get_snapshot()
    actual_gpu = detected_gpu if detected_gpu is not None else snap.get("gpu", {})
    return {"hardware": {**snap, "gpu": actual_gpu}, "target": resolve_gpu_profile(gpu_profile, actual_gpu),
            "runtime": runtime if runtime is not None else get_runtime_capabilities()}


def _number(value: Any) -> Optional[float]:
    try:
        value = float(value)
        return value if value >= 0 and value < float("inf") else None
    except (TypeError, ValueError):
        return None


def _fit(required: Optional[float], available: Optional[float]) -> str:
    if required is None or available is None:
        return "unknown"
    return "fits_estimate" if available >= required else "insufficient"


def assess_definition(defn: Dict[str, Any], context: Dict[str, Any], *, catalog: Any,
                      include_base: bool = True) -> Dict[str, Any]:
    engine = str(defn.get("engine") or "")
    candidate = defn.get("validation_state") == "candidate" or (defn.get("is_custom") and defn.get("validation_state") != "reviewed_recipe")
    if defn.get("category") == "lora" and include_base and not candidate:
        return _assess_adapter(defn, context, catalog)
    contract = ENGINE_CONTRACTS.get(engine)
    supported = bool(contract and not candidate)
    hardware, target = context["hardware"], context["target"]
    gpu = hardware.get("gpu", {})
    ram, disk = hardware.get("ram", {}), hardware.get("disk", {})
    budget = model_budget(defn, has_gpu=bool(target.get("has_gpu")))
    cpu = budget["cpu"]
    min_vram, min_ram = budget["vram_gb"], budget["ram_gb"]
    downloaded = bool(catalog.is_downloaded(defn["id"]))
    estimated_disk = _number(defn.get("estimated_gb"))
    needed_disk = 0.0 if downloaded or defn.get("builtin_no_download") else (
        round(estimated_disk * 1.1 + 1.0, 2) if estimated_disk is not None else None)
    vram_total = _number(target.get("vram_gb"))
    ram_total, ram_free = _number(ram.get("total_gb")), _number(ram.get("available_gb"))
    disk_free = _number(disk.get("free_gb"))
    vram_free = vram_total if target.get("is_simulated") else _number(gpu.get("vram_free_gb"))
    # Preserve at least 1 GiB for outputs/temporary files even with local weights.
    output_reserve = 1.0
    total_disk_budget = max(output_reserve, needed_disk) if needed_disk is not None else None
    reserve = reserves(ram_total or 0, vram_total or 0)
    ram_admission = min_ram + reserve["ram_gb"] if min_ram is not None else None
    vram_admission = min_vram + reserve["vram_gb"] if min_vram else min_vram
    fits = {"vram": _fit(vram_admission, vram_total), "ram": _fit(ram_admission, ram_total), "disk": _fit(total_disk_budget, disk_free)}
    reasons, warnings = [], []
    if not supported:
        reasons.append("No hay receta de ejecución revisada para este registro. Guardarlo no habilita un cargador.")
    for resource, status in fits.items():
        if status == "insufficient":
            reasons.append(f"Capacidad de {resource.upper()} insuficiente para la estimación de esta receta.")
        elif status == "unknown":
            reasons.append(f"Estimación o disponibilidad de {resource.upper()} desconocida; no se declara compatible.")
    bf16 = gpu.get("bf16_supported")
    tier = str(target.get("tier", "")).upper()
    if target.get("is_simulated") or bf16 is None:
        if tier in {"T4", "V100", "CPU"}:
            bf16 = False
        elif tier in {"L4", "A100", "H100"}:
            bf16 = True
    needs_native_bf16 = bool((contract or {}).get("native_bf16") or defn.get("ltx_fp8_cast"))
    dtype_ok = not (needs_native_bf16 and bf16 is False)
    if not dtype_ok:
        reasons.append("Esta receta exige BF16 nativo (Ampere/Ada/Hopper); no se convierte ni cuantiza automáticamente para T4/V100.")
    dtype_unknown = bool(needs_native_bf16 and bf16 is None)
    if dtype_unknown:
        warnings.append("Soporte BF16 de la GPU no comprobado.")
    needs_unload = bool(min_vram and vram_free is not None and vram_free < vram_admission and fits["vram"] == "fits_estimate")
    ram_pressure = bool(min_ram and ram_free is not None and ram_free < ram_admission and fits["ram"] == "fits_estimate")
    if needs_unload:
        warnings.append("VRAM libre insuficiente ahora: descargar de memoria el modelo anterior, conservar los pesos en disco y volver a medir.")
    if ram_pressure:
        warnings.append("RAM disponible inferior al presupuesto: liberar modelos/procesos y volver a medir antes de cargar.")
    rt = runtime_for_engine(engine, context["runtime"], with_adapter=bool(defn.get("_with_adapter")))
    if not rt["ready"]:
        warnings.append("Dependencias/clases pendientes de instalar o comprobar; consultar runtime.")
    if (contract or {}).get("network_service"):
        warnings.append("Servicio de red externo: 0 GB de pesos locales no significa funcionamiento sin internet.")
    compatible = supported and dtype_ok and not dtype_unknown and all(s == "fits_estimate" for s in fits.values())
    current_memory_known = ram_free is not None and (cpu or not min_vram or vram_free is not None)
    if not current_memory_known:
        warnings.append("Disponibilidad actual de memoria no comprobada; no se declara ejecutable ahora.")
    runnable = (compatible and rt["ready"] and downloaded and current_memory_known
                and not needs_unload and not ram_pressure and not target.get("is_simulated"))
    if not downloaded:
        warnings.append("Pesos aún no preparados: confirmar licencia/acceso y presupuesto de descarga antes de ejecutar.")
    if candidate:
        status, label, tier_name = "candidate", "🧪 Candidato sin receta revisada", "candidate"
    elif not compatible:
        status, tier_name = "blocked", "incompatible"
        if supported and dtype_ok and not dtype_unknown and fits["vram"] == "fits_estimate":
            label = "🟡 VRAM con encaje estimado · revisar RAM/disco"
        elif fits["vram"] == "insufficient":
            label = "🟠 VRAM por debajo de la estimación · revisar"
        else:
            label = "🟠 Requisitos o datos por revisar"
    elif target.get("is_simulated"):
        status, label, tier_name = "simulated_fit", "🧮 Ajuste estimado · GPU simulada", "normal"
    elif not runnable:
        status, label, tier_name = "needs_preparation", "🟡 Encaje estimado · revisar preparación", "normal"
    else:
        status, label, tier_name = "estimated_fit", "🟢 Receta con encaje estimado", "excellent"
    return {
        "status": status, "compatible": compatible, "runnable_now": runnable, "tier": tier_name,
        "tier_rank": 0 if runnable else 1 if compatible else 2 if candidate else 3,
        "sticker": label, "label": label, "support": "reviewed_recipe" if supported else "candidate",
        "gpu_tested": bool(defn.get("gpu_tested", False)), "confidence": defn.get("resource_confidence", "engineering_estimate"),
        "gpu_target": target.get("gpu_name"), "is_simulated": bool(target.get("is_simulated")),
        "simulation_scope": "GPU simulada; RAM y disco son los de la sesión real" if target.get("is_simulated") else None,
        "resources": {
            "vram": {"status": fits["vram"], "required_gb": min_vram, "reserve_gb": reserve["vram_gb"] if min_vram else 0, "admission_required_gb": vram_admission, "recommended_gb": defn.get("recommended_vram_gb"), "total_gb": vram_total, "free_gb": vram_free, "needs_unload": needs_unload},
            "ram": {"status": fits["ram"], "required_gb": min_ram, "reserve_gb": reserve["ram_gb"], "admission_required_gb": ram_admission, "total_gb": ram_total, "available_gb": ram_free, "needs_release": ram_pressure},
            "disk": {"status": fits["disk"], "download_budget_gb": needed_disk, "estimated_weights_gb": estimated_disk, "free_gb": disk_free, "already_downloaded": downloaded,
                     "output_reserve_gb": output_reserve, "total_required_free_gb": total_disk_budget},
        },
        "runtime": rt, "weights_prepared": downloaded, "reasons": reasons, "warnings": warnings,
        "memory_strategy": defn.get("memory_strategy") or (contract or {}).get("quantization", "CPU" if cpu else "Precisión del pipeline + CPU offload; sin cuantización automática"),
        "limits": "Estimación para los valores predeterminados. Resolución, frames, VAE, encoders, LoRAs y procesos externos cambian el pico; no garantiza ausencia de OOM ni tiempos.",
    }


def assess_model_stack(base: Dict[str, Any], adapters: List[Dict[str, Any]], context: Dict[str, Any], *, catalog: Any) -> Dict[str, Any]:
    """Assess one exact base plus ALL resident adapters, including scale zero.

    Read-only planning. A compatible stack still needs real admission after
    unloading idle models; a loaded base is never charged twice by the runtime.
    """
    unique = {item["id"]: item for item in adapters}
    adapters = list(unique.values())
    budget = model_budget(base, has_gpu=bool(context["target"].get("has_gpu")), adapters=adapters)
    definitions = [base, *adapters]
    prepared = all(catalog.is_downloaded(item["id"]) for item in definitions)
    weights = [_number(item.get("estimated_gb")) for item in definitions]
    missing = [0.0 if catalog.is_downloaded(item["id"]) or item.get("builtin_no_download") else weight
               for item, weight in zip(definitions, weights)]
    combined = {**base, "_with_adapter": bool(adapters),
                "estimated_gb": sum(missing) if None not in missing else None,
                "min_ram_gb": budget["ram_gb"], "min_vram_gb": budget["vram_gb"]}
    preparation = SimpleNamespace(is_downloaded=lambda _mid: prepared)
    result = assess_definition(combined, context, catalog=preparation, include_base=False)
    result["resources"]["disk"].update(
        estimated_weights_gb=sum(weights) if None not in weights else None,
        base_prepared=bool(catalog.is_downloaded(base["id"])),
        adapters_prepared=all(catalog.is_downloaded(item["id"]) for item in adapters))
    invalid = [item["id"] for item in adapters if item.get("category") != "lora"
               or item.get("validation_state") == "candidate" or base["id"] not in item.get("compatible_models", [])]
    if invalid:
        result.update(compatible=False, runnable_now=False, status="blocked", tier="incompatible", tier_rank=3,
                      label="🔴 Adaptadores sin base exacta revisada", sticker="🔴 Adaptadores sin base exacta revisada")
        result["reasons"].append(f"Adaptadores sin base exacta revisada: {', '.join(invalid)}")
    result["stack"] = {"base_model_id": base["id"], "adapter_ids": list(unique),
                       "zero_scale_adapters_still_consume_memory": True, "budget": budget}
    result["gpu_tested"] = all(item.get("gpu_tested", False) for item in definitions)
    return result


def _assess_adapter(defn: Dict[str, Any], context: Dict[str, Any], catalog: Any) -> Dict[str, Any]:
    base_ids: List[str] = list(defn.get("compatible_models") or [])
    if not base_ids and defn.get("parent_model"):
        base_ids = [defn["parent_model"]]
    options = []
    for mid in base_ids:
        base = catalog.get_definition(mid)
        if not base or base.get("category") == "lora" or base.get("validation_state") == "candidate":
            continue
        result = assess_model_stack(base, [defn], context, catalog=catalog)
        result["resources"]["disk"]["adapter_prepared"] = bool(catalog.is_downloaded(defn["id"]))
        result["requires_base"] = base_ids
        result["assessed_base_model_id"] = mid
        result["incompatible_submodes"] = defn.get("incompatible_submodes", [])
        result["gpu_tested"] = bool(base.get("gpu_tested") and defn.get("gpu_tested"))
        result["runtime"]["adapter_note"] = "PEFT y base/submodo exactos requeridos; precargar LoRA antes de CPU offload. Presupuesto incremental estimado, no pico medido."
        result["warnings"].append("El encaje es para una base y este LoRA. Sumar otros adaptadores/frames/resolución exige reevaluar; no se infiere compatibilidad por tamaño.")
        options.append((base, result))
    if not options:
        result = assess_definition({**defn, "validation_state": "candidate"}, context, catalog=catalog, include_base=False)
        result["reasons"].append("No hay un modelo base exacto declarado para el adaptador; su tamaño por sí solo no prueba compatibilidad.")
        return result
    options.sort(key=lambda row: (not row[1]["compatible"], not row[1]["runnable_now"],
                                  not catalog.is_downloaded(row[0]["id"]), row[0].get("min_vram_gb", 0)))
    result = options[0][1]
    result["base_options"] = [{"model_id": base["id"], "compatible": assessment["compatible"],
                                "runnable_now": assessment["runnable_now"], "reasons": assessment["reasons"]}
                               for base, assessment in options]
    return result
