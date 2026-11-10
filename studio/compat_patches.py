"""Compatibility for Colab's Transformers / PEFT / bitsandbytes combination.

PEFT (and older Accelerate) rebuild parameters on the meta device by passing
``param.__dict__`` to their constructor. Transformers adds private metadata such
as ``_is_hf_initialized`` which neither Parameter nor Params4bit accepts.
Keep that metadata on the new parameter, not in its constructor arguments.
No model weights, tensor devices or quantization settings are changed here.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import logging
import sys
import threading
from contextlib import contextmanager
from functools import wraps
from typing import Any, Iterable, Optional

_PATCH_LOCK = threading.RLock()
LOGGER = logging.getLogger("studio.compat")
REIMPORT_TARGETS = (
    "transformers",
    "diffusers",
    "bitsandbytes",
    "accelerate",
    "peft",
)


def _wrap_parameter_subclass_new(target_cls: type) -> None:
    """Tolerate private HF metadata while preserving the real constructor signature."""
    orig_new = target_cls.__dict__.get("__new__") or getattr(target_cls, "__new__", None)
    raw_fn = orig_new.__func__ if isinstance(orig_new, staticmethod) else orig_new
    if raw_fn is None or getattr(raw_fn, "_studio_safe_wrapped", False):
        return

    try:
        sig = inspect.signature(raw_fn)
        allowed = set(sig.parameters)
        accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    except (TypeError, ValueError):
        # Parameter.__new__ in older PyTorch has no inspectable signature.
        allowed = {"data", "requires_grad"}
        accepts_kwargs = False

    @wraps(raw_fn)
    def _safe_new(cls: type, *args: Any, **kwargs: Any) -> Any:
        metadata = {
            key: value
            for key, value in kwargs.items()
            if key == "_is_hf_initialized"
            or (key.startswith("_") and not accepts_kwargs and key not in allowed)
        }
        clean_kwargs = {key: value for key, value in kwargs.items() if key not in metadata}
        obj = raw_fn(cls, *args, **clean_kwargs)
        if metadata:
            obj.__dict__.update(metadata)
        return obj

    _safe_new._studio_safe_wrapped = True  # type: ignore[attr-defined]
    target_cls.__new__ = staticmethod(_safe_new)


def _safe_find_spec(name: str) -> Any:
    try:
        return importlib.util.find_spec(name)
    except (ImportError, ValueError, AttributeError):
        return None


def _mark_runtime_rebuilt(modules: Iterable[str]) -> str:
    from studio import runtime_flags

    runtime_flags.mark_rebuilt(sorted(set(modules)))
    return runtime_flags.snapshot()["runtime_rebuild_notice"]


def ensure_patchable_runtime(*, modules: Optional[Iterable[str]] = None) -> dict[str, Any]:
    """Apply the compatibility patches before heavy frameworks build their classes.

    Patching after the model class was imported can leave already-created classes
    (or instances such as a cached 4-bit VLM) without the guard, which is not a
    quantization problem. When that is detected, the affected framework modules
    are re-imported once so the next load starts from a consistent state.
    Returns a report; never downloads weights, never touches saved files.
    """
    names = tuple(modules or REIMPORT_TARGETS)
    apply_runtime_compatibility_patches()
    from studio import runtime_flags

    already_rebuilt = set(runtime_flags.rebuilt_modules())
    stale = [
        name
        for name in names
        if name in already_rebuilt
        and (name in sys.modules or _safe_find_spec(name) is not None)
    ]
    if stale:
        return {
            "ok": False,
            "reimported": False,
            "notice": runtime_flags.snapshot()["runtime_rebuild_notice"],
        }
    loaded_late = [name for name in names if name in sys.modules]
    if not loaded_late:
        return {"ok": True, "reimported": False, "notice": ""}
    for name in loaded_late:
        try:
            importlib.reload(importlib.import_module(name))
        except Exception as exc:  # a partially reloaded framework is worse than a notice
            LOGGER.warning("No se pudo recargar %s tras los parches: %s", name, exc)
            notice = _mark_runtime_rebuilt(loaded_late)
            return {"ok": False, "reimported": False, "notice": notice}
    apply_runtime_compatibility_patches()
    notice = _mark_runtime_rebuilt(loaded_late)
    LOGGER.warning("%s", notice)
    return {"ok": True, "reimported": True, "notice": notice}


def _patch_peft_integrations() -> None:
    # NO importar `peft` aquí si aún no está cargado. En algunos entornos su primera
    # importación abre conexiones de red (descubrimiento de integraciones) y un fallo
    # de red tumbaba el arranque COMPLETO de la aplicación antes de existir, con un
    # ConnectionRefusedError que parecía venir del lanzador. Es un parche opcional:
    # si peft llega después, se aplica al cargar el modelo.
    integrations = sys.modules.get("peft.utils.integrations")
    if integrations is None:
        return

    original = getattr(integrations, "init_empty_weights", None)
    if original is None or getattr(original, "_studio_safe_wrapped", False):
        return

    @contextmanager
    @wraps(original)
    def _safe_init_empty_weights(*args: Any, **kwargs: Any):
        # Delegate, don't reimplement PEFT's context. In particular preserve
        # requires_grad, include_buffers, nested contexts and _skip_init_on_device.
        from studio.memory_manager import GPU_MODEL_OP_LOCK

        with GPU_MODEL_OP_LOCK:
            with original(*args, **kwargs) as context:
                yield context

    _safe_init_empty_weights._studio_safe_wrapped = True  # type: ignore[attr-defined]
    integrations.init_empty_weights = _safe_init_empty_weights


def apply_runtime_compatibility_patches() -> None:
    """Idempotent, including when optional GPU dependencies are installed later.

    Cada paso es PRESCINDIBLE: si un parche opcional falla (por ejemplo, porque una
    dependencia intenta usar la red), se registra un aviso y la aplicación sigue
    arrancando. Antes, una excepción aquí impedía abrir el servidor y el lanzador
    solo mostraba un «Connection refused» que ocultaba la causa real.
    """
    with _PATCH_LOCK:
        try:
            import torch  # type: ignore
        except ImportError:
            return

        def _safe(label: str, fn) -> None:
            try:
                fn()
            except Exception as exc:  # un parche nunca debe bloquear el arranque
                LOGGER.warning("Parche de compatibilidad omitido (%s): %s", label, exc)

        parameter_cls = getattr(getattr(torch, "nn", None), "Parameter", None)
        if isinstance(parameter_cls, type):
            _safe("Parameter.__new__", lambda: _wrap_parameter_subclass_new(parameter_cls))

        def _patch_bnb() -> None:
            import bitsandbytes as bnb  # type: ignore
            # These subclasses override __new__ and bypass Parameter.__new__.
            for name in ("Params4bit", "Int8Params"):
                cls = getattr(getattr(bnb, "nn", None), name, None)
                if isinstance(cls, type):
                    _wrap_parameter_subclass_new(cls)

        try:
            import bitsandbytes  # noqa: F401
            _safe("bitsandbytes", _patch_bnb)
        except Exception:
            pass

        _safe("peft.integrations", _patch_peft_integrations)

        def _hooks() -> None:
            from studio.memory_safety_hooks import install_memory_safety_hooks
            install_memory_safety_hooks()

        _safe("memory_safety_hooks", _hooks)
