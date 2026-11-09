"""Compatibility for Colab's Transformers / PEFT / bitsandbytes combination.

PEFT (and older Accelerate) rebuild parameters on the meta device by passing
``param.__dict__`` to their constructor. Transformers adds private metadata such
as ``_is_hf_initialized`` which neither Parameter nor Params4bit accepts.
Keep that metadata on the new parameter, not in its constructor arguments.
No model weights, tensor devices or quantization settings are changed here.
"""

from __future__ import annotations

import inspect
import threading
from contextlib import contextmanager
from functools import wraps
from typing import Any

_PATCH_LOCK = threading.RLock()


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


def _patch_peft_integrations() -> None:
    try:
        import peft.utils.integrations as integrations  # type: ignore
    except Exception:
        # Optional in the CPU application; try again on the next model load.
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
    """Idempotent, including when optional GPU dependencies are installed later."""
    with _PATCH_LOCK:
        try:
            import torch  # type: ignore
        except ImportError:
            return

        _wrap_parameter_subclass_new(torch.nn.Parameter)
        try:
            import bitsandbytes as bnb  # type: ignore
        except Exception:
            bnb = None
        if bnb is not None:
            # These subclasses override __new__ and bypass Parameter.__new__.
            for name in ("Params4bit", "Int8Params"):
                cls = getattr(bnb.nn, name, None)
                if isinstance(cls, type):
                    _wrap_parameter_subclass_new(cls)

        _patch_peft_integrations()
        from studio.memory_safety_hooks import install_memory_safety_hooks
        install_memory_safety_hooks()
