"""Thread-scoped cooperative fuses at native model-loading boundaries.

Never kill a thread, discard live tensors or call to_empty. These checks only act
inside Studio's managed engine scopes, and keep optional imports optional.
A single allocation/kernel can still outrun any Python checkpoint.
"""
from __future__ import annotations

import sys
import inspect
from contextlib import contextmanager
import threading
from functools import wraps
from typing import Any

_LOCK = threading.RLock()


def _wrap(owner: Any, name: str) -> None:
    original = getattr(owner, name, None)
    if not callable(original) or getattr(original, "_studio_memory_guard", False):
        return

    @wraps(original)
    def checked(*args: Any, **kwargs: Any) -> Any:
        from studio.memory_manager import checkpoint_current_operation
        checkpoint_current_operation()
        result = original(*args, **kwargs)
        checkpoint_current_operation()
        return result

    checked._studio_memory_guard = True  # type: ignore[attr-defined]
    setattr(owner, name, checked)


def install_memory_safety_hooks() -> None:
    # Do not import frameworks from the watchdog, API telemetry or discovery.
    # Revisit modules on every load because Diffusers/Transformers are lazy.
    targets = {
        "accelerate.utils": ("set_module_tensor_to_device",),
        "accelerate.utils.modeling": ("set_module_tensor_to_device",),
        "accelerate.big_modeling": ("set_module_tensor_to_device",),
        "diffusers.models.model_loading_utils": ("set_module_tensor_to_device", "load_state_dict", "_determine_param_device"),
        "transformers.modeling_utils": ("set_module_tensor_to_device", "load_state_dict", "_load_state_dict_into_meta_model", "_infer_parameter_dtype", "_load_parameter_into_model", "load_shard_file"),
        "safetensors.torch": ("load_file",),
    }
    with _LOCK:
        for module_name, names in targets.items():
            module = sys.modules.get(module_name)
            if module is not None:
                for name in names:
                    _wrap(module, name)
        torch = sys.modules.get("torch")
        module_class = getattr(getattr(torch, "nn", None), "Module", None) if torch else None
        if module_class is not None:
            _wrap(module_class, "_apply")


def guarded_model_load(loader: Any, *args: Any, **kwargs: Any) -> Any:
    """Resolve the class first, install optional hooks, then invoke its loader."""
    from studio.memory_manager import checkpoint_current_operation, native_model_load_scope
    install_memory_safety_hooks()
    with native_model_load_scope():
        checkpoint_current_operation(force=True)
        try:
            result = loader(*args, **kwargs)
        except Exception as exc:
            # Some upstream loaders translate all exceptions to OSError. Preserve
            # our safety signal rather than treating it as missing/corrupt weights.
            from studio.memory_policy import MemoryPressureError
            cause, visited = exc, set()
            while cause is not None and id(cause) not in visited:
                visited.add(id(cause))
                if isinstance(cause, MemoryPressureError):
                    if cause is exc:
                        raise
                    raise MemoryPressureError(str(cause)) from exc
                cause = cause.__cause__ or cause.__context__
            raise
        checkpoint_current_operation(force=True)
        return result


@contextmanager
def cleanup_partial_load(cleanup: Any) -> Any:
    """Unwind an owner's private load lock before dropping partial resources."""
    try:
        yield
    except BaseException:
        cleanup()
        raise


def generation_memory_criteria(job: Any = None) -> Any:
    from transformers import StoppingCriteria, StoppingCriteriaList

    class MemoryCheckpoint(StoppingCriteria):
        def __call__(self, *args: Any, **kwargs: Any) -> bool:
            from studio.memory_manager import checkpoint_current_operation
            checkpoint_current_operation(job=job, force=True)
            if job is not None:
                job.check_cancelled()
            return False

    return StoppingCriteriaList([MemoryCheckpoint()])


def diffusion_memory_callback_kwargs(pipe: Any, job: Any = None) -> dict[str, Any]:
    """Select a supported callback by signature; never replay a failed inference."""
    from studio.memory_manager import checkpoint_current_operation

    def check() -> None:
        checkpoint_current_operation(job=job, force=True)
        if job is not None:
            job.check_cancelled()

    def modern(_pipe: Any, _index: int, _time: Any, values: Any) -> Any:
        check()
        return values

    def legacy(_index: int, _time: Any, _latents: Any) -> None:
        check()

    try:
        params = inspect.signature(pipe.__call__).parameters
    except (ValueError, TypeError):
        return {}
    if "callback_on_step_end" in params:
        return {"callback_on_step_end": modern}
    if "callback" in params and "callback_steps" in params:
        return {"callback": legacy, "callback_steps": 1}
    return {}
