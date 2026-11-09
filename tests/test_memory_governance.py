"""Automatic memory-policy regressions. No weights, Hub, CUDA or tunnel access.

Hardware/cgroup doubles are explicit. Native-load hooks exercise their actual
wrappers and thread ownership, not a monitor flag pretending to interrupt C++.
"""
from __future__ import annotations

import gc
import json
import sys
import threading
import time
import types
import weakref
import zipfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.usefixtures("temp_studio_env")
GIB = 1024 ** 3


@pytest.fixture
def guardian(colab_memory, monkeypatch):
    import studio.memory_manager as mm
    state = colab_memory()
    events = []
    monkeypatch.setattr(mm, "flush_ram_and_vram", lambda: events.append("cache_cleanup"))
    manager = mm.RuntimeMemoryManager()
    yield manager, state, events
    manager.stop()


def resident(manager, state, events, slot, *, ram=0, vram=0, model_id=None):
    class Handle:
        def unload(self):
            events.append(slot)
            state["available"] += ram
            state["vram_free"] += vram
    handle = {"model": Handle()}
    manager.register_loaded(slot, model_id or slot, slot, handle)
    return handle


def test_effective_ram_honors_parent_quotas_and_only_clean_inactive_cache(monkeypatch):
    from studio import memory_policy as policy
    monkeypatch.setattr(policy.psutil, "virtual_memory", lambda: types.SimpleNamespace(total=128 * GIB, available=100 * GIB))
    cgroup = {"version": 2, "memory.max": 8 * GIB, "memory.current": 7 * GIB,
              "stat": {"inactive_file": 2 * GIB, "file_dirty": GIB // 2},
              "ancestors": [{"memory.max": 4 * GIB, "memory.current": 3.9 * GIB}]}
    monkeypatch.setattr(policy, "read_cgroup_memory", lambda: cgroup)
    result = policy.host_memory()
    assert result["total_gb"] == 4 and result["available_gb"] == pytest.approx(0.1)
    assert result["host_total_gb"] == 128
    assert result["cgroup_ceilings"][0]["clean_inactive_cache_gb"] == 1.5
    cgroup["ancestors"] = []
    assert policy.host_memory()["available_gb"] == pytest.approx(2.5)
    cgroup["stat"] = {"inactive_file": 4 * GIB, "file_dirty": 5 * GIB}
    assert policy.host_memory()["available_gb"] == 1


@pytest.mark.parametrize("limit,current", [("max", 7 * GIB), ("unknown", None), (256 * GIB, 7 * GIB)])
def test_unbounded_or_unreadable_cgroup_does_not_invent_a_ceiling(monkeypatch, limit, current):
    from studio import memory_policy as policy
    monkeypatch.setattr(policy.psutil, "virtual_memory", lambda: types.SimpleNamespace(total=64 * GIB, available=52 * GIB))
    monkeypatch.setattr(policy, "read_cgroup_memory", lambda: {"memory.max": limit, "memory.current": current})
    assert policy.host_memory()["available_gb"] == 52


def test_cgroup_reader_resolves_visible_ancestors(monkeypatch):
    from studio.diagnostics import cgroup_memory_snapshot
    files = {"/proc/self/cgroup": "0::/users/studio\n"}
    for folder, limit, current in (("/sys/fs/cgroup", "max", 10 * GIB),
                                   ("/sys/fs/cgroup/users", 8 * GIB, 7 * GIB),
                                   ("/sys/fs/cgroup/users/studio", "max", GIB)):
        files[folder + "/memory.max"] = str(limit)
        files[folder + "/memory.current"] = str(current)
        files[folder + "/memory.events"] = "oom 0\noom_kill 0\n"
        files[folder + "/memory.stat"] = "inactive_file 1024\nfile_dirty 0\n"
    original = Path.read_text
    def read(path, *args, **kwargs):
        if str(path).startswith(("/proc/self/cgroup", "/sys/fs/cgroup")):
            if str(path) not in files:
                raise FileNotFoundError(str(path))
            return files[str(path)]
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", read)
    result = cgroup_memory_snapshot()
    assert result["path"] == "/sys/fs/cgroup/users"
    assert result["memory.max"] == 8 * GIB and len(result["ancestors"]) == 2
    assert result["stat"]["inactive_file"] == 1024


def test_cgroup_v1_reader_normalizes_legacy_quota(monkeypatch):
    from studio.diagnostics import cgroup_memory_snapshot
    files = {"/proc/self/cgroup": "5:memory:/studio\n",
             "/sys/fs/cgroup/memory/studio/memory.limit_in_bytes": str(4 * GIB),
             "/sys/fs/cgroup/memory/studio/memory.usage_in_bytes": str(3 * GIB),
             "/sys/fs/cgroup/memory/studio/memory.stat": "total_inactive_file 1024\n"}
    def read(path, *args, **kwargs):
        if str(path) not in files:
            raise FileNotFoundError(str(path))
        return files[str(path)]
    monkeypatch.setattr(Path, "read_text", read)
    result = cgroup_memory_snapshot()
    assert result["version"] == 1 and result["memory.max"] == 4 * GIB
    assert result["memory.current"] == 3 * GIB
    assert result["events"] == {}  # No invented kernel OOM evidence.


@pytest.mark.parametrize("available,free,expected", [
    (52, 21, "normal"), (5, 21, "pressure"), (3, 21, "critical"),
    (52, 1, "pressure"), (52, 0.5, "critical"), (52, None, "unknown"),
])
def test_pressure_uses_separate_ram_and_device_headroom(guardian, available, free, expected):
    manager, state, _ = guardian
    state.update(available=available, vram_free=free)
    # The complete GPU stub computes used bytes too; None explicitly represents
    # a driver telemetry failure rather than zero GPU capacity.
    if free is None:
        snapshot = {"ram": {"total_gb": 64, "available_gb": available},
                    "gpu": {"has_gpu": True, "vram_total_gb": 22, "vram_free_gb": None}}
        from studio.memory_policy import pressure
        result = pressure(snapshot)
    else:
        result = manager.sample()["pressure"]
    assert result["level"] == expected
    assert result["reserve"]["ram_gb"] == 6.4
    assert result["reserve"]["vram_gb"] == pytest.approx(1.76)


def test_full_stack_budget_deduplicates_and_counts_zero_scale_residency():
    from studio.catalog import catalog_manager
    from studio.memory_policy import model_budget
    base = catalog_manager.get_definition("wan21-vace-14b-fp8")
    ids = ["lora-wan21-14b-fusionx-phantom", "lora-wan21-14b-unianimate-char", "lora-wan21-14b-moviigen-cinema"]
    adapters = [catalog_manager.get_definition(mid) for mid in ids]
    result = model_budget(base, has_gpu=True, adapters=[*adapters, adapters[0]])
    assert result["adapter_ids"] == ids
    assert result["ram_gb"] == pytest.approx(44.47) and result["vram_gb"] == 17
    unknown = model_budget(base, has_gpu=True, adapters=[{"id": "unreviewed"}])
    assert unknown["ram_gb"] is None and unknown["vram_gb"] is None
    lipsync = catalog_manager.get_definition("edtalk-256-lipsync-hd")
    assert model_budget(lipsync, has_gpu=True)["vram_gb"] > 0
    assert model_budget(lipsync, has_gpu=False)["vram_gb"] == 0


def test_lru_admission_remeasures_and_stops_after_enough_idle_evictions(guardian, tmp_path):
    manager, state, events = guardian
    state["available"] = 10
    old = resident(manager, state, events, "old", ram=10)
    resident(manager, state, events, "middle", ram=7)
    newest = resident(manager, state, events, "new", ram=5)
    with manager._lock:
        for i, slot in enumerate(("old", "middle", "new")):
            manager._slots[slot]["last_used_at"] = i
    weights = tmp_path / "weights.bin"
    weights.write_bytes(b"never deleted by RAM management")
    snapshot = manager.ensure_capacity(ram_gb=20, vram_gb=5, stage="siguiente modelo")
    assert snapshot["ram"]["available_gb"] == 27
    assert [event for event in events if event != "cache_cleanup"] == ["old", "middle"]
    assert events[0] == "cache_cleanup" and old == {} and newest
    assert manager.loaded_model_ids() == ["new"] and weights.read_bytes().startswith(b"never")
    assert len(manager.guardian_status()["recent_events"]) == 2


def test_get_slot_marks_real_use_but_status_poll_does_not(guardian):
    manager, state, events = guardian
    resident(manager, state, events, "older", ram=2)
    resident(manager, state, events, "newer", ram=2)
    with manager._lock:
        manager._slots["older"]["last_used_at"] = 1
        manager._slots["newer"]["last_used_at"] = 2
    manager.status_summary()
    assert manager._slots["older"]["last_used_at"] == 1
    manager.get_slot("older")
    state["available"] = 5
    manager.tick()
    assert "newer" in events and "older" not in events


def test_healthy_watchdog_keeps_warm_models_and_does_not_gc(guardian):
    manager, state, events = guardian
    handle = resident(manager, state, events, "image")
    manager.tick()
    assert events == [] and handle and manager.get_slot("image")


def test_owner_checkpoint_reclaims_idle_slot_without_touching_active_model(guardian):
    from studio.jobs import JobRecord
    from studio.memory_manager import checkpoint_current_operation
    manager, state, events = guardian
    active = resident(manager, state, events, "image", ram=20)
    idle = resident(manager, state, events, "tts", ram=8)
    job = JobRecord("image", "seguro")
    with manager.operation(slots=("image",), job=job):
        assert manager.status_summary()[0]["in_use"]
        state["available"] = 2
        checkpoint_current_operation(job=job, force=True)
        assert active and idle == {} and manager.get_slot("image")
        assert "image" not in events and job.memory_protection["level"] == "normal"
    assert manager.guardian_status()["protected_slots"] == []
    assert not manager.status_summary()[0]["in_use"]


def test_critical_pressure_unwinds_owner_then_discards_only_its_handle(guardian):
    from studio.jobs import JobRecord
    from studio.memory_manager import checkpoint_current_operation
    from studio.memory_policy import MemoryPressureError
    manager, state, events = guardian
    handle = resident(manager, state, events, "video", ram=20)
    job = JobRecord("video", "proteger servidor")
    with pytest.raises(MemoryPressureError, match="MEMORY_SAFETY_STOP"):
        with manager.operation(slots=("video",), job=job):
            state["available"] = 2
            checkpoint_current_operation(job=job, force=True)
            pytest.fail("The next native allocation must not run")
    assert handle == {} and "video" in events
    assert not manager.loaded_model_ids() and not manager.guardian_status()["protected_slots"]
    assert job.memory_protection["code"] == "MEMORY_SAFETY_STOP"
    assert job.memory_protection["observed_peaks"]["ram_used_gb"] >= 62


def test_lru_unload_is_exact_slot_even_when_a_model_id_is_shared(guardian):
    manager, state, events = guardian
    active = resident(manager, state, events, "image", model_id="shared")
    idle = resident(manager, state, events, "vision", model_id="shared", ram=8)
    with manager.operation(slots=("image",)):
        state["available"] = 2
        manager.checkpoint(force=True)
        assert active and idle == {} and manager.get_slot("image")
        assert "image" not in events


def test_watchdog_thread_defers_to_owner_instead_of_unloading_live_handles(guardian):
    manager, state, events = guardian
    active = resident(manager, state, events, "video")
    resident(manager, state, events, "tts", ram=8)
    entered, checkpoint = threading.Event(), threading.Event()
    errors = []
    def infer():
        try:
            with manager.operation(slots=("video",)):
                entered.set()
                assert checkpoint.wait(3)
                manager.checkpoint(force=True)
                assert active
        except BaseException as exc:
            errors.append(exc)
    worker = threading.Thread(target=infer, daemon=True)
    worker.start()
    try:
        assert entered.wait(2)
        state["available"] = 2
        start = time.monotonic()
        manager.tick()
        assert time.monotonic() - start < 0.5
        assert manager.guardian_status()["last_action"] == "pressure_pending_safe_checkpoint"
        assert active and "tts" not in events and "video" not in events
    finally:
        checkpoint.set()
        worker.join(3)
    assert not worker.is_alive() and errors == []
    assert "tts" in events and "video" not in events


def test_manual_unload_waits_until_the_engine_scope_finishes(guardian):
    manager, state, events = guardian
    handle = resident(manager, state, events, "image")
    entered, release, attempted, unloaded = (threading.Event() for _ in range(4))
    errors = []
    def infer():
        try:
            with manager.operation(slots=("image",)):
                entered.set()
                assert release.wait(3)
                assert handle
        except BaseException as exc:
            errors.append(exc)
    def unload():
        attempted.set()
        manager.unload_all()
        unloaded.set()
    owner, waiter = threading.Thread(target=infer, daemon=True), threading.Thread(target=unload, daemon=True)
    owner.start()
    try:
        assert entered.wait(2)
        waiter.start()
        assert attempted.wait(2) and not unloaded.wait(0.05)
    finally:
        release.set()
        owner.join(3)
        if waiter.ident:
            waiter.join(3)
    assert not owner.is_alive() and not waiter.is_alive() and errors == []
    assert handle == {} and unloaded.is_set()


def test_unknown_vram_stops_admission_without_blind_eviction(guardian, monkeypatch):
    import studio.memory_manager as mm
    from studio.memory_policy import MemoryPressureError
    manager, state, events = guardian
    handle = resident(manager, state, events, "image")
    monkeypatch.setattr(mm, "query_gpu_info", lambda: {"has_gpu": True, "vram_total_gb": 22, "vram_free_gb": None})
    assert manager.tick()["pressure"]["level"] == "unknown"
    with pytest.raises(MemoryPressureError, match="MEMORY_BUDGET_EXCEEDED"):
        manager.ensure_capacity(ram_gb=5, vram_gb=2, stage="sin telemetría")
    assert handle and events == []


@pytest.mark.parametrize("bad", [-1, float("inf"), float("nan")])
def test_invalid_budget_does_not_nuke_cache(guardian, bad):
    manager, state, events = guardian
    handle = resident(manager, state, events, "image")
    with pytest.raises(ValueError):
        manager.ensure_capacity(ram_gb=bad, stage="invalid")
    assert handle and events == []


def test_inactive_caches_cannot_rescue_an_over_capacity_stack(guardian):
    from studio.memory_policy import MemoryPressureError
    manager, state, events = guardian
    state.update(ram=48, available=46)
    with pytest.raises(MemoryPressureError, match="base \\+"):
        manager.prepare_for_adapters({"id": "base", "engine": "wan_vace_fp8", "min_ram_gb": 40, "min_vram_gb": 15},
                                     [{"id": "adapter", "estimated_gb": 3, "min_vram_gb": 1}])
    assert events == []


def test_reused_base_only_charges_incremental_adapter_memory(guardian):
    from studio.catalog import catalog_manager
    manager, state, events = guardian
    state["available"] = 12  # The 40 GiB base is already resident, not reloaded.
    base = catalog_manager.get_definition("wan21-vace-14b-fp8")
    adapter = catalog_manager.get_definition("lora-wan21-14b-fusionx-phantom")
    resident(manager, state, events, "video")
    with manager.operation(slots=("video",)):
        result = manager.prepare_for_adapters(base, [adapter])
    assert result["ram"]["available_gb"] == 12 and events == []


def test_guardian_background_lifecycle_is_idempotent_and_reclaims(guardian):
    manager, state, events = guardian
    released = threading.Event()
    class Handle:
        def unload(self):
            state["available"] += 8
            released.set()
    manager.register_loaded("image", "qwen", "Qwen", Handle())
    manager.poll_seconds = 0.25
    manager.start()
    thread = manager._thread
    manager.start()
    assert manager._thread is thread and manager.guardian_status()["running"]
    state["available"] = 2
    assert released.wait(2)
    manager.stop()
    assert not manager.guardian_status()["running"]
    manager.start()
    assert manager._thread is not thread
    manager.stop()


def test_malformed_poll_interval_does_not_break_app_import(monkeypatch):
    from studio.memory_manager import RuntimeMemoryManager
    monkeypatch.setenv("STUDIO_MEMORY_POLL_SECONDS", "not-a-number")
    assert RuntimeMemoryManager().poll_seconds == 1


def test_tensor_hooks_are_idempotent_thread_scoped_and_abort_before_next_allocation(guardian, monkeypatch):
    from studio.memory_policy import MemoryPressureError
    from studio.memory_safety_hooks import install_memory_safety_hooks
    manager, state, _ = guardian
    calls = []
    module = types.ModuleType("accelerate.utils.modeling")
    def setter(model, name, device="cpu", *, value=None):
        calls.append(name)
        if name == "first":
            state["available"] = 2
            manager.sample()  # Background measurement signals the owning scope.
        return value
    module.set_module_tensor_to_device = setter
    monkeypatch.setitem(sys.modules, "accelerate.utils.modeling", module)
    install_memory_safety_hooks()
    wrapped = module.set_module_tensor_to_device
    install_memory_safety_hooks()
    assert module.set_module_tensor_to_device is wrapped
    assert wrapped(None, "outside", value=7) == 7  # No Studio scope: no global side effects.
    with pytest.raises(MemoryPressureError):
        with manager.operation(slots=("image",)):
            wrapped(None, "first", value=8)
            wrapped(None, "must_not_allocate", value=9)
    assert calls == ["outside", "first"]


def test_lazy_loader_installs_transformers_per_parameter_boundary(guardian, monkeypatch):
    from studio.memory_policy import MemoryPressureError
    from studio.memory_safety_hooks import guarded_model_load
    manager, state, _ = guardian
    module = types.ModuleType("transformers.modeling_utils")
    calls = []
    def infer(*args, **kwargs):
        calls.append("dtype")
        return "unchanged_dtype"
    module._infer_parameter_dtype = infer
    # Simulate the real module being resolved only by the model class import.
    monkeypatch.setitem(sys.modules, "transformers.modeling_utils", module)
    def loader():
        assert module._infer_parameter_dtype is not infer
        assert module._infer_parameter_dtype() == "unchanged_dtype"
        state["available"] = 2
        manager.sample()
        module._infer_parameter_dtype()
        pytest.fail("Next parameter must not be allocated")
    with pytest.raises(MemoryPressureError):
        with manager.operation(slots=("vision",)):
            guarded_model_load(loader)
    assert calls == ["dtype"]


def test_generation_callbacks_check_even_without_a_job(guardian, monkeypatch):
    from studio.memory_safety_hooks import diffusion_memory_callback_kwargs, generation_memory_criteria
    from studio.memory_policy import MemoryPressureError
    manager, state, _ = guardian
    class Pipe:
        def __call__(self, callback_on_step_end=None):
            pass
    transformer = types.ModuleType("transformers")
    transformer.StoppingCriteria = type("StoppingCriteria", (), {})
    transformer.StoppingCriteriaList = list
    monkeypatch.setitem(sys.modules, "transformers", transformer)
    callback = diffusion_memory_callback_kwargs(Pipe())["callback_on_step_end"]
    criteria = generation_memory_criteria()[0]
    for invoke in (lambda: callback(None, 0, 0, {}), lambda: criteria(None, None)):
        state["available"] = 52
        with pytest.raises(MemoryPressureError):
            with manager.operation(slots=("video",)):
                state["available"] = 2
                invoke()


def test_gpu_allocator_is_not_mistaken_for_free_device_memory(monkeypatch):
    from studio import hardware
    props = types.SimpleNamespace(total_memory=22 * GIB, major=8, minor=9)
    def fail(*args):
        raise RuntimeError("driver telemetry unavailable")
    torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True,
        get_device_properties=lambda _: props, get_device_name=lambda _: "NVIDIA L4",
        memory_allocated=lambda _: GIB, memory_reserved=lambda _: 2 * GIB,
        mem_get_info=fail, device_count=lambda: 1))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(hardware, "_query_nvidia_smi", lambda: None)
    result = hardware.query_gpu_info()
    assert result["has_gpu"] and result["vram_free_gb"] is None
    assert result["measurement_source"] == "allocator_only_device_free_unknown"
    monkeypatch.setattr(hardware, "_query_nvidia_smi", lambda: {"has_gpu": True, "vram_free_gb": 0.5, "measurement_source": "nvidia_smi_driver"})
    assert hardware.query_gpu_info()["vram_free_gb"] == 0.5


@pytest.mark.parametrize("engine_name,slot", [("audio", "audio_bgm"), ("lipsync", "lipsync")])
def test_reregistering_self_owned_cache_does_not_unload_its_new_state(guardian, engine_name, slot):
    from studio.catalog import catalog_manager
    from studio.engines.audio_studio_engine import AudioStudioEngine
    from studio.engines.lipsync_engine import LipSyncEngine
    manager, _, _ = guardian
    owner = AudioStudioEngine(catalog_manager) if engine_name == "audio" else LipSyncEngine(catalog_manager)
    marker = object()
    manager.register_loaded(slot, "old", "old", owner)
    if engine_name == "audio":
        owner._pipe = marker
    else:
        owner._sessions = {"edtalk": marker}
    owner.loaded_model_id = "new"
    manager.register_loaded(slot, "new", "new", owner)
    assert owner.loaded_model_id == "new"
    assert (owner._pipe if engine_name == "audio" else owner._sessions["edtalk"]) is marker


def test_partial_onnx_load_releases_private_sessions_after_unwinding_lock(guardian, monkeypatch, tmp_path):
    from studio.catalog import catalog_manager
    from studio.engines import lipsync_engine as engine
    from studio.memory_policy import MemoryPressureError
    manager, _, _ = guardian
    monkeypatch.setattr(engine, "memory_manager", manager)
    definition = catalog_manager.get_definition("edtalk-256-lipsync-hd")
    catalog = types.SimpleNamespace(get_definition=lambda _: definition, is_downloaded=lambda _: True,
                                    model_dir=lambda _: tmp_path)
    owner = engine.LipSyncEngine(catalog)
    original = owner.unload
    def checked_unload():
        assert not owner._lock.locked(), "Do not recursively take a private non-reentrant lock"
        original()
    monkeypatch.setattr(owner, "unload", checked_unload)
    refs = []
    class Session:
        def __init__(self, path, **kwargs):
            if "edtalk" in path:
                raise MemoryPressureError("MEMORY_SAFETY_STOP: simulated native boundary")
            refs.append(weakref.ref(self))
    ort = types.SimpleNamespace(get_available_providers=lambda: ["CPUExecutionProvider"],
        SessionOptions=lambda: types.SimpleNamespace(),
        GraphOptimizationLevel=types.SimpleNamespace(ORT_ENABLE_ALL=1), InferenceSession=Session)
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    with pytest.raises(MemoryPressureError):
        owner._ensure_loaded(definition["id"])
    gc.collect()
    assert owner._sessions == {} and owner.loaded_model_id is None
    assert len(refs) == 1 and refs[0]() is None and not manager.loaded_model_ids()


def test_partial_musicgen_load_clears_processor_without_private_lock_deadlock(guardian, monkeypatch, tmp_path):
    from studio.catalog import catalog_manager
    from studio.jobs import JobRecord
    from studio.engines import audio_studio_engine as engine
    manager, _, _ = guardian
    monkeypatch.setattr(engine, "memory_manager", manager)
    definition = catalog_manager.get_definition("musicgen-small-bgm")
    catalog = types.SimpleNamespace(get_definition=lambda _: definition, is_downloaded=lambda _: True,
                                    model_dir=lambda _: tmp_path)
    owner = engine.AudioStudioEngine(catalog)
    original = owner.unload
    def checked_unload():
        assert not owner._lock.locked(), "Never call unload while holding the private lock"
        original()
    monkeypatch.setattr(owner, "unload", checked_unload)
    def fail(*args, **kwargs):
        raise RuntimeError("invalid local weights")
    transformers = types.SimpleNamespace(AutoProcessor=types.SimpleNamespace(from_pretrained=lambda *a, **k: object()),
        MusicgenForConditionalGeneration=types.SimpleNamespace(from_pretrained=fail))
    torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True, empty_cache=lambda: None),
                                  float16="fp16", float32="fp32")
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "torch", torch)
    with pytest.raises(RuntimeError, match="invalid local weights"):
        owner.generate_music_or_ambience(JobRecord("audio", "partial"), prompt="jazz", model_id=definition["id"])
    assert owner._processor is None and owner._pipe is None and owner.loaded_model_id is None
    assert not manager.loaded_model_ids()


def test_preset_assesses_all_three_video_adapters_and_never_changes_recipe():
    from studio.agent_presets import compile_agent_workflow_preset
    gpu = {"has_gpu": True, "gpu_name": "NVIDIA L4", "gpu_tier": "L4", "bf16_supported": True,
           "vram_total_gb": 22.03, "vram_free_gb": 21}
    hardware = {"gpu": gpu, "ram": {"total_gb": 48, "available_gb": 46}, "disk": {"free_gb": 256}}
    plan = compile_agent_workflow_preset(gpu_override=gpu, hardware_override=hardware)
    stack = plan["detected_gpu"]["selected_video_stack"]
    assert stack["model_id"] == "wan21-vace-14b-fp8"  # No fallback download cascade.
    assert stack["memory_assessment"]["resources"]["ram"]["required_gb"] == pytest.approx(44.47)
    assert not plan["video_stack_capacity_ok"] and plan["planning_only"]
    assert all(step["do_not_execute"] for step in plan["execution_plan"] if step.get("tool") == "studio_generate_video")
    assert not plan["execution_policy"]["allow_automatic_fallback_downloads"]
    hardware["ram"] = {"total_gb": 53, "available_gb": 50.5}
    assert compile_agent_workflow_preset(gpu_override=gpu, hardware_override=hardware)["video_stack_capacity_ok"]


def test_mcp_rejects_combined_stack_before_submitting_or_downloading(monkeypatch):
    from studio import mcp_server as mcp
    from studio.model_compatibility import assessment_context
    snapshot = {"gpu": {"has_gpu": True, "gpu_tier": "L4", "vram_total_gb": 22, "vram_free_gb": 21,
                         "bf16_supported": True},
                "ram": {"total_gb": 48, "available_gb": 46}, "disk": {"free_gb": 256}}
    monkeypatch.setattr(mcp, "assessment_context", lambda **kw: assessment_context(snapshot=snapshot))
    def forbidden(*args, **kwargs):
        pytest.fail("Rejected admission must not enqueue inference or download weights")
    monkeypatch.setattr(mcp.job_manager, "submit", forbidden)
    monkeypatch.setattr(mcp.catalog_manager, "download_model", forbidden)
    monkeypatch.setattr(mcp.catalog_manager, "check_disk_space", lambda *a, **kw: {"ok": True})
    result = mcp._execute_mcp_tool("studio_generate_video", {"model_id": "wan21-vace-14b-fp8", "video_submode": "t2v",
        "prompt": "two friends walking", "video_lora_id": "lora-wan21-14b-fusionx-phantom",
        "extra_video_lora_ids": ["lora-wan21-14b-unianimate-char", "lora-wan21-14b-moviigen-cinema"]})
    payload = json.loads(result["content"][0]["text"])
    assert result["isError"] and payload["error_code"] == "MODEL_RESOURCE_CONSTRAINT"
    assert len(payload["assessment"]["stack"]["adapter_ids"]) == 3


def test_status_ui_and_mcp_expose_same_running_guardian(temp_studio_env):
    from studio import mcp_server as mcp
    client, _ = temp_studio_env
    status = client.get("/api/status").json()["memory"]["guardian"]
    memory = client.get("/api/memory/status").json()["guardian"]
    assert status["enabled"] and status["running"] and memory["running"]
    assert memory["poll_seconds"] >= 0.25 and "last_observation" in memory
    html = client.get("/").text
    assert 'id="memory-guardian-state"' in html and 'id="memory-guardian-reserve"' in html
    result = mcp._execute_mcp_tool("studio_get_state_and_manual", {"include_recent_assets": False})
    assert json.loads(result["content"][0]["text"])["memory_guardian"]["running"]


def test_guardian_events_are_bounded_redacted_and_available_offline(guardian, temp_studio_env, tmp_path):
    from studio.diagnostics import export_diagnostic_bundle
    manager, _, _ = guardian
    for index in range(45):
        manager._event("test", f"event {index} Bearer TESTSECRET123")
    snapshot = manager.guardian_status()
    assert len(snapshot["recent_events"]) == 12 and len(manager._events) == 40
    snapshot["recent_events"].clear()
    assert len(manager.guardian_status()["recent_events"]) == 12
    _, root = temp_studio_env
    journal = root / "state" / "memory_guardian.json"
    assert "TESTSECRET123" not in journal.read_text() and "REDACTED" in journal.read_text()
    bundle = export_diagnostic_bundle(root, tmp_path / "diagnostic.zip")
    with zipfile.ZipFile(bundle) as archive:
        assert "state/memory_guardian.json" in archive.namelist()
        assert b"TESTSECRET123" not in archive.read("state/memory_guardian.json")


def test_loader_translation_cannot_hide_memory_safety_exception(guardian):
    from studio.memory_safety_hooks import guarded_model_load
    from studio.memory_policy import MemoryPressureError
    manager, _, _ = guardian
    def translating_loader():
        try:
            raise MemoryPressureError("MEMORY_SAFETY_STOP: checkpoint")
        except Exception as exc:
            raise OSError("weights could not be loaded") from exc
    with pytest.raises(MemoryPressureError, match="MEMORY_SAFETY_STOP"):
        with manager.operation(slots=("image",)):
            guarded_model_load(translating_loader)


def test_tensor_boundary_still_honors_manual_cancellation(guardian, monkeypatch):
    from studio.memory_safety_hooks import install_memory_safety_hooks
    from studio.jobs import JobRecord, JobCancelledError
    manager, _, _ = guardian
    job = JobRecord("image", "cancel")
    called = []
    module = types.ModuleType("accelerate.utils.modeling")
    module.set_module_tensor_to_device = lambda *a, **kw: called.append(True)
    monkeypatch.setitem(sys.modules, "accelerate.utils.modeling", module)
    install_memory_safety_hooks()
    with pytest.raises(JobCancelledError):
        with manager.operation(slots=("image",), job=job):
            job.cancel_requested = True
            module.set_module_tensor_to_device(None, "weight", "cpu")
    assert called == []


def test_memory_stopped_job_keeps_diagnostics_and_worker_usable(guardian, temp_studio_env):
    from studio.jobs import JobManager
    manager, state, events = guardian
    resident(manager, state, events, "video", ram=8)
    jobs = JobManager(max_workers=1)
    def fail(job):
        with manager.operation(slots=("video",), job=job):
            state["available"] = 2
            manager.sample()
            job.check_cancelled()
        return {}
    try:
        job = jobs.submit("video", "pressure", fail)
        deadline = time.monotonic() + 3
        while job.status in ("queued", "running") and time.monotonic() < deadline:
            time.sleep(0.01)
        assert job.status == "failed" and "MEMORY_SAFETY_STOP" in job.error
        assert job.to_dict()["memory_protection"]["action"] == "safety_stop"
        assert job.current_step == "Protección de memoria"
        client, _ = temp_studio_env
        assert client.get("/api/status").status_code == 200
        next_job = jobs.submit("test", "still usable", lambda _: {"ok": True})
        deadline = time.monotonic() + 3
        while next_job.status in ("queued", "running") and time.monotonic() < deadline:
            time.sleep(0.01)
        assert next_job.status == "completed"
    finally:
        jobs._executor.shutdown(wait=True)


def test_tts_factory_holds_same_gpu_lease_as_image_and_video(guardian, monkeypatch, tmp_path):
    from studio.engines import tts_engine as engine
    from studio.catalog import catalog_manager
    manager, _, _ = guardian
    definition = catalog_manager.get_definition("kokoro-82m-tts")
    catalog = types.SimpleNamespace(get_definition=lambda _: definition, is_downloaded=lambda _: True,
                                    model_dir=lambda _: tmp_path)
    monkeypatch.setattr(engine, "catalog_manager", catalog)
    monkeypatch.setattr(engine, "memory_manager", manager)
    entered, release, waiting, unloaded = (threading.Event() for _ in range(4))
    failures = []
    class Model:
        def __init__(self, **kwargs):
            entered.set()
            assert release.wait(3)
        def to(self, device):
            return self
        def eval(self):
            return self
    monkeypatch.setitem(sys.modules, "kokoro", types.SimpleNamespace(KModel=Model, KPipeline=lambda **kw: object()))
    def load():
        try:
            engine.load_tts_model_into_memory(definition["id"])
        except BaseException as exc:
            failures.append(exc)
    def unload():
        waiting.set()
        manager.unload_all()
        unloaded.set()
    owner = threading.Thread(target=load, daemon=True)
    waiter = threading.Thread(target=unload, daemon=True)
    owner.start()
    try:
        assert entered.wait(2)
        assert manager.guardian_status()["protected_slots"] == ["tts"]
        waiter.start()
        assert waiting.wait(2) and not unloaded.wait(0.05)
    finally:
        release.set()
        owner.join(3)
        if waiter.ident:
            waiter.join(3)
    assert not owner.is_alive() and not waiter.is_alive() and failures == []
    assert unloaded.is_set() and not manager.loaded_model_ids()


def test_native_load_stops_at_preventive_reserve_but_running_inference_is_not_ejected(guardian):
    from studio.memory_safety_hooks import guarded_model_load
    from studio.memory_policy import MemoryPressureError
    manager, state, events = guardian
    model = resident(manager, state, events, "video")
    with manager.operation(slots=("video",)):
        state["available"] = 5  # Soft <6.4 GiB, but still above critical 3.2.
        manager.checkpoint(force=True)
        assert model and "video" not in events
    with pytest.raises(MemoryPressureError, match="reserva completa"):
        with manager.operation(slots=("video",)):
            guarded_model_load(lambda: pytest.fail("No new model tensor may consume the reserve"))
    assert model == {}


def test_cooperative_hook_stops_a_real_rss_allocating_subprocess(tmp_path):
    """Physical bytearrays + real RSS, with an explicit synthetic quota, not CUDA.

    The test commits only ~128 MiB in an isolated child. It never exhausts the
    sandbox or claims that a watchdog can interrupt an arbitrary native call.
    """
    import os
    import subprocess
    script = r'''
import json, sys, types
import psutil
from studio import memory_manager as mm
from studio.memory_policy import MemoryPressureError
from studio.memory_safety_hooks import guarded_model_load
GIB = 1024 ** 3
process = psutil.Process()
baseline = process.memory_info().rss
quota = baseline + int(1.125 * GIB)
def host():
    current = process.memory_info().rss
    available = max(0, quota - current) / GIB
    return {"total_gb": quota / GIB, "available_gb": available,
            "used_gb": current / GIB, "source": "synthetic_quota_real_rss"}
mm.host_memory = host
mm.query_gpu_info = lambda: {"has_gpu": False, "vram_total_gb": 0,
                             "vram_free_gb": 0, "vram_used_gb": 0}
manager = mm.RuntimeMemoryManager()
blocks = []
module = types.ModuleType("accelerate.utils.modeling")
def setter(*args, **kwargs):
    blocks.append(bytearray(16 * 1024 ** 2))
    manager.sample()
module.set_module_tensor_to_device = setter
sys.modules[module.__name__] = module
def load():
    for index in range(20):
        module.set_module_tensor_to_device(None, str(index), "cpu")
try:
    with manager.operation(slots=("video",)):
        guarded_model_load(load)
except MemoryPressureError as exc:
    peak = (process.memory_info().rss - baseline) / GIB
    assert 1 <= len(blocks) <= 12, len(blocks)
    assert peak < 0.3, peak
    print(json.dumps({"stopped": True, "allocations": len(blocks), "rss_growth_gib": peak,
                      "code": exc.code}))
else:
    raise AssertionError("The real allocating loop was not stopped")
'''
    environment = {**os.environ, "STUDIO_DATA_DIR": str(tmp_path / "child-data"), "STUDIO_FORCE_LOCAL": "1"}
    result = subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).resolve().parent.parent,
                            env=environment, text=True, capture_output=True, timeout=12)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["stopped"] and report["code"] == "MEMORY_SAFETY_STOP"


def test_measurement_failure_stops_allocations_but_cannot_block_cleanup(guardian, monkeypatch):
    import studio.memory_manager as mm
    from studio.memory_policy import MemoryPressureError
    manager, state, events = guardian
    handle = resident(manager, state, events, "video")
    def unavailable():
        raise OSError("telemetry unavailable")
    with pytest.raises(MemoryPressureError, match="MEMORY_MEASUREMENT_UNAVAILABLE"):
        with manager.operation(slots=("video",)):
            monkeypatch.setattr(mm, "host_memory", unavailable)
            manager.checkpoint(force=True)
    assert handle == {} and not manager.loaded_model_ids()
    assert manager.guardian_status()["recent_events"][-1]["action"] == "safety_stop"
    # A daemon surviving an error is not the same as healthy measurements.
    manager._event("monitor_error", "telemetry unavailable")
    assert not manager.guardian_status()["measurements_healthy"]


def test_nested_scopes_keep_parent_pins_until_the_last_owner_exits(guardian):
    manager, state, events = guardian
    resident(manager, state, events, "image")
    resident(manager, state, events, "vision")
    with manager.operation(slots=("image",)):
        with manager.operation(slots=("image", "vision")):
            assert manager._pins["image"] == 2
            assert manager.guardian_status()["protected_slots"] == ["image", "vision"]
        assert manager._pins["image"] == 1
        assert manager.guardian_status()["protected_slots"] == ["image"]
    assert manager.guardian_status()["protected_slots"] == []


def test_nested_no_job_callback_inherits_parent_cancellation(guardian):
    from studio.jobs import JobRecord, JobCancelledError
    from studio.memory_manager import checkpoint_current_operation
    manager, _, _ = guardian
    job = JobRecord("vision", "parent")
    with pytest.raises(JobCancelledError):
        with manager.operation(slots=("vision",), job=job):
            with manager.operation(slots=("vision",)):
                assert manager.guardian_status()["active_operations"][-1]["job_id"] == job.id
                job.cancel_requested = True
                checkpoint_current_operation(force=True)
    assert not manager.guardian_status()["protected_slots"]


def test_live_memory_helper_enforces_loading_reserve_not_only_telemetry(guardian):
    from studio.memory_manager import checkpoint_current_operation, run_with_live_memory_progress
    from studio.memory_policy import MemoryPressureError
    manager, state, _ = guardian
    def factory():
        state["available"] = 5  # Depleted preventive reserve, not yet critical.
        checkpoint_current_operation(force=True)
        pytest.fail("Do not continue allocating through the loading reserve")
    with pytest.raises(MemoryPressureError, match="MEMORY_SAFETY_STOP"):
        with manager.operation(slots=("video",)):
            run_with_live_memory_progress(None, 0, 100, "load", factory)
