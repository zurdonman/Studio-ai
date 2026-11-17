"""Resident mode is opt-in; test cache pruning and fail-closed lifecycle without GPU."""
from types import SimpleNamespace

import pytest

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('resident_core', Path(__file__).resolve().parents[1] / 'studio/engines/comfy_resident_node/core.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
prune_completed = module.prune_completed


def fake_executor():
    cache = {f'key{i}': object() for i in range(1, 8)}
    keys = SimpleNamespace(get_data_key=lambda node: 'key' + node)
    outputs = SimpleNamespace(initialized=True, cache_key_set=keys, cache=cache,
                              subcaches={'nested': object()}, used_generation=dict.fromkeys(cache, 1),
                              timestamps=dict.fromkeys(cache, 1), children={'key5': []})
    objects = SimpleNamespace(cache={'any': object()}, subcaches={'nested': object()})
    graph = {str(i): {'class_type': cls} for i, cls in enumerate(
        ('UnetLoaderGGUF', 'CLIPLoader', 'VAELoader', 'VAELoader'), 1)}
    return SimpleNamespace(caches=SimpleNamespace(outputs=outputs, objects=objects)), graph


def test_only_base_loaders_retained_and_bookkeeping_pruned():
    executor, graph = fake_executor()
    original = executor.caches.outputs.cache.copy()
    result = prune_completed(executor, graph)
    out = executor.caches.outputs
    assert result == {'retained_loaders': 4, 'total_loaders': 4}
    assert set(out.cache) == {'key1', 'key2', 'key3', 'key4'}
    assert all(out.cache[k] is original[k] for k in out.cache)
    assert set(out.used_generation) == set(out.timestamps) == set(out.cache)
    assert out.children == out.subcaches == executor.caches.objects.cache == executor.caches.objects.subcaches == {}


def test_unrecognized_graph_fails_closed():
    executor, graph = fake_executor()
    graph['1']['class_type'] = 'Other'
    with pytest.raises(ValueError):
        prune_completed(executor, graph)
    assert len(executor.caches.outputs.cache) == 7


def test_ambiguous_or_unsupported_cache_fails_closed():
    executor, graph = fake_executor()
    executor.caches.outputs.cache_key_set.get_data_key = lambda node: 'same'
    with pytest.raises(RuntimeError):
        prune_completed(executor, graph)
    assert len(executor.caches.outputs.cache) == 7


@pytest.mark.parametrize('success,clean,cancel,alive,expected', [
    (True, True, False, True, True), (True, False, False, True, False),
    (False, True, False, True, False), (True, True, True, True, False),
    (True, True, False, False, False),
])
def test_finish_keeps_only_confirmed_live_success(temp_studio_env, monkeypatch, tmp_path, success, clean, cancel, alive, expected):
    from studio.engines import ltx_gguf_backend as backend
    calls = []
    monkeypatch.setattr(backend.memory_manager, 'unload_slot', lambda slot: calls.append(slot))
    root = tmp_path / 'session'
    for p in ('in', 'out', 'temp/temp'):
        (root / p).mkdir(parents=True)
        (root / p / 'scratch').write_text('temporary')
    session = SimpleNamespace(keep_loaded=True, cleanup_ok=clean, alive=lambda: alive,
                              root=root, input_dir=root/'in', output_dir=root/'out',
                              guard=SimpleNamespace(job='old'))
    job = SimpleNamespace(comfy_keep_loaded=True, cancel_requested=cancel, log=lambda *a, **k: None)
    backend._finish_session(session, job, success)
    assert (calls == []) is expected
    if expected:
        assert not list((root/'in').iterdir()) and not list((root/'out').iterdir())
        assert (root/'temp'/'temp').is_dir() and not list((root/'temp'/'temp').iterdir())
        assert session.guard.job is None


def test_cleanup_error_closes_process(temp_studio_env, monkeypatch, tmp_path):
    from studio.engines import ltx_gguf_backend as backend
    calls = []
    monkeypatch.setattr(backend.memory_manager, 'unload_slot', lambda slot: calls.append(slot))
    root = tmp_path / 'session'; root.mkdir()
    (root/'in').symlink_to(tmp_path, target_is_directory=True)
    session = SimpleNamespace(keep_loaded=True, cleanup_ok=True, alive=lambda: True,
                              root=root, input_dir=root/'in', output_dir=root/'out')
    job = SimpleNamespace(comfy_keep_loaded=True, cancel_requested=False, log=lambda *a, **k: None)
    backend._finish_session(session, job, True)
    assert calls == ['video']


def test_api_option_propagates_and_defaults_off(temp_studio_env):
    import app
    job = SimpleNamespace()
    assert app.LTXGGUFReq(prompt='test').comfy_keep_loaded is True
    app._comfy_job_function(lambda j: j.comfy_keep_loaded, ('auto', False), True)(job)
    assert job.comfy_keep_loaded is True and job.comfy_options == ('auto', False)


def test_optional_plugin_command_and_conflict(temp_studio_env, tmp_path):
    from studio.engines import ltx_gguf_backend as backend
    normal = backend._server_command(12345, tmp_path/'in', tmp_path/'out', tmp_path)
    experimental = backend._server_command(12345, tmp_path/'in', tmp_path/'out', tmp_path, keep_loaded=True)
    assert 'StudioResident' not in normal and 'StudioResident' in experimental
    backend._install_resident_plugin(tmp_path)
    backend._install_resident_plugin(tmp_path)
    target = tmp_path/'comfy'/'custom_nodes'/'StudioResident'/'core.py'
    target.write_text('other file, do not overwrite')
    with pytest.raises(RuntimeError, match='CONFLICT'):
        backend._install_resident_plugin(tmp_path)
    assert target.read_text() == 'other file, do not overwrite'


@pytest.mark.parametrize('engine,model,extra', [('wan','StudioWanAnimate2GGUF',False), ('wan','UNETLoader',False), ('scail','UnetLoaderGGUF',True)])
def test_motion_prunes_intermediates_and_retains_compatible_loaders(engine, model, extra):
    executor, graph = fake_executor()
    graph['1']['class_type'] = model
    graph['4']['class_type'] = 'CLIPVisionLoader'
    graph['14'] = {'class_type': 'WanSCAILToVideo' if extra else 'WanAnimate2ToVideo'}
    if extra:
        graph['20'] = {'class_type':'CheckpointLoaderSimple'}
        executor.caches.outputs.cache['key20'] = object()
    result = prune_completed(executor, graph)
    assert result['total_loaders'] == (5 if extra else 4)
    assert set(executor.caches.outputs.cache) == ({'key1','key2','key3','key4','key20'} if extra else {'key1','key2','key3','key4'})


def test_film_graph_keeps_interpolation_loader_only():
    executor, graph = fake_executor()
    graph = {'1':{'class_type':'LoadVideo'}, '2':{'class_type':'GetVideoComponents'},
             '3':{'class_type':'FrameInterpolationModelLoader'}, '4':{'class_type':'FrameInterpolate'},
             '5':{'class_type':'CreateVideo'}, '6':{'class_type':'SaveVideo'}}
    assert prune_completed(executor, graph) == {'retained_loaders':1, 'total_loaders':1}
    assert set(executor.caches.outputs.cache) == {'key3'}
