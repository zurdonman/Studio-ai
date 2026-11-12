"""Synthetic and CPU-only coverage; never downloads weights or runs AI."""
import hashlib
import subprocess

import pytest

from studio.engines.ltx_gguf_workflow import make_graph


def graph(**overrides):
    kw = dict(prompt="A person speaks", negative_prompt="", width=512, height=320,
              frames=49, fps=24, seed=42, image_name="frame.png", model="model.gguf",
              text_encoder="gemma.gguf", connectors="connectors.safetensors",
              video_vae="video.safetensors", audio_vae="audio.safetensors")
    kw.update(overrides)
    return make_graph(**kw)


def test_gguf_graph_has_joint_audio_video_and_no_injected_user_nodes():
    g = graph()
    assert g["1"]["class_type"] == "UnetLoaderGGUF"
    assert g["2"]["class_type"] == "DualCLIPLoaderGGUF"
    assert g["10"]["inputs"]["video_latent"] == ["22", 0]
    assert g["19"]["inputs"]["audio"] == ["18", 0]
    assert len(g["14"]["inputs"]["sigmas"].split(",")) == 9
    assert graph(image_name=None)["10"]["inputs"]["video_latent"] == ["8", 0]
    assert graph(image_strength=.7)["22"]["inputs"]["strength"] == .7
    assert graph(frames=601)["9"]["inputs"]["frames_number"] == 601
    for kw in ({"frames":609}, {"image_strength":.49}, {"image_strength":float("nan")}):
        with pytest.raises(ValueError):
            graph(**kw)
    for kw in ({"model": "../evil.gguf"}, {"image_name":"/tmp/x.png"}, {"width":513}, {"frames":48}, {"fps":float("nan")}):
        with pytest.raises(ValueError):
            graph(**kw)


def test_download_hash_verification(tmp_path):
    from studio.engines.ltx_gguf_download import verify
    p = tmp_path / 'file'
    p.write_bytes(b'abc')
    digest = hashlib.sha256(b'abc').hexdigest()
    assert verify(p, 3, digest)
    assert not verify(p, 2, digest)
    assert not verify(p, 3, '0' * 64)


def test_gguf_cpu_does_not_install_or_download(temp_studio_env, monkeypatch):
    from studio.engines import ltx_gguf_backend as b
    from studio.jobs import JobRecord
    calls = []
    monkeypatch.setattr(b, 'query_gpu_info', lambda: {'has_gpu': False})
    monkeypatch.setattr(b, '_install', lambda *a: calls.append('install'))
    monkeypatch.setattr(b, '_prepare_models', lambda *a: calls.append('download'))
    with pytest.raises(RuntimeError, match='GGUF_GPU_UNSUPPORTED'):
        b.generate_gguf(image_path=None, prompt='test', negative_prompt='', seed=42, job=JobRecord('video', 'test'))
    assert calls == []


def test_gguf_real_api_route_with_synthetic_av(temp_studio_env, monkeypatch):
    from studio.engines import ltx_gguf_backend as b
    from studio.engines.ffmpeg_utils import get_ffmpeg_bin, probe_media_info
    from studio.jobs import JobRecord
    from PIL import Image
    _, root = temp_studio_env
    img = root / 'uploads' / 'frame.png'
    Image.new('RGB', (256, 256), 'blue').save(img)
    monkeypatch.setattr(b, 'query_gpu_info', lambda: {'has_gpu':True, 'bf16_supported':True, 'vram_total_gb':22})
    monkeypatch.setattr(b.memory_manager, 'get_slot', lambda *_: None)
    capacity_calls = []
    monkeypatch.setattr(b.memory_manager, 'ensure_capacity', lambda **kwargs: capacity_calls.append(kwargs))
    monkeypatch.setattr(b, '_install', lambda *_: None)
    monkeypatch.setattr(b, '_probe_runtime', lambda *_: None)
    monkeypatch.setattr(b, '_prepare_models', lambda *_: None)
    monkeypatch.setattr(b, '_make_model_links', lambda *_: {'dit':'x.gguf', 'text_encoder':'g.gguf', 'connectors':'c.safetensors', 'video_vae':'v.safetensors', 'audio_vae':'a.safetensors'})
    monkeypatch.setattr(b.shutil, 'disk_usage', lambda *_: type('D', (), {'free': 100 * 1024**3})())
    seen = []
    from types import SimpleNamespace
    inp, out = root / 'fake_input', root / 'fake_output'
    inp.mkdir(exist_ok=True); out.mkdir(exist_ok=True)
    monkeypatch.setattr(b, '_open_resident', lambda *_: SimpleNamespace(input_dir=inp, output_dir=out))
    def synth(g, inp, out, job):
        assert (inp / g['21']['inputs']['image']).is_file()
        assert g['10']['inputs']['video_latent'] == ['22', 0]
        seen.append(g)
        path = out / 'studio_gguf_00001_.mp4'
        subprocess.run([get_ffmpeg_bin(), '-v', 'error', '-f','lavfi','-i','color=c=blue:s=320x256:r=24:d=1',
                        '-f','lavfi','-i','sine=frequency=440:sample_rate=24000:duration=1',
                        '-c:v','libx264','-pix_fmt','yuv420p','-c:a','aac','-shortest','-y',str(path)], check=True)
        return path
    monkeypatch.setattr(b, '_infer', synth)
    result = b.generate_gguf(image_path=img, prompt='test', negative_prompt='', seed=42, job=JobRecord('video','test'),
                             duration_sec=20, fps=24, width=512, height=320, speech_language='es', image_strength=.7)
    info = probe_media_info(root / result['rel_path'])
    assert info['has_video'] and info['has_audio'] and result['audio_native'] is True and len(seen) == 1
    assert seen[0]['8']['inputs']['length'] == 481 and seen[0]['9']['inputs']['frames_number'] == 481
    assert seen[0]['22']['inputs']['strength'] == .7
    assert 'spoken dialogue is in Spanish' in seen[0]['5']['inputs']['text']
    assert result['frames'] == 481 and result['requested_duration_sec'] == 20
    assert result['speech_language_hint'] == 'es' and result['image_strength'] == .7
    assert any(c['ram_gb'] > 32 and c['vram_gb'] > 14 for c in capacity_calls)


def test_gguf_api_input_validation_and_panel(temp_studio_env):
    client, _ = temp_studio_env
    assert client.post('/api/video/ltx-gguf', json={'prompt':'   '}).status_code == 400
    assert client.post('/api/video/ltx-gguf', json={'prompt':'test', 'image_rel':'../escape'}).status_code in (400,404)
    assert 'id="ltx-gguf-panel"' in client.get('/').text


def test_gguf_http_submits_job_with_image(temp_studio_env, monkeypatch):
    import time
    from studio.engines import ltx_gguf_backend as b
    from PIL import Image
    client, root = temp_studio_env
    image = root / 'uploads' / 'frame.png'
    Image.new('RGB', (16, 16)).save(image)
    calls = []
    def fake(**kw):
        calls.append(kw)
        return {'media_url':'/media/outputs/videos/synthetic.mp4'}
    monkeypatch.setattr(b, 'generate_gguf', fake)
    r = client.post('/api/video/ltx-gguf', json={'prompt':'Hola', 'image_rel':'uploads/frame.png', 'seed':7, 'duration_sec':20, 'speech_language':'es', 'image_strength':.75})
    assert r.status_code == 200
    ident = r.json()['job']['id']
    for _ in range(50):
        if calls:
            break
        time.sleep(.02)
    assert calls and calls[0]['image_path'] == image and calls[0]['seed'] == 7
    assert calls[0]['duration_sec'] == 20 and calls[0]['speech_language'] == 'es' and calls[0]['image_strength'] == .75
    assert ident


def test_setup_child_cancellation(temp_studio_env, tmp_path):
    import sys
    import threading
    from studio.jobs import JobCancelledError, JobRecord
    from studio.engines.ltx_gguf_backend import _command
    job = JobRecord('video', 'cancel')
    sentinel = tmp_path / 'never'
    timer = threading.Timer(.4, lambda: setattr(job, 'cancel_requested', True))
    timer.start()
    try:
        with pytest.raises(JobCancelledError):
            _command([sys.executable, '-c', f'import time;time.sleep(3);open({str(sentinel)!r},"w").write("x")'], job, stage='test', timeout=10)
    finally:
        timer.cancel()
    assert not sentinel.exists()


def test_gguf_catalog_filter_and_prepare_endpoint(temp_studio_env, monkeypatch):
    import time
    from studio.engines import ltx_gguf_backend as b
    client, _ = temp_studio_env
    model_id = "ltx-2.3-distilled-gguf-q4-experimental"
    unfiltered = client.get('/api/catalog?gpu_profile=L4').json()['models']
    entry = next(m for m in unfiltered if m['id'] == model_id)
    assert entry['engine'] == 'ltx_gguf' and entry['compatibility']['compatible'] is False
    assert not any(m['id'] == model_id for m in client.get('/api/catalog?gpu_profile=L4&only_compatible=true').json()['models'])
    calls = []
    monkeypatch.setattr(b, 'prepare_gguf', lambda *, job: calls.append(job) or {'prepared': True})
    res = client.post('/api/video/ltx-gguf/prepare')
    assert res.status_code == 200 and res.json()['job']['id']
    for _ in range(50):
        if calls:
            break
        time.sleep(.02)
    assert calls


def test_gguf_prepare_checks_guard_and_does_not_infer(temp_studio_env, monkeypatch):
    from studio.engines import ltx_gguf_backend as b
    from studio.jobs import JobRecord
    calls = []
    monkeypatch.setattr(b, 'query_gpu_info', lambda: {'has_gpu': False})
    monkeypatch.setattr(b, '_install', lambda *_: calls.append('install'))
    monkeypatch.setattr(b, '_infer', lambda *_: calls.append('infer'))
    with pytest.raises(RuntimeError, match='GGUF_GPU_UNSUPPORTED'):
        b.prepare_gguf(job=JobRecord('download', 'prepare'))
    assert calls == []
    monkeypatch.setattr(b, '_prepare_gguf', lambda *_: calls.append('prepare'))
    assert b.prepare_gguf(job=JobRecord('download', 'prepare'))['prepared']
    assert calls == ['prepare']


def test_gguf_repairs_partial_venv_without_touching_data(temp_studio_env, monkeypatch, tmp_path):
    import os
    import sys
    from studio.engines import ltx_gguf_backend as b
    from studio.jobs import JobRecord
    root = tmp_path / 'runtime'
    root.mkdir()
    monkeypatch.setattr(b, 'ROOT', root)
    subprocess.run([sys.executable, '-m', 'venv', '--without-pip', str(root / 'venv')], check=True)
    saved = root / 'venv' / 'preserved.txt'
    saved.write_text('keep')
    monkeypatch.setenv('PIP_USER', '1')
    monkeypatch.setenv('PYTHONPATH', '/nonexistent')
    py = b._prepare_venv(JobRecord('download', 'repair'))
    assert saved.read_text() == 'keep'
    assert 'include-system-site-packages = true' in (root / 'venv' / 'pyvenv.cfg').read_text()
    script = 'import pathlib, sys, pip; print(pathlib.Path(pip.__file__).is_relative_to(pathlib.Path(sys.prefix)))'
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    assert subprocess.check_output([str(py), '-c', script], env=env, text=True).strip() == 'True'


def test_gguf_venv_bootstrap_fallback_is_isolated(temp_studio_env, monkeypatch, tmp_path):
    import ensurepip
    import os
    from pathlib import Path
    from studio.engines import ltx_gguf_backend as b
    from studio.jobs import JobRecord
    bundled = Path(getattr(ensurepip, '_WHEEL_PKG_DIR', None) or Path(ensurepip.__file__).parent / '_bundled')
    if not list(bundled.glob('pip*.whl')):
        pytest.skip('Python has no bundled pip wheel')
    root = tmp_path / 'runtime'
    root.mkdir()
    monkeypatch.setattr(b, 'ROOT', root)
    monkeypatch.setenv('PIP_NO_INDEX', '1')
    monkeypatch.setenv('PIP_FIND_LINKS', str(bundled))
    monkeypatch.setenv('PIP_USER', '1')
    original = b._command
    calls = []
    def fail_ensurepip(args, job, **kwargs):
        calls.append(args)
        if '-m' in args and 'ensurepip' in args:
            raise RuntimeError('simulated ensurepip failure')
        return original(args, job, **kwargs)
    monkeypatch.setattr(b, '_command', fail_ensurepip)
    py = b._prepare_venv(JobRecord('download', 'fallback'))
    assert any('--python' in args and str(py) in args for args in calls)
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    code = 'import pathlib, sys, pip; print(pathlib.Path(pip.__file__).is_relative_to(pathlib.Path(sys.prefix)))'
    assert subprocess.check_output([str(py), '-c', code], env=env, text=True).strip() == 'True'


def test_gguf_refuses_unmanaged_venv(temp_studio_env, monkeypatch, tmp_path):
    from studio.engines import ltx_gguf_backend as b
    from studio.jobs import JobRecord
    monkeypatch.setattr(b, 'ROOT', tmp_path)
    venv = tmp_path / 'venv'
    venv.mkdir()
    (venv / 'user-file').write_text('keep')
    with pytest.raises(RuntimeError, match='GGUF_VENV_CONFLICT'):
        b._prepare_venv(JobRecord('download', 'conflict'))
    assert (venv / 'user-file').read_text() == 'keep'


def test_gguf_bootstrap_reports_both_errors_and_does_not_retry_cancellation(temp_studio_env, monkeypatch, tmp_path):
    from studio.engines import ltx_gguf_backend as b
    from studio.jobs import JobRecord, JobCancelledError
    monkeypatch.setattr(b, 'ROOT', tmp_path)
    calls = []
    def fail(args, job, **kwargs):
        calls.append(args)
        if 'ensurepip' in args:
            raise RuntimeError('ensurepip detail')
        if '--python' in args:
            raise RuntimeError('fallback detail')
    monkeypatch.setattr(b, '_command', fail)
    with pytest.raises(RuntimeError, match='ensurepip detail.*fallback detail'):
        b._prepare_venv(JobRecord('download', 'failure'))
    calls.clear()
    def cancel(args, job, **kwargs):
        calls.append(args)
        if 'ensurepip' in args:
            raise JobCancelledError('cancelled')
    monkeypatch.setattr(b, '_command', cancel)
    with pytest.raises(JobCancelledError):
        b._prepare_venv(JobRecord('download', 'cancel'))
    assert not any('--python' in args for args in calls)


def test_gguf_duration_quantization_and_api_validation(temp_studio_env):
    from studio.engines.ltx_gguf_backend import frames_for_duration
    client, _ = temp_studio_env
    assert frames_for_duration(10, 24) == 241
    assert frames_for_duration(2, 24) == 49
    assert abs(frames_for_duration(7.5, 24) / 24 - 7.5) <= 4 / 24
    assert client.post('/api/video/ltx-gguf', json={'prompt': 'hola', 'width': 510}).status_code == 400
    assert frames_for_duration(20, 24) == 481
    assert frames_for_duration(20, 30) == 601
    assert client.post('/api/video/ltx-gguf', json={'prompt': 'hola', 'duration_sec': 20.1}).status_code == 422
    assert client.post('/api/video/ltx-gguf', json={'prompt': 'hola', 'image_strength': .4}).status_code == 422
    assert client.post('/api/video/ltx-gguf', json={'prompt': 'hola', 'speech_language': 'fr'}).status_code == 422
    assert client.post('/api/video/ltx-gguf', json={'prompt': 'hola', 'width': 768, 'height': 768}).status_code == 400


def test_gguf_resident_reuses_process_progress_and_unloads(temp_studio_env, monkeypatch, tmp_path):
    import sys
    from pathlib import Path
    from studio.engines import ltx_gguf_backend as b
    from studio.jobs import JobRecord
    from studio.memory_manager import RuntimeMemoryManager
    script = tmp_path / 'fake_comfy.py'
    script.write_text('''
import asyncio, json, sys
from pathlib import Path
from aiohttp import web
port, out = int(sys.argv[1]), Path(sys.argv[2])
clients = set(); history = {}; counter = 0
async def ws(request):
    s = web.WebSocketResponse(); await s.prepare(request); clients.add(s)
    async for msg in s: pass
    clients.discard(s); return s
async def prompt(request):
    global counter
    d = await request.json(); counter += 1; ident = str(counter)
    async def execute():
        await asyncio.sleep(.1)
        for payload in [dict(type='executing', data=dict(prompt_id=ident, node='15')),
                        dict(type='progress', data=dict(prompt_id=ident, node='15', value=4, max=8))]:
            for c in list(clients): await c.send_json(payload)
        await asyncio.sleep(1.3)
        (out / (d['prompt']['20']['inputs']['filename_prefix'] + '_00001_.mp4')).write_bytes(b'ok')
        history[ident] = {'status': {'completed': True}}
    asyncio.create_task(execute()); return web.json_response({'prompt_id': ident})
async def hist(request):
    key = request.match_info['id']; return web.json_response({key: history[key]} if key in history else {})
async def stats(request): return web.json_response({})
app = web.Application(); app.add_routes([web.get('/ws', ws), web.post('/prompt', prompt), web.get('/history/{id}', hist), web.get('/system_stats', stats)])
web.run_app(app, host='127.0.0.1', port=port, print=None)
''')
    root = tmp_path / 'runtime'; (root / 'comfy').mkdir(parents=True)
    monkeypatch.setattr(b, 'ROOT', root)
    monkeypatch.setattr(b, 'TEMP_DIR', tmp_path)
    monkeypatch.setattr(b, '_check_pressure', lambda: None)
    monkeypatch.setattr(b, '_server_command', lambda port, inp, out: [sys.executable, str(script), str(port), str(out)])
    manager = RuntimeMemoryManager()
    monkeypatch.setattr(b, 'memory_manager', manager)
    job = JobRecord('video', 'mock'); job.status = 'running'
    session = b._open_resident(job, {'dit': 'x'})
    proc = session.proc
    try:
        g = graph(image_name=None)
        g['20']['inputs']['filename_prefix'] = 'studio_gguf_one'
        first = b._infer(g, session.input_dir, session.output_dir, job)
        assert first.read_bytes() == b'ok'
        assert job.progress >= 67
        assert b._open_resident(job, {'dit':'x'}) is session
        assert session.proc.pid == proc.pid
        g['20']['inputs']['filename_prefix'] = 'studio_gguf_two'
        assert b._infer(g, session.input_dir, session.output_dir, job).read_bytes() == b'ok'
        assert len(manager.status_summary()) == 1
    finally:
        manager.unload_slot('video')
    assert proc.poll() is not None
    assert not session.root.exists()
    assert manager.get_slot('video') is None


def test_gguf_download_reports_real_bytes_and_verifies_hash(monkeypatch, tmp_path, capsys):
    import hashlib, json
    from studio.engines import ltx_gguf_download as d
    payload = b'0123456789'
    sha = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(d, 'FILES', (('dit', 'repo', 'rev', 'model.gguf', len(payload), sha),))
    def download(*, repo_id, revision, filename, local_dir, token, tqdm_class):
        bar = tqdm_class(total=len(payload), unit='B')
        bar.update(4); bar.update_transfer(9); bar.update(6); bar.close()
        from pathlib import Path
        (Path(local_dir) / filename).write_bytes(payload)
    monkeypatch.setattr(d, 'hf_hub_download', download)
    d.main(tmp_path)
    lines = capsys.readouterr().out.splitlines()
    progress = [json.loads(x.split(':', 1)[1]) for x in lines if x.startswith('GGUF_PROGRESS:')]
    assert any(p['phase'] == 'descarga' and p['done'] > 0 for p in progress)
    assert progress[-1]['phase'] == 'verificación' and progress[-1]['done'] == len(payload)
    assert progress[-1]['overall'] == 1.0
    (tmp_path / 'ltx' / 'model.gguf').write_bytes(b'X' * len(payload))
    with pytest.raises(RuntimeError, match='GGUF_CORRUPT_EXISTING'):
        d.main(tmp_path)
    assert (tmp_path / 'ltx' / 'model.gguf').read_bytes() == b'X' * len(payload)


def test_gguf_idle_slot_is_evicted_under_pressure_not_while_pinned(temp_studio_env, monkeypatch):
    from studio.memory_manager import RuntimeMemoryManager
    from studio import memory_manager as mm
    manager = RuntimeMemoryManager()
    monkeypatch.setattr(mm, 'flush_ram_and_vram', lambda: None)
    class Handle:
        unloaded = False
        def unload(self): self.unloaded = True
    h = Handle()
    manager.register_loaded('video', 'ltx-2.3-distilled-gguf-q4-experimental', 'GGUF', h)
    under_pressure = False
    def sample():
        present = manager.get_slot('video') is not None
        level = 'pressure' if under_pressure and present else 'normal'
        return {'ts': 0, 'ram': {'available_gb': 2 if level == 'pressure' else 10},
                'gpu': {'vram_free_gb': 5},
                'pressure': {'level': level, 'reserve': {'ram_gb': 3, 'vram_gb': 1}}}
    monkeypatch.setattr(manager, 'sample', sample)
    with manager.operation(slots=('video',), label='test'):
        under_pressure = True
        manager._reclaim(stage='test')
        assert not h.unloaded
        under_pressure = False
    under_pressure = True
    manager.tick()
    assert h.unloaded and manager.get_slot('video') is None


def test_gguf_generation_cancellation_unloads_session(temp_studio_env, monkeypatch, tmp_path):
    from types import SimpleNamespace
    from studio.engines import ltx_gguf_backend as b
    from studio.jobs import JobRecord, JobCancelledError
    inp, out = tmp_path / 'in', tmp_path / 'out'
    inp.mkdir(); out.mkdir()
    fake_session = SimpleNamespace(input_dir=inp, output_dir=out)
    monkeypatch.setattr(b, '_prepare_gguf', lambda *a, **k: {'dit':'x', 'text_encoder':'g', 'connectors':'c', 'video_vae':'v', 'audio_vae':'a'})
    monkeypatch.setattr(b, '_resident', lambda: fake_session)
    monkeypatch.setattr(b, '_open_resident', lambda *a: fake_session)
    monkeypatch.setattr(b.memory_manager, 'ensure_capacity', lambda **k: None)
    released = []
    monkeypatch.setattr(b.memory_manager, 'unload_slot', lambda slot: released.append(slot))
    def cancel(*args): raise JobCancelledError('cancelled')
    monkeypatch.setattr(b, '_infer', cancel)
    with pytest.raises(JobCancelledError):
        b.generate_gguf(image_path=None, prompt='test', negative_prompt='', seed=1, job=JobRecord('video', 'cancel'))
    assert released == ['video']


def test_gguf_disk_inventory_and_session_log(temp_studio_env, monkeypatch, tmp_path):
    from studio.engines import ltx_gguf_backend as b
    client, _ = temp_studio_env
    monkeypatch.setattr(b, 'FILES', (('dit', 'r', 'rev', 'a', 3, 'hash'),
                                    ('text_encoder', 'r', 'rev', 'b', 4, 'hash')))
    files = {'dit': tmp_path / 'a', 'text_encoder': tmp_path / 'b'}
    files['dit'].write_bytes(b'abc')
    files['text_encoder'].write_bytes(b'xx')
    monkeypatch.setattr(b, '_artifact_path', lambda role: files[role])
    info = b.gguf_disk_inventory()
    assert info == {'present': 2, 'complete_sizes': 1, 'total': 2, 'size_bytes': 5}
    report = client.get('/api/logs/export?format=md')
    assert report.status_code == 200
    assert b.MODEL_ID in report.text
    assert '1/2 componentes con tamaño esperado; hashes no comprobados' in report.text
