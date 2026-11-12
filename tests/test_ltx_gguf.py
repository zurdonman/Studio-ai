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
    assert entry['compatibility']['tier'] == 'candidate'
    assert entry['execution_validation']['status'] == 'observed'
    assert 'I2V probada en L4' in entry['execution_validation']['label']
    assert '20 s' in entry['execution_validation']['scope']
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
    monkeypatch.setattr(b, '_server_command', lambda port, inp, out, *args: [sys.executable, str(script), str(port), str(out)])
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


def test_ltx25_graph_has_its_own_encoder_and_joint_audio_video():
    from studio.engines.ltx25_gguf import make_graph25
    names = {'dit':'model.gguf', 'text_encoder':'gemma4.safetensors', 'video_vae':'video.safetensors', 'audio_vae':'audio.safetensors'}
    g = make_graph25(prompt='A person speaks', negative_prompt='', width=512, height=320, frames=49, fps=24,
                     seed=42, image_name='frame.png', names=names, image_strength=.7)
    assert g['1']['inputs']['unet_name'] == 'model.gguf'
    assert g['2']['class_type'] == 'CLIPLoader' and g['2']['inputs']['clip_name'] == 'gemma4.safetensors'
    assert g['11']['class_type'] == 'LTXVDualCFGGuider' and g['11']['inputs']['audio_cfg'] == 1
    assert g['13']['inputs']['sampler_name'] == 'euler_ancestral'
    assert g['17']['class_type'] == 'VAEDecodeTiled' and g['19']['inputs']['audio'] == ['18', 0]
    assert g['22']['inputs']['strength'] == .7
    assert make_graph25(prompt='test', negative_prompt='', width=512, height=320, frames=49, fps=24,
                        seed=1, image_name=None, names=names)['10']['inputs']['video_latent'] == ['8', 0]
    with pytest.raises(ValueError):
        make_graph25(prompt='test', negative_prompt='', width=513, height=320, frames=49, fps=24,
                     seed=1, image_name=None, names=names)


def test_ltx25_metadata_and_api_routing(temp_studio_env, monkeypatch):
    import time
    from studio.engines import ltx25_gguf as b
    client, _ = temp_studio_env
    q2, q4 = [mid for mid in b.MODELS if mid not in b.Q5_IDS]
    entries = {x['id']: x for x in client.get('/api/catalog').json()['models']}
    assert entries[q2]['engine'] == 'ltx_gguf' and entries[q4]['engine'] == 'ltx_gguf'
    assert entries[q2]['estimated_gb'] < entries[q4]['estimated_gb']
    assert not entries[q2]['compatibility']['compatible'] and 'execution_validation' not in entries[q2]
    assert not any(x['id'] in b.MODELS for x in client.get('/api/catalog?gpu_profile=L4&only_compatible=true').json()['models'])
    assert b.files_for(q2)[0][2] == b.REV and b.files_for(q2)[0][4] == 9500270976
    assert b.files_for(q4)[0][4] == 14000234880 and b.files_for(q2)[1:] == b.files_for(q4)[1:]
    assert client.post('/api/video/ltx-gguf/prepare?model_id=invalid').status_code == 400
    assert client.post('/api/video/ltx-gguf', json={'prompt': 'test', 'model_id':'invalid'}).status_code == 422
    assert client.post('/api/video/ltx-gguf', json={'prompt': 'test', 'model_id':q2, 'reference_voice_rel':'uploads/voice.wav'}).status_code == 400
    assert client.post('/api/video/ltx-gguf', json={'prompt': 'test', 'audio_rel':'uploads/voice.wav'}).status_code == 400
    calls = []
    monkeypatch.setattr(b, 'prepare', lambda mid, *, job: calls.append(('prepare',mid)) or {'prepared':True})
    monkeypatch.setattr(b, 'generate', lambda **kwargs: calls.append(('generate',kwargs)) or {'media_url':'/fake.mp4'})
    assert client.post('/api/video/ltx-gguf/prepare', params={'model_id':q2}).status_code == 200
    assert client.post('/api/video/ltx-gguf', json={'prompt':'Hola', 'model_id':q4, 'speech_language':'es', 'duration_sec':10}).status_code == 200
    for _ in range(50):
        if len(calls) == 2: break
        time.sleep(.02)
    assert ('prepare',q2) in calls
    request = next(v for k,v in calls if k == 'generate')
    assert request['model_id'] == q4 and request['speech_language'] == 'es' and request['duration_sec'] == 10


def test_ltx25_guards_gpu_before_downloading_and_switches_models(temp_studio_env, monkeypatch):
    from studio.engines import ltx25_gguf as b
    from studio.jobs import JobRecord
    q2 = next(iter(b.MODELS))
    installed = []
    monkeypatch.setattr(b.common, 'query_gpu_info', lambda: {'has_gpu':False})
    monkeypatch.setattr(b.common, '_install', lambda *_: installed.append('install'))
    with pytest.raises(RuntimeError, match='GGUF_GPU_UNSUPPORTED'):
        b._prepare(q2, JobRecord('video','test'))
    assert installed == []
    with pytest.raises(ValueError, match='GGUF_MODEL_UNSUPPORTED'):
        b._prepare('unknown', JobRecord('video','test'))


def test_ltx25_end_to_end_with_synthetic_audio_and_video(temp_studio_env, monkeypatch):
    import subprocess
    from types import SimpleNamespace
    from studio.engines import ltx25_gguf as b
    from studio.engines.ffmpeg_utils import get_ffmpeg_bin
    from studio.jobs import JobRecord
    from PIL import Image
    _, root = temp_studio_env
    image = root / 'uploads' / 'frame.png'
    Image.new('RGB', (256,256), 'blue').save(image)
    inp, out = root / 'fake_in25', root / 'fake_out25'
    inp.mkdir(); out.mkdir()
    mid = next(iter(b.MODELS))
    monkeypatch.setattr(b, '_prepare', lambda *a, **k: {'dit':'q2.gguf', 'text_encoder':'g4.safetensors', 'video_vae':'v.safetensors', 'audio_vae':'a.safetensors'})
    monkeypatch.setattr(b.common, '_open_resident', lambda job,names,model_id: SimpleNamespace(input_dir=inp, output_dir=out))
    graphs = []
    def infer(graph, i, o, job, model_id):
        assert model_id == mid
        assert (i / graph['21']['inputs']['image']).is_file()
        graphs.append(graph)
        path = o / 'test.mp4'
        subprocess.run([get_ffmpeg_bin(), '-v','error','-f','lavfi','-i','color=c=blue:s=320x256:r=24:d=1', '-f','lavfi','-i','sine=frequency=440:sample_rate=24000:duration=1', '-c:v','libx264','-pix_fmt','yuv420p','-c:a','aac','-shortest','-y',str(path)], check=True)
        return path
    monkeypatch.setattr(b.common, '_infer', infer)
    result = b.generate(model_id=mid, image_path=image, prompt='Hola', negative_prompt='', seed=42, job=JobRecord('video','test'),
                        width=512, height=320, duration_sec=2, fps=24, speech_language='es', image_strength=.8)
    assert result['model_id'] == mid and result['audio_native'] and result['frames'] == 49
    assert result['duration_warning'] and result['image_strength'] == .8
    assert 'spoken dialogue is in Spanish' in graphs[0]['5']['inputs']['text']
    assert not list(inp.iterdir()) and not list(out.iterdir())


def test_gguf_resident_cache_never_reuses_a_different_variant(temp_studio_env, monkeypatch):
    from types import SimpleNamespace
    from studio.engines import ltx_gguf_backend as common, ltx25_gguf as ltx25
    q2, q4 = [mid for mid in ltx25.MODELS if mid not in ltx25.Q5_IDS]
    session = object.__new__(common._ResidentComfy)
    session.proc = SimpleNamespace(poll=lambda: None)
    monkeypatch.setattr(common.memory_manager, 'get_slot', lambda *_: {'model_id':q2, 'handle':session})
    assert common._resident(q2) is session
    assert common._resident(q4) is None and common._resident() is None


def test_ltx25_memory_rejection_precedes_install_and_download(temp_studio_env, monkeypatch):
    from studio.engines import ltx25_gguf as b
    from studio.jobs import JobRecord
    mid = next(iter(b.MODELS))
    calls = []
    monkeypatch.setattr(b.common, 'query_gpu_info', lambda: {'has_gpu':True, 'bf16_supported':True, 'vram_total_gb':22})
    monkeypatch.setattr(b.common, '_resident', lambda *_: None)
    monkeypatch.setattr(b.memory_manager, 'get_slot', lambda *_: None)
    def reject(**kw):
        calls.append((kw['ram_gb'],kw['vram_gb']))
        raise RuntimeError('insufficient memory')
    monkeypatch.setattr(b.memory_manager, 'ensure_capacity', reject)
    monkeypatch.setattr(b.common, '_install', lambda *_: pytest.fail('must not install'))
    with pytest.raises(RuntimeError, match='insufficient memory'):
        b._prepare(mid, JobRecord('video','test'), extra_ram=2, extra_vram=.5)
    assert calls == [(47,16.5)]


def test_ltx25_cancel_unloads_resident_process(temp_studio_env, monkeypatch, tmp_path):
    from types import SimpleNamespace
    from studio.engines import ltx25_gguf as b
    from studio.jobs import JobRecord, JobCancelledError
    inp, out = tmp_path/'in25', tmp_path/'out25'
    inp.mkdir(); out.mkdir()
    mid = next(iter(b.MODELS))
    monkeypatch.setattr(b, '_prepare', lambda *a, **k: {'dit':'x.gguf', 'text_encoder':'g.safetensors', 'video_vae':'v.safetensors','audio_vae':'a.safetensors'})
    monkeypatch.setattr(b.common, '_open_resident', lambda *a: SimpleNamespace(input_dir=inp, output_dir=out))
    def fail(*args): raise JobCancelledError('cancelled')
    monkeypatch.setattr(b.common, '_infer', fail)
    released=[]
    monkeypatch.setattr(b.memory_manager, 'unload_slot', lambda slot: released.append(slot))
    with pytest.raises(JobCancelledError):
        b.generate(model_id=mid, image_path=None, prompt='test', negative_prompt='', seed=1, job=JobRecord('video','test'),
                   width=512, height=320, duration_sec=2, fps=24, speech_language='auto', image_strength=1)
    assert released == ['video']


def test_ltx25_q5_manifest_graph_catalog_and_routing(temp_studio_env, monkeypatch):
    import time
    from studio.engines import ltx25_gguf as b
    client, _ = temp_studio_env
    q2 = next(iter(b.MODELS))
    files = b.files_for(b.Q5_ID)
    assert files[0] == b.files_for(q2)[0] and files[2:] == b.files_for(q2)[2:]
    assert files[1] == b.Q5_ENCODER and files[1][2] == '2a18e836d286eb0570ec9f013c3591eb8c614d57'
    assert round(sum(x[4] for x in files) / 1024**3, 1) == 19.4
    names = {'dit':'model.gguf', 'text_encoder': files[1][3], 'video_vae':'v.safetensors', 'audio_vae':'a.safetensors'}
    graph = b.make_graph25(prompt='Hola', negative_prompt='', width=512, height=320, frames=49, fps=24, seed=1, image_name=None, names=names)
    assert graph['2'] == {'class_type':'CLIPLoaderGGUF', 'inputs': {'clip_name': files[1][3], 'type':'ltxv'}}
    entry = next(x for x in client.get('/api/catalog').json()['models'] if x['id'] == b.Q5_ID)
    assert 'Gemma 4 GGUF Q5' in entry['name'] and entry['estimated_gb'] == 19.4 and not entry['compatibility']['compatible']
    assert not any(x['id'] == b.Q5_ID for x in client.get('/api/catalog?gpu_profile=L4&only_compatible=true').json()['models'])
    calls = []
    monkeypatch.setattr(b, 'prepare', lambda mid, *, job: calls.append(('prepare',mid)) or {'prepared':True})
    monkeypatch.setattr(b, 'generate', lambda **kw: calls.append(('generate',kw['model_id'])) or {'media_url':'/mock.mp4'})
    assert client.post('/api/video/ltx-gguf/prepare', params={'model_id': b.Q5_ID}).status_code == 200
    assert client.post('/api/video/ltx-gguf', json={'model_id':b.Q5_ID, 'prompt':'Hola'}).status_code == 200
    for _ in range(50):
        if len(calls) == 2: break
        time.sleep(.02)
    assert ('prepare',b.Q5_ID) in calls and ('generate',b.Q5_ID) in calls


@pytest.mark.parametrize('variant,ram,vram', [('Q5_ID', 32, 14), ('Q4_Q5_ID', 35, 16), ('UDQ4_Q5_ID', 35, 16)])
def test_ltx25_q5_uses_separate_runtime_and_memory_guard(temp_studio_env, monkeypatch, tmp_path, variant, ram, vram):
    from studio.engines import ltx25_gguf as b
    from studio.jobs import JobRecord
    calls=[]
    monkeypatch.setattr(b, 'Q5_RUNTIME', tmp_path/'isolated')
    monkeypatch.setattr(b.common, 'query_gpu_info', lambda: {'has_gpu':True,'bf16_supported':True,'vram_total_gb':22})
    monkeypatch.setattr(b.common, '_resident', lambda *_: None)
    monkeypatch.setattr(b.memory_manager, 'get_slot', lambda *_: None)
    monkeypatch.setattr(b.memory_manager, 'ensure_capacity', lambda **kw: calls.append(('capacity', kw['ram_gb'], kw['vram_gb'])))
    monkeypatch.setattr(b.common, '_install', lambda job,root,rev: calls.append(('install',root,rev)))
    monkeypatch.setattr(b.common, '_probe_runtime', lambda job,nodes,root: calls.append(('probe',root,nodes)))
    monkeypatch.setattr(b.common, '_command', lambda args,job,**kw: calls.append(('command',args,kw['stage'])))
    monkeypatch.setattr(b, '_links', lambda mid: {'dit':'d.gguf'})
    monkeypatch.setattr(b.shutil, 'disk_usage', lambda p: type('Disk',(),{'free':100*1024**3})())
    mid = getattr(b, variant)
    b._prepare(mid, JobRecord('video','test'), extra_ram=2, extra_vram=.5)
    assert ('capacity',ram + 2,vram + .5) in calls
    assert ('install', b.Q5_RUNTIME, b.Q5_COMFY_COMMIT) in calls
    assert any(c[0]=='probe' and c[1] == b.Q5_RUNTIME and 'CLIPLoaderGGUF' in c[2] for c in calls)
    assert any(c[0]=='command' and 'studio.engines.ltx25_gguf_patch' in c[1] for c in calls)
    assert any(c[0]=='command' and 'studio.engines.ltx_gguf_download' in c[1] and mid in c[1] for c in calls)


def test_gemma4_patch_applies_only_verified_patch_and_preserves_changes(tmp_path, monkeypatch):
    import json
    import subprocess
    from studio.engines import ltx25_gguf_patch as p
    plugin = tmp_path/'plugin'; plugin.mkdir()
    def git(*args):
        return subprocess.run(['git','-C',str(plugin),*args], check=True, capture_output=True, text=True).stdout
    git('init','-q'); git('config','user.email','test@example.invalid'); git('config','user.name','Test')
    loader=plugin/'loader.py'; loader.write_text('original\n')
    git('add','loader.py'); git('commit','-qm','initial')
    monkeypatch.setattr(p, 'GGUF_COMMIT', git('rev-parse','HEAD').strip())
    loader.write_text('patched\n')
    patch=tmp_path/'test.patch'; patch.write_text(git('diff','--','loader.py'))
    git('checkout','--','loader.py')
    monkeypatch.setattr(p, 'PATCH_SHA256', p.digest(patch))
    monkeypatch.setattr(p, 'hf_hub_download', lambda **kw: str(patch))
    stamp=tmp_path/'stamp.json'
    p.run(plugin,stamp)
    assert loader.read_text() == 'patched\n' and json.loads(stamp.read_text())['patch_sha256'] == p.digest(patch)
    p.run(plugin,stamp)  # idempotent
    loader.write_text('user edits\n')
    with pytest.raises(RuntimeError, match='PATCH_CONFLICT'):
        p.run(plugin,stamp)
    assert loader.read_text() == 'user edits\n'
    patch.write_text('tampered')
    with pytest.raises(RuntimeError, match='PATCH_HASH'):
        p.run(plugin,stamp)
    assert loader.read_text() == 'user edits\n'


def test_gemma4_patch_gated_access_error_is_actionable(tmp_path, monkeypatch):
    from studio.engines import ltx25_gguf_patch as p
    class GatedRepoError(Exception): pass
    def denied(**kw): raise GatedRepoError('denied with secret')
    monkeypatch.setattr(p, 'hf_hub_download', denied)
    with pytest.raises(RuntimeError, match='GGUF_GEMMA4_ACCESS') as error:
        p.run(tmp_path, tmp_path/'stamp')
    assert 'secret' not in str(error.value)


def test_gemma4_download_errors_distinguish_auth_network_and_unknown_without_secrets(tmp_path, monkeypatch):
    from studio.engines import ltx25_gguf_patch as p
    class LocalEntryNotFoundError(Exception): pass
    class RevisionNotFoundError(Exception): pass
    class GatedRepoError(Exception): pass
    class FakeHttpError(Exception):
        response = type('Response', (), {'status_code':403})()
    for error, code in [(LocalEntryNotFoundError('hf_secret'), 'GGUF_GEMMA4_NETWORK'),
                        (RevisionNotFoundError('hf_secret'), 'GGUF_GEMMA4_RESOURCE'),
                        (FakeHttpError('hf_secret'), 'GGUF_GEMMA4_ACCESS'),
                        (ValueError('hf_secret'), 'GGUF_GEMMA4_DOWNLOAD')]:
        message = p._download_error(error, True)
        assert code in message and 'hf_secret' not in message and 'token de Studio en el proceso: sí' in message
    wrapped = LocalEntryNotFoundError('outer')
    wrapped.__cause__ = GatedRepoError('secret')
    assert 'GGUF_GEMMA4_ACCESS' in p._download_error(wrapped, False)


def test_comfy_server_command_supports_both_pinned_runtimes(temp_studio_env, monkeypatch, tmp_path):
    from studio.engines import ltx_gguf_backend as b, ltx25_gguf as b25
    monkeypatch.setattr(b, 'ROOT', tmp_path/'base')
    monkeypatch.setattr(b25, 'Q5_RUNTIME', tmp_path/'q5')
    for root in [b.ROOT, b25.Q5_RUNTIME]:
        cmd = b._server_command(18700, tmp_path/'in', tmp_path/'out', root)
        assert cmd[0] == str(root/'venv/bin/python') and cmd[1] == str(root/'comfy/main.py')
        assert '--disable-api-nodes' in cmd and '--offline' not in cmd
        assert cmd[cmd.index('--listen')+1] == '127.0.0.1'
        assert '--disable-all-custom-nodes' in cmd
        assert cmd[cmd.index('--whitelist-custom-nodes')+1] == 'ComfyUI-GGUF'
        assert '--reserve-vram' in cmd and '--disable-metadata' in cmd

@pytest.mark.parametrize('variant,expected_size,sha', [
    ('Q4_Q5_ID', 13584703872, 'ba40d53770314e69e88361661e55c6cc3b12a363646ba2ea53c8bcd459b66747'),
    ('UDQ4_Q5_ID', 14000234880, '55442750eb0013d94fffaa4dadb577c5f4dc07107ec04389d41444f6e517d980'),
])
def test_ltx25_q4_gemma5_manifest_catalog_and_routing(temp_studio_env, monkeypatch, variant, expected_size, sha):
    import time
    from studio.engines import ltx25_gguf as b
    client, _ = temp_studio_env
    mid = getattr(b, variant)
    files = b.files_for(mid)
    assert files[0][4:] == (expected_size, sha)
    assert files[1] == b.Q5_ENCODER and files[2:] == b.files_for(b.Q5_ID)[2:]
    assert b.path_for(files[1]) == b.path_for(b.files_for(b.Q5_ID)[1])
    entry = next(m for m in client.get('/api/catalog').json()['models'] if m['id'] == mid)
    assert entry['engine'] == 'ltx_gguf' and not entry['compatibility']['compatible']
    assert entry['estimated_gb'] == round(sum(x[4] for x in files)/1024**3, 1)
    calls = []
    monkeypatch.setattr(b, 'prepare', lambda model_id, *, job: calls.append(('prepare',model_id)))
    monkeypatch.setattr(b, 'generate', lambda **kw: calls.append(('generate',kw['model_id'])) or {})
    assert client.post('/api/video/ltx-gguf/prepare', params={'model_id':mid}).status_code == 200
    assert client.post('/api/video/ltx-gguf', json={'model_id':mid,'prompt':'Un retrato'}).status_code == 200
    for _ in range(50):
        if len(calls) == 2: break
        time.sleep(.02)
    assert ('prepare',mid) in calls and ('generate',mid) in calls


def test_face_identity_candidates_are_informative_and_blocked(temp_studio_env):
    client, _ = temp_studio_env
    entries = {m['id']:m for m in client.get('/api/catalog').json()['models']}
    for mid in ('lora-ltx23-best-face-id-candidate', 'lora-ltx23-character-sheet-candidate'):
        assert entries[mid]['validation_state'] == 'candidate'
        assert entries[mid]['compatible_models'] == []
        assert client.post(f'/api/catalog/{mid}/download').status_code == 400
        assert client.post('/api/memory/load', json={'model_id':mid}).status_code == 400
