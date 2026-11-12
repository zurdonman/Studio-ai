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


@pytest.mark.parametrize('model_id', [
    'ltx-2.3-distilled-gguf-q4-experimental',
    'ltx-2.5-distilled-gguf-ud-q2-gemma-q5-experimental',
    'ltx-2.5-distilled-gguf-q4km-gemma-q5-experimental',
    'ltx-2.5-distilled-gguf-ud-q4-gemma-q5-experimental',
])
def test_gguf_idle_slot_is_evicted_under_pressure_not_while_pinned(temp_studio_env, monkeypatch, tmp_path, model_id):
    from studio.memory_manager import RuntimeMemoryManager
    from studio import memory_manager as mm
    manager = RuntimeMemoryManager()
    monkeypatch.setattr(mm, 'flush_ram_and_vram', lambda: None)
    class Handle:
        unloaded = False
        def unload(self): self.unloaded = True
    h = Handle()
    weights = tmp_path / 'weights.gguf'
    weights.write_bytes(b'pesos preservados')
    manager.register_loaded('video', model_id, 'GGUF', h)
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
    assert weights.read_bytes() == b'pesos preservados'


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
    client, root = temp_studio_env
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
    # An output is not useful unless the gallery lists it and the browser can
    # request it (including the byte ranges used by video players).
    assert (root / result['rel_path']).is_file()
    listed = client.get('/api/assets').json()['videos']
    assert any(v['rel_path'] == result['rel_path'] and v['media_url'] == result['media_url'] for v in listed)
    response = client.get(result['media_url'])
    assert response.status_code == 200 and response.headers['content-type'].startswith('video/mp4')
    partial = client.get(result['media_url'], headers={'Range': 'bytes=0-31'})
    assert partial.status_code == 206 and len(partial.content) == 32


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


@pytest.mark.parametrize('variant', ['Q5_ID', 'Q4_Q5_ID', 'UDQ4_Q5_ID'])
@pytest.mark.parametrize('reason', ['cancel', 'critical'])
def test_ltx25_failure_unloads_resident_process(temp_studio_env, monkeypatch, tmp_path, variant, reason):
    from types import SimpleNamespace
    from studio.engines import ltx25_gguf as b
    from studio.jobs import JobRecord, JobCancelledError
    inp, out = tmp_path/'in25', tmp_path/'out25'
    inp.mkdir(); out.mkdir()
    mid = getattr(b, variant)
    monkeypatch.setattr(b, '_prepare', lambda *a, **k: {'dit':'x.gguf', 'text_encoder':'g.gguf', 'video_vae':'v.safetensors','audio_vae':'a.safetensors'})
    monkeypatch.setattr(b.common, '_open_resident', lambda *a: SimpleNamespace(input_dir=inp, output_dir=out))
    failure = JobCancelledError('cancelled') if reason == 'cancel' else RuntimeError('GGUF_MEMORY_CRITICAL')
    def fail(*args): raise failure
    monkeypatch.setattr(b.common, '_infer', fail)
    released=[]
    monkeypatch.setattr(b.memory_manager, 'unload_slot', lambda slot: released.append(slot))
    with pytest.raises(type(failure)):
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


@pytest.mark.parametrize('variant', ['Q5_ID', 'Q4_Q5_ID', 'UDQ4_Q5_ID'])
def test_ltx25_q5_resident_reuses_session_without_reinstall_or_download(temp_studio_env, monkeypatch, variant):
    from types import SimpleNamespace
    from studio.engines import ltx25_gguf as b
    from studio.jobs import JobRecord
    mid = getattr(b, variant)
    session = SimpleNamespace(names={'dit':'cached.gguf'})
    monkeypatch.setattr(b.common, 'query_gpu_info', lambda: {'has_gpu':True,'bf16_supported':True,'vram_total_gb':22})
    monkeypatch.setattr(b.common, '_resident', lambda model_id: session if model_id == mid else None)
    calls = []
    monkeypatch.setattr(b.memory_manager, 'ensure_capacity', lambda **kw: calls.append((kw['ram_gb'],kw['vram_gb'])))
    monkeypatch.setattr(b.common, '_install', lambda *a, **kw: pytest.fail('resident must not reinstall'))
    monkeypatch.setattr(b.common, '_command', lambda *a, **kw: pytest.fail('resident must not redownload or rehash'))
    result = b._prepare(mid, JobRecord('video','reuse'), extra_ram=2, extra_vram=.5)
    assert result is session.names and calls == [(2,.5)]


@pytest.mark.parametrize('variant', ['Q5_ID', 'Q4_Q5_ID', 'UDQ4_Q5_ID'])
def test_ltx25_q5_switch_unloads_previous_before_admission_and_does_not_download_on_failure(temp_studio_env, monkeypatch, variant):
    from studio.engines import ltx25_gguf as b
    from studio.jobs import JobRecord
    mid = getattr(b, variant)
    calls = []
    monkeypatch.setattr(b.common, 'query_gpu_info', lambda: {'has_gpu':True,'bf16_supported':True,'vram_total_gb':22})
    monkeypatch.setattr(b.common, '_resident', lambda *_: None)
    monkeypatch.setattr(b.memory_manager, 'get_slot', lambda *_: {'model_id':'old'})
    monkeypatch.setattr(b.memory_manager, 'unload_slot', lambda slot: calls.append(('unload',slot)))
    def reject(**kw):
        calls.append(('admission',kw['ram_gb'],kw['vram_gb']))
        raise RuntimeError('not enough')
    monkeypatch.setattr(b.memory_manager, 'ensure_capacity', reject)
    monkeypatch.setattr(b.common, '_install', lambda *a, **kw: pytest.fail('must not install before admission'))
    monkeypatch.setattr(b.common, '_command', lambda *a, **kw: pytest.fail('must not download before admission'))
    with pytest.raises(RuntimeError, match='not enough'):
        b._prepare(mid, JobRecord('video','switch'))
    assert calls[0] == ('unload','video') and calls[1][0] == 'admission'


def test_voice_graph_routes_audio_and_reference_without_muting_video():
    from studio.engines.ltx23_voice import add_voice_graph, frames_for_audio
    g = add_voice_graph(graph(), 'audio', 'voice.wav')
    assert g['10']['inputs']['audio_latent'] == ['33', 0]
    assert g['10']['inputs']['video_latent'] == ['22', 0]
    assert g['33']['inputs']['samples'] == ['31', 0] and g['32']['inputs']['value'] == 0
    assert g['31']['class_type'] == 'LTXVAudioVAEEncode' and '9' not in g
    g = add_voice_graph(graph(), 'voice_reference', 'voice.wav')
    assert g['31']['class_type'] == 'StudioStrictVoiceLoRA'
    assert g['32']['class_type'] == 'LTXVReferenceAudio' and g['32']['inputs']['identity_guidance_scale'] == 3
    assert g['11']['inputs']['model'] == ['32', 0] and g['11']['inputs']['positive'] == ['32', 1]
    assert g['10']['inputs']['audio_latent'] == ['9', 0]
    assert frames_for_audio(2.1,24) == 57 and frames_for_audio(20,30) == 601
    for mode, filename in [('audio','../bad'), ('bad','valid.wav')]:
        with pytest.raises(ValueError): add_voice_graph(graph(), mode, filename)


@pytest.mark.parametrize('mode', ['audio', 'voice_reference'])
def test_voice_generation_graph_cleanup_and_budget(temp_studio_env, monkeypatch, tmp_path, mode):
    from types import SimpleNamespace
    from studio.engines import ltx_gguf_backend as b
    from studio.engines.ffmpeg_utils import get_ffmpeg_bin, probe_media_info
    from studio.jobs import JobRecord
    from PIL import Image
    image = tmp_path/'image.png'; Image.new('RGB', (64,64), 'blue').save(image)
    voice = tmp_path/'voice.wav'
    subprocess.run([get_ffmpeg_bin(),'-v','error','-f','lavfi','-i','sine=frequency=440:duration=2.1', '-y',str(voice)], check=True)
    inp, out = tmp_path/'input', tmp_path/'output'; inp.mkdir(); out.mkdir()
    session=SimpleNamespace(input_dir=inp, output_dir=out)
    monkeypatch.setattr(b, 'VIDEOS_OUT_DIR', tmp_path/'videos')
    budgets, graphs, loras = [], [], []
    monkeypatch.setattr(b, '_prepare_gguf', lambda job, **kw: budgets.append(kw) or {'dit':'x.gguf','text_encoder':'g.gguf','connectors':'c.safetensors','video_vae':'v.safetensors','audio_vae':'a.safetensors'})
    monkeypatch.setattr(b, '_prepare_voice_lora', lambda job: loras.append('loaded'))
    monkeypatch.setattr(b, '_resident', lambda: session)
    monkeypatch.setattr(b, '_open_resident', lambda *args: session)
    monkeypatch.setattr(b.memory_manager, 'ensure_capacity', lambda **kw: None)
    def infer(g, inp, out, job):
        wav = inp / g['30']['inputs']['audio']
        info = probe_media_info(wav)
        assert info['has_audio']
        if mode == 'audio': assert abs(info['duration_seconds'] - 57/24) < .02
        graphs.append(g)
        path=out/'synthetic.mp4'
        subprocess.run([get_ffmpeg_bin(),'-v','error','-f','lavfi','-i','color=c=blue:s=320x256:r=24:d=1','-f','lavfi','-i','sine=frequency=440:duration=1','-c:v','libx264','-pix_fmt','yuv420p','-c:a','aac','-shortest','-y',str(path)], check=True)
        return path
    monkeypatch.setattr(b, '_infer', infer)
    result = b.generate_gguf(image_path=image, voice_path=voice, voice_mode=mode, prompt='Una persona habla', negative_prompt='',seed=7,job=JobRecord('video','test'),duration_sec=2,fps=24)
    assert result['voice_mode'] == mode and result['frames'] == (57 if mode == 'audio' else 49)
    assert loras == (['loaded'] if mode == 'voice_reference' else [])
    assert budgets[0]['extra_ram'] >= (4 if mode == 'voice_reference' else 1)
    assert graphs and not list(inp.iterdir()) and not list(out.iterdir())


def test_voice_api_guards_and_routes(temp_studio_env, monkeypatch):
    import time
    from studio.engines import ltx_gguf_backend as b
    from studio.engines.ffmpeg_utils import get_ffmpeg_bin
    client, root = temp_studio_env
    voice=root/'uploads'/'voice_test.wav'
    subprocess.run([get_ffmpeg_bin(),'-v','error','-f','lavfi','-i','sine=frequency=440:duration=2','-y',str(voice)],check=True)
    calls=[]
    monkeypatch.setattr(b,'generate_gguf',lambda **kw: calls.append(kw) or {})
    assert client.post('/api/video/ltx-gguf', json={'prompt':'hi','voice_mode':'audio'}).status_code == 400
    assert client.post('/api/video/ltx-gguf', json={'prompt':'hi','audio_rel':'uploads/voice_test.wav'}).status_code == 400
    assert client.post('/api/video/ltx-gguf', json={'prompt':'hi','voice_mode':'audio','audio_rel':'uploads/voice_test.wav','reference_voice_rel':'uploads/voice_test.wav'}).status_code == 400
    assert client.post('/api/video/ltx-gguf', json={'prompt':'hi','model_id':'ltx-2.5-distilled-gguf-ud-q2-experimental','voice_mode':'voice_reference','reference_voice_rel':'uploads/voice_test.wav'}).status_code == 400
    assert client.post('/api/video/ltx-gguf', json={'prompt':'hi','voice_mode':'audio','audio_rel':'../escape'}).status_code == 400
    for mode, field in [('audio','audio_rel'), ('voice_reference','reference_voice_rel')]:
        assert client.post('/api/video/ltx-gguf',json={'prompt':'hi','voice_mode':mode, field:'uploads/voice_test.wav'}).status_code == 200
    for _ in range(50):
        if len(calls)==2: break
        time.sleep(.02)
    assert {c['voice_mode'] for c in calls} == {'audio','voice_reference'}
    assert all(c['voice_path']==voice for c in calls)


def test_voice_strict_loader_fails_closed_on_unmapped_or_unapplied(monkeypatch):
    import importlib.util
    import sys
    import types
    from pathlib import Path
    comfy = types.ModuleType('comfy'); comfy.__path__ = []
    lora = types.ModuleType('comfy.lora')
    convert = types.ModuleType('comfy.lora_convert')
    utils = types.ModuleType('comfy.utils')
    paths = types.ModuleType('folder_paths')
    comfy.lora, comfy.lora_convert, comfy.utils = lora, convert, utils
    for name, mod in [('comfy',comfy),('comfy.lora',lora),('comfy.lora_convert',convert),('comfy.utils',utils),('folder_paths',paths)]:
        monkeypatch.setitem(sys.modules,name,mod)
    paths.get_full_path_or_raise = lambda *a: '/voice.safetensors'
    paths.get_filename_list = lambda *a: ['voice.safetensors']
    convert.convert_lora = lambda x: x
    utils.load_torch_file = lambda *a,**k: {'layer.lora_A.weight':1,'layer.lora_B.weight':2}
    lora.model_lora_keys_unet = lambda *args: {'layer':'weight'}
    class Adapter:
        loaded_keys = {'layer.lora_A.weight','layer.lora_B.weight'}
    lora.load_lora = lambda *a,**k: {'weight':Adapter()}
    class Model:
        model=object()
        def clone(self): return self
        def add_patches(self, patches, strength): return {'weight'}
    file=Path('studio/engines/comfy_voice_node/__init__.py')
    spec=importlib.util.spec_from_file_location('strict_voice_test',file)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    node=module.StudioStrictVoiceLoRA(); model=Model()
    assert node.load(model,'voice.safetensors',1)[0] is model
    utils.load_torch_file = lambda *a,**k: {'layer.lora_A.weight':1,'layer.lora_B.weight':2,'unknown':3}
    with pytest.raises(RuntimeError,match='tensores no reconocidos'): node.load(model,'voice.safetensors',1)
    utils.load_torch_file = lambda *a,**k: {'layer.lora_A.weight':1,'layer.lora_B.weight':2}
    model.add_patches = lambda *a: set()
    with pytest.raises(RuntimeError,match='parches no aplicados'): node.load(model,'voice.safetensors',1)


def test_voice_adapter_is_pinned_cached_linked_and_never_overwrites(temp_studio_env, monkeypatch, tmp_path):
    import hashlib
    from studio.engines import ltx_gguf_backend as b, ltx23_voice as voice
    from studio.jobs import JobRecord
    from types import SimpleNamespace
    runtime = tmp_path/'runtime'; weights = tmp_path/'weights'; weights.mkdir()
    payload = b'test adapter'
    item = ('voice_lora','repo','fixedrev','lora.safetensors',len(payload),hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(b,'ROOT',runtime); monkeypatch.setattr(b,'WEIGHTS',weights)
    monkeypatch.setattr(b,'_voice_lora_fingerprint',None)
    monkeypatch.setattr(voice,'LORA',item)
    monkeypatch.setattr(b.shutil,'disk_usage',lambda *_: SimpleNamespace(free=10*1024**3))
    monkeypatch.setattr(b,'_resident',lambda: None)
    commands=[]
    def download(*args, **kw):
        commands.append(args[0]); (weights/'ltx').mkdir(exist_ok=True)
        (weights/'ltx'/'lora.safetensors').write_bytes(payload)
    monkeypatch.setattr(b,'_command',download)
    b._prepare_voice_lora(JobRecord('video','test'))
    link=runtime/'comfy'/'models'/'loras'/voice.LORA_LINK
    assert link.is_symlink() and link.read_bytes() == payload
    plugin=runtime/'comfy'/'custom_nodes'/'StudioVoice'/'__init__.py'
    assert plugin.is_file() and len(commands)==1 and 'studio.engines.ltx23_voice_download' in commands[0]
    b._prepare_voice_lora(JobRecord('video','test'))
    assert len(commands)==1
    plugin.write_text('foreign code')
    with pytest.raises(RuntimeError,match='VOICE_NODE_CONFLICT'):
        b._prepare_voice_lora(JobRecord('video','test'))
    assert plugin.read_text() == 'foreign code' and link.read_bytes() == payload


@pytest.mark.parametrize('mode', ['audio', 'voice_reference'])
def test_ltx25_audio_modes_synthetic(temp_studio_env, monkeypatch, tmp_path, mode):
    from types import SimpleNamespace
    from PIL import Image
    from studio.engines import ltx25_gguf as b, ltx25_msr as msr
    from studio.engines.ffmpeg_utils import get_ffmpeg_bin, probe_media_info
    from studio.jobs import JobRecord
    image = tmp_path/'face.png'; Image.new('RGB',(64,64),'blue').save(image)
    audio = tmp_path/'reference.wav'
    subprocess.run([get_ffmpeg_bin(),'-v','error','-f','lavfi','-i','sine=frequency=440:duration=2.1','-y',str(audio)],check=True)
    inp,out=tmp_path/'in',tmp_path/'out'; inp.mkdir(); out.mkdir()
    session=SimpleNamespace(input_dir=inp,output_dir=out,base='http://unused')
    mid = b.Q5_ID
    budgets=[]; prepared=[]; graphs=[]
    monkeypatch.setattr(b,'VIDEOS_OUT_DIR',tmp_path/'videos')
    monkeypatch.setattr(b,'_prepare',lambda *args,**kw: budgets.append(kw) or {'dit':'d.gguf','text_encoder':'g.gguf','video_vae':'v.safetensors','audio_vae':'a.safetensors'})
    monkeypatch.setattr(msr,'prepare_msr',lambda *a,**kw: prepared.append(kw) or False)
    monkeypatch.setattr(b.common,'_open_resident',lambda *args: session)
    monkeypatch.setattr(b.common.requests,'get',lambda *a,**kw: SimpleNamespace(raise_for_status=lambda:None,json=lambda:{'StudioStrictMSRLoader':{},'ComfyUILTX25MSRMultiReferenceGuide':{},'LTXVCropGuides':{}}))
    def infer(g,i,o,job,model_id):
        wav=i/g['30' if mode=='audio' else '41']['inputs']['audio']
        duration=probe_media_info(wav)['duration_seconds']
        assert abs(duration-(57/24 if mode=='audio' else 2.1)) < .02
        if mode=='audio':
            assert g['10']['inputs']['audio_latent']==['33',0]
        else:
            assert g['44']['inputs']['audio_ref1']==['42',0]
            assert g['44']['inputs']['msr_parameters']==['43',1]
            assert g['10']['inputs']['video_latent']==['44',2]
            assert g['11']['inputs']['model']==['43',0]
            assert g['17']['inputs']['samples']==['45',2]
        graphs.append(g)
        result=o/'synthetic.mp4'
        subprocess.run([get_ffmpeg_bin(),'-v','error','-f','lavfi','-i','color=c=blue:s=320x256:r=24:d=1','-f','lavfi','-i','sine=frequency=440:duration=1','-c:v','libx264','-pix_fmt','yuv420p','-c:a','aac','-shortest','-y',str(result)],check=True)
        return result
    monkeypatch.setattr(b.common,'_infer',infer)
    result=b.generate(model_id=mid,image_path=image,voice_path=audio,voice_mode=mode,prompt='Hola',negative_prompt='',seed=42,job=JobRecord('video','test'),width=512,height=320,duration_sec=2,fps=24,speech_language='es',image_strength=.8)
    assert result['voice_mode']==mode and result['frames']==(57 if mode=='audio' else 49)
    assert bool(prepared)==(mode=='voice_reference')
    assert budgets[0]['extra_ram'] >= (5 if mode=='voice_reference' else 1)
    assert graphs and not list(inp.iterdir()) and not list(out.iterdir())
    assert (tmp_path/'videos'/result['filename']).is_file()


def test_ltx25_voice_api_routes_and_image_guard(temp_studio_env, monkeypatch):
    import time
    from PIL import Image
    from studio.engines import ltx25_gguf as b
    from studio.engines.ffmpeg_utils import get_ffmpeg_bin
    client,root=temp_studio_env
    image=root/'uploads'/'face25.png'; Image.new('RGB',(64,64),'blue').save(image)
    audio=root/'uploads'/'ref25.wav'
    subprocess.run([get_ffmpeg_bin(),'-v','error','-f','lavfi','-i','sine=frequency=440:duration=2','-y',str(audio)],check=True)
    calls=[]; monkeypatch.setattr(b,'generate',lambda **kw: calls.append(kw) or {})
    base={'model_id':b.Q5_ID,'prompt':'hola'}
    assert client.post('/api/video/ltx-gguf',json={**base,'voice_mode':'voice_reference','reference_voice_rel':'uploads/ref25.wav'}).status_code==400
    for mode,field in [('audio','audio_rel'),('voice_reference','reference_voice_rel')]:
        response=client.post('/api/video/ltx-gguf',json={**base,'image_rel':'uploads/face25.png','voice_mode':mode,field:'uploads/ref25.wav'})
        assert response.status_code==200, response.text
    for _ in range(50):
        if len(calls)==2: break
        time.sleep(.02)
    assert {call['voice_mode'] for call in calls}=={'audio','voice_reference'}
    assert all(call['voice_path']==audio and call['image_path']==image for call in calls)


def test_msr_strict_loader_rejects_unknown_and_unapplied(monkeypatch):
    import importlib.util
    import sys
    import types
    from pathlib import Path
    comfy=types.ModuleType('comfy'); comfy.__path__=[]
    lora=types.ModuleType('comfy.lora'); convert=types.ModuleType('comfy.lora_convert'); sd=types.ModuleType('comfy.sd')
    comfy.lora=lora; comfy.lora_convert=convert; comfy.sd=sd
    for name,mod in [('comfy',comfy),('comfy.lora',lora),('comfy.lora_convert',convert),('comfy.sd',sd)]: monkeypatch.setitem(sys.modules,name,mod)
    upstream=types.SimpleNamespace(ComfyUILTX25MSRICLoRALoader=type('Loader',(),{}),ComfyUILTX25MSRMultiReferenceGuide=type('Guide',(),{}))
    spec=object()
    monkeypatch.setattr(importlib.util,'spec_from_file_location',lambda *a,**k:spec)
    monkeypatch.setattr(importlib.util,'module_from_spec',lambda _:upstream)
    # A separate fake spec supplies the execution hook.
    spec=types.SimpleNamespace(loader=types.SimpleNamespace(exec_module=lambda _:None))
    module={'__file__':str(Path('studio/engines/comfy_msr_node/__init__.py').resolve())}
    exec(compile(Path(module['__file__']).read_text(),module['__file__'],'exec'),module)
    convert.convert_lora=lambda weights:weights
    lora.model_lora_keys_unet=lambda *a:{'layer':'weight'}
    lora.load_lora=lambda *a,**kw:{'weight':types.SimpleNamespace(loaded_keys={'a','b'})}
    class Model:
        model=object()
        def clone(self): return self
        def add_patches(self,*a): return {'weight'}
        def set_attachments(self,*a): self.metadata=a
    model=Model(); load=module['_strict_load']
    assert load(model,None,{'a':1,'b':2},1,0,{'meta':1})[0] is model
    with pytest.raises(RuntimeError,match='tensores sin correspondencia'): load(model,None,{'a':1,'b':2,'unknown':3},1,0)
    model.add_patches=lambda *a:set()
    with pytest.raises(RuntimeError,match='no se aplicaron'): load(model,None,{'a':1,'b':2},1,0)


def test_msr_installer_cache_and_conflicts(temp_studio_env, monkeypatch, tmp_path):
    from types import SimpleNamespace
    from studio.engines import ltx25_msr as msr
    from studio.jobs import JobRecord
    runtime=tmp_path/'runtime'; weights=tmp_path/'weights'; weights.mkdir()
    source=runtime/'studio-msr-source'; (source/'.git').mkdir(parents=True)
    item=('msr_voice','repo','rev','adapter.safetensors',4,'abc')
    monkeypatch.setattr(msr,'LORA',item); monkeypatch.setattr(msr,'_verified',None)
    monkeypatch.setattr(msr.shutil,'disk_usage',lambda *_:SimpleNamespace(free=10*1024**3))
    monkeypatch.setattr(msr.subprocess,'run',lambda cmd,**kw:SimpleNamespace(stdout=msr.REV if 'rev-parse' in cmd else ''))
    monkeypatch.setattr(msr.common,'_resident',lambda *_:None)
    calls=[]
    def download(cmd,*args,**kw):
        calls.append(cmd); (weights/'ltx').mkdir(exist_ok=True); (weights/'ltx'/item[3]).write_bytes(b'test')
    monkeypatch.setattr(msr.common,'_command',download)
    job=JobRecord('video','test')
    assert msr.prepare_msr(job,runtime=runtime,weights=weights,model_id='test') is False
    assert (runtime/'comfy/models/loras'/msr.LINK).read_bytes()==b'test'
    assert len(calls)==1 and 'studio.engines.ltx25_msr_download' in calls[0]
    msr.prepare_msr(job,runtime=runtime,weights=weights,model_id='test')
    assert len(calls)==1
    wrapper=runtime/'comfy/custom_nodes/StudioStrictMSR/__init__.py'; wrapper.write_text('foreign')
    with pytest.raises(RuntimeError,match='MSR_NODE_CONFLICT'): msr.prepare_msr(job,runtime=runtime,weights=weights,model_id='test')
    assert wrapper.read_text()=='foreign'
