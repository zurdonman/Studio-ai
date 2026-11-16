"""CPU/synthetic checks only; no model download and no GPU inference."""
from types import SimpleNamespace
import json
import subprocess
import threading

import pytest


class FakeGuard:
    def __init__(self, *args, **kwargs): self.done = threading.Event()
    def raise_if_stopped(self): pass
    def close(self): pass


def make_clip(path, audio=False, duration=1):
    from studio.engines.ffmpeg_utils import get_ffmpeg_bin
    args = [get_ffmpeg_bin(), '-v', 'error', '-y', '-f', 'lavfi', '-i', f'color=blue:s=128x96:r=24:d={duration}']
    if audio:
        args += ['-f', 'lavfi', '-i', f'sine=frequency=440:duration={duration}', '-c:a', 'aac', '-shortest']
    subprocess.run(args + ['-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(path)], check=True)


def test_graph_is_fixed_and_uses_official_nodes(temp_studio_env):
    from studio.engines.freefield_film import make_graph, MODEL
    graph = make_graph('input.mp4', 'film_123')
    assert graph['4']['inputs'] == {'interp_model': ['3', 0], 'images': ['2', 0], 'multiplier': 2}
    assert graph['5']['inputs']['fps'] == 48 and graph['6']['inputs']['format.codec'] == 'auto'
    assert MODEL[2] == 'aa7009be4d1471ea0bac91d6fe4d7e4b637cbbfc'
    assert len(MODEL[-1]) == 64
    for name in ('../x', '/x', 'a/b', 'a\\b'):
        with pytest.raises(ValueError): make_graph(name, 'out')


def test_api_restricts_input_and_enqueues(temp_studio_env, monkeypatch):
    import app
    from studio.engines import freefield_film as f
    client, root = temp_studio_env
    clip = root / 'uploads' / 'clip.mp4'; clip.write_bytes(b'fake')
    invalid = root / 'models' / 'bad.mp4'; invalid.parent.mkdir(parents=True, exist_ok=True); invalid.write_bytes(b'fake')
    assert client.post('/api/community/film', json={'video_rel': '../escape'}).status_code in (400, 404)
    assert client.post('/api/community/film', json={'video_rel': 'models/bad.mp4'}).status_code == 400
    jobs = []
    monkeypatch.setattr(app.job_manager, 'submit', lambda **kw: jobs.append(kw) or SimpleNamespace(to_dict=lambda **kw: {'id':'j'}))
    result = client.post('/api/community/film', json={'video_rel': 'uploads/clip.mp4'})
    assert result.status_code == 200 and result.json()['job']['id'] == 'j'
    assert jobs[0]['kind'] == 'video' and not hasattr(f, 'last_result')


@pytest.mark.parametrize('audio', [False, True])
def test_synthetic_pipeline_preserves_optional_audio_and_cleans(temp_studio_env, monkeypatch, tmp_path, audio):
    from studio.engines import freefield_film as f
    from studio.engines.ffmpeg_utils import probe_media_info
    from studio.jobs import JobRecord
    _, root = temp_studio_env
    clip = root / 'uploads' / 'source.mp4'; make_clip(clip, audio)
    work = tmp_path / 'session'; work.mkdir()
    inp, out = work / 'input', work / 'output'; inp.mkdir(); out.mkdir()
    closed = []
    session = SimpleNamespace(root=work, input_dir=inp, output_dir=out, unload=lambda: closed.append(True),
                              guard=SimpleNamespace(raise_if_stopped=lambda: None))
    monkeypatch.setattr(f.common, '_ComfyMemoryGuard', FakeGuard)
    monkeypatch.setattr(f.common, 'query_gpu_info', lambda: {'has_gpu': True})
    monkeypatch.setattr(f.memory_manager, 'get_slot', lambda *a: None)
    monkeypatch.setattr(f.common, '_comfy_start_admission', lambda *a, **k: None)
    monkeypatch.setattr(f, '_prepare', lambda *a: None)
    monkeypatch.setattr(f.common, '_ResidentComfy', lambda *a: session)
    def fake_infer(sess, graph, job):
        assert probe_media_info(inp / 'input.mp4')['has_audio'] == audio
        fake = out / 'result.mp4'; make_clip(fake)
        return fake
    monkeypatch.setattr(f, '_infer', fake_infer)
    result = f.interpolate(clip, job=JobRecord('video','film test'))
    actual = probe_media_info(root / result['rel_path'])
    assert actual['has_video'] and actual['has_audio'] is audio and result['has_audio'] is audio
    assert result['fps'] == 48 and json.loads((root / 'outputs/videos' / (result['id']+'.json')).read_text())['model_id'] == 'freefield-film'
    assert closed == [True]


def test_failure_closes_worker(temp_studio_env, monkeypatch, tmp_path):
    from studio.engines import freefield_film as f
    from studio.jobs import JobRecord
    _, root = temp_studio_env
    clip = root / 'uploads' / 'source.mp4'; make_clip(clip)
    inp, out = tmp_path / 'input', tmp_path / 'output'; inp.mkdir(); out.mkdir()
    closed = []
    sess = SimpleNamespace(root=tmp_path, input_dir=inp, output_dir=out, unload=lambda: closed.append(True),
                           guard=SimpleNamespace(raise_if_stopped=lambda: None))
    monkeypatch.setattr(f.common, '_ComfyMemoryGuard', FakeGuard)
    monkeypatch.setattr(f.common, 'query_gpu_info', lambda: {'has_gpu': True})
    monkeypatch.setattr(f.memory_manager, 'get_slot', lambda *a: None)
    monkeypatch.setattr(f.common, '_comfy_start_admission', lambda *a, **k: None)
    monkeypatch.setattr(f, '_prepare', lambda *a: None)
    monkeypatch.setattr(f.common, '_ResidentComfy', lambda *a: sess)
    before = set((root / 'outputs/videos').glob('film*'))
    def fail(*args): raise RuntimeError('failed inference')
    monkeypatch.setattr(f, '_infer', fail)
    with pytest.raises(RuntimeError, match='failed inference'):
        f.interpolate(clip, job=JobRecord('video','film test'))
    assert closed == [True]
    assert set((root / 'outputs/videos').glob('film*')) == before


def test_input_limits_and_cpu_no_download(temp_studio_env, monkeypatch):
    from studio.engines import freefield_film as f
    from studio.jobs import JobRecord
    _, root = temp_studio_env
    clip = root / 'uploads' / 'clip.mp4'; make_clip(clip, duration=11)
    with pytest.raises(ValueError, match='FILM_INPUT_LIMIT'): f.validate_input(clip)
    make_clip(clip)
    called = []
    monkeypatch.setattr(f.common, 'query_gpu_info', lambda: {'has_gpu': False})
    monkeypatch.setattr(f, '_prepare', lambda *a: called.append(1))
    with pytest.raises(RuntimeError, match='FILM_GPU_REQUIRED'):
        f.interpolate(clip, job=JobRecord('video','film test'))
    assert not called
