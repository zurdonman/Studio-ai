"""CPU/synthetic tests only: no real checkpoints, CUDA or inference."""
from types import SimpleNamespace
import json
import subprocess
import threading
import pytest


class FakeGuard:
    def __init__(self, *args, **kwargs): self.done = threading.Event()
    def raise_if_stopped(self): pass
    def close(self): pass


def clip(path, audio=False, duration=3):
    from studio.engines.ffmpeg_utils import get_ffmpeg_bin
    args = [get_ffmpeg_bin(), '-v', 'error', '-y', '-f', 'lavfi', '-i', f'color=blue:s=128x96:r=24:d={duration}']
    if audio:
        args += ['-f', 'lavfi', '-i', f'sine=frequency=440:duration={duration}', '-c:a', 'aac', '-shortest']
    subprocess.run(args + ['-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(path)], check=True)


def test_api_requires_both_consents_and_restricts_path(temp_studio_env, monkeypatch):
    import app
    client, root = temp_studio_env
    path = root / 'uploads/clip.mp4'; clip(path, audio=True)
    submitted = []
    monkeypatch.setattr(app.job_manager, 'submit', lambda **kw: submitted.append(kw) or SimpleNamespace(to_dict=lambda **kw: {'id':'j'}))
    body = {'video_rel': 'uploads/clip.mp4'}
    assert client.post('/api/community/video-audio', json=body).status_code == 400
    body['research_license_accepted'] = True
    assert client.post('/api/community/video-audio', json=body).status_code == 400
    body['install_accepted'] = True
    assert client.post('/api/community/video-audio', json=body).status_code == 400  # audio not authorized
    body['replace_audio'] = True
    assert client.post('/api/community/video-audio', json=body).status_code == 200
    assert len(submitted) == 1 and submitted[0]['kind'] == 'video'
    body['video_rel'] = '../outside.mp4'
    assert client.post('/api/community/video-audio', json=body).status_code in (400, 404)
    body['video_rel'] = 'uploads/clip.mp4'; body['steps'] = 100
    assert client.post('/api/community/video-audio', json=body).status_code == 422


@pytest.mark.parametrize('audio,duration', [(False, 3), (True, 3), (False, 9), (False, 16.25), (False, .5)])
def test_synthetic_mux_preserves_source(temp_studio_env, monkeypatch, audio, duration):
    from studio.engines import mmaudio as m
    from studio.engines.ffmpeg_utils import probe_media_info, get_ffmpeg_bin
    from studio.jobs import JobRecord
    _, root = temp_studio_env
    path = root / 'uploads/clip.mp4'; clip(path, audio=audio, duration=duration)
    original = path.read_bytes()
    monkeypatch.setattr(m.common, 'query_gpu_info', lambda: {'has_gpu': True, 'bf16_supported': True})
    monkeypatch.setattr(m.common, '_switch_to_idle', lambda *args: False)
    monkeypatch.setattr(m.memory_manager, 'ensure_capacity', lambda **kwargs: None)
    monkeypatch.setattr(m, '_prepare', lambda job: 'python')
    monkeypatch.setattr(m.common, '_ComfyMemoryGuard', FakeGuard)
    real_command = m.common._command
    calls = []
    def fake_worker(args, job, **kwargs):
        if 'mmaudio_worker.py' in args[1]:
            input_info = probe_media_info(__import__('pathlib').Path(args[4]))
            assert input_info['duration_seconds'] >= float(args[6]) - .1
            calls.append((float(args[6]), int(args[9])))
            subprocess.run([get_ffmpeg_bin(), '-v', 'error', '-y', '-f', 'lavfi', '-i', f'sine=frequency=880:duration={args[6]}', args[5]], check=True)
        else:
            real_command(args, job, **kwargs)
    monkeypatch.setattr(m.common, '_command', fake_worker)
    result = m.generate(path, prompt='wind', negative='', seed=42, steps=25, replace_audio=audio, job=JobRecord('video', 'test'))
    assert path.read_bytes() == original
    output = root / result['rel_path']
    details = probe_media_info(output)
    import math
    expected = probe_media_info(path)['duration_seconds']
    assert details['has_video'] and details['has_audio'] and abs(details['duration_seconds'] - expected) < .3
    assert len(calls) == math.ceil(expected / 8) == result['segments']
    assert [s for _, s in calls] == list(range(42, 42 + len(calls)))
    assert all(2 <= d <= 8 for d, _ in calls)
    assert result['original_audio_replaced'] is audio
    assert json.loads(output.with_suffix('.json').read_text())['model_id'] == m.MODEL_ID


def test_input_limits_no_download(temp_studio_env, monkeypatch):
    from studio.engines import mmaudio as m
    from studio.jobs import JobRecord
    _, root = temp_studio_env
    path = root / 'uploads/clip.mp4'; clip(path, duration=9)
    assert m.validate_input(path)['duration_seconds'] == 9
    clip(path)
    monkeypatch.setattr(m.common, 'query_gpu_info', lambda: {'has_gpu': False, 'bf16_supported': False})
    monkeypatch.setattr(m, '_prepare', lambda job: pytest.fail('must not download on CPU'))
    with pytest.raises(RuntimeError, match='V2A_GPU_UNSUPPORTED'):
        m.generate(path, prompt='', negative='', seed=42, steps=25, replace_audio=False, job=JobRecord('video', 'test'))


def test_manifest_includes_all_local_components():
    from studio.engines.mmaudio_download import FILES
    assert {x[0] for x in FILES} == {'mmaudio', 'clip', 'vocoder'}
    assert all(len(x[2]) == 40 and len(x[-1]) in (40, 64) for x in FILES)
    assert sum(x[4] for x in FILES) > 10_000_000_000
