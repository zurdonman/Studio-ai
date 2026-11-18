"""Experimental H3 contracts; no models, network, or GPU required."""
import json
from pathlib import Path

import pytest
from PIL import Image


def test_manifest_pinned_and_variants_share_vaes(temp_studio_env):
    from studio.engines.minimax_h3 import files_for
    q2, q4 = files_for('h3-fl2va-udq2'), files_for('h3-fl2va-q4')
    assert [a[0] for a in q2] == ['dit', 'text_encoder', 'video_vae', 'audio_vae']
    assert q2[2:] == q4[2:] and q2[1] != q4[1]
    assert all(len(x[2]) == 40 and len(x[5]) == 64 and x[4] > 0 for x in q2 + q4)
    with pytest.raises(ValueError, match='H3_MODEL_UNSUPPORTED'):
        files_for('../../bad')


def test_command_has_expected_components_and_frames(temp_studio_env, tmp_path):
    from studio.engines.minimax_h3 import build_command
    image, end, binary, output = (tmp_path / x for x in ('first.png', 'last.png', 'sd-cli', 'out.avi'))
    cmd = build_command(binary, 'h3-fl2va-udq2', 'hello; no shell', 640, 384, 22, 4, 42, output, image, end, True)
    assert cmd[0] == str(binary)
    assert cmd[cmd.index('--prompt') + 1] == 'hello; no shell'
    assert cmd[cmd.index('--mode') + 1] == 'vid_gen'
    assert cmd[cmd.index('--backend') + 1] == 'te=cpu'
    assert cmd[cmd.index('--video-frames') + 1] == '22'
    assert cmd[cmd.index('--cfg-scale') + 1] == '1.0'
    assert cmd[cmd.index('--init-img') + 1] == str(image)
    assert cmd[cmd.index('--end-img') + 1] == str(end)
    assert '--offload-to-cpu' in cmd and '--audio-vae' in cmd
    minimal = build_command(binary, 'h3-fl2va-q4', 'hello', 640, 384, 39, 8, 5, output, None, None, False)
    assert '--end-img' not in minimal and '--init-img' not in minimal and '--offload-to-cpu' not in minimal


def test_h3_api_license_and_inputs_fail_before_job_submission(temp_studio_env, monkeypatch):
    from app import job_manager
    client, root = temp_studio_env
    submitted = []
    class Record:
        def to_dict(self): return {'id': 'job-1', 'status': 'queued'}
    monkeypatch.setattr(job_manager, 'submit', lambda **kw: submitted.append(kw) or Record())
    base = {'model_id': 'h3-fl2va-udq2', 'prompt': 'Una escena'}
    assert client.post('/api/video/h3/prepare', json={'model_id': base['model_id']}).status_code == 400
    assert client.post('/api/video/h3/generate', json=base).status_code == 400
    assert client.post('/api/video/h3/generate', json={**base, 'license_authorized': True}).status_code == 400
    assert not submitted
    allowed = {**base, 'license_authorized': True, 'download_authorized': True}
    assert client.post('/api/video/h3/generate', json={**allowed, 'end_image_rel':'uploads/last.png'}).status_code == 400
    assert client.post('/api/video/h3/generate', json={**allowed, 'width': 513}).status_code == 400
    assert client.post('/api/video/h3/generate', json={**allowed, 'frames': 25}).status_code == 422
    assert client.post('/api/video/h3/generate', json={**allowed, 'image_rel':'../../etc/passwd'}).status_code == 400
    assert client.post('/api/video/h3/prepare', json={'model_id':'unknown', 'license_authorized':True}).status_code == 422
    assert not submitted
    uploads = root / 'uploads'; uploads.mkdir(exist_ok=True)
    Image.new('RGB', (32, 32), 'red').save(uploads / 'h3_test.png')
    response = client.post('/api/video/h3/generate', json={**allowed, 'image_rel':'uploads/h3_test.png'})
    assert response.status_code == 200 and submitted[-1]['kind'] == 'video'
    assert client.post('/api/video/h3/prepare', json={'model_id': base['model_id'], 'license_authorized':True}).status_code == 200
    assert submitted[-1]['dedup_key'] == 'h3-prepare:h3-fl2va-udq2'


def test_preparation_rejects_non_l4_before_install(temp_studio_env, monkeypatch):
    from studio.engines import minimax_h3 as h3
    from studio.jobs import JobRecord
    monkeypatch.setattr(h3.common, 'query_gpu_info', lambda: {'has_gpu': True, 'gpu_name': 'NVIDIA T4'})
    monkeypatch.setattr(h3, '_install', lambda job: pytest.fail('must not install'))
    with pytest.raises(RuntimeError, match='H3_GPU_UNSUPPORTED'):
        h3._prepare('h3-fl2va-udq2', JobRecord('download', 'test'))


def test_generation_prepares_missing_model_and_propagates_setup_errors(temp_studio_env, monkeypatch, tmp_path):
    from studio.engines import minimax_h3 as h3
    from studio.jobs import JobRecord
    monkeypatch.setattr(h3, 'ROOT', tmp_path / 'empty')
    calls = []
    def prepare(model_id, job):
        calls.append(model_id)
        raise RuntimeError('setup failure')
    monkeypatch.setattr(h3, '_prepare', prepare)
    with pytest.raises(RuntimeError, match='setup failure'):
        h3.generate('h3-fl2va-q4', prompt='hola', image=None, end=None, width=640, height=384, frames=22, steps=4, seed=0, offload=True, job=JobRecord('video', 'test'))
    assert calls == ['h3-fl2va-q4']
    with pytest.raises(ValueError, match='H3_INPUT_INVALID'):
        h3.generate('h3-fl2va-q4', prompt='', image=None, end=None, width=640, height=384, frames=22, steps=4, seed=0, offload=True, job=JobRecord('video', 'test'))
    assert calls == ['h3-fl2va-q4']


@pytest.mark.parametrize('conversion_fails', [False, True])
def test_synthetic_pipeline_keeps_audio_and_recovers_on_conversion_failure(temp_studio_env, tmp_path, monkeypatch, conversion_fails):
    from studio.engines import minimax_h3 as h3
    from studio.engines.ffmpeg_utils import get_ffmpeg_bin, probe_media_info
    from studio.jobs import JobRecord
    runtime = tmp_path / 'runtime'; binary = runtime / 'build' / 'bin' / 'sd-cli'
    binary.parent.mkdir(parents=True)
    binary.write_text('''#!/usr/bin/env python3
import subprocess, sys
out = sys.argv[sys.argv.index("--output") + 1]
subprocess.run([%r, "-v", "error", "-f", "lavfi", "-i", "color=c=blue:s=320x256:r=24:d=1", "-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-c:v", "mjpeg", "-threads", "1", "-c:a", "pcm_s16le", "-shortest", "-y", out], check=True)
''' % get_ffmpeg_bin())
    binary.chmod(0o755)  # fake preparer below; no ready marker or weights exist
    monkeypatch.setattr(h3, 'ROOT', runtime)
    monkeypatch.setattr(h3, 'TEMP_DIR', tmp_path / 'temp')
    monkeypatch.setattr(h3, 'VIDEOS_OUT_DIR', tmp_path / 'videos')
    monkeypatch.setattr(h3, 'fingerprint', lambda *_: (1,))
    monkeypatch.setattr(h3, '_prepare', lambda *a: binary)
    monkeypatch.setattr(h3.memory_manager, 'get_slot', lambda *_: None)
    monkeypatch.setattr(h3.memory_manager, 'ensure_capacity', lambda **kw: None)
    class Guard:
        def __init__(self, *a, **kw):
            import threading
            self.done = threading.Event()
        def close(self): pass
        def raise_if_stopped(self): pass
    monkeypatch.setattr(h3.common, '_ComfyMemoryGuard', Guard)
    if conversion_fails:
        monkeypatch.setattr(h3.common, '_command', lambda *a, **kw: (_ for _ in ()).throw(RuntimeError('ffmpeg failed')))
    kw = dict(model_id='h3-fl2va-udq2', prompt='test', image=None, end=None, width=640, height=384,
              frames=22, steps=4, seed=42, offload=True, job=JobRecord('video', 'test'))
    if conversion_fails:
        with pytest.raises(RuntimeError, match='ffmpeg failed'):
            h3.generate.__wrapped__(**kw)
        recovered = list((tmp_path / 'videos').glob('*_recuperacion.avi'))
        assert len(recovered) == 1 and probe_media_info(recovered[0])['has_audio']
    else:
        result = h3.generate.__wrapped__(**kw)
        output = tmp_path / 'videos' / result['filename']
        info = probe_media_info(output)
        assert info['has_audio'] and info['has_video'] and output.suffix == '.mp4'
        assert json.loads(output.with_suffix('.json').read_text())['model_id'] == kw['model_id']
        assert not list((tmp_path / 'videos').glob('*_recuperacion.avi'))
