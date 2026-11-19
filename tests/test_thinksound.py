"""CPU-only and synthetic integration tests; not real model inference."""
import json
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace


def test_manifest_pins_encoders_and_light_model():
    from studio.engines.thinksound_download import FILES
    assert {row[3] for row in FILES} >= {'thinksound_light.ckpt', 'vae.ckpt', 'synchformer_state_dict.pth', 'model.safetensors', 'pytorch_model.bin', 'spiece.model'}
    assert sum(row[4] for row in FILES) > 24_000_000_000
    assert all(len(row[2]) == 40 and len(row[6]) in (40, 64) for row in FILES)


def test_thinksound_license_is_separate_from_apple(temp_studio_env, monkeypatch):
    import app
    from studio.engines.ffmpeg_utils import get_ffmpeg_bin
    client, root = temp_studio_env
    path = root/'uploads/think.mp4'
    subprocess.run([get_ffmpeg_bin(), '-v','error','-y','-f','lavfi','-i','color=s=128x96:r=25:d=2', '-c:v','libx264', str(path)], check=True)
    monkeypatch.setattr(app.job_manager, 'submit', lambda **kw: SimpleNamespace(to_dict=lambda **kw: {'id':'j'}))
    body = dict(video_rel='uploads/think.mp4', engine='thinksound', install_accepted=True)
    assert client.post('/api/community/video-audio', json={**body, 'research_license_accepted':True}).status_code == 400
    assert client.post('/api/community/video-audio', json={**body, 'model_license_accepted':True}).status_code == 200


def test_synthetic_full_video_and_eight_second_padding(temp_studio_env, monkeypatch):
    from studio.engines import mmaudio as m, thinksound as t
    from studio.engines.ffmpeg_utils import get_ffmpeg_bin, probe_media_info
    from studio.jobs import JobRecord
    _, root = temp_studio_env
    path = root/'uploads/think.mp4'
    subprocess.run([get_ffmpeg_bin(), '-v','error','-y','-f','lavfi','-i','color=s=128x96:r=25:d=9', '-c:v','libx264', str(path)], check=True)
    original = path.read_bytes()
    monkeypatch.setattr(m.common, 'query_gpu_info', lambda: {'has_gpu':True, 'bf16_supported':True})
    monkeypatch.setattr(m.common, '_switch_to_idle', lambda *args: False)
    monkeypatch.setattr(m.memory_manager, 'ensure_capacity', lambda **kw:None)
    monkeypatch.setattr(t, 'prepare', lambda job:'python')
    class Guard:
        def __init__(self, *a, **kw): self.done = threading.Event()
        def raise_if_stopped(self): pass
        def close(self): pass
    monkeypatch.setattr(m.common, '_ComfyMemoryGuard', Guard)
    real = m.common._command; calls = []
    def command(args, job, **kwargs):
        if 'thinksound_worker.py' not in args[1]: return real(args, job, **kwargs)
        mode = args[2]; request = Path(args[-1]); work = request.parent
        data = json.loads(request.read_text()); calls.append((mode, data['seed']))
        assert abs(probe_media_info(work/'videos/demo.mp4')['duration_seconds']-8) < .2
        if mode == 'extract':
            (work/'features').mkdir(); (work/'features/demo.npz').write_bytes(b'fake')
        else:
            subprocess.run([get_ffmpeg_bin(), '-v','error','-y','-f','lavfi','-i','sine=frequency=440:duration=8',data['output']], check=True)
    monkeypatch.setattr(m.common, '_command', command)
    result = m.generate(path, engine='thinksound', prompt='wind', negative='', seed=42, steps=25, replace_audio=False, job=JobRecord('video','test'))
    assert calls == [('extract',42), ('infer',42), ('extract',43), ('infer',43)]
    assert result['model_id'] == t.MODEL_ID and result['steps'] == 24 and result['segments'] == 2
    assert abs(probe_media_info(root/result['rel_path'])['duration_seconds']-9) < .3
    assert path.read_bytes() == original


def test_av_video_reader_rates_and_truncated_input(tmp_path):
    """Exercise the real decoder without GPU or model weights, in isolation."""
    import importlib.util
    import sys
    import pytest
    if importlib.util.find_spec('av') is None or importlib.util.find_spec('torch') is None:
        pytest.skip('PyAV and torch are needed for the decoder smoke test')
    from studio.engines.ffmpeg_utils import get_ffmpeg_bin
    video = tmp_path/'frames.mp4'
    subprocess.run([get_ffmpeg_bin(), '-v','error','-y','-f','lavfi','-i',
                    'testsrc2=s=64x64:r=25:d=8', '-c:v','libx264',str(video)], check=True)
    script = '''
import sys, torch
from studio.engines.thinksound_worker import AvVideoReader
r=AvVideoReader(sys.argv[1])
for fps in (4,8,25): r.add_basic_video_stream(frames_per_chunk=8*fps,frame_rate=fps,format='rgb24')
r.fill_buffer()
for fps, chunk in zip((4,8,25),r.pop_chunks()):
 assert tuple(chunk.shape)==(8*fps,3,64,64) and chunk.dtype==torch.uint8
try:
 r=AvVideoReader(sys.argv[1]); r.add_basic_video_stream(frames_per_chunk=201,frame_rate=25,format='rgb24'); r.fill_buffer()
except RuntimeError: pass
else: raise AssertionError('missing frames must be rejected')
'''
    subprocess.run([sys.executable, '-c', script, str(video)], check=True, timeout=60)
