import math
from pathlib import Path
import subprocess
import threading
import pytest


@pytest.mark.parametrize('duration', [0.04, .5, 2, 7.99, 8, 8.01, 9, 15.99, 16, 16.25, 28.5, 60])
def test_segments_cover_whole_clip_without_gaps(duration):
    from studio.engines.mmaudio import plan_segments
    parts = list(plan_segments(duration))
    assert len(parts) == math.ceil(duration / 8)
    assert parts[0][0] == 0
    assert abs(parts[-1][0] + parts[-1][1] - duration) < 1e-7
    for i, (start, length) in enumerate(parts):
        assert 0 < length <= 8
        if duration >= 2: assert length >= 2 - 1e-7
        if i: assert abs(parts[i-1][0] + parts[i-1][1] - start) < 1e-7


def test_mid_segment_failure_never_publishes_partial_video(temp_studio_env, monkeypatch):
    from studio.engines import mmaudio as m
    from studio.engines.ffmpeg_utils import get_ffmpeg_bin
    from studio.jobs import JobRecord
    _, root = temp_studio_env
    clip = root / 'uploads/clip.mp4'
    subprocess.run([get_ffmpeg_bin(), '-v', 'error', '-y', '-f', 'lavfi', '-i', 'color=s=128x96:r=24:d=10',
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(clip)], check=True)
    before = clip.read_bytes()
    outputs_before = set((root / 'outputs/videos').glob('mmaudio*'))
    monkeypatch.setattr(m.common, 'query_gpu_info', lambda: {'has_gpu': True, 'bf16_supported': True})
    monkeypatch.setattr(m.common, '_switch_to_idle', lambda *a: False)
    monkeypatch.setattr(m.memory_manager, 'ensure_capacity', lambda **k: None)
    monkeypatch.setattr(m, '_prepare', lambda job: 'python')
    class Guard:
        def __init__(self, *a, **kw): self.done = threading.Event()
        def raise_if_stopped(self): pass
        def close(self): pass
    monkeypatch.setattr(m.common, '_ComfyMemoryGuard', Guard)
    real = m.common._command
    seen = []
    def worker(args, job, **kwargs):
        if 'mmaudio_worker.py' not in args[1]: return real(args, job, **kwargs)
        seen.append(1)
        if len(seen) == 2: raise RuntimeError('second segment failure')
        subprocess.run([get_ffmpeg_bin(), '-v', 'error', '-y', '-f', 'lavfi', '-i',
                        f'sine=frequency=880:duration={args[6]}', args[5]], check=True)
    monkeypatch.setattr(m.common, '_command', worker)
    with pytest.raises(RuntimeError, match='second segment failure'):
        m.generate(clip, prompt='', negative='', seed=1, steps=25, replace_audio=False, job=JobRecord('video', 'test'))
    assert len(seen) == 2 and clip.read_bytes() == before
    assert set((root / 'outputs/videos').glob('mmaudio*')) == outputs_before
    assert not list((root / 'temp').glob('mmaudio-*'))
