"""ControlFoley integration contract; no weights, CUDA or real inference."""
import json
import subprocess
import threading
from types import SimpleNamespace
import pytest


def test_manifest_includes_v2a_weights_and_pinned_components():
    from studio.engines.controlfoley_download import FILES
    paths = {(row[0], row[3]) for row in FILES}
    assert ('control', 'weights/controlfoley.pth') in paths
    assert ('control', 'ext_weights/cav_mae_st.pth') in paths
    assert ('clip', 'open_clip_pytorch_model.bin') in paths
    assert ('vocoder', 'bigvgan_generator.pt') in paths
    assert all(len(row[2]) == 40 and len(row[-1]) in (40, 64) for row in FILES)
    assert not any('music_speech' in p for _, p in paths)  # reference-audio mode is not exposed


def test_worker_binds_namespace_and_reference_mode_is_disabled(tmp_path, monkeypatch):
    import sys
    from studio.engines import controlfoley_worker as w
    old = {k:v for k,v in sys.modules.items() if k in ('controlfoley', 'lib', 'laion_clap', 'lib.audiocraft.audiocraft.models')}
    try:
        w.bind_source_packages(tmp_path); w.install_unused_reference_stubs()
        assert list(sys.modules['controlfoley'].__path__) == [str(tmp_path/'controlfoley')]
        assert list(sys.modules['lib'].__path__) == [str(tmp_path/'lib')]
        with pytest.raises(RuntimeError, match='REFERENCE_AUDIO_UNSUPPORTED'):
            sys.modules['laion_clap'].CLAP_Module()
        assert sys.modules['lib.audiocraft.audiocraft.models'].MusicGen.get_pretrained('anything') is None
    finally:
        for k in ('controlfoley', 'lib', 'laion_clap', 'lib.audiocraft.audiocraft.models'):
            if k in old: sys.modules[k] = old[k]
            else: sys.modules.pop(k, None)


def test_api_selects_controlfoley_and_rejects_unknown(temp_studio_env, monkeypatch):
    import app
    from studio.engines.ffmpeg_utils import get_ffmpeg_bin
    client, root = temp_studio_env
    path = root/'uploads/a.mp4'
    subprocess.run([get_ffmpeg_bin(), '-v','error','-y','-f','lavfi','-i','color=s=128x96:r=24:d=2',
                    '-c:v','libx264','-pix_fmt','yuv420p',str(path)], check=True)
    submitted = []
    monkeypatch.setattr(app.job_manager, 'submit', lambda **kw: submitted.append(kw) or SimpleNamespace(to_dict=lambda **kw: {'id':'j'}))
    body = dict(video_rel='uploads/a.mp4', engine='controlfoley', research_license_accepted=True, install_accepted=True)
    assert client.post('/api/community/video-audio', json=body).status_code == 200
    assert 'controlfoley' in submitted[0]['title']
    body['engine'] = 'unknown-engine'
    assert client.post('/api/community/video-audio', json=body).status_code == 422


def test_controlfoley_synthetic_multisegment_preserves_video(temp_studio_env, monkeypatch):
    from studio.engines import mmaudio as m, controlfoley as c
    from studio.engines.ffmpeg_utils import get_ffmpeg_bin, probe_media_info
    from studio.jobs import JobRecord
    _, root = temp_studio_env
    path = root/'uploads/a.mp4'
    subprocess.run([get_ffmpeg_bin(), '-v','error','-y','-f','lavfi','-i','color=s=128x96:r=24:d=10',
                    '-c:v','libx264','-pix_fmt','yuv420p',str(path)], check=True)
    original = path.read_bytes()
    monkeypatch.setattr(m.common, 'query_gpu_info', lambda: {'has_gpu':True,'bf16_supported':True})
    monkeypatch.setattr(m.common, '_switch_to_idle', lambda *args: False)
    monkeypatch.setattr(m.memory_manager, 'ensure_capacity', lambda **kwargs: None)
    monkeypatch.setattr(c, 'prepare', lambda job: 'python')
    class Guard:
        def __init__(self, *a, **kw): self.done = threading.Event()
        def raise_if_stopped(self): pass
        def close(self): pass
    monkeypatch.setattr(m.common, '_ComfyMemoryGuard', Guard)
    real = m.common._command; calls = []
    def fake_worker(args, job, **kwargs):
        if 'controlfoley_worker.py' in args[1]:
            calls.append(args)
            subprocess.run([get_ffmpeg_bin(), '-v','error','-y','-f','lavfi','-i',f'sine=frequency=880:duration={args[6]}',args[5]], check=True)
        else: real(args, job, **kwargs)
    monkeypatch.setattr(m.common, '_command', fake_worker)
    result = m.generate(path, engine='controlfoley', prompt='wind', negative='', seed=42, steps=25,
                        replace_audio=False, job=JobRecord('video', 'test'))
    output = root/result['rel_path']
    assert result['model_id'] == c.MODEL_ID and result['segments'] == len(calls) == 2
    assert probe_media_info(output)['has_audio'] and abs(probe_media_info(output)['duration_seconds']-10) < .3
    assert path.read_bytes() == original
    assert json.loads(output.with_suffix('.json').read_text())['model_name'].startswith('ControlFoley')
