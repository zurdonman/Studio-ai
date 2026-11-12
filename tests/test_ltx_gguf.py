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
    monkeypatch.setattr(b.memory_manager, 'ensure_capacity', lambda **kwargs: None)
    monkeypatch.setattr(b, '_install', lambda *_: None)
    monkeypatch.setattr(b, '_probe_runtime', lambda *_: None)
    monkeypatch.setattr(b, '_prepare_models', lambda *_: None)
    monkeypatch.setattr(b, '_make_model_links', lambda *_: {'dit':'x.gguf', 'text_encoder':'g.gguf', 'connectors':'c.safetensors', 'video_vae':'v.safetensors', 'audio_vae':'a.safetensors'})
    monkeypatch.setattr(b.shutil, 'disk_usage', lambda *_: type('D', (), {'free': 100 * 1024**3})())
    seen = []
    def synth(g, inp, out, job):
        assert (inp / 'frame.png').is_file()
        assert g['10']['inputs']['video_latent'] == ['22', 0]
        seen.append(g)
        path = out / 'studio_gguf_00001_.mp4'
        subprocess.run([get_ffmpeg_bin(), '-v', 'error', '-f','lavfi','-i','color=c=blue:s=320x256:r=24:d=1',
                        '-f','lavfi','-i','sine=frequency=440:sample_rate=24000:duration=1',
                        '-c:v','libx264','-pix_fmt','yuv420p','-c:a','aac','-shortest','-y',str(path)], check=True)
        return path
    monkeypatch.setattr(b, '_infer', synth)
    result = b.generate_gguf(image_path=img, prompt='test', negative_prompt='', seed=42, job=JobRecord('video','test'))
    info = probe_media_info(root / result['rel_path'])
    assert info['has_video'] and info['has_audio'] and result['audio_native'] is True and len(seen) == 1


def test_gguf_api_input_validation_and_panel(temp_studio_env):
    client, _ = temp_studio_env
    assert client.post('/api/video/ltx-gguf', json={'prompt':'   '}).status_code == 400
    assert client.post('/api/video/ltx-gguf', json={'prompt':'test', 'image_rel':'../escape'}).status_code in (400,404)
    assert 'id="btn-run-ltx-gguf"' in client.get('/').text


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
    r = client.post('/api/video/ltx-gguf', json={'prompt':'Hola', 'image_rel':'uploads/frame.png', 'seed':7})
    assert r.status_code == 200
    ident = r.json()['job']['id']
    for _ in range(50):
        if calls:
            break
        time.sleep(.02)
    assert calls and calls[0]['image_path'] == image and calls[0]['seed'] == 7
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
