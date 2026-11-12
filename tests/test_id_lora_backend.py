"""Test wiring with synthetic media only: no AI, no model weights, no GPU."""
import json
import subprocess
from pathlib import Path

from PIL import Image


def make_inputs(root):
    from studio.engines.ffmpeg_utils import get_ffmpeg_bin
    img = root / 'uploads' / 'id_lora_frame.png'
    voice = root / 'uploads' / 'id_lora_reference.wav'
    Image.new('RGB', (256, 256), 'blue').save(img)
    subprocess.run([get_ffmpeg_bin(), '-v', 'error', '-f', 'lavfi', '-i', 'sine=frequency=350:duration=2.5', '-y', str(voice)], check=True)
    return img, voice


def payload(reply):
    return json.loads(reply['content'][0]['text'])


def test_preflight_rejects_cpu_before_any_setup(temp_studio_env, monkeypatch):
    import studio.engines.id_lora_backend as engine
    from studio.jobs import JobRecord
    import pytest
    _, root = temp_studio_env
    image, voice = make_inputs(root)
    monkeypatch.setattr(engine, 'query_gpu_info', lambda: {'has_gpu': False})
    called = []
    monkeypatch.setattr(engine, '_prepare_weights', lambda *a, **kw: called.append('weights'))
    monkeypatch.setattr(engine, '_prepare_runtime', lambda *a, **kw: called.append('runtime'))
    job = JobRecord(kind='video', title='test')
    with pytest.raises(RuntimeError, match='ID_LORA_GPU_NOT_VALIDATED'):
        engine.generate_id_lora_clip(image, voice, 'Habla', 42, job)
    assert called == []


def test_id_lora_routes_two_synthetic_clips_and_montage(temp_studio_env, monkeypatch):
    import studio.engines.id_lora_backend as engine
    import studio.mcp_server as mcp
    from studio.engines.ffmpeg_utils import get_ffmpeg_bin, probe_media_info
    _, root = temp_studio_env
    image, voice = make_inputs(root)
    monkeypatch.setattr(engine, '_hardware_admission', lambda j: None)
    monkeypatch.setattr(engine, '_prepare_runtime', lambda j: None)
    monkeypatch.setattr(engine, '_prepare_weights', lambda j, **kw: None)
    calls = []
    ffmpeg = get_ffmpeg_bin()
    def fake_child(cmd, job, **kwargs):
        if kwargs['stage'] == 'Preparando voz de referencia':
            subprocess.run(cmd, check=True, capture_output=True)
            return
        assert kwargs['stage'] == 'Generando vídeo y voz con ID-LoRA'
        def arg(name):
            return cmd[cmd.index(name) + 1]
        assert arg('--first-frame') == str(image)
        assert Path(arg('--reference-audio')).is_file()
        assert arg('--num-frames') == '121' and arg('--num-inference-steps') == '30'
        calls.append(cmd)
        outdir = Path(arg('--output-dir'))
        outdir.mkdir()
        tone = 440 if len(calls) == 1 else 880
        subprocess.run([ffmpeg, '-v', 'error', '-f', 'lavfi', '-i', 'color=c=blue:s=256x256:r=25:d=1',
                        '-f', 'lavfi', '-i', f'sine=frequency={tone}:sample_rate=48000:duration=1',
                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-shortest', '-y',
                        str(outdir / 'output_0000.mp4')], check=True)
    monkeypatch.setattr(engine, '_run_child', fake_child)
    clips = []
    for seed in (42, 43):
        result = payload(mcp._execute_mcp_tool('studio_generate_id_lora_video', {
            'image_rel': 'uploads/id_lora_frame.png', 'reference_voice_rel': 'uploads/id_lora_reference.wav',
            'prompt': '[VISUAL]: Persona hablando. [SPEECH]: Hola.', 'seed': seed, 'wait': True}))
        assert result['status'] == 'completed', result
        assert result['result']['audio_native'] is True
        clips.append(result['result']['rel_path'])
    result = payload(mcp._execute_mcp_tool('studio_render_montage', {
        'items': [{'media_rel': x} for x in clips], 'width': 256, 'height': 256, 'fps': 25, 'fade_seconds': 0, 'wait': True}))
    assert result['status'] == 'completed', result
    info = probe_media_info(root / result['result']['rel_path'])
    assert info['has_video'] and info['has_audio'] and 1.9 <= info['duration_seconds'] <= 2.2
    assert len(calls) == 2


def test_id_lora_inputs_rejected_and_dedicated_ui(temp_studio_env):
    client, root = temp_studio_env
    make_inputs(root)
    resp = client.post('/api/video/id-lora', json={'image_rel': 'uploads/id_lora_frame.png', 'reference_voice_rel': 'uploads/id_lora_reference.wav', 'prompt': ' '})
    assert resp.status_code == 400
    resp = client.post('/api/video/id-lora', json={'image_rel': '../other', 'reference_voice_rel': 'uploads/id_lora_reference.wav', 'prompt': 'Habla'})
    assert resp.status_code in (400, 404)
    html = client.get('/').text
    assert 'id="btn-id-lora"' in html and 'id="id-lora-voice"' in html


def test_weights_marker_detects_corruption(temp_studio_env, monkeypatch, tmp_path):
    import studio.engines.id_lora_backend as engine
    from studio.engines.id_lora_download import BASE_NAME, LORA_NAME
    root = tmp_path / 'weights'
    for folder, name, content in [('base', BASE_NAME, b'base'), ('lora', LORA_NAME, b'lora'),
                                  ('gemma', 'config.json', b'{}'), ('gemma', 'model-1.safetensors', b'gemma')]:
        path = root / folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    files = {str(p.relative_to(root)): p.stat().st_size for p in root.rglob('*') if p.is_file()}
    (root / 'complete.json').write_text(json.dumps({'files': files}))
    monkeypatch.setattr(engine, 'WEIGHTS_ROOT', root)
    assert engine._weights_ready()
    (root / 'gemma/model-1.safetensors').write_bytes(b'x')
    assert not engine._weights_ready()


def test_child_process_cancelled(temp_studio_env, tmp_path):
    import sys
    import threading
    import pytest
    from studio.jobs import JobCancelledError, JobRecord
    from studio.engines.id_lora_backend import _run_child
    job = JobRecord(kind='video', title='cancel test')
    sentinel = tmp_path / 'must-not-exist'
    timer = threading.Timer(.5, lambda: setattr(job, 'cancel_requested', True))
    timer.start()
    try:
        with pytest.raises(JobCancelledError):
            _run_child([sys.executable, '-c', f'import time;time.sleep(3);open({str(sentinel)!r},"w").write("bad")'], job, timeout=10, stage='Prueba')
    finally:
        timer.cancel()
    assert not sentinel.exists()
