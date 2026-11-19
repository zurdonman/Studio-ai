"""Fish integration tests with synthetic audio; no weights or GPU required."""
import json
import time
import wave
from pathlib import Path

import pytest


def wav(path, seconds=5):
    with wave.open(str(path), 'wb') as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(8000)
        f.writeframes(b'\0\0' * int(8000 * seconds))


def test_fish_api_consent_and_reference(temp_studio_env, monkeypatch):
    from studio.engines import fish_s2 as fish
    client, root = temp_studio_env
    reference = root/'uploads'/'fish_ref.wav'; wav(reference)
    calls = []
    monkeypatch.setattr(fish, 'synthesize', lambda **kw: calls.append(kw) or {})
    body = {'text': 'Hola', 'license_accepted': True, 'install_accepted': True}
    assert client.post('/api/tts/fish', json={'text':'Hola'}).status_code == 400
    assert client.post('/api/tts/fish', json={**body, 'reference_rel':'uploads/fish_ref.wav', 'reference_text':'hola'}).status_code == 400
    assert client.post('/api/tts/fish', json={**body, 'reference_rel':'uploads/fish_ref.wav', 'authorized':True}).status_code == 400
    assert client.post('/api/tts/fish', json={**body, 'reference_text':'hola'}).status_code == 400
    assert client.post('/api/tts/fish', json={**body, 'reference_rel':'../secret', 'reference_text':'hola', 'authorized':True}).status_code == 400
    assert client.post('/api/tts/fish', json=body).status_code == 200
    assert client.post('/api/tts/fish', json={**body, 'reference_rel':'uploads/fish_ref.wav', 'reference_text':'hola', 'authorized':True}).status_code == 200
    for _ in range(100):
        if len(calls) == 2: break
        time.sleep(.02)
    assert len(calls) == 2
    assert calls[0]['reference_path'] is None
    assert calls[1]['reference_path'] == reference


@pytest.mark.parametrize('clone', [False, True])
def test_fish_synthetic_output_and_cleanup(temp_studio_env, monkeypatch, tmp_path, clone):
    from studio.engines import fish_s2 as fish, voice_clone_reference as ref
    from studio.jobs import JobRecord
    client, root = temp_studio_env
    source = root/'uploads'/'ref.wav'; wav(source)
    monkeypatch.setattr(fish, 'TEMP_DIR', tmp_path/'fish')
    monkeypatch.setattr(fish, 'AUDIO_OUT_DIR', root/'outputs/audio')
    monkeypatch.setattr(ref, 'TEMP_DIR', tmp_path/'references')
    monkeypatch.setattr(fish.common, 'query_gpu_info', lambda: {'has_gpu':True, 'bf16_supported':True})
    monkeypatch.setattr(fish.common, '_switch_to_idle', lambda job:None)
    monkeypatch.setattr(fish.memory_manager, 'ensure_capacity', lambda **kw:None)
    monkeypatch.setattr(fish.memory_manager, 'get_slot', lambda slot:None)
    monkeypatch.setattr(fish, 'prepare', lambda job:Path('/fake/python'))
    seen = []
    def command(args, job, **kwargs):
        request = json.loads(Path(args[-1]).read_text())
        seen.append(request)
        if clone:
            assert Path(request['reference']).exists() and request['transcript'] == 'muestra'
        else:
            assert request['reference'] is None and not request['transcript']
        wav(request['output'], 1)
    monkeypatch.setattr(fish.common, '_command', command)
    result = fish.synthesize(text='hola', reference_path=source if clone else None, reference_text='muestra' if clone else '', job=JobRecord('tts','test'))
    assert result['model_id'] == fish.ENGINE and result['duration_seconds'] == 1
    assert client.get(result['media_url']).status_code == 200
    assert not list((tmp_path/'fish').iterdir())
    if clone: assert not list((tmp_path/'references').iterdir())
    assert 'muestra' not in (root/'outputs/audio'/(result['id']+'.json')).read_text()


def test_fish_no_gpu_and_missing_transcript_never_install(monkeypatch, tmp_path):
    from studio.engines import fish_s2 as fish
    from studio.jobs import JobRecord
    monkeypatch.setattr(fish.common, 'query_gpu_info', lambda: {'has_gpu':False})
    monkeypatch.setattr(fish, 'prepare', lambda job: pytest.fail('must not install'))
    with pytest.raises(ValueError, match='transcripción'):
        fish.synthesize(text='hola', reference_path=tmp_path/'a.wav', reference_text='', job=JobRecord('tts','test'))
    with pytest.raises(RuntimeError, match='FISH_GPU_UNSUPPORTED'):
        fish.synthesize(text='hola', reference_path=None, reference_text='', job=JobRecord('tts','test'))
