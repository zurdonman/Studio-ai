"""CPU, synthetic and mocked tests: no claim of GPU/model inference."""
import hashlib
import subprocess
import time
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest


def wav(path, seconds=5):
    with wave.open(str(path),'wb') as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(8000); f.writeframes(b'\x00\x00'*int(8000*seconds))


def test_hashes_validate_git_blob_and_lfs(tmp_path):
    from studio.engines.qwen_clone_download import verify
    p=tmp_path/'a'; p.write_bytes(b'abc')
    assert verify(p,3,'git',hashlib.sha1(b'blob 3\0abc').hexdigest())
    assert verify(p,3,'sha256',hashlib.sha256(b'abc').hexdigest())
    assert not verify(p,4,'git',hashlib.sha1(b'blob 3\0abc').hexdigest())
    assert not verify(p,3,'sha256','wrong')


def test_clone_api_requires_permission_and_valid_audio(temp_studio_env, monkeypatch):
    from studio.engines import qwen_clone as q
    client,root=temp_studio_env
    reference=root/'uploads'/'qwen_ref.wav'; wav(reference)
    calls=[]; monkeypatch.setattr(q,'synthesize',lambda **kwargs: calls.append(kwargs) or {})
    body={'text':'Hola','reference_rel':'uploads/qwen_ref.wav'}
    assert client.post('/api/tts/clone',json=body).status_code==400
    assert client.post('/api/tts/clone',json={**body,'authorized':True,'reference_rel':'../x'}).status_code==400
    assert client.post('/api/tts/clone',json={**body,'authorized':True,'text':' '}).status_code==400
    response=client.post('/api/tts/clone',json={**body,'authorized':True,'reference_text':'muestra'})
    assert response.status_code==200, response.text
    for _ in range(50):
        if calls: break
        time.sleep(.02)
    assert calls and calls[0]['reference_path']==reference and calls[0]['reference_text']=='muestra'


def test_clone_reuses_slot_and_publishes_audio(temp_studio_env,monkeypatch,tmp_path):
    from studio.engines import qwen_clone as q, voice_clone_reference as ref
    from studio.jobs import JobRecord
    client,root=temp_studio_env
    source=root/'uploads'/'refq.wav'; wav(source)
    monkeypatch.setattr(q,'AUDIO_OUT_DIR',root/'outputs/audio')
    monkeypatch.setattr(ref,'TEMP_DIR',tmp_path/'refs')
    monkeypatch.setattr(q,'query_gpu_info',lambda:{'has_gpu':True,'bf16_supported':True})
    monkeypatch.setattr(q.memory_manager,'ensure_capacity',lambda **kw:None)
    monkeypatch.setattr(q,'_prepare',lambda job:pytest.fail('cached session should not download'))
    class Session:
        directory=tmp_path
        def alive(self): return True
        def generate(self, job, text, transcript, reference, output):
            assert reference.exists() and transcript=='referencia' and text=='hola'
            wav(output,1)
    session=Session()
    monkeypatch.setattr(q.memory_manager,'get_slot',lambda slot:{'model_id':q.ENGINE,'handle':session})
    result=q.synthesize(text='hola',reference_text='referencia',reference_path=source,job=JobRecord('tts','test'))
    assert result['model_id']==q.ENGINE and result['duration_seconds']==1
    assert client.get(result['media_url']).status_code==200
    assert any(a['rel_path']==result['rel_path'] for a in client.get('/api/assets').json()['audio'])
    assert not list((tmp_path/'refs').iterdir())
    sidecar=(root/'outputs/audio'/(result['id']+'.json')).read_text()
    assert 'referencia' not in sidecar and 'refq.wav' not in sidecar


def test_clone_no_gpu_does_not_install(temp_studio_env, monkeypatch,tmp_path):
    from studio.engines import qwen_clone as q
    from studio.jobs import JobRecord
    monkeypatch.setattr(q,'query_gpu_info',lambda:{'has_gpu':False})
    monkeypatch.setattr(q,'_prepare',lambda job:pytest.fail('must not download'))
    with pytest.raises(RuntimeError,match='QWEN_GPU_UNSUPPORTED'):
        q.synthesize(text='hola',reference_text=None,reference_path=tmp_path/'x',job=JobRecord('tts','test'))


def test_clone_failure_unloads_session(temp_studio_env,monkeypatch,tmp_path):
    from studio.engines import qwen_clone as q, voice_clone_reference as ref
    from studio.jobs import JobRecord
    source=tmp_path/'input.wav'; wav(source)
    monkeypatch.setattr(ref,'TEMP_DIR',tmp_path/'refs')
    monkeypatch.setattr(q,'AUDIO_OUT_DIR',tmp_path/'output')
    monkeypatch.setattr(q,'query_gpu_info',lambda:{'has_gpu':True,'bf16_supported':True})
    monkeypatch.setattr(q.memory_manager,'ensure_capacity',lambda **kw:None)
    class Session:
        directory=tmp_path
        def alive(self): return True
        def generate(self,*args): raise RuntimeError('worker failed')
    monkeypatch.setattr(q.memory_manager,'get_slot',lambda slot:{'model_id':q.ENGINE,'handle':Session()})
    unloaded=[]; monkeypatch.setattr(q.memory_manager,'unload_slot',lambda slot:unloaded.append(slot))
    with pytest.raises(RuntimeError,match='worker failed'):
        q.synthesize(text='hola',reference_text=None,reference_path=source,job=JobRecord('tts','test'))
    assert 'tts' in unloaded and not list((tmp_path/'refs').iterdir())


def test_worker_protocol_modes_without_real_model(tmp_path):
    import json
    import os
    import sys
    import time
    source=tmp_path/'source'; source.mkdir()
    (source/'torch.py').write_text('''
from contextlib import nullcontext
class cuda:
    @staticmethod
    def is_available(): return True
    @staticmethod
    def is_bf16_supported(): return True
bfloat16='bf16'
def inference_mode(): return nullcontext()
''')
    (source/'soundfile.py').write_text('''
def write(path, data, sr, subtype=None):
    from pathlib import Path
    Path(path).write_text(str(data))
''')
    (source/'qwen_tts.py').write_text('''
class Qwen3TTSModel:
    @classmethod
    def from_pretrained(cls, *args, **kw): return cls()
    def generate_voice_clone(self, **kw):
        assert kw['language']=='Spanish'
        assert kw['x_vector_only_mode'] == (not bool(kw['ref_text']))
        return [[1,2,3]], 24000
''')
    directory=tmp_path/'private'; directory.mkdir(); output=directory/'out.wav'
    worker=Path('studio/engines/qwen_clone_worker.py').resolve()
    proc=subprocess.Popen([sys.executable,str(worker),str(source),str(tmp_path/'model'),str(directory)],stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,text=True)
    def wait_for(path):
        for _ in range(100):
            if path.exists(): return json.loads(path.read_text())
            if proc.poll() is not None: break
            time.sleep(.05)
        pytest.fail('worker did not respond')
    try:
        assert wait_for(directory/'ready.json')['ok']
        for index,transcript in enumerate(['','transcripción']):
            req={'id':str(index),'text':'hola','transcript':transcript,'reference':str(tmp_path/'reference.wav'),'output':str(output)}
            proc.stdin.write(json.dumps(req)+'\n'); proc.stdin.flush()
            assert wait_for(directory/(str(index)+'.json'))['ok'] and output.exists()
    finally:
        proc.stdin.close(); proc.wait(timeout=5)
