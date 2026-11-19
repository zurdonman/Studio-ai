"""PrismAudio contracts: mock GPU inference and verify pinned LFS pointers."""
import hashlib
import json
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace


def test_gated_lfs_pointer_verification(tmp_path):
    from studio.engines.prismaudio_download import verify_lfs_pointer
    data = b'private weights'; path = tmp_path/'weights'; path.write_bytes(data)
    sha = hashlib.sha256(data).hexdigest()
    pointer = f'version https://git-lfs.github.com/spec/v1\noid sha256:{sha}\nsize {len(data)}\n'.encode()
    blob = hashlib.sha1(f'blob {len(pointer)}\0'.encode()+pointer).hexdigest()
    assert verify_lfs_pointer(path, len(data), blob)
    assert not verify_lfs_pointer(path, len(data)+1, blob)
    path.write_bytes(b'x'*len(data))
    assert not verify_lfs_pointer(path, len(data), blob)


def test_prism_requires_specific_license(temp_studio_env, monkeypatch):
    import app
    from studio.engines.ffmpeg_utils import get_ffmpeg_bin
    client, root = temp_studio_env
    path = root/'uploads/prism.mp4'
    subprocess.run([get_ffmpeg_bin(), '-v','error','-y','-f','lavfi','-i','color=s=128x96:r=25:d=2', '-c:v','libx264', str(path)], check=True)
    monkeypatch.setattr(app.job_manager, 'submit', lambda **kw:SimpleNamespace(to_dict=lambda **kw:{'id':'j'}))
    body = dict(video_rel='uploads/prism.mp4', engine='prismaudio', install_accepted=True)
    assert client.post('/api/community/video-audio', json={**body, 'research_license_accepted':True}).status_code == 400
    assert client.post('/api/community/video-audio', json={**body, 'model_license_accepted':True}).status_code == 400
    assert client.post('/api/community/video-audio', json={**body, 'prism_license_accepted':True}).status_code == 200


def test_prism_synthetic_clip_full_duration(temp_studio_env, monkeypatch):
    from studio.engines import mmaudio as m, prismaudio as p
    from studio.engines.ffmpeg_utils import get_ffmpeg_bin, probe_media_info
    from studio.jobs import JobRecord
    _, root = temp_studio_env
    path = root/'uploads/prism.mp4'
    subprocess.run([get_ffmpeg_bin(), '-v','error','-y','-f','lavfi','-i','color=s=128x96:r=25:d=9', '-c:v','libx264', str(path)], check=True)
    original = path.read_bytes()
    monkeypatch.setattr(m.common, 'query_gpu_info', lambda: {'has_gpu':True, 'bf16_supported':True})
    monkeypatch.setattr(m.common, '_switch_to_idle', lambda *args:False)
    budgets=[]
    monkeypatch.setattr(m.memory_manager, 'ensure_capacity', lambda **kw:budgets.append(kw))
    monkeypatch.setattr(p, 'prepare', lambda job:'python')
    class Guard:
        def __init__(self, *a, **kw): self.done=threading.Event()
        def raise_if_stopped(self): pass
        def close(self): pass
    monkeypatch.setattr(m.common, '_ComfyMemoryGuard', Guard)
    real = m.common._command; calls=[]
    def command(args, job, **kw):
        if 'prismaudio_worker.py' not in args[1]: return real(args, job, **kw)
        mode=args[2]; request=Path(args[-1]); data=json.loads(request.read_text()); calls.append((mode, data['seed']))
        assert abs(probe_media_info(request.parent/'videos/demo.mp4')['duration_seconds']-8) < .2
        if mode == 'extract':
            (request.parent/'features').mkdir(); (request.parent/'features/demo.npz').write_bytes(b'fake')
        else:
            subprocess.run([get_ffmpeg_bin(),'-v','error','-y','-f','lavfi','-i','sine=frequency=440:duration=8', data['output']],check=True)
    monkeypatch.setattr(m.common, '_command', command)
    result=m.generate(path,engine='prismaudio',prompt='wind',negative='',seed=10,steps=25,replace_audio=False,job=JobRecord('video','test'))
    assert calls==[('extract',10),('infer',10),('extract',11),('infer',11)]
    assert result['segments']==2 and result['steps']==24 and result['model_id']==p.MODEL_ID
    assert budgets[0]['vram_gb']==32
    assert abs(probe_media_info(root/result['rel_path'])['duration_seconds']-9) < .3
    assert path.read_bytes()==original
