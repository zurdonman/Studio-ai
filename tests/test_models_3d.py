"""3D catalog, job/API integration and publication; no GPU or real model inference."""
import io
import json
import struct
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image


def fake_glb():
    data = json.dumps({'asset': {'version':'2.0'}, 'meshes':[{'primitives':[]}]}).encode()
    data += b' ' * (-len(data) % 4)
    return struct.pack('<4sII', b'glTF', 2, 20 + len(data)) + struct.pack('<I4s', len(data), b'JSON') + data


class Job:
    kind = 'model3d'
    def log(self, *a, **k): pass
    def check_cancelled(self): pass


def configure(monkeypatch, tmp_path, engine):
    prefix = 'STUDIO_SF3D' if engine == 'sf3d' else 'STUDIO_TRIPOSR'
    repo, weights = tmp_path / 'repo', tmp_path / 'weights'
    repo.mkdir(exist_ok=True); weights.mkdir(exist_ok=True)
    (repo / 'run.py').write_text('# test stub')
    (weights / 'config.yaml').write_text('test')
    (weights / ('model.safetensors' if engine == 'sf3d' else 'model.ckpt')).write_bytes(b'fake')
    monkeypatch.setenv(prefix + '_REPO', str(repo))
    monkeypatch.setenv(prefix + '_PYTHON', '/usr/bin/python3')
    monkeypatch.setenv(prefix + '_WEIGHTS', str(weights))
    return repo, weights


def test_catalog_and_api(temp_studio_env, monkeypatch, tmp_path):
    import app
    client, root = temp_studio_env
    for name in ('STUDIO_SF3D_REPO','STUDIO_SF3D_PYTHON','STUDIO_SF3D_WEIGHTS'):
        monkeypatch.delenv(name, raising=False)
    models = client.get('/api/3d/catalog').json()['models']
    assert {m['id'] for m in models} == {'sf3d','triposr','hunyuan2','hunyuanmini','hunyuan21','trellis2','idol'}
    assert not models[0]['configured'] and 'UE' in models[2]['license'] and 'PLY' in models[-1]['output']
    (root/'uploads'/'ref.png').write_bytes(b'fake')
    body = {'engine':'sf3d', 'image_rel':'uploads/ref.png', 'license_accepted':True}
    assert client.post('/api/3d/generate', json={**body,'license_accepted':False}).status_code == 400
    jobs = []
    monkeypatch.setattr(app.job_manager, 'submit', lambda **kw: jobs.append(kw) or SimpleNamespace(to_dict=lambda **kw:{'id':'test'}))
    assert client.post('/api/3d/generate', json=body).status_code == 200
    assert jobs[0]['kind'] == 'model3d'
    configure(monkeypatch, tmp_path, 'sf3d')
    assert client.post('/api/3d/generate', json={**body,'image_rel':'../escape'}).status_code in (400,404)
    assert client.post('/api/3d/generate', json={**body,'engine':'nonexistent'}).status_code == 422
    assert client.post('/api/3d/generate', json={**body,'texture_resolution':999}).status_code == 422
    jobs = []
    monkeypatch.setattr(app.job_manager, 'submit', lambda **kw: jobs.append(kw) or SimpleNamespace(to_dict=lambda **kw:{'id':'test'}))
    assert client.post('/api/3d/generate', json=body).status_code == 200
    assert jobs[0]['kind'] == 'model3d'
    folder = root/'outputs'/'3d'; folder.mkdir(parents=True, exist_ok=True)
    (folder/'a.glb').write_bytes(fake_glb())
    assert client.get('/api/assets').json()['models3d'][0]['filename'] == 'a.glb'
    assert client.get('/media/outputs/3d/a.glb').content == fake_glb()
    with zipfile.ZipFile(io.BytesIO(client.get('/api/export/zip').content)) as zf:
        assert zf.read('modelos_3d/a.glb') == fake_glb()


@pytest.mark.parametrize('engine', ['sf3d','triposr'])
def test_cli_publication_and_cleanup(temp_studio_env, monkeypatch, tmp_path, engine):
    from studio.engines import models_3d as m
    configure(monkeypatch, tmp_path, engine)
    image = tmp_path/'image.png'; Image.new('RGBA', (24,24), (10,20,30,255)).save(image)
    calls = []
    def command(args, job, **kwargs):
        calls.append((args, kwargs))
        result = Path(args[args.index('--output-dir')+1])/'0'/'mesh.glb'
        result.parent.mkdir(parents=True)
        result.write_bytes(fake_glb())
    monkeypatch.setattr(m.common, '_command', command)
    result = m.generate.__wrapped__(engine, image, 1024, job=Job())
    assert Path(m.MODELS_OUT_DIR/result['filename']).read_bytes() == fake_glb()
    assert json.loads((m.MODELS_OUT_DIR/result['filename']).with_suffix('.json').read_text())['model_id'] == engine
    assert calls[0][1]['guard_label'] and '--output-dir' in calls[0][0]
    assert ('--texture-resolution' in calls[0][0]) == (engine == 'sf3d')
    assert not list(m.TEMP_DIR.glob('studio_3d_*'))
    def bad(args, job, **kwargs):
        result = Path(args[args.index('--output-dir')+1])/'0'/'mesh.glb'
        result.parent.mkdir(parents=True); result.write_bytes(b'bad')
    monkeypatch.setattr(m.common, '_command', bad)
    before = set(m.MODELS_OUT_DIR.iterdir())
    with pytest.raises(RuntimeError, match='3D_OUTPUT_INVALID'): m.generate.__wrapped__(engine, image, 1024, job=Job())
    assert set(m.MODELS_OUT_DIR.iterdir()) == before
    assert not list(m.TEMP_DIR.glob('studio_3d_*'))


def test_cancel_before_publication_and_bad_headers(temp_studio_env, monkeypatch, tmp_path):
    from studio.engines import models_3d as m
    configure(monkeypatch, tmp_path, 'sf3d')
    image = tmp_path/'image.png'; Image.new('RGB', (8,8)).save(image)
    def command(args, job, **kwargs):
        target = Path(args[args.index('--output-dir')+1])/'0'/'mesh.glb'
        target.parent.mkdir(parents=True); target.write_bytes(fake_glb())
    monkeypatch.setattr(m.common, '_command', command)
    class Cancel(Job):
        def check_cancelled(self): raise RuntimeError('cancelled')
    before = set(m.MODELS_OUT_DIR.iterdir()) if m.MODELS_OUT_DIR.exists() else set()
    with pytest.raises(RuntimeError, match='cancelled'):
        m.generate.__wrapped__('sf3d', image, 512, job=Cancel())
    assert (set(m.MODELS_OUT_DIR.iterdir()) if m.MODELS_OUT_DIR.exists() else set()) == before
    assert not list(m.TEMP_DIR.glob('studio_3d_*'))
    invalid = tmp_path/'invalid.glb'; invalid.write_bytes(fake_glb()[:-1])
    with pytest.raises(RuntimeError, match='3D_OUTPUT_INVALID'): m.validate_glb(invalid)


def test_subprocess_contract_without_gpu(temp_studio_env, monkeypatch, tmp_path):
    import sys
    import threading
    from studio.engines import models_3d as m
    repo, _ = configure(monkeypatch, tmp_path, 'sf3d')
    monkeypatch.setenv('STUDIO_SF3D_PYTHON', sys.executable)
    # This is a fake CLI, not the upstream model or inference.
    (repo/'run.py').write_text('''import pathlib,sys,struct,json
args=sys.argv
assert "--pretrained-model" in args and "--texture-resolution" in args
out=pathlib.Path(args[args.index("--output-dir")+1])/"0"/"mesh.glb"
out.parent.mkdir(parents=True)
data=json.dumps({"asset":{"version":"2.0"},"meshes":[{"primitives":[]}]}).encode()
data+=b" "*(-len(data)%4)
out.write_bytes(struct.pack("<4sII",b"glTF",2,20+len(data))+struct.pack("<I4s",len(data),b"JSON")+data)
''')
    class Guard:
        def __init__(self, *a, **k): self.done=threading.Event()
        def raise_if_stopped(self): pass
        def close(self): pass
    class ProcessJob(Job):
        def heartbeat(self, **kw): pass
    monkeypatch.setattr(m.common, '_ComfyMemoryGuard', Guard)
    image = tmp_path/'input.png'; Image.new('RGB',(10,10)).save(image)
    result = m.generate.__wrapped__('sf3d', image, 1024, job=ProcessJob())
    assert (m.MODELS_OUT_DIR/result['filename']).read_bytes() == fake_glb()
