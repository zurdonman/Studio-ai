"""ONNX inference contract tests use synthetic sessions, never real model weights or GPU."""
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
import numpy as np
import pytest
from PIL import Image


@pytest.mark.parametrize('engine', ['sf3d', 'hunyuanmini'])
def test_onnx_graph_contract_and_surface(temp_studio_env, monkeypatch, tmp_path, engine):
    pytest.importorskip('trimesh')
    pytest.importorskip('skimage')
    from studio.engines import scene_onnx_worker as worker
    from studio.engines.models_3d import validate_glb
    calls=[]
    class Session:
        def __init__(self, path, providers):
            self.name=Path(path).name
            assert providers[0]=='CUDAExecutionProvider'
        def get_providers(self): return ['CUDAExecutionProvider', 'CPUExecutionProvider']
        def get_outputs(self): return [SimpleNamespace(name='density'),SimpleNamespace(name='vertex_offset')]
        def run(self, _, feeds):
            calls.append((self.name,tuple(feeds)))
            if self.name.startswith('image_tokenizer'): return [np.zeros((1,1,1),dtype=np.float32)]
            if self.name.startswith('backbone'): return [np.zeros((1,3,40,2,2),dtype=np.float32)]
            if self.name.startswith('decoder'):
                p=feeds['positions']; return [10.4-(p*p).sum(-1), np.zeros_like(p)]
            if self.name.startswith('dinov2'): return [np.zeros((1,1,1536),dtype=np.float16)]
            if self.name.startswith('dit'): return [np.zeros_like(feeds['x'])]
            if self.name.startswith('vae_kv'): return [np.zeros((1,8,1,64),dtype=np.float16)]*2
            p=feeds['points']; return [.4-(p*p).sum(-1)]
    ort=ModuleType('onnxruntime'); ort.preload_dlls=lambda:None; ort.InferenceSession=Session; ort.get_available_providers=lambda:['CUDAExecutionProvider']
    monkeypatch.setitem(sys.modules,'onnxruntime',ort)
    im=tmp_path/'ref.png'; Image.new('RGBA',(32,24),(220,0,0,255)).save(im)
    worker.run({'engine':engine,'image':str(im),'weights':str(tmp_path),'output':str(tmp_path/'out'),'seed':42})
    validate_glb(tmp_path/'out/result.glb')
    if engine=='hunyuanmini':
        assert len([x for x in calls if x[0]=='dit_turbo.onnx'])==5
        assert all('points' in c[1] for c in calls if c[0]=='vae_query.onnx')
    else: assert any(x[0]=='decoder_single.onnx' for x in calls)


def test_onnx_requires_cuda(monkeypatch):
    from studio.engines import scene_onnx_worker as worker
    ort=ModuleType('onnxruntime'); ort.preload_dlls=lambda:None; ort.get_available_providers=lambda:['CPUExecutionProvider']
    monkeypatch.setitem(sys.modules,'onnxruntime',ort)
    with pytest.raises(RuntimeError,match='ONNX_CUDA_UNAVAILABLE'): worker._sessions(Path('/unused'),['x'])


def test_api_onnx_and_fast_routes(temp_studio_env, monkeypatch):
    import app
    from studio.engines import scene_setup
    client,root=temp_studio_env
    (root/'uploads').mkdir(exist_ok=True)
    (root/'uploads/ref.png').write_bytes(b'fake')
    jobs=[]
    monkeypatch.setattr(app.job_manager,'submit',lambda **kw: jobs.append(kw) or SimpleNamespace(to_dict=lambda **kw:{'id':'test'}))
    for engine in ('sf3d','hunyuanmini'):
        body={'engine':engine,'variant':'onnx','image_rel':'uploads/ref.png','license_accepted':True}
        assert client.post('/api/3d/prepare',json=body).status_code==200
        assert client.post('/api/3d/generate',json=body).status_code==200
    assert client.post('/api/3d/prepare',json={'engine':'worldsplat','variant':'fast','license_accepted':True}).status_code==200
    assert client.post('/api/worlds/generate',json={'engine':'worldsplat','variant':'fast','prompt':'forest','license_accepted':True}).status_code==200
    assert scene_setup.runtime_key('worldsplat','fast')=='diffsplat'
    assert scene_setup.runtime_key('sf3d','onnx')=='sf3donnx'
    assert scene_setup.runtime_key('hunyuanmini','onnx')=='hunyuanonnx'
    assert client.post('/api/3d/generate',json={'engine':'triposr','variant':'onnx','image_rel':'uploads/ref.png','license_accepted':True}).status_code==400


def test_worldsplat_fast_worker_uses_object_pipeline(monkeypatch, tmp_path):
    from studio.engines import scene_worker
    import subprocess
    output=tmp_path/'out'; weights=tmp_path/'weights'; weights.mkdir()
    repo=tmp_path/'repo'; repo.mkdir()
    def fake_run(args, cwd, env, check):
        assert check and cwd==str(repo) and env['PYTHONPATH'].split(':')[0]==str(repo)
        assert '--save_ply' in args and '--prompt_file' in args
        prompt=Path(args[args.index('--prompt_file')+1])
        assert prompt.read_text()=='a forest\n'
        run_dir=Path(args[args.index('--output_dir')+1])
        assert (run_dir/'gsrecon_gobj265k_cnp_even4/checkpoints').is_symlink()
        result=run_dir/'gsdiff_gobj83k_sd15__render/inference'; result.mkdir()
        (result/'000_000.ply').write_text('ply\n')
    monkeypatch.setattr(subprocess,'run',fake_run)
    scene_worker.main({'engine':'worldsplat','variant':'fast','repo':str(repo),'output':str(output),
                       'weights':str(weights),'prompt':'a forest','seed':42})
    assert (output/'result.ply').read_text()=='ply\n'
