"""Contract tests with fake runtimes; these do not run upstream models or CUDA."""
import io
import json
import sys
import zipfile
from pathlib import Path
from types import ModuleType, SimpleNamespace
import pytest
from PIL import Image

from tests.test_models_3d import fake_glb

PLY = b'ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n0 0 0\n'

class Job:
    def log(self, *args, **kwargs): pass
    def check_cancelled(self): pass


def config(monkeypatch, tmp_path, engine, variant='base'):
    from studio.engines import scene_jobs as s
    prefix, _, marker, _ = s.SPECS[engine]
    repo = tmp_path/engine; (repo/marker).parent.mkdir(parents=True, exist_ok=True); (repo/marker).write_text('fake')
    monkeypatch.setenv(prefix+'_REPO', str(repo)); monkeypatch.setenv(prefix+'_PYTHON', sys.executable)
    if engine == 'world2':
        monkeypatch.setenv('STUDIO_WORLD2_VLM_HOST','127.0.0.1'); monkeypatch.setenv('STUDIO_WORLD2_VLM_PORT','8000'); monkeypatch.setenv('STUDIO_WORLD2_VLM_MODEL','vlm')
    if variant in ('q4','q8'):
        weights=tmp_path/variant; weights.mkdir(exist_ok=True); (weights/'ss_flow.gguf').write_bytes(b'fake')
        monkeypatch.setenv('STUDIO_TRELLIS_GGUF_EXE',sys.executable); monkeypatch.setenv('STUDIO_TRELLIS_GGUF_'+variant.upper()+'_WEIGHTS',str(weights))
    assert s.runtime_status(engine, variant)['configured']
    return repo


@pytest.mark.parametrize('engine,variant',[(x,'base') for x in ('hunyuan2','hunyuanmini','hunyuan21','trellis2','idol','sharp','worldgen','mirror','mirror2','world1','world2','worldsplat')]+[('trellis2','q4'),('trellis2','q8')])
def test_fake_pipeline_publishes_expected_artifact(temp_studio_env, monkeypatch, tmp_path, engine, variant):
    from studio.engines import scene_jobs as s
    config(monkeypatch,tmp_path,engine,variant)
    image=tmp_path/'ref.png'; Image.new('RGB',(10,10)).save(image)
    recorded=[]
    def command(args,job,**kw):
        recorded.append((args,kw))
        if engine == 'mirror': out=Path(args[args.index('--output_path')+1])/'images'
        elif engine == 'mirror2': out=Path(args[args.index('--strict_output_path')+1])
        elif variant in ('q4','q8'): out=Path(args[2]).parent
        else: out=Path(json.loads(Path(args[2]).read_text())['output'])
        out.mkdir(parents=True,exist_ok=True)
        if engine in ('hunyuan2','hunyuanmini','hunyuan21','trellis2'): (out/'result.glb').write_bytes(fake_glb())
        elif engine in ('mirror','mirror2'): (out/'gaussians.ply').write_bytes(PLY)
        elif engine == 'world1': (out/'mesh_layer0.ply').write_bytes(PLY)
        elif engine == 'world2': (out/'training').mkdir(); (out/'training'/'final.ply').write_bytes(PLY)
        else: (out/'result.ply').write_bytes(PLY)
    monkeypatch.setattr(s.common,'_command',command)
    images = [] if engine == 'worldsplat' else [image]
    result=s.generate.__wrapped__(engine,images,'a forest',variant,True,1024,42,30,True,False,'real',False,job=Job())
    file=(s.MODELS_OUT_DIR if engine in s.OBJECTS else s.WORLDS_OUT_DIR)/result['filename']
    assert file.is_file() and recorded[0][1]['guard_label']
    assert result['model_id']==engine and not list(s.TEMP_DIR.glob('studio_scene_*'))
    if engine in ('mirror','mirror2','world1','world2'):
        with zipfile.ZipFile(file) as z: assert any(x.endswith('.ply') for x in z.namelist())
    else: assert file.read_bytes() in (PLY,fake_glb())


def test_world_api_and_zip(temp_studio_env, monkeypatch, tmp_path):
    import app
    client,root=temp_studio_env
    assert set(x['id'] for x in client.get('/api/worlds/catalog').json()['models']) == {'sharp','worldgen','mirror','mirror2','world1','world2','worldsplat'}
    config(monkeypatch,tmp_path,'mirror2')
    (root/'uploads'/'ref.png').write_bytes(b'fake')
    body={'engine':'mirror2','images_rel':['uploads/ref.png'],'license_accepted':True}
    assert client.post('/api/worlds/generate',json={**body,'license_accepted':False}).status_code==400
    assert client.post('/api/worlds/generate',json={**body,'images_rel':['../escape.png']}).status_code in (400,404)
    assert client.post('/api/worlds/generate',json={**body,'images_rel':[]}).status_code==400
    assert client.post('/api/worlds/generate',json={**body,'images_rel':['uploads/ref.png']*18}).status_code==422
    jobs=[]
    monkeypatch.setattr(app.job_manager,'submit',lambda **kw: jobs.append(kw) or SimpleNamespace(to_dict=lambda **kw:{'id':'job'}))
    assert client.post('/api/worlds/generate',json=body).status_code==200 and jobs[0]['kind']=='world3d'
    assert client.post('/api/3d/generate',json={'engine':'hunyuan2','image_rel':'uploads/ref.png','license_accepted':True,'variant':'q4'}).status_code==400
    folder=root/'outputs'/'worlds'; folder.mkdir(parents=True,exist_ok=True); (folder/'w.ply').write_bytes(PLY)
    assert client.get('/api/assets').json()['worlds'][0]['filename']=='w.ply'
    assert client.get('/media/outputs/worlds/w.ply').content==PLY
    with zipfile.ZipFile(io.BytesIO(client.get('/api/export/zip').content)) as z: assert z.read('mundos_3d/w.ply')==PLY


def test_worker_worldgen_and_world2_stages(monkeypatch,tmp_path):
    from studio.engines import scene_worker as w
    calls=[]
    class WG:
        def __init__(self, **kw): calls.append(kw)
        def generate_world(self, **kw):
            calls.append(kw)
            return SimpleNamespace(save=lambda path:Path(path).write_bytes(PLY))
    fake=ModuleType('worldgen'); fake.WorldGen=WG
    monkeypatch.setitem(sys.modules,'worldgen',fake)
    torch=ModuleType('torch'); torch.manual_seed=lambda n: calls.append(n)
    monkeypatch.setitem(sys.modules,'torch',torch)
    cfg={'engine':'worldgen','repo':str(tmp_path),'output':str(tmp_path/'o'),'image':'','seed':42,'low_vram':True,'prompt':'forest','mesh':False}
    w.main(cfg)
    assert (tmp_path/'o'/'result.ply').read_bytes()==PLY and calls[1]['mode']=='t2s' and calls[1]['low_vram']
    import subprocess
    cmds=[]
    monkeypatch.setattr(subprocess,'run',lambda args,**kw:cmds.append((args,kw)))
    inp=tmp_path/'pano.png'; Image.new('RGB',(20,10)).save(inp)
    cfg={'engine':'world2','repo':str(tmp_path),'output':str(tmp_path/'o2'),'image':str(inp),'gpus':4,'vlm_host':'localhost','vlm_port':'8000','vlm_model':'vlm'}
    w.main(cfg)
    assert len(cmds)==5 and 'traj_generate.py' in cmds[0][0] and '--max_steps' in cmds[-1][0]
    assert cmds[-1][0][cmds[-1][0].index('--max_steps')+1]=='2000'


@pytest.mark.parametrize('engine,variant,subfolder', [('hunyuan2','fast','hunyuan3d-dit-v2-0-fast'),('hunyuanmini','turbo','hunyuan3d-dit-v2-mini-turbo'),('hunyuan21','base','hunyuan3d-dit-v2-1')])
def test_hunyuan_worker_shape_and_paint_contract(monkeypatch, tmp_path, engine, variant, subfolder):
    from studio.engines import scene_worker as w
    calls=[]
    class Mesh:
        def export(self,path): Path(path).write_bytes(fake_glb())
    class Shape:
        @classmethod
        def from_pretrained(cls, weights, **kw): calls.append(('shape',weights,kw)); return cls()
        def __call__(self, **kw): calls.append(('infer',kw)); return [Mesh()]
    class Remover:
        def __call__(self, image): calls.append(('rembg',)); return image.convert('RGBA')
    class Painter2:
        @classmethod
        def from_pretrained(cls, path): calls.append(('paint',path)); return cls()
        def __call__(self, mesh, **kw): return mesh
    class Config:
        def __init__(self,*a): pass
    class Painter21:
        def __init__(self, cfg): calls.append(('paint21',cfg.multiview_pretrained_path))
        def __call__(self, **kw):
            calls.append(('paint_output',kw['output_mesh_path']))
            Path(kw['output_mesh_path']).with_suffix('.glb').write_bytes(fake_glb())
    for name in ('hy3dgen','hy3dshape'):
        pkg=ModuleType(name);pkg.__path__=[]; monkeypatch.setitem(sys.modules,name,pkg)
        mod=ModuleType(name+('.shapegen' if name=='hy3dgen' else '.pipelines')); mod.Hunyuan3DDiTFlowMatchingPipeline=Shape; monkeypatch.setitem(sys.modules,mod.__name__,mod)
        rem=ModuleType(name+'.rembg');rem.BackgroundRemover=Remover;monkeypatch.setitem(sys.modules,rem.__name__,rem)
    tex=ModuleType('hy3dgen.texgen');tex.Hunyuan3DPaintPipeline=Painter2;monkeypatch.setitem(sys.modules,tex.__name__,tex)
    tex21=ModuleType('textureGenPipeline');tex21.Hunyuan3DPaintConfig=Config;tex21.Hunyuan3DPaintPipeline=Painter21;monkeypatch.setitem(sys.modules,tex21.__name__,tex21)
    torch=ModuleType('torch');torch.cuda=SimpleNamespace(empty_cache=lambda:None);torch.Generator=lambda device:SimpleNamespace(manual_seed=lambda seed:seed);monkeypatch.setitem(sys.modules,'torch',torch)
    image=tmp_path/'input.png'; Image.new('RGB',(8,8)).save(image)
    cfg={'engine':engine,'repo':str(tmp_path),'output':str(tmp_path/'out'),'image':str(image),'variant':variant,'weights':'shape','texture':True,'paint_weights':'paint','steps':5,'seed':42}
    w.main(cfg)
    assert calls[1][0] == 'shape' and calls[1][2]['subfolder'] == subfolder
    assert (tmp_path/'out'/'result.glb').read_bytes()==fake_glb()
    if engine=='hunyuan21': assert any(c[0]=='paint_output' and c[1].endswith('.obj') for c in calls if isinstance(c,tuple))
