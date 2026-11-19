"""Setup orchestration uses fakes: no network downloads or GPU inference."""
import json
from types import SimpleNamespace
import pytest

class Job:
    def log(self, *args, **kwargs): pass
    def heartbeat(self, *args, **kwargs): pass
    def check_cancelled(self): pass

@pytest.fixture
def setup(temp_studio_env, monkeypatch, tmp_path):
    from studio.engines import scene_setup as s
    monkeypatch.setattr(s, 'ROOT', tmp_path/'runtimes')
    monkeypatch.setattr(s, 'WEIGHTS', tmp_path/'weights')
    monkeypatch.setattr(s.shutil, 'disk_usage', lambda p: SimpleNamespace(free=100*1024**3))
    def clone(job, repo, *args, **kw):
        repo.mkdir(parents=True); (repo/'run.py').write_text('')
    def install(job, key, root, repo):
        p=root/'venv/bin/python'; p.parent.mkdir(parents=True); p.write_text('fake'); return p
    def weights(job, engine, key, variant, texture):
        p=s.WEIGHTS/key; p.mkdir(parents=True, exist_ok=True)
        for f in ('config.yaml','model.safetensors' if key == 'sf3d' else 'model.ckpt'): (p/f).write_text('fake')
    monkeypatch.setattr(s, '_clone', clone)
    monkeypatch.setattr(s, '_install_env', install)
    monkeypatch.setattr(s, '_get_weights', weights)
    monkeypatch.setattr(s.common, '_command', lambda *a, **k: None)
    return s

def test_prepare_success_and_isolation(setup, monkeypatch):
    from studio.engines import models_3d
    for name in ('STUDIO_SF3D_REPO','STUDIO_SF3D_PYTHON','STUDIO_SF3D_WEIGHTS'):
        monkeypatch.delenv(name, raising=False)
    before = dict(__import__('os').environ)
    assert setup.prepare('sf3d','base',False,Job())['prepared']
    status=models_3d.runtime_status('sf3d')
    assert status['configured'] and status['managed']
    cfg=setup.managed_config('sf3d')
    assert cfg['repo'] != cfg['weights'] and cfg['python'].startswith(str(setup.ROOT))
    assert dict(__import__('os').environ) == before
    assert not models_3d.runtime_status('triposr')['configured']
    monkeypatch.setenv('STUDIO_SF3D_REPO', '/invalid')
    assert not models_3d.runtime_status('sf3d')['configured']

def test_failure_or_cancel_never_marks_ready(setup, monkeypatch):
    from studio.jobs import JobCancelledError
    def fail(*args, **kw): raise RuntimeError('install error')
    monkeypatch.setattr(setup, '_install_env', fail)
    with pytest.raises(RuntimeError, match='install error'): setup.prepare('sf3d','base',False,Job())
    assert not (setup.ROOT/'sf3d/ready.json').exists()
    def cancel(*args, **kw): raise JobCancelledError()
    monkeypatch.setattr(setup, '_install_env', cancel)
    monkeypatch.setattr(setup, '_clone', lambda *a, **k: None)
    with pytest.raises(JobCancelledError): setup.prepare('sf3d','base',False,Job())
    assert not (setup.ROOT/'sf3d/ready.json').exists()

def test_api_validation(temp_studio_env, monkeypatch):
    import app
    client, _ = temp_studio_env
    body={'engine':'sf3d','license_accepted':True}
    assert client.post('/api/3d/prepare',json={**body,'license_accepted':False}).status_code == 400
    assert client.post('/api/3d/prepare',json={**body,'variant':'q4'}).status_code == 400
    assert client.post('/api/3d/prepare',json={**body,'engine':'../../x'}).status_code == 422
    jobs=[]
    monkeypatch.setattr(app.job_manager,'submit',lambda **kwargs: jobs.append(kwargs) or SimpleNamespace(to_dict=lambda **kw:{'id':'setup'}))
    assert client.post('/api/3d/prepare',json=body).status_code == 200
    assert jobs[0]['kind'] == 'scene_setup'
    assert client.post('/api/3d/prepare',json={**body,'engine':'world2'}).status_code == 200

def test_managed_runtime_used_by_generation(setup, monkeypatch, tmp_path):
    from PIL import Image
    from studio.engines import models_3d
    from tests.test_models_3d import fake_glb
    for suffix in ('_REPO','_PYTHON','_WEIGHTS'):
        monkeypatch.delenv('STUDIO_SF3D'+suffix, raising=False)
    setup.prepare('sf3d','base',False,Job())
    py=__import__('pathlib').Path(setup.managed_config('sf3d')['python'])
    py.unlink(); py.symlink_to('/usr/bin/python3')
    image=tmp_path/'input.png'; Image.new('RGB',(8,8)).save(image)
    seen=[]
    def command(args, job, **kwargs):
        seen.append((args,kwargs))
        out=__import__('pathlib').Path(args[args.index('--output-dir')+1])/'0/mesh.glb'
        out.parent.mkdir(parents=True); out.write_bytes(fake_glb())
    monkeypatch.setattr(models_3d.common, '_command', command)
    models_3d.generate.__wrapped__('sf3d',image,1024,job=Job())
    assert seen[0][0][0] == setup.managed_config('sf3d')['python']
    assert seen[0][1]['cwd'] == setup.managed_config('sf3d')['repo']
    assert seen[0][0][seen[0][0].index('--pretrained-model')+1] == setup.managed_config('sf3d')['weights']

@pytest.mark.parametrize('variant', ['q4', 'q8'])
def test_gguf_variant_prepare_and_dispatch(setup, monkeypatch, tmp_path, variant):
    from studio.engines import scene_jobs
    from tests.test_models_3d import fake_glb
    from pathlib import Path
    for k in ('STUDIO_TRELLIS2_REPO', 'STUDIO_TRELLIS2_PYTHON', 'STUDIO_TRELLIS_GGUF_EXE', 'STUDIO_TRELLIS_GGUF_Q4_WEIGHTS', 'STUDIO_TRELLIS_GGUF_Q8_WEIGHTS'):
        monkeypatch.delenv(k, raising=False)
    calls=[]
    def clone(job, repo, *args, **kwargs): repo.mkdir(parents=True); calls.append(kwargs)
    def command(args, job, **kwargs):
        calls.append(args)
        if args[:2] == ['cmake','--build']:
            exe=Path(args[2])/'trellis-cli'; exe.parent.mkdir(parents=True); exe.write_text('stub')
    def weights(job, engine, key, v, texture):
        path=setup.WEIGHTS/key/v; path.mkdir(parents=True,exist_ok=True)
        for f in setup.GGUF_FILES: (path/f).write_bytes(b'gguf')
    monkeypatch.setattr(setup, '_clone', clone)
    monkeypatch.setattr(setup, '_get_weights', weights)
    monkeypatch.setattr(setup.common, '_command', command)
    assert setup.prepare('trellis2',variant,False,Job())['prepared']
    assert calls[0]['recursive']
    assert any(isinstance(x,list) and '-DGGML_CUDA=ON' in x for x in calls)
    cfg=setup.managed_config('trellis2',variant)
    assert scene_jobs.runtime_status('trellis2',variant)['managed']
    other='q8' if variant=='q4' else 'q4'
    assert not scene_jobs.runtime_status('trellis2',other)['configured']
    monkeypatch.setenv('STUDIO_TRELLIS_GGUF_EXE','/invalid')
    assert not scene_jobs.runtime_status('trellis2',variant)['configured']
    monkeypatch.delenv('STUDIO_TRELLIS_GGUF_EXE')
    from PIL import Image
    image=tmp_path/'image.png'; Image.new('RGB',(8,8)).save(image)
    def generate_command(args, job, **kwargs):
        calls.append(args); out=Path(args[2]); out.write_bytes(fake_glb())
    monkeypatch.setattr(setup.common, '_command', generate_command)
    scene_jobs.generate.__wrapped__('trellis2',[image],'',variant,False,1024,42,12,True,False,'real',False,job=Job())
    assert calls[-1][0] == cfg['exe'] and calls[-1][calls[-1].index('--models')+1] == cfg['weights']

@pytest.mark.parametrize('engine,variant', [('hunyuan2','fast'),('hunyuanmini','turbo')])
def test_hunyuan_prepare_variants_and_paint(setup, monkeypatch, engine, variant):
    from studio.engines import scene_jobs
    prefix=scene_jobs.SPECS[engine][0]
    for suffix in ('_REPO','_PYTHON','_WEIGHTS','_PAINT_WEIGHTS'): monkeypatch.delenv(prefix+suffix, raising=False)
    def clone(job, repo, *args, **kwargs):
        p=repo/'hy3dgen/shapegen/pipelines.py'; p.parent.mkdir(parents=True); p.write_text('')
    def weights(job, engine, key, v, texture):
        base=setup.WEIGHTS/key/setup._shape_folder(key,v); base.mkdir(parents=True,exist_ok=True)
        for f in ('config.yaml','model.fp16.safetensors'): (base/f).write_bytes(b'x')
        paint=setup.WEIGHTS/'hunyuan2-paint'
        for f in ('hunyuan3d-delight-v2-0/model_index.json','hunyuan3d-paint-v2-0-turbo/unet/diffusion_pytorch_model.bin'):
            p=paint/f; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(b'x')
    monkeypatch.setattr(setup, '_clone', clone); monkeypatch.setattr(setup, '_get_weights', weights)
    assert setup.prepare(engine,variant,True,Job())['prepared']
    assert scene_jobs.runtime_status(engine,variant)['configured']
    assert not scene_jobs.runtime_status(engine,'base')['configured']
    assert setup.managed_config(engine,variant)['paint_weights'].endswith('hunyuan2-paint')

def test_new_variants_api(temp_studio_env, monkeypatch):
    import app
    client,_=temp_studio_env
    jobs=[]
    monkeypatch.setattr(app.job_manager,'submit',lambda **kw: jobs.append(kw) or SimpleNamespace(to_dict=lambda **kw:{'id':'setup'}))
    for engine,variant in [('trellis2','q4'),('trellis2','q8'),('hunyuanmini','turbo'),('mirror2','base')]:
        assert client.post('/api/3d/prepare',json={'engine':engine,'variant':variant,'license_accepted':True}).status_code==200
    assert len(jobs)==4
    assert client.post('/api/3d/prepare',json={'engine':'trellis2','variant':'base','license_accepted':True}).status_code==200

def test_generate_prepares_first_and_respects_override(temp_studio_env, monkeypatch):
    import app
    from studio.engines import scene_setup, models_3d, scene_jobs
    client,root=temp_studio_env
    (root/'uploads').mkdir(parents=True,exist_ok=True)
    (root/'uploads'/'setup_input.png').write_bytes(b'x')
    calls=[]; jobs=[]
    monkeypatch.setattr(scene_setup,'prepare', lambda engine,variant,texture,job: calls.append(('prepare',engine,variant)))
    monkeypatch.setattr(models_3d,'generate', lambda engine,path,resolution,job: calls.append(('generate',engine)) or {'ok':True})
    monkeypatch.setattr(scene_jobs,'generate', lambda engine,*args,**kw: calls.append(('world',engine)) or {'ok':True})
    monkeypatch.setattr(app.job_manager,'submit',lambda **kw: jobs.append(kw) or SimpleNamespace(to_dict=lambda **kw:{'id':'setup'}))
    for suffix in ('_REPO','_PYTHON','_WEIGHTS'): monkeypatch.delenv('STUDIO_SF3D'+suffix,raising=False)
    assert client.post('/api/3d/generate', json={'engine':'sf3d','image_rel':'uploads/setup_input.png','license_accepted':True}).status_code == 200
    assert jobs[-1]['fn'](Job()) == {'ok':True}
    assert calls == [('prepare','sf3d','base'),('generate','sf3d')]
    monkeypatch.setenv('STUDIO_SF3D_REPO','/nonexistent')
    assert client.post('/api/3d/generate', json={'engine':'sf3d','image_rel':'uploads/setup_input.png','license_accepted':True}).status_code == 400
    for suffix in ('_REPO','_PYTHON'): monkeypatch.delenv('STUDIO_MIRROR2'+suffix,raising=False)
    calls.clear()
    assert client.post('/api/worlds/generate',json={'engine':'mirror2','images_rel':['uploads/setup_input.png'],'license_accepted':True}).status_code==200
    assert jobs[-1]['fn'](Job()) == {'ok':True}
    assert calls == [('prepare','mirror2','base'),('world','mirror2')]

@pytest.mark.parametrize('engine,source,files', [
    ('mirror','infer.py',('config.json','model.safetensors')),
    ('mirror2','hyworld2/worldrecon/pipeline.py',('HY-WorldMirror-2.0/config.json','HY-WorldMirror-2.0/model.safetensors')),
])
def test_mirror_preparation_uses_pinned_local_weights(setup, monkeypatch, tmp_path, engine, source, files):
    from PIL import Image
    from studio.engines import scene_jobs
    import zipfile
    prefix=scene_jobs.SPECS[engine][0]
    for suffix in ('_REPO','_PYTHON','_WEIGHTS'): monkeypatch.delenv(prefix+suffix,raising=False)
    def clone(job, repo, *args, **kwargs):
        p=repo/source; p.parent.mkdir(parents=True); p.write_text('')
    def weights(job, engine, key, variant, texture):
        for f in files:
            p=setup.WEIGHTS/key/f; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(b'fake')
    monkeypatch.setattr(setup,'_clone',clone); monkeypatch.setattr(setup,'_get_weights',weights)
    setup.prepare(engine,'base',False,Job())
    config=setup.managed_config(engine)
    assert config['weights'] == str(setup.WEIGHTS/engine)
    image=tmp_path/'input.png'; Image.new('RGB',(8,8)).save(image)
    observed=[]
    def command(args, job, **kwargs):
        observed.append(args)
        if engine == 'mirror':
            req=json.loads(__import__('pathlib').Path(args[-1]).read_text())
            assert req['weights'] == config['weights']
            out=__import__('pathlib').Path(req['output'])/'images'
        else:
            assert args[args.index('--pretrained_model_name_or_path')+1] == config['weights']
            out=__import__('pathlib').Path(args[args.index('--strict_output_path')+1])
        out.mkdir(parents=True,exist_ok=True)
        (out/'x.ply').write_text('ply\nformat ascii 1.0\nelement vertex 1\nend_header\n0 0 0\n')
    monkeypatch.setattr(setup.common,'_command',command)
    result=scene_jobs.generate.__wrapped__(engine,[image],'','base',False,1024,1,10,True,False,'real',False,job=Job())
    with zipfile.ZipFile(scene_jobs.WORLDS_OUT_DIR/result['filename']) as z: assert any(n.endswith('.ply') for n in z.namelist())


def test_hf_token_passed_privately_to_snapshot_and_worker(temp_studio_env, monkeypatch, tmp_path):
    from PIL import Image
    from studio.catalog import catalog_manager
    from studio.engines import scene_setup, models_3d
    from tests.test_models_3d import configure, fake_glb
    monkeypatch.setattr(catalog_manager, 'get_hf_token', lambda: 'test-token-123')
    calls=[]
    monkeypatch.setattr(scene_setup.common, '_command', lambda args, job, **kwargs: calls.append((args, kwargs)))
    scene_setup._snapshot(Job(), 'sf3d')
    assert calls[-1][1]['env']['HF_TOKEN'] == 'test-token-123'
    assert 'test-token-123' not in repr(calls[-1][0])
    configure(monkeypatch, tmp_path, 'sf3d')
    image = tmp_path/'ref.png'; Image.new('RGB',(8,8)).save(image)
    def run(args, job, **kwargs):
        assert kwargs['env']['HF_TOKEN'] == 'test-token-123'
        assert 'test-token-123' not in repr(args)
        result=__import__('pathlib').Path(args[args.index('--output-dir')+1])/'0/mesh.glb'
        result.parent.mkdir(parents=True); result.write_bytes(fake_glb())
    monkeypatch.setattr(scene_setup.common, '_command', run)
    models_3d.generate.__wrapped__('sf3d',image,1024,job=Job())


def test_worldsplat_automatic_setup_uses_pinned_mast3r_sources(setup, monkeypatch):
    from studio.engines import scene_jobs
    from pathlib import Path
    for name in ('STUDIO_WORLDSPLAT_REPO', 'STUDIO_WORLDSPLAT_PYTHON'):
        monkeypatch.delenv(name, raising=False)
    calls=[]
    def clone(job, repo, name, rev, **kw):
        calls.append((name, rev, kw)); (repo/'worldsplat').mkdir(parents=True); (repo/'worldsplat/pipeline.py').write_text('')
    def install(job, key, root, repo):
        py=root/'venv/bin/python'; py.parent.mkdir(parents=True); py.write_text('')
        for filename in ('mast3r/mast3r/model.py', 'mast3r/dust3r/dust3r/model.py'):
            f=root/filename; f.parent.mkdir(parents=True, exist_ok=True); f.write_text('')
        return py
    monkeypatch.setattr(setup, '_clone', clone)
    monkeypatch.setattr(setup, '_install_env', install)
    monkeypatch.setattr(setup, '_get_weights', lambda *a: None)
    monkeypatch.setattr(setup.common, '_command', lambda *a, **kw: None)
    assert setup.prepare('worldsplat','base',False,Job())['prepared']
    cfg=setup.managed_config('worldsplat')
    assert cfg['mast3r'] == str(setup.ROOT/'worldsplat/mast3r')
    assert scene_jobs.runtime_status('worldsplat')['configured']
    assert calls[0][:2] == setup.REPOS['worldsplat']
    Path(cfg['mast3r'], 'dust3r/dust3r/model.py').unlink()
    assert not scene_jobs.runtime_status('worldsplat')['configured']


def test_worldgen_setup_and_fixed_dependencies(temp_studio_env, monkeypatch, tmp_path):
    from studio.engines import scene_setup as setup
    calls=[]
    monkeypatch.setattr(setup.shutil,'which',lambda n:'/usr/bin/python3.11')
    monkeypatch.setattr(setup.common,'_command',lambda args, job, **kw: calls.append(args))
    root=tmp_path/'worldgen'; root.mkdir(parents=True)
    repo=root/'source'; repo.mkdir()
    setup._install_env(Job(),'worldgen',root,repo)
    rendered=' '.join(' '.join(x) for x in calls)
    assert 'torch==2.7.1' in rendered and 'xformers==0.0.31.post1' in rendered
    assert 'DA-2.git@d659838585f2bc9967c7e9367af573795271dbbb' in rendered
    assert 'pytorch3d.git@ddfc55127fed1a74514cd24db08a7c2e62e76b08' in rendered
    assert 'worldgen' in setup.AUTOMATED


def test_trellis2_official_preparation_requires_complete_files(setup, monkeypatch):
    from studio.engines import scene_jobs
    for name in ('STUDIO_TRELLIS2_REPO', 'STUDIO_TRELLIS2_PYTHON'):
        monkeypatch.delenv(name, raising=False)
    def clone(job, repo, *args, **kw):
        p=repo/'trellis2/pipelines/trellis2_image_to_3d.py'; p.parent.mkdir(parents=True); p.write_text('')
    def weights(job, engine, key, variant, texture):
        p=setup.WEIGHTS/key; p.mkdir(parents=True,exist_ok=True); (p/'pipeline.json').write_text('{}')
        (p/'ckpts').mkdir()
        for name in setup.TRELLIS_MODELS:
            for ext in ('json','safetensors'):
                (p/'ckpts'/f'{name}.{ext}').write_text('fake')
    monkeypatch.setattr(setup,'_clone',clone)
    monkeypatch.setattr(setup,'_get_weights',weights)
    assert setup.prepare('trellis2','base',False,Job())['prepared']
    cfg=setup.managed_config('trellis2')
    assert scene_jobs.runtime_status('trellis2')['managed']
    from pathlib import Path
    Path(cfg['weights'],'ckpts',setup.TRELLIS_MODELS[-1]+'.json').unlink()
    assert not scene_jobs.runtime_status('trellis2')['configured']
