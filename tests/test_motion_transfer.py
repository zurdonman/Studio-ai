"""Motion graphs, API and lifecycles; all models and Comfy results are fake."""
import importlib.util
import json
import subprocess
import sys
import threading
from types import ModuleType, SimpleNamespace
import pytest


def test_manifests_and_graph_wiring(temp_studio_env):
    from studio.engines.motion_manifest import files_for, WAN, SCAIL
    from studio.engines.motion_transfer import make_graph
    for engine,table in [('wan',WAN),('scail',SCAIL)]:
        for q in table:
            files=files_for(engine,q)
            assert files[0][3].endswith('.gguf') and all(len(x[-1]) == 64 and len(x[2]) == 40 and x[4]>0 for x in files)
    for engine,quant in [('wan','q4'),('wan','int8'),('scail','q4'),('scail','fp8')]:
        names={x[0]:'model.gguf' if x[0]=='dit' and quant=='q4' else 'model.safetensors' for x in files_for(engine,quant)}
        g=make_graph(engine=engine,quant=quant,image_name='ref.png',video_name='drive.mp4',names=names,prompt='dance',negative='',width=512,height=320,frames=25,seed=1,steps=10,cfg=1)
        assert g['19']['inputs']['filename_prefix']=='studio_motion'
        assert g['14']['inputs']['reference_image']==['5',0] and g['14']['inputs']['pose_video']==['7',0]
        if engine=='wan':
            assert g['1']['class_type']==('StudioWanAnimate2GGUF' if quant=='q4' else 'UNETLoader')
            assert g['16']['inputs']['trim_amount']==['14',3]
        else:
            assert g['24']['inputs']['replacement_mode'] is False
            assert g['14']['inputs']['pose_video_mask']==['24',0] and g['14']['inputs']['reference_image_mask']==['24',1]
    with pytest.raises(ValueError): files_for('wan','fp8')
    with pytest.raises(ValueError): files_for('other','q4')


def test_ltx_control_graph_conditions_and_crops(temp_studio_env):
    from studio.engines.ltx_gguf_workflow import make_graph
    from studio.engines.ltx_motion import add_union_graph
    g=make_graph(prompt='dance',negative_prompt='',width=512,height=320,frames=25,fps=24,seed=1,image_name='ref.png',model='dit',text_encoder='text',connectors='connector',video_vae='video',audio_vae='audio')
    add_union_graph(g,'guide.mp4','union.safetensors')
    assert g['34']['inputs']['latent']==['22',0]
    assert g['34']['inputs']['iclora_parameters']==['31',0]
    assert g['11']['inputs']['positive']==['34',0] and g['11']['inputs']['model']==['30',0]
    assert g['35']['inputs']['latent']==['16',0] and g['17']['inputs']['samples']==['35',2]
    assert g['19']['inputs']['audio']==['18',0]
    with pytest.raises(ValueError): add_union_graph(g,'../x','a')


def test_adapter_injects_metadata_and_fails_closed(temp_studio_env, monkeypatch):
    # Import the private node with a tiny fake Comfy registry, not Comfy itself.
    fake=ModuleType('nodes'); seen={}
    namespace={'json':json,'SimpleNamespace':SimpleNamespace,'seen':seen}
    exec("""def gguf_sd_loader(path): return {}, {'metadata': {'config': '{\"transformer\":{\"existing\":7}}'}}
class Loader:
    @classmethod
    def INPUT_TYPES(cls): return {'required': {'unet_name': (['test'],)}}
    def load_unet(self, unet_name):
        _,extra=gguf_sd_loader(unet_name)
        seen['config']=json.loads(extra['metadata']['config'])
        return (SimpleNamespace(model=type('WAN_Animate2', (), {})()),)
""",namespace)
    Loader=namespace['Loader']; reader=namespace['gguf_sd_loader']
    fake.NODE_CLASS_MAPPINGS={'UnetLoaderGGUF':Loader}
    monkeypatch.setitem(sys.modules,'nodes',fake)
    path='studio/engines/comfy_motion_node/__init__.py'
    spec=importlib.util.spec_from_file_location('test_motion_node',path); module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    out=module.StudioWanAnimate2GGUF().load('test')
    assert type(out[0].model).__name__ == 'WAN_Animate2'
    assert seen['config']['transformer']=={'existing':7,'model_type':'animate2'}
    assert Loader.load_unet.__globals__['gguf_sd_loader'] is reader
    # Exercise missing architecture detection without installing GPU weights.
    def wrong(self, name):
        return (SimpleNamespace(model=object()),)
    Loader.load_unet=wrong
    # Adapter expects the original loader's gguf_sd_loader global.
    wrong.__globals__['gguf_sd_loader']=reader
    with pytest.raises(RuntimeError,match='WRONG_ARCH'): module.StudioWanAnimate2GGUF().load('test')


def test_api_rejects_wrong_inputs_and_enqueues(temp_studio_env, monkeypatch):
    import app
    client,root=temp_studio_env
    (root/'uploads'/'motion.png').write_bytes(b'fake')
    (root/'uploads'/'motion.mp4').write_bytes(b'fake')
    data={'engine':'scail','quant':'q4','image_rel':'uploads/motion.png','video_rel':'uploads/motion.mp4','prompt':'dance'}
    assert client.post('/api/community/motion',json=data).status_code==400
    data['sam_license_accepted']=True
    assert client.post('/api/community/motion',json={**data,'quant':'bf16'}).status_code==400
    assert client.post('/api/community/motion',json={**data,'width':513}).status_code==400
    assert client.post('/api/community/motion',json={**data,'image_rel':'../escape.png'}).status_code in (400,404)
    jobs=[]
    monkeypatch.setattr(app.job_manager,'submit',lambda **kwargs: jobs.append(kwargs) or SimpleNamespace(to_dict=lambda **kw:{'id':'job'}))
    res=client.post('/api/community/motion',json=data)
    assert res.status_code==200 and jobs[0]['kind']=='video'
    assert client.post('/api/community/motion',json={**data,'engine':'ltx','quant':'q5'}).status_code==400


class FakeGuard:
    def __init__(self,*a,**k): self.done=threading.Event()
    def raise_if_stopped(self): pass
    def close(self): pass


def make_clip(path, audio=False):
    from studio.engines.ffmpeg_utils import get_ffmpeg_bin
    args=[get_ffmpeg_bin(),'-v','error','-y','-f','lavfi','-i','color=blue:s=256x256:r=24:d=2']
    if audio: args+=['-f','lavfi','-i','sine=frequency=440:duration=2','-c:a','aac','-shortest']
    subprocess.run(args+['-c:v','libx264','-pix_fmt','yuv420p',str(path)],check=True)


@pytest.mark.parametrize('engine',['wan','scail','ltx'])
def test_synthetic_pipeline_closes_worker_and_publishes(temp_studio_env,monkeypatch,tmp_path,engine):
    from PIL import Image
    from studio.engines import motion_transfer as m, ltx_motion as l
    from studio.jobs import JobRecord
    from studio.engines.ffmpeg_utils import probe_media_info
    _,root=temp_studio_env
    img=root/'uploads'/'motion_ref.png';Image.new('RGB',(256,256)).save(img)
    video=root/'uploads'/'motion_guide.mp4';make_clip(video)
    inp,out=tmp_path/'input',tmp_path/'output';inp.mkdir();out.mkdir()
    closed=[]
    session=SimpleNamespace(root=tmp_path,input_dir=inp,output_dir=out,unload=lambda:closed.append(True),guard=SimpleNamespace(raise_if_stopped=lambda:None))
    monkeypatch.setattr(m.common,'_ComfyMemoryGuard',FakeGuard)
    monkeypatch.setattr(m.common,'query_gpu_info',lambda:{'has_gpu':True})
    monkeypatch.setattr(m.memory_manager,'get_slot',lambda *_:None)
    monkeypatch.setattr(m.common,'_comfy_start_admission',lambda *a,**k:None)
    monkeypatch.setattr(m.common,'_ResidentComfy',lambda *a:session)
    names={'dit':'dit','umt5':'text','vision':'vision','vae':'vae','sam':'sam'}
    monkeypatch.setattr(m,'_prepare',lambda *a:names)
    monkeypatch.setattr(l,'_prepare_lora',lambda *a:'union.safetensors')
    monkeypatch.setattr(m.common,'_prepare_gguf',lambda *a,**k:{'dit':'dit','text_encoder':'text','connectors':'connector','video_vae':'video','audio_vae':'audio'})
    def fake_infer(sess,graph,job,**kwargs):
        assert (inp/('guide.mp4' if engine=='ltx' else 'drive.mp4')).exists()
        clip=out/'generated.mp4';make_clip(clip,audio=engine=='ltx');return clip
    monkeypatch.setattr(m,'_infer',fake_infer);monkeypatch.setattr(l,'_infer',fake_infer)
    common=dict(image_path=img,video_path=video,prompt='dance',negative='',width=256,height=256,frames=25,seed=1,strength=1,job=JobRecord('video','test'))
    result=l.generate(**common) if engine=='ltx' else m.generate(engine=engine,quant='q4',steps=10,cfg=1,person='human',**common)
    final=probe_media_info(root/result['rel_path'])
    assert final['has_video'] and final['has_audio']==(engine=='ltx') and closed==[True]
    assert json.loads((root/'outputs/videos'/f"{result['id']}.json").read_text())['model_id']==result['model_id']


@pytest.mark.parametrize('cancel', [False, True])
def test_motion_failure_closes_worker(temp_studio_env, monkeypatch,tmp_path,cancel):
    from PIL import Image
    from studio.engines import motion_transfer as m
    from studio.jobs import JobRecord
    _,root=temp_studio_env
    img=root/'uploads'/'motion_ref.png';Image.new('RGB',(256,256)).save(img)
    video=root/'uploads'/'motion_guide.mp4';make_clip(video)
    inp,out=tmp_path/'input',tmp_path/'output';inp.mkdir();out.mkdir()
    closed=[];session=SimpleNamespace(root=tmp_path,input_dir=inp,output_dir=out,unload=lambda:closed.append(True),guard=SimpleNamespace(raise_if_stopped=lambda:None))
    monkeypatch.setattr(m.common,'_ComfyMemoryGuard',FakeGuard)
    monkeypatch.setattr(m.common,'query_gpu_info',lambda:{'has_gpu':True})
    monkeypatch.setattr(m.memory_manager,'get_slot',lambda *_:None)
    monkeypatch.setattr(m.common,'_comfy_start_admission',lambda *a,**k:None)
    monkeypatch.setattr(m,'_prepare',lambda *a:{'dit':'a','umt5':'b','vision':'c','vae':'d'})
    monkeypatch.setattr(m.common,'_ResidentComfy',lambda *a:session)
    from studio.jobs import JobCancelledError
    def fail(*a,**k):
        if cancel: raise JobCancelledError('cancelled')
        raise RuntimeError('broken')
    monkeypatch.setattr(m,'_infer',fail)
    with pytest.raises(JobCancelledError if cancel else RuntimeError):
        m.generate(engine='wan',quant='q4',image_path=img,video_path=video,prompt='dance',negative='',width=256,height=256,frames=25,seed=1,steps=10,cfg=1,person='human',strength=1,job=JobRecord('video','test'))
    assert closed==[True]


def test_infer_reports_node_and_safe_traceback(temp_studio_env, monkeypatch, tmp_path):
    from studio.engines import motion_transfer as m
    from studio.jobs import JobRecord
    class Reply:
        ok = True
        def __init__(self, data): self.data = data
        def json(self): return self.data
        def raise_for_status(self): pass
    event = ['execution_error', {'node_id':'1', 'node_type':'Untrusted',
             'exception_type':'builtins.TypeError', 'exception_message':"'int' object is not subscriptable",
             'traceback':['  File "/private/home/ComfyUI/loader.py", line 23, in get_orig_shape\n    user_secret_input\n'],
             'current_inputs':{'prompt':'do not log this'}}]
    def get(url, **kwargs):
        if url.endswith('/object_info'): return Reply({'StudioWanAnimate2GGUF':{}})
        return Reply({'id': {'status': {'status_str':'error', 'messages':[event]}}})
    monkeypatch.setattr(m.common.requests, 'get', get)
    monkeypatch.setattr(m.common.requests, 'post', lambda *a,**k: Reply({'prompt_id':'id'}))
    session=SimpleNamespace(base='http://test', output_dir=tmp_path, alive=lambda:True,
                            guard=SimpleNamespace(raise_if_stopped=lambda:None))
    job=JobRecord('video','test')
    with pytest.raises(RuntimeError,match=r'nodo 1 \(StudioWanAnimate2GGUF\), builtins.TypeError'):
        m._infer(session, {'1':{'class_type':'StudioWanAnimate2GGUF','inputs':{}}}, job)
    logs=str(job.logs)
    assert 'loader.py:23 en get_orig_shape' in logs
    assert '/private/home' not in logs and 'user_secret_input' not in logs and 'do not log this' not in logs


def test_comfy_error_handles_missing_or_malformed_details(temp_studio_env):
    from studio.engines.motion_transfer import _report_comfy_error
    from studio.jobs import JobRecord
    job=JobRecord('video','test')
    text=_report_comfy_error([None, [], ['execution_error', {'node_id':'../x', 'traceback':42}]], {}, job)
    assert 'nodo ? (?)' in text and 'Error desconocido' in text
