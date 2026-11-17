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
    exec("""class GGMLTensor:
    def new_empty(self, size, *args, **kwargs): return size, args, kwargs
def gguf_sd_loader(path): return {}, {'metadata': {'config': '{\"transformer\":{\"existing\":7}}'}}
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
        guide = 'guide.mp4' if engine == 'ltx' else graph['6']['inputs']['file']
        assert (inp/guide).exists()
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


@pytest.mark.parametrize('frames', [1, 2, 24, 80, 81, 82, 161, 192, 720])
def test_wan_duration_plan_covers_frames_exactly(frames):
    from studio.engines.motion_transfer import _wan_plan, _duration_frames
    plan=list(_wan_plan(frames))
    assert sum(x['take'] for x in plan)==frames
    done=0
    for i, part in enumerate(plan):
        assert part['start']==done-(i>0)
        assert part['overlap']==(i>0)
        assert 5 <= part['window'] <= 81 and (part['window']-1)%4==0
        assert part['window']>=part['take']+part['overlap']
        done+=part['take']
    assert _duration_frames(frames/24)==frames


@pytest.mark.parametrize('total', [192, 720])
def test_wan_long_segments_real_ffmpeg_fake_model(temp_studio_env, monkeypatch, tmp_path, total):
    from studio.engines import motion_transfer as m
    from studio.engines.ffmpeg_utils import get_ffmpeg_bin, probe_media_info
    from studio.jobs import JobRecord
    from PIL import Image
    inp=tmp_path/'input'; out=tmp_path/'output'; inp.mkdir();out.mkdir()
    source=tmp_path/'reference_video.mp4'
    subprocess.run([get_ffmpeg_bin(),'-hide_banner','-loglevel','error','-y','-f','lavfi',
                    '-i',f'color=blue:s=256x256:r=24:d={total/24}','-c:v','libx264','-pix_fmt','yuv420p',str(source)],check=True)
    image=tmp_path/'ref.png'; Image.new('RGB',(256,256)).save(image)
    seen=[]
    def fake_infer(sess, graph, job, prefix):
        i=len(seen); seen.append(graph)
        part=graph['14']['inputs']
        if i:
            assert part['continue_motion']==['24',0]
            assert (inp/graph['23']['inputs']['file']).is_file()
            assert part['video_frame_offset']==0
        else: assert 'continue_motion' not in part
        frames=part['length']; clip=out/f'{prefix}_00001_.mp4'
        subprocess.run([get_ffmpeg_bin(),'-hide_banner','-loglevel','error','-y','-f','lavfi','-i',
                        f'color={"red" if i%2==0 else "green"}:s=256x256:r=24', '-frames:v',str(frames),
                        '-c:v','libx264','-pix_fmt','yuv420p',str(clip)],check=True)
        return clip
    monkeypatch.setattr(m,'_infer',fake_infer)
    monkeypatch.setattr(m.common,'_ComfyMemoryGuard',FakeGuard)
    session=SimpleNamespace(root=tmp_path,input_dir=inp,output_dir=out,guard=SimpleNamespace(raise_if_stopped=lambda:None))
    names={'dit':'a','umt5':'b','vision':'c','vae':'d'}
    output,count=m._generate_wan_segments(session, JobRecord('video','test'),total_frames=total,
        image_path=image,video_path=source,names=names,quant='q4',prompt='dance',negative='',width=256,height=256,
        seed=42,steps=10,cfg=1,person='human',strength=1)
    plan=list(m._wan_plan(total))
    assert count==len(plan)==len(seen) and probe_media_info(output)['duration_seconds']==total/24
    decoded=subprocess.run([get_ffmpeg_bin(),'-hide_banner','-loglevel','error','-i',str(output),
                            '-map','0:v:0','-f','framemd5','-'],check=True,capture_output=True,text=True).stdout
    frames=[line for line in decoded.splitlines() if line and not line.startswith('#')]
    assert len(frames)==total
    assert [int(line.split(',')[2]) for line in frames] == list(range(total))
    assert [g['14']['inputs']['length'] for g in seen]==[p['window'] for p in plan]
    assert [g['15']['inputs']['noise_seed'] for g in seen]==list(range(42,42+count))


def test_api_duration_validation_and_legacy_modes(temp_studio_env, monkeypatch):
    import app
    client,root=temp_studio_env
    (root/'uploads'/'ref.png').write_bytes(b'x'); (root/'uploads'/'drive.mp4').write_bytes(b'x')
    data={'engine':'wan','quant':'q4','image_rel':'uploads/ref.png','video_rel':'uploads/drive.mp4','prompt':'dance', 'duration_sec':30.2}
    jobs=[]
    monkeypatch.setattr(app.job_manager,'submit',lambda **kw: jobs.append(kw) or SimpleNamespace(to_dict=lambda **kw:{'id':'job'}))
    assert client.post('/api/community/motion',json=data).status_code==200
    assert client.post('/api/community/motion',json={**data,'duration_sec':86401}).status_code==422
    assert client.post('/api/community/motion',json={**data,'engine':'ltx'}).status_code==400
    assert client.post('/api/community/motion',json={**data,'engine':'ltx','duration_sec':None}).status_code==200


def test_duration_rejects_short_guide_before_model_setup(temp_studio_env, monkeypatch, tmp_path):
    from studio.engines import motion_transfer as m
    from studio.jobs import JobRecord
    from PIL import Image
    image=tmp_path/'ref.png'; Image.new('RGB',(256,256)).save(image)
    video=tmp_path/'drive.mp4';make_clip(video)
    monkeypatch.setattr(m, '_prepare', lambda *a: pytest.fail('must not download'))
    with pytest.raises(ValueError, match='MOTION_VIDEO_TOO_SHORT'):
        m.generate(engine='wan',quant='q4',image_path=image,video_path=video,prompt='dance',negative='',
                   width=256,height=256,frames=49,duration_sec=30,seed=1,steps=10,cfg=1,person='human',strength=1,job=JobRecord('video','test'))


def test_comfy_events_log_node_execution(temp_studio_env, monkeypatch, tmp_path):
    from studio.engines import motion_transfer as m
    from studio.jobs import JobRecord
    class Reply:
        ok=True
        def __init__(self, data): self.data=data
        def json(self): return self.data
        def raise_for_status(self): pass
    def listener(base, client_id, events, stop, ready):
        events.put({'type':'executing','data':{'prompt_id':'id','node':'1'}})
        events.put({'type':'progress','data':{'prompt_id':'id','node':'1','value':2,'max':10}})
        ready.set()
    monkeypatch.setattr(m.common,'_events_listener',listener)
    monkeypatch.setattr(m.common.requests,'get',lambda url,**k: Reply({'StudioWanAnimate2GGUF':{}}) if url.endswith('object_info') else Reply({'id':{'status':{'status_str':'error','messages':[]}}}))
    monkeypatch.setattr(m.common.requests,'post',lambda *a,**k: Reply({'prompt_id':'id'}))
    session=SimpleNamespace(base='http://test',output_dir=tmp_path,alive=lambda:True,guard=SimpleNamespace(raise_if_stopped=lambda:None))
    job=JobRecord('video','test')
    with pytest.raises(RuntimeError,match='MOTION_INFERENCE_FAILED'):
        m._infer(session,{'1':{'class_type':'StudioWanAnimate2GGUF','inputs':{}}},job)
    assert any('Comfy ejecutando nodo 1: StudioWanAnimate2GGUF' in item['message'] for item in job.logs)
    assert job.current_step=='Comfy StudioWanAnimate2GGUF: 2/10'


def test_gguf_positional_new_empty_compatibility(temp_studio_env, monkeypatch):
    """Reproduce the pinned GGUF subclass + Wan attention buffer shape failure on CPU."""
    torch=pytest.importorskip('torch')
    fake=ModuleType('nodes'); monkeypatch.setitem(sys.modules,'nodes',fake)
    spec=importlib.util.spec_from_file_location('motion_shape_fix','studio/engines/comfy_motion_node/__init__.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    # Minimal copy of the pinned GGUF tensor's actual allocation/shape methods.
    class GGMLTensor(torch.Tensor):
        def __new__(cls, *args, tensor_type, tensor_shape, patches=None, **kwargs):
            return super().__new__(cls, *args, **kwargs)
        def __init__(self, *args, tensor_type, tensor_shape, patches=None, **kwargs):
            self.tensor_type, self.tensor_shape = tensor_type, tensor_shape
            self.patches = [] if patches is None else patches
        def new_empty(self, size, *args, **kwargs):
            new_tensor=super().new_empty(size, *args, **kwargs)
            return GGMLTensor(new_tensor, tensor_type=getattr(self,'tensor_type',None),
                              tensor_shape=size, patches=getattr(self,'patches',[]).copy())
        @property
        def shape(self):
            if not hasattr(self,'tensor_shape'): self.tensor_shape=self.size()
            return self.tensor_shape
    def reader(): pass
    reader.__globals__['GGMLTensor']=GGMLTensor
    try:
        x=GGMLTensor(torch.zeros(1,4,2),tensor_type='marker',tensor_shape=(1,4,2),patches=['patch'])
        broken=x.new_empty(1,6,1,2)
        with pytest.raises(TypeError,match='not subscriptable'): _=broken.shape[1]
        module._fix_gguf_new_empty(reader)
        first=GGMLTensor.new_empty; module._fix_gguf_new_empty(reader)
        assert GGMLTensor.new_empty is first  # repeated loads don't stack wrappers
        kbuf=x.new_empty(1,6,1,2,dtype=torch.float64)
        assert tuple(kbuf.shape)==(1,6,1,2) and kbuf.shape[1]==6
        assert kbuf.dtype==torch.float64 and kbuf.tensor_type=='marker' and kbuf.patches==['patch']
        assert tuple(x.new_empty((1,6,1,2)).shape)==(1,6,1,2)
        assert tuple(x.new_empty(3).shape)==(3,)
        # The expression used by Wan's forward_gen now handles the buffer branch.
        assert kbuf.reshape(1,kbuf.shape[1],2).size()==torch.Size([1,6,2])
    finally:
        reader.__globals__.pop('GGMLTensor',None)


def test_motion_plugin_upgrades_only_known_versions(temp_studio_env, monkeypatch, tmp_path):
    import hashlib
    from studio.engines import motion_transfer as m
    monkeypatch.setattr(m.common,'ROOT',tmp_path)
    dest=tmp_path/'comfy/custom_nodes/StudioMotion';dest.mkdir(parents=True)
    target=dest/'__init__.py'
    old=b'previous known version'
    target.write_bytes(old)
    monkeypatch.setattr(m,'_MOTION_PLUGIN_PREVIOUS_HASHES',frozenset({hashlib.sha256(old).hexdigest()}))
    m._install_motion_plugin()
    assert target.read_bytes()==(m.Path(m.__file__).with_name('comfy_motion_node')/'__init__.py').read_bytes()
    m._install_motion_plugin()  # already installed
    target.write_text('other user plugin')
    with pytest.raises(RuntimeError,match='MOTION_PLUGIN_CONFLICT'): m._install_motion_plugin()
    assert target.read_text()=='other user plugin'
    target.unlink(); target.symlink_to(tmp_path/'other')
    with pytest.raises(RuntimeError,match='MOTION_PLUGIN_CONFLICT'): m._install_motion_plugin()


@pytest.mark.parametrize('width,height',[(512,288),(288,512),(512,320),(320,576)])
def test_motion_framing_keeps_top_and_bottom_of_inputs(tmp_path,width,height):
    from PIL import Image, ImageDraw
    from studio.engines.motion_framing import fit_reference_image, fit_video_filter
    from studio.engines.ffmpeg_utils import get_ffmpeg_bin
    # Portrait image with clearly identifiable areas at both extremes.
    original=Image.new('RGB',(200,400),'green')
    draw=ImageDraw.Draw(original)
    draw.rectangle((0,0,199,59),fill=(255,0,0))
    draw.rectangle((0,340,199,399),fill=(0,0,255))
    source=tmp_path/'portrait.png'; original.save(source)
    fitted=tmp_path/'fitted.png'; fit_reference_image(source,fitted,width,height)
    assert Image.open(fitted).size == (width,height)
    for point,color in [((width//2,10),'red'),((width//2,height-10),'blue')]:
        pixel=Image.open(fitted).getpixel(point)
        assert pixel[0 if color=='red' else 2]>200
    video_frame=tmp_path/'ffmpeg.png'
    subprocess.run([get_ffmpeg_bin(),'-hide_banner','-loglevel','error','-y','-i',str(source),
                    '-vf',fit_video_filter(width,height),'-frames:v','1','-threads','1',str(video_frame)],check=True)
    video_image=Image.open(video_frame).convert('RGB')
    assert video_image.size == (width,height)
    assert video_image.getpixel((width//2,10))[0] > 200
    assert video_image.getpixel((width//2,height-10))[2] > 200
    if width / height > 0.5:
        assert all(abs(c-127)<5 for c in video_image.getpixel((2,height//2)))


def test_motion_verified_receipt_reuses_hash_and_detects_changes(temp_studio_env, monkeypatch, tmp_path):
    import hashlib, os
    from studio.engines import ltx_gguf_download as download
    root=tmp_path/'weights'; root.mkdir()
    target=root/'model'; target.write_bytes(b'abc')
    digest=hashlib.sha256(b'abc').hexdigest()
    calls=[]; real=download.verify
    monkeypatch.setattr(download,'verify',lambda *args,**kwargs: calls.append(1) or real(*args,**kwargs))
    assert download._verify_motion_cached(root,target,3,digest)
    assert download._verify_motion_cached(root,target,3,digest)
    assert calls == [1]
    old=target.stat().st_mtime_ns
    target.write_bytes(b'bad'); os.utime(target, ns=(old+2_000_000_000,old+2_000_000_000))
    assert not download._verify_motion_cached(root,target,3,digest)
    assert calls == [1,1]
    target.unlink(); assert not download._verify_motion_cached(root,target,3,digest)
    target.write_bytes(b'abc'); assert download._verify_motion_cached(root,target,3,digest)
    assert len(calls) == 3
    assert not download._verify_motion_cached(root,target,3,'0'*64)


def test_motion_resident_lifecycle_reuses_and_invalidates(temp_studio_env, monkeypatch, tmp_path):
    from PIL import Image
    from studio.engines import motion_transfer as m
    from studio.jobs import JobRecord, JobCancelledError
    _,root=temp_studio_env
    img=root/'uploads'/'ref.png'; Image.new('RGB',(256,256)).save(img)
    video=root/'uploads'/'guide.mp4'; make_clip(video)
    slot={}; starts=[]; destroyed=[]; fingerprint=['one']; failing=[False]
    def unload(name):
        session=slot.pop('value',None)
        if session: destroyed.append(session)
    def resident(model):
        value=slot.get('value')
        return value if value and value.model_id==model else None
    def open_session(job,names,model):
        session=resident(model)
        if session is None:
            base=tmp_path/f'session{len(starts)}'; base.mkdir()
            (base/'in').mkdir(); (base/'out').mkdir()
            session=SimpleNamespace(model_id=model,root=base,input_dir=base/'in',output_dir=base/'out',
                options=('auto',False), keep_loaded=True, cleanup_ok=False,
                guard=SimpleNamespace(raise_if_stopped=lambda:None))
            slot['value']=session; starts.append(session)
        else: session.cleanup_ok=False
        return session
    def finish(session,job,success):
        if not success or not session.cleanup_ok: unload('video')
    def infer(session,graph,job,**kwargs):
        if failing[0]: raise JobCancelledError('cancelled')
        session.cleanup_ok=True
        clip=session.output_dir/'generated.mp4';make_clip(clip);return clip
    monkeypatch.setattr(m.common,'query_gpu_info',lambda:{'has_gpu':True})
    monkeypatch.setattr(m.common,'_ComfyMemoryGuard',FakeGuard)
    monkeypatch.setattr(m.common,'_comfy_start_admission',lambda *a,**kw:None)
    monkeypatch.setattr(m.memory_manager,'get_slot',lambda *_: slot.get('value'))
    monkeypatch.setattr(m.memory_manager,'unload_slot',unload)
    monkeypatch.setattr(m.common,'_resident',resident)
    monkeypatch.setattr(m.common,'_open_resident',open_session)
    monkeypatch.setattr(m.common,'_finish_session',finish)
    monkeypatch.setattr(m,'_prepare',lambda *a:{'dit':'dit','umt5':'text','vision':'vision','vae':'vae'})
    monkeypatch.setattr(m,'_motion_fingerprint',lambda *a:fingerprint[0])
    monkeypatch.setattr(m,'_infer',infer)
    def run(quant='q4'):
        return m.generate(engine='wan',quant=quant,image_path=img,video_path=video,prompt='person',negative='',
            width=256,height=256,frames=25,seed=1,steps=10,cfg=1,person='human',strength=1,job=JobRecord('video','test'))
    run();run();assert len(starts)==1 and not destroyed
    fingerprint[0]='two';run();assert len(starts)==2 and len(destroyed)==1
    run('q5');assert len(starts)==3 and len(destroyed)==2
    failing[0]=True
    with pytest.raises(JobCancelledError): run('q5')
    assert len(destroyed)==3 and not slot


@pytest.mark.parametrize('valid', [True, False])
def test_motion_infer_resident_confirms_cleanup_before_reuse(temp_studio_env, monkeypatch, tmp_path, valid):
    from studio.engines import motion_transfer as m
    from studio.jobs import JobRecord
    out=tmp_path/'out';out.mkdir();(out/'studio_motion_00001.mp4').write_bytes(b'video')
    calls=[]
    class Reply:
        ok=True
        def __init__(self, data): self.data=data
        def json(self): return self.data
        def raise_for_status(self): pass
    def get(url, **kwargs):
        if url.endswith('/object_info'): return Reply({'Test':{}})
        if '/history/' in url: return Reply({'id':{'status':{'completed':True}}})
        if url.endswith('/studio-resident/status'): return Reply({'version':1,'prompt_id':'id','ok':valid,'retained_loaders':4})
        raise AssertionError(url)
    def post(url, **kwargs):
        calls.append(url)
        return Reply({'prompt_id':'id'})
    monkeypatch.setattr(m.common.requests,'get',get)
    monkeypatch.setattr(m.common.requests,'post',post)
    monkeypatch.setattr(m.common,'_events_listener',lambda base,cid,events,stop,ready: ready.set())
    session=SimpleNamespace(base='http://private',output_dir=out,keep_loaded=True,cleanup_ok=False,
        guard=SimpleNamespace(raise_if_stopped=lambda:None),alive=lambda:True)
    m._infer(session,{'1':{'class_type':'Test'}},JobRecord('video','test'))
    assert session.cleanup_ok is valid
    assert any(x.endswith('/history') for x in calls) is valid
    if not valid: assert session.motion_cleanup_failed


def test_motion_bad_metadata_invalidates_receipt_without_deleting_model(temp_studio_env, monkeypatch, tmp_path):
    import hashlib
    from studio.engines import motion_transfer as m, ltx_gguf_download as download
    root=tmp_path/'weights'; root.mkdir()
    monkeypatch.setattr(m,'WEIGHTS',root)
    monkeypatch.setattr(m,'files_for',lambda *args:[('dit','repo','revision','model',3,hashlib.sha256(b'abc').hexdigest())])
    path=root/'ltx'/'model';path.parent.mkdir();path.write_bytes(b'abc')
    assert download._verify_motion_cached(root,path,3,hashlib.sha256(b'abc').hexdigest())
    assert len(list((root/'.motion-verified').iterdir()))==1
    m._invalidate_motion_receipts('wan','q4')
    assert not list((root/'.motion-verified').iterdir()) and path.read_bytes()==b'abc'


def test_ltx_union_prepare_reuses_own_resident_not_unrelated(temp_studio_env, monkeypatch):
    from studio.engines import ltx_gguf_backend as backend
    from studio.jobs import JobRecord
    resident=SimpleNamespace(options=('auto',False),keep_loaded=True,names={'dit':'existing'})
    asked=[]
    monkeypatch.setattr(backend,'query_gpu_info',lambda:{'has_gpu':True,'bf16_supported':True,'vram_total_gb':22})
    monkeypatch.setattr(backend,'_resident',lambda model: asked.append(model) or resident)
    monkeypatch.setattr(backend,'_install',lambda *a,**k:pytest.fail('no reinstall'))
    job=JobRecord('video','union');job.comfy_keep_loaded=True
    assert backend._prepare_gguf(job,model_id='ltx23-union-control-q4') == {'dit':'existing'}
    assert asked == ['ltx23-union-control-q4']
