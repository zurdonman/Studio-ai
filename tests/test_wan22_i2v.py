"""Wan two-expert graph, API and lifecycle contracts; no real model inference."""
import json
import subprocess
from types import SimpleNamespace
import pytest
from PIL import Image


def args():
    return dict(quant='q4',names={'high':'high.gguf','low':'low.gguf','umt5':'t5.safetensors','vae':'vae.safetensors'},
                image_name='reference.png',prompt='A person walking',negative='',width=512,height=320,frames=49,seed=42,steps=40,cfg=3.5)


def test_manifest_and_two_expert_graph(temp_studio_env):
    from studio.engines.wan22_i2v import make_graph
    from studio.engines.wan22_i2v_manifest import files_for
    for quant in ('q2','q3s','q3','q4s','q4','q5s','q5_0','q5','q5_1','q6','q8','fp8','nvfp4'):
        files=files_for(quant)
        assert [x[0] for x in files] == ['high','low','umt5','vae']
        assert all(len(x[2])==40 and x[4]>0 and len(x[5])==64 for x in files)
        assert files[0][3] != files[1][3] and files[0][5] != files[1][5]
        g=make_graph(**{**args(),'quant':quant})
        assert g['1']['class_type'] == ('UnetLoaderGGUF' if quant.startswith('q') else 'UNETLoader')
        assert g['8']['inputs']['start_image']==['5',0]
        first, second = g['11']['inputs'],g['12']['inputs']
        assert first['model']==['9',0] and second['model']==['10',0]
        assert (first['start_at_step'],first['end_at_step'],first['return_with_leftover_noise'])==(0,20,'enable')
        assert (second['start_at_step'],second['end_at_step'],second['add_noise'],second['latent_image'])==(20,40,'disable',['11',0])
        assert g['14']['inputs']['fps']==16
    with pytest.raises(ValueError): make_graph(**{**args(),'quant':'fp4'})
    with pytest.raises(ValueError): make_graph(**{**args(),'frames':50})
    with pytest.raises(ValueError): make_graph(**{**args(),'width':513})
    with pytest.raises(ValueError): make_graph(**{**args(),'image_name':'../image.png'})


def test_api_requires_consent_and_valid_input(temp_studio_env,monkeypatch):
    import app
    client,root=temp_studio_env
    (root/'uploads').mkdir(parents=True, exist_ok=True)
    img=root/'uploads'/'wan22.png'; Image.new('RGB',(256,256)).save(img)
    jobs=[]
    monkeypatch.setattr(app.job_manager,'submit',lambda **kw:jobs.append(kw) or SimpleNamespace(to_dict=lambda **kw:{'id':'j'}))
    body={'quant':'q4','image_rel':'uploads/wan22.png','prompt':'A wave'}
    assert client.post('/api/video/wan22-i2v',json=body).status_code==400
    body['install_accepted']=True
    assert client.post('/api/video/wan22-i2v',json={**body,'frames':50}).status_code==400
    assert client.post('/api/video/wan22-i2v',json={**body,'quant':'unknown'}).status_code==422
    assert client.post('/api/video/wan22-i2v',json={**body,'image_rel':'../x.png'}).status_code in (400,404)
    assert client.post('/api/video/wan22-i2v',json=body).status_code==200
    assert jobs[0]['kind']=='video'


def test_synthetic_output_and_worker_cleanup(temp_studio_env,monkeypatch,tmp_path,colab_memory):
    from studio.engines import wan22_i2v as w
    from studio.engines.ffmpeg_utils import get_ffmpeg_bin
    from studio.jobs import JobRecord
    colab_memory()
    _,root=temp_studio_env
    img=root/'uploads'/'wan22_test.png';Image.new('RGB',(512,320)).save(img)
    (tmp_path/'input').mkdir(); (tmp_path/'output').mkdir()
    closed=[]
    session=SimpleNamespace(input_dir=tmp_path/'input',output_dir=tmp_path/'output',guard=SimpleNamespace(raise_if_stopped=lambda:None))
    monkeypatch.setattr(w.common,'query_gpu_info',lambda:{'has_gpu':True,'bf16_supported':True})
    monkeypatch.setattr(w.common,'_switch_to_idle',lambda *a:False)
    monkeypatch.setattr(w.common,'_comfy_start_admission',lambda *a,**k:None)
    monkeypatch.setattr(w,'_prepare',lambda *a:{k:k for k in ('high','low','umt5','vae')})
    monkeypatch.setattr(w.common,'_open_resident',lambda *a:session)
    monkeypatch.setattr(w.common,'_finish_session',lambda *a:closed.append(a[-1]))
    def fake_infer(sess,graph,job,**kw):
        assert (sess.input_dir/'reference.png').is_file()
        assert graph['12']['inputs']['latent_image']==['11',0]
        path=sess.output_dir/'fake.mp4'
        subprocess.run([get_ffmpeg_bin(),'-v','error','-y','-f','lavfi','-i','color=s=512x320:r=16:d=3.0625','-c:v','libx264',str(path)],check=True)
        return path
    monkeypatch.setattr(w,'_infer',fake_infer)
    result=w.generate(quant='q4',image_path=img,prompt='walk',negative='',width=512,height=320,frames=49,steps=40,cfg=3.5,seed=42,job=JobRecord('video','test'))
    assert closed==[True] and (root/result['rel_path']).is_file()
    assert json.loads((root/'outputs/videos'/f"{result['id']}.json").read_text())['model_id']==result['model_id']

@pytest.mark.parametrize('quant', ['q2','q3s','q3','q4_0','q4_1','q4s','q4','q5s','q5_0','q5','q5_1','q6','q8','fp8'])
@pytest.mark.parametrize('mode', ['t2v','i2v','flf2v'])
def test_fun_modes_and_quantized_experts(temp_studio_env, quant, mode):
    from studio.engines.wan22_i2v import make_graph
    from studio.engines.wan22_i2v_manifest import files_for
    files=files_for(quant,'fun')
    assert [f[0] for f in files]==['high','low','umt5','vae']
    assert all(len(f[2])==40 and len(f[5])==64 and f[4]>0 for f in files)
    assert 'fun' in files[0][3].lower() and 'fun' in files[1][3].lower()
    assert files[0][5] != files[1][5]
    g=make_graph(**{**args(),'recipe':'fun','mode':mode,'quant':quant,
        'image_name':None if mode=='t2v' else 'reference.png', 'end_image_name':'end.png' if mode=='flf2v' else None})
    assert g['8']['class_type']=='WanFunInpaintToVideo'
    assert ('start_image' in g['8']['inputs']) == (mode!='t2v')
    assert ('end_image' in g['8']['inputs']) == (mode=='flf2v')
    assert g['11']['inputs']['end_at_step']==g['12']['inputs']['start_at_step']


def test_fun_invalid_modes_and_api(temp_studio_env, monkeypatch):
    import app
    from studio.engines.wan22_i2v import make_graph
    from studio.engines.wan22_i2v_manifest import files_for
    with pytest.raises(ValueError): files_for('nvfp4','fun')
    with pytest.raises(ValueError): files_for('q4_0','i2v')
    with pytest.raises(ValueError): make_graph(**{**args(),'recipe':'fun','mode':'flf2v'})
    with pytest.raises(ValueError): make_graph(**{**args(),'recipe':'i2v','mode':'t2v','image_name':None})
    with pytest.raises(ValueError): make_graph(**{**args(),'recipe':'fun','mode':'t2v'})
    client,root=temp_studio_env
    (root/'uploads').mkdir(exist_ok=True)
    for n in ('begin','end'): Image.new('RGB',(256,256)).save(root/'uploads'/f'{n}.png')
    jobs=[]
    monkeypatch.setattr(app.job_manager,'submit',lambda **kw:jobs.append(kw) or SimpleNamespace(to_dict=lambda **kw:{'id':'j'}))
    b={'recipe':'fun','quant':'fp8','mode':'t2v','prompt':'wave','install_accepted':True}
    assert client.post('/api/video/wan22-i2v',json=b).status_code==200
    assert client.post('/api/video/wan22-i2v',json={**b,'mode':'i2v'}).status_code==400
    assert client.post('/api/video/wan22-i2v',json={**b,'mode':'flf2v','image_rel':'uploads/begin.png'}).status_code==400
    assert client.post('/api/video/wan22-i2v',json={**b,'mode':'flf2v','image_rel':'uploads/begin.png','end_image_rel':'uploads/end.png'}).status_code==200
    assert client.post('/api/video/wan22-i2v',json={**b,'mode':'t2v','image_rel':'uploads/begin.png'}).status_code==400
    assert client.post('/api/video/wan22-i2v',json={**b,'quant':'nvfp4'}).status_code==400
    assert len(jobs)==2

@pytest.mark.parametrize('mode', ['t2v','i2v','flf2v'])
def test_fun_synthetic_lifecycle(temp_studio_env, monkeypatch, tmp_path, colab_memory, mode):
    from studio.engines import wan22_i2v as w
    from studio.jobs import JobRecord
    colab_memory()
    _,root=temp_studio_env
    a=tmp_path/'a.png'; b=tmp_path/'b.png'
    Image.new('RGB',(256,256),'red').save(a); Image.new('RGB',(256,256),'blue').save(b)
    (tmp_path/'input').mkdir(); (tmp_path/'output').mkdir()
    sess=SimpleNamespace(input_dir=tmp_path/'input',output_dir=tmp_path/'output',guard=SimpleNamespace(raise_if_stopped=lambda:None))
    closed=[]
    monkeypatch.setattr(w.common,'query_gpu_info',lambda:{'has_gpu':True,'bf16_supported':True})
    monkeypatch.setattr(w.common,'_switch_to_idle',lambda *a:False)
    monkeypatch.setattr(w.common,'_comfy_start_admission',lambda *a,**k:None)
    monkeypatch.setattr(w,'_prepare',lambda q,j,r: {k:k for k in ('high','low','umt5','vae')} if r=='fun' else pytest.fail('wrong recipe'))
    monkeypatch.setattr(w.common,'_open_resident',lambda *a:sess)
    monkeypatch.setattr(w.common,'_finish_session',lambda *a:closed.append(a[-1]))
    monkeypatch.setattr(w,'probe_media_info',lambda p:{'has_video':True,'width':512,'height':320,'duration_seconds':49/16})
    def infer(session,graph,job,**kwargs):
        assert graph['8']['class_type']=='WanFunInpaintToVideo'
        assert ('start_image' in graph['8']['inputs']) == (mode!='t2v')
        assert ('end_image' in graph['8']['inputs']) == (mode=='flf2v')
        assert (session.input_dir/'reference.png').exists() == (mode!='t2v')
        assert (session.input_dir/'end.png').exists() == (mode=='flf2v')
        p=session.output_dir/'result.mp4'; p.write_bytes(b'synthetic'); return p
    monkeypatch.setattr(w,'_infer',infer)
    result=w.generate(quant='fp8',recipe='fun',mode=mode,image_path=None if mode=='t2v' else a,
        end_image_path=b if mode=='flf2v' else None,prompt='walk',negative='',width=512,height=320,
        frames=49,steps=40,cfg=3.5,seed=42,job=JobRecord('video','test'))
    assert closed==[True] and result['video_submode']==mode and 'fun' in result['model_id']
    assert (root/result['rel_path']).is_file()
