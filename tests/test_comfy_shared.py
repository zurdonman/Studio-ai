"""Shared Comfy lifecycle and fallback, with fake HTTP; no GPU required."""
import hashlib
import json
from types import SimpleNamespace

import pytest


def make_manager(monkeypatch, tmp_path, *, clean=True, matching=True):
    from studio.engines import ltx_gguf_backend as b
    from studio.jobs import JobRecord
    class Worker:
        def __init__(self):
            self.runtime_root = (tmp_path / ('common' if matching else 'other')).resolve()
            self.keep_loaded = True
            self.options = ('auto', False)
            self.cleanup_ok = clean
            self.base = 'http://127.0.0.1:test'
            self.proc = SimpleNamespace(pid=1234)
            self.guard = SimpleNamespace(job=None, raise_if_stopped=lambda: None)
            self.names = {'old':'weights'}
        def alive(self): return True
    monkeypatch.setattr(b, '_ResidentComfy', Worker)
    monkeypatch.setattr(b, 'ROOT', tmp_path/'common')
    session=Worker()
    slots={'video':{'model_id':'wan-motion-q4','handle':session}}
    calls=[]
    monkeypatch.setattr(b.memory_manager, 'get_slot', lambda slot: slots.get(slot))
    monkeypatch.setattr(b.memory_manager, 'unload_slot', lambda slot: calls.append('stop') or slots.pop(slot, None))
    def register(slot, model_id, name, handle):
        slots[slot]={'model_id':model_id,'handle':handle}; calls.append('register:'+model_id)
    monkeypatch.setattr(b.memory_manager, 'register_loaded', register)
    class Reply:
        def __init__(self,data): self.data=data
        def raise_for_status(self): pass
        def json(self): return self.data
    def get(url, **kwargs):
        if '/history/' in url: return Reply({'reset-id':{'status':{'completed':True}}})
        if url.endswith('/studio-resident/status'): return Reply({'version':1,'prompt_id':'reset-id','ok':True,'total_loaders':0})
        if url.endswith('/object_info'): return Reply({'StudioWanAnimate2GGUF':{}})
        raise AssertionError(url)
    def post(url, **kwargs):
        calls.append(url.rsplit('/',1)[-1])
        return Reply({'prompt_id':'reset-id'})
    monkeypatch.setattr(b.requests,'get',get)
    monkeypatch.setattr(b.requests,'post',post)
    job=JobRecord('video','shared'); job.comfy_keep_loaded=True
    return b,job,session,slots,calls


def test_switch_same_runtime_reuses_process_after_confirmed_reset(temp_studio_env, monkeypatch, tmp_path):
    b,job,session,slots,calls=make_manager(monkeypatch,tmp_path)
    assert b._switch_to_idle(job) is True
    assert slots['video']['model_id']==b._IDLE_COMFY and session.names=={}
    assert 'prompt' in calls and 'history' in calls and 'stop' not in calls
    assert session.guard.job is None
    job.studio_motion_plugin=True
    opened=b._open_resident(job, {'dit':'new'}, 'scail-motion-q4')
    assert opened is session and slots['video']['model_id']=='scail-motion-q4'
    assert opened.names=={'dit':'new'} and opened.guard.job is job and not opened.cleanup_ok


@pytest.mark.parametrize('clean,matching', [(False,True),(True,False)])
def test_switch_closes_unconfirmed_or_incompatible_process(temp_studio_env, monkeypatch, tmp_path, clean, matching):
    b,job,session,slots,calls=make_manager(monkeypatch,tmp_path,clean=clean,matching=matching)
    assert not b._switch_to_idle(job)
    assert calls==['stop'] and not slots


def test_switch_failure_falls_back_to_new_process(temp_studio_env, monkeypatch, tmp_path):
    b,job,session,slots,calls=make_manager(monkeypatch,tmp_path)
    monkeypatch.setattr(b.requests,'post',lambda *a,**kw: (_ for _ in ()).throw(RuntimeError('offline')))
    assert b._switch_to_idle(job) is False and calls==['stop'] and not slots


def test_missing_new_plugin_restarts_instead_of_reusing_stale_server(temp_studio_env, monkeypatch, tmp_path):
    b,job,session,slots,calls=make_manager(monkeypatch,tmp_path)
    assert b._switch_to_idle(job)
    job.comfy_required_nodes={'NewNode'}
    class NewWorker:
        def __init__(self, job, names): self.names=names
        def unload(self): pass
    monkeypatch.setattr(b,'_ResidentComfy',NewWorker)
    # _resident validates instance; make the existing live session visible to _open_resident.
    monkeypatch.setattr(b,'_resident',lambda model=b.MODEL_ID: session if model==b._IDLE_COMFY else None)
    opened=b._open_resident(job,{'dit':'new'},'new-model')
    assert isinstance(opened,NewWorker) and 'stop' in calls


@pytest.mark.parametrize('probe_ok', [True, False])
def test_shared_gemma_patch_success_or_safe_rollback(temp_studio_env, monkeypatch, tmp_path, probe_ok):
    from studio.engines import ltx25_gguf as q, ltx25_gguf_patch as patch
    from studio.catalog import catalog_manager
    from studio.jobs import JobRecord
    root=tmp_path/'shared';loader=root/'comfy/custom_nodes/ComfyUI-GGUF/loader.py'
    loader.parent.mkdir(parents=True);loader.write_bytes(b'original')
    monkeypatch.setattr(q.common,'ROOT',root)
    monkeypatch.setattr(q.common,'_install',lambda *a:None)
    monkeypatch.setattr(q.memory_manager,'get_slot',lambda *_:None)
    monkeypatch.setattr(catalog_manager,'get_hf_token',lambda:None)
    def apply(*args,**kwargs):
        loader.write_bytes(b'patched')
        (root/'gemma4-patch.json').write_text(json.dumps({'patch_sha256':patch.PATCH_SHA256,'loader_sha256':hashlib.sha256(b'patched').hexdigest()}))
    monkeypatch.setattr(q.common,'_command',apply)
    def probe(*args):
        if not probe_ok: raise RuntimeError('nodes incompatible')
    monkeypatch.setattr(q.common,'_probe_runtime',probe)
    job=JobRecord('video','test')
    assert q._try_shared_gemma(job,q.Q5_ID) is probe_ok
    assert loader.read_bytes()==(b'patched' if probe_ok else b'original')
    assert (root/'gemma4-patch.json').exists() is probe_ok


def test_shared_gemma_never_rolls_back_unknown_edit(temp_studio_env, monkeypatch, tmp_path):
    from studio.engines import ltx25_gguf as q
    from studio.catalog import catalog_manager
    from studio.jobs import JobRecord
    root=tmp_path/'shared';loader=root/'comfy/custom_nodes/ComfyUI-GGUF/loader.py'
    loader.parent.mkdir(parents=True);loader.write_bytes(b'original')
    monkeypatch.setattr(q.common,'ROOT',root)
    monkeypatch.setattr(q.common,'_install',lambda *a:None)
    monkeypatch.setattr(catalog_manager,'get_hf_token',lambda:None)
    def conflict(*a,**kw):
        loader.write_bytes(b'foreign-edit')
        raise RuntimeError('conflict')
    monkeypatch.setattr(q.common,'_command',conflict)
    with pytest.raises(RuntimeError,match='ROLLBACK_FAILED'):
        q._try_shared_gemma(JobRecord('video','test'),q.Q5_ID)
    assert loader.read_bytes()==b'foreign-edit'


@pytest.mark.parametrize('message, retry', [
    ('GGUF_RUNTIME_MISSING_NODES: CLIPLoaderGGUF', True),
    ('GGUF_INFERENCE_FAILED: unknown architecture gemma4', True),
    ('GGUF_INFERENCE_FAILED: out of memory', False),
    ('GGUF_INFERENCE_FAILED: invalid user input', False),
])
def test_q5_retries_isolated_only_for_explicit_compatibility_failure(temp_studio_env, monkeypatch, message, retry):
    from studio.engines import ltx25_gguf as q
    from studio.jobs import JobRecord
    calls=[]
    def once(**kw):
        job=kw['job']; calls.append(getattr(job, 'ltx25_force_fallback', False))
        job.ltx25_runtime=q.common.ROOT
        if len(calls)==1: raise RuntimeError(message)
        return {'ok':True}
    monkeypatch.setattr(q, '_generate_once', once)
    kwargs=dict(model_id=q.Q5_ID,image_path=None,prompt='test',negative_prompt='',seed=1,
                job=JobRecord('video','test'),width=512,height=320,duration_sec=2,fps=24,speech_language='auto',image_strength=1)
    if retry:
        assert q.generate(**kwargs)=={'ok':True}
        assert calls==[False,True] and not kwargs['job'].ltx25_force_fallback
    else:
        with pytest.raises(RuntimeError,match='GGUF_'): q.generate(**kwargs)
        assert calls==[False]


@pytest.mark.parametrize('shared', [True, False])
def test_gemma_selects_shared_or_isolated_runtime(temp_studio_env, monkeypatch, tmp_path, shared):
    from studio.engines import ltx25_gguf as q
    from studio.jobs import JobRecord
    root=tmp_path/'shared'; fallback=tmp_path/'isolated'; called=[]
    monkeypatch.setattr(q.common,'ROOT',root)
    monkeypatch.setattr(q,'Q5_RUNTIME',fallback)
    monkeypatch.setattr(q.common,'query_gpu_info',lambda:{'has_gpu':True,'bf16_supported':True,'vram_total_gb':22})
    monkeypatch.setattr(q.common,'_resident',lambda *_:None)
    monkeypatch.setattr(q.common,'_switch_to_idle',lambda job,root=None:called.append(('switch',root)))
    monkeypatch.setattr(q.common,'_comfy_start_admission',lambda *a,**kw:None)
    monkeypatch.setattr(q,'_check_q5_access',lambda job:None)
    monkeypatch.setattr(q,'_try_shared_gemma',lambda job,model:shared)
    monkeypatch.setattr(q.common,'_install',lambda *args:called.append(('install',args[1] if len(args)>1 else root)))
    monkeypatch.setattr(q.common,'_probe_runtime',lambda *a:None)
    monkeypatch.setattr(q.common,'_command',lambda *a,**kw:None)
    monkeypatch.setattr(q.shutil,'disk_usage',lambda *_:SimpleNamespace(free=100*1024**3))
    monkeypatch.setattr(q,'_links',lambda mid,runtime:{'runtime':str(runtime)})
    job=JobRecord('video','test')
    result=q._prepare(q.Q5_ID,job)
    expected=root if shared else fallback
    assert job.ltx25_runtime==expected and result['runtime']==str(expected)
    assert (('install',fallback) in called) is not shared


def test_switch_cancel_stops_worker_and_propagates(temp_studio_env, monkeypatch, tmp_path):
    from studio.jobs import JobCancelledError
    b,job,session,slots,calls=make_manager(monkeypatch,tmp_path)
    job.cancel_requested=True
    with pytest.raises(JobCancelledError): b._switch_to_idle(job)
    assert calls==['stop'] and not slots


def test_switch_requires_matching_reset_identity(temp_studio_env, monkeypatch, tmp_path):
    b,job,session,slots,calls=make_manager(monkeypatch,tmp_path)
    class Reply:
        def raise_for_status(self): pass
        def json(self): return {'version':1,'prompt_id':'old-job','ok':True,'total_loaders':0}
    old=b.requests.get
    monkeypatch.setattr(b.requests,'get',lambda url,**kw: Reply() if url.endswith('/studio-resident/status') else old(url,**kw))
    assert not b._switch_to_idle(job)
    assert 'stop' in calls and not slots


def test_switch_memory_pressure_is_not_retried(temp_studio_env, monkeypatch, tmp_path):
    from studio.memory_policy import MemoryPressureError
    b,job,session,slots,calls=make_manager(monkeypatch,tmp_path)
    monkeypatch.setattr(b.requests,'post',lambda *a,**kw: (_ for _ in ()).throw(MemoryPressureError('low memory')))
    with pytest.raises(MemoryPressureError): b._switch_to_idle(job)
    assert calls==['stop'] and not slots
