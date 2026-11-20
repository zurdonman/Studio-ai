"""Wan Fun adapter compatibility and lifecycles; no GPU/model downloads."""
from types import SimpleNamespace
import pytest


def test_official_fun_rewards_have_exact_base_and_files(temp_studio_env):
    from studio.catalog import DEFAULT_CATALOG
    models = {m['id']:m for m in DEFAULT_CATALOG}
    for size, base in [('14b','wan21-fun-inp-14b-fp8'), ('1.3b','wan21-fun-1.3b-inp')]:
        for reward, label in [('hps21','HPS2.1'), ('mps','MPS')]:
            d=models[f'lora-wan21-fun-{size}-{reward}']
            assert d['compatible_models']==[base] and d['stackable_creative']
            assert d['artifacts'][0]['filename']==f'Wan2.1-Fun-{size.upper()}-InP-{label}_reward_lora_comfy.safetensors'
            assert not d.get('recommended_steps')
    client,_=temp_studio_env
    ids={x['id'] for x in client.get('/api/loras?model_id=wan21-flf2v-14b-fp8').json()['video_loras']}
    assert 'lora-wan21-fun-14b-mps' not in ids


@pytest.mark.parametrize('failure',[False,True])
def test_fun_diffusers_weights_released_after_success_or_error(temp_studio_env,monkeypatch,failure):
    from studio.engines import video_engine as v
    slots={'video':{'model_id':'wan21-fun-inp-14b-fp8','handle':object()}}
    freed=[]
    monkeypatch.setattr(v.memory_manager,'get_slot',lambda s:slots.get(s))
    monkeypatch.setattr(v.memory_manager,'unload_slot',lambda s:freed.append(s) or slots.pop(s))
    @v._release_video_after_generation
    def fake(mode='ai',model_id=None,job=None):
        if failure: raise RuntimeError('synthetic out of memory')
        return 'output-saved'
    job=SimpleNamespace(log=lambda *a,**kw:None)
    if failure:
        with pytest.raises(RuntimeError,match='out of memory'): fake(model_id='wan21-fun-inp-14b-fp8',job=job)
    else:
        assert fake(model_id='wan21-fun-inp-14b-fp8',job=job)=='output-saved'
    assert freed==['video'] and not slots


def test_other_models_and_ffmpeg_not_evicted(temp_studio_env,monkeypatch):
    from studio.engines import video_engine as v
    freed=[]
    monkeypatch.setattr(v.memory_manager,'get_slot',lambda s:{'model_id':'wan21-fun-inp-14b-fp8'})
    monkeypatch.setattr(v.memory_manager,'unload_slot',lambda s:freed.append(s))
    @v._release_video_after_generation
    def fake(mode='ai',model_id=None,job=None): return 'ok'
    assert fake(mode='ffmpeg_motion') == 'ok'
    assert fake(model_id='other-model') == 'ok'
    assert not freed


@pytest.mark.parametrize("model", ["wan21-fun-inp-14b-fp8", "wan21-fun-1.3b-inp", "ltx-video-13b-distilled"])
@pytest.mark.parametrize("failure", [False, True])
def test_video_slot_released_across_models_and_repeated_runs(temp_studio_env, monkeypatch, model, failure):
    from studio.engines import video_engine as v
    slots = {}
    monkeypatch.setattr(v.memory_manager, 'get_slot', lambda slot: slots.get(slot))
    monkeypatch.setattr(v.memory_manager, 'unload_slot', lambda slot: slots.pop(slot))
    @v._release_video_after_generation
    def fake(mode='ai', model_id=None, job=None):
        slots['video'] = {'model_id':model_id, 'handle':object()}
        if failure: raise RuntimeError('cancelled or failed')
        return 'ok'
    for _ in range(3):
        if failure:
            with pytest.raises(RuntimeError): fake(model_id=model)
        else: assert fake(model_id=model) == 'ok'
        assert not slots


def test_video_lora_validation_rejects_unknown_base_and_submode_before_loading(temp_studio_env, monkeypatch):
    from studio.engines import video_engine as v
    defs = {
        'base': {'id':'base', 'name':'Base', 'category':'video', 'video_modes':['i2v','flf2v']},
        'bad': {'id':'bad', 'name':'Bad', 'category':'lora', 'lora_target':'video', 'compatible_models':['other']},
        'nomodel': {'id':'nomodel', 'name':'No declared base', 'category':'lora', 'lora_target':'video'},
        'wrongmode': {'id':'wrongmode', 'name':'Wrong mode', 'category':'lora', 'lora_target':'video', 'compatible_models':['base'], 'incompatible_submodes':['flf2v']},
    }
    monkeypatch.setattr(v.catalog_manager, 'get_definition', lambda ident: defs.get(ident))
    monkeypatch.setattr(v.memory_manager, 'get_slot', lambda name: None)
    monkeypatch.setattr(v, 'load_video_model_into_memory', lambda **kw: pytest.fail('loaded before validation'))
    image = temp_studio_env[1] / 'dummy.png'
    image.write_bytes(b'test')
    for lid, mode in [('bad','i2v'), ('nomodel','i2v'), ('wrongmode','flf2v')]:
        with pytest.raises(ValueError, match='VIDEO_LORA_INCOMPATIBLE'):
            v.generate_video_clip.__wrapped__.__wrapped__(model_id='base', video_submode=mode, image_path=image, end_image_path=image, video_lora_id=lid)
        with pytest.raises(ValueError, match='VIDEO_LORA_INCOMPATIBLE'):
            v.generate_video_clip.__wrapped__.__wrapped__(model_id='base', video_submode=mode, image_path=image, end_image_path=image, extra_video_lora_ids=[lid])


def test_compatibility_endpoint_does_not_advertise_unverified_fun_crossovers(temp_studio_env):
    client,_=temp_studio_env
    items = client.get('/api/loras?model_id=wan21-fun-inp-14b-fp8').json()['video_loras']
    assert items and all('wan21-fun-inp-14b-fp8' in d.get('compatible_models',[]) for d in items)
    ids = {d['id'] for d in items}
    assert {'lora-wan21-fun-14b-mps','lora-wan21-fun-14b-hps21'} <= ids
    assert not ids.intersection({'lora-wan21-14b-fusionx-i2v','lora-wan21-14b-lightx2v-i2v',
                                 'lora-wan21-14b-unianimate-char','lora-wan21-14b-moviigen-cinema',
                                 'lora-wan21-14b-detailz','lora-wan21-14b-snapshot-real'})
