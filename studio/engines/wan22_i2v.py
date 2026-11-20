"""Experimental Wan 2.2 I2V A14B two-expert Comfy recipe (no real GPU validation)."""
from __future__ import annotations
import json
import math
import os
import shutil
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from studio.config import DATA_ROOT, VIDEOS_OUT_DIR
from studio.engines import ltx_gguf_backend as common
from studio.engines.ffmpeg_utils import probe_media_info, verify_image_file
from studio.engines.motion_framing import fit_reference_image
from studio.engines.motion_transfer import _infer
from studio.engines.wan22_i2v_manifest import files_for, recipe_files
from studio.memory_manager import memory_manager, serialized_gpu_operation

WEIGHTS = DATA_ROOT/'models'/'wan22-i2v-a14b'


def make_graph(*, quant, names, image_name, prompt, negative, width, height, frames, seed, steps, cfg, recipe='i2v', mode='i2v', end_image_name=None, speed='none'):
    recipe_files(quant, recipe, speed)
    if speed != 'none' and (mode != 'flf2v' or steps != 4 or cfg != 1.0):
        raise ValueError('WAN22_LORA_CONFIG: LightX2V solo inicio-final, 4 pasos y CFG 1.')
    if mode not in ('t2v','i2v','flf2v') or (recipe == 'i2v' and mode != 'i2v'): raise ValueError('WAN22_MODE_INVALID')
    if (mode in ('i2v','flf2v') and not image_name) or (mode == 'flf2v' and not end_image_name): raise ValueError('WAN22_IMAGE_REQUIRED')
    if (mode == 't2v' and (image_name or end_image_name)) or (mode == 'i2v' and end_image_name): raise ValueError('WAN22_IMAGE_MODE_MISMATCH')
    if not prompt.strip() or len(prompt) > 2000 or len(negative) > 2000: raise ValueError('WAN22_PROMPT_INVALID')
    if width < 256 or height < 256 or width % 32 or height % 32 or width * height > 1280*720 or max(width,height)>1280:
        raise ValueError('WAN22_SIZE_INVALID: múltiplos de 32, área máxima 1280×720.')
    if frames < 5 or frames > 81 or (frames-1)%4 or not 4 <= steps <= 60 or not math.isfinite(cfg) or not 1 <= cfg <= 10 or not 0 <= seed < 2**32:
        raise ValueError('WAN22_CONFIG_INVALID: frames 4n+1 entre 5 y 81; pasos 4–60.')
    for value in [*(v for v in (image_name,end_image_name) if v), *(names.get(k, '') for k in (('high','low','umt5','vae','lora_high','lora_low') if speed != 'none' else ('high','low','umt5','vae')))]:
        if not isinstance(value, str) or not value or '/' in value or chr(92) in value or value in ('.','..'): raise ValueError('WAN22_FILENAME_INVALID')
    def n(kind, **inputs): return {'class_type':kind,'inputs':inputs}
    loader = 'UnetLoaderGGUF' if quant.startswith('q') else 'UNETLoader'
    def load(name): return n(loader, unet_name=name, **({} if quant.startswith('q') else {'weight_dtype':'default'}))
    split = steps // 2
    sampler = {'steps':steps, 'cfg':cfg, 'sampler_name':'euler', 'scheduler':'simple', 'positive':['8',0], 'negative':['8',1], 'noise_seed':seed}
    graph = {
        '1':load(names['high']), '2':load(names['low']),
        '3':n('CLIPLoader', clip_name=names['umt5'], type='wan', device='default'),
        '4':n('VAELoader', vae_name=names['vae']),
        '6':n('CLIPTextEncode', clip=['3',0], text=prompt),
        '7':n('CLIPTextEncode', clip=['3',0], text=negative),
        '8':n('WanFunInpaintToVideo' if recipe == 'fun' else 'WanImageToVideo', positive=['6',0], negative=['7',0], vae=['4',0], width=width, height=height, length=frames, batch_size=1),
        '9':n('ModelSamplingSD3', model=['1',0], shift=8.0 if speed == 'lightx2v' else 5.0),
        '10':n('ModelSamplingSD3', model=['2',0], shift=8.0 if speed == 'lightx2v' else 5.0),
        '11':n('KSamplerAdvanced', model=['9',0], add_noise='enable', start_at_step=0, end_at_step=split, return_with_leftover_noise='enable', latent_image=['8',2], **sampler),
        '12':n('KSamplerAdvanced', model=['10',0], add_noise='disable', start_at_step=split, end_at_step=steps, return_with_leftover_noise='disable', latent_image=['11',0], **sampler),
        '13':n('VAEDecode', samples=['12',0], vae=['4',0]),
        '14':n('CreateVideo', images=['13',0], fps=16.0),
        '15':n('SaveVideo', video=['14',0], filename_prefix='studio_wan22_i2v', format='auto', **{'format.codec':'auto'}),
    }
    if speed == 'lightx2v':
        graph['17'] = n('LoraLoaderModelOnly', model=['1',0], lora_name=names['lora_high'], strength_model=1.0)
        graph['18'] = n('LoraLoaderModelOnly', model=['2',0], lora_name=names['lora_low'], strength_model=1.0)
        graph['9']['inputs']['model'] = ['17',0]
        graph['10']['inputs']['model'] = ['18',0]
    if image_name:
        graph['5'] = n('LoadImage', image=image_name)
        graph['8']['inputs']['start_image'] = ['5',0]
    if end_image_name:
        graph['16'] = n('LoadImage', image=end_image_name)
        graph['8']['inputs']['end_image'] = ['16',0]
    return graph


def _prepare(quant, job, recipe='i2v', speed='none'):
    files = recipe_files(quant, recipe, speed); WEIGHTS.mkdir(parents=True, exist_ok=True)
    for _, _, _, filename, _, _ in files:
        if (WEIGHTS/'ltx'/filename).is_symlink(): raise RuntimeError('WAN22_FILE_CONFLICT: archivo de modelo es un enlace.')
    missing = sum(size for _, _, _, name, size, _ in files if not (WEIGHTS/'ltx'/name).is_file())
    reserve = (2 if (common.ROOT/'ready.json').is_file() else 10)*1024**3
    if shutil.disk_usage(WEIGHTS).free < missing + reserve:
        raise RuntimeError(f'WAN22_DISK_LOW: faltan {missing/1e9:.1f} GB más {reserve/1024**3:.0f} GiB de margen.')
    common._install(job)
    env = common._setup_env(); env['PYTHONPATH'] = str(Path(__file__).resolve().parents[2])
    from studio.catalog import catalog_manager
    token = catalog_manager.get_hf_token()
    if token: env['HF_TOKEN'] = token
    common._command([sys.executable,'-m','studio.engines.wan22_i2v_download',str(WEIGHTS),quant,recipe,speed],job,
                    stage='Descargando y verificando los dos expertos Wan, UMT5 y VAE',timeout=28800,env=env)
    names = {}
    for role, _, _, filename, _, _ in files:
        src = WEIGHTS/'ltx'/filename
        folder = {'high':'diffusion_models','low':'diffusion_models','umt5':'text_encoders','vae':'vae','lora_high':'loras','lora_low':'loras'}[role]
        dst = common.ROOT/'comfy/models'/folder/src.name; dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.is_symlink():
            if dst.resolve() != src.resolve(): raise RuntimeError('WAN22_LINK_CONFLICT: enlace ajeno.')
        elif dst.exists(): raise RuntimeError('WAN22_LINK_CONFLICT: archivo existente no se sobrescribe.')
        else: dst.symlink_to(src.resolve())
        names[role] = src.name
    return names


@serialized_gpu_operation(slots=('video',), managed=True)
def generate(*, quant, image_path: Path | None, prompt: str, negative: str, width: int, height: int,
             frames: int, steps: int, cfg: float, seed: int, job: Any, recipe='i2v', mode='i2v', end_image_path: Path | None=None, speed='none'):
    started = time.monotonic(); files = recipe_files(quant, recipe, speed)
    make_graph(quant=quant,names={x[0]:'placeholder' for x in files},image_name='reference.png' if image_path else None,
               end_image_name='end.png' if end_image_path else None, recipe=recipe, mode=mode, prompt=prompt,negative=negative,
               width=width,height=height,frames=frames,seed=seed,steps=steps,cfg=cfg,speed=speed)
    for path in (image_path, end_image_path):
        if path is None: continue
        if not path.is_file() or path.stat().st_size > 50*1024**2: raise ValueError('WAN22_IMAGE_INVALID')
        w,h = verify_image_file(path)
        if w*h > 32_000_000: raise ValueError('WAN22_IMAGE_TOO_LARGE')
    gpu = common.query_gpu_info()
    if not gpu.get('has_gpu') or not gpu.get('bf16_supported'): raise RuntimeError('WAN22_GPU_UNSUPPORTED: GPU NVIDIA CUDA BF16 requerida.')
    job.log(f'Wan 2.2 {recipe} A14B · {mode}: dos expertos, sin audio; no probado con pesos/GPU. Más resolución y pasos aumentan coste y memoria.',level='warning')
    if quant == 'nvfp4': job.log('NVFP4 mixto comunitario: sin FP4 nativo en L4; compatibilidad y velocidad no comprobadas.',level='warning')
    # Keep only the server alive; after each job reset its cached weights.
    job.comfy_keep_loaded = True
    common._switch_to_idle(job)
    ram = {'q2':18,'q3s':21,'q3':22,'q4s':25,'q4_0':25,'q4_1':26,'q4':27,'q5s':28,'q5_0':28,'q5':29,'q5_1':30,'q6':32,'q8':39,'fp8':38,'nvfp4':29}[quant]
    if speed != 'none': ram += 3  # conservative planning for the two LoRA files; not measured
    memory_manager.ensure_capacity(ram_gb=ram,vram_gb=12,stage='Wan A14B: presupuesto preventivo no medido',job=job)
    names = _prepare(quant, job) if recipe == 'i2v' and speed == 'none' else (_prepare(quant, job, recipe) if speed == 'none' else _prepare(quant, job, recipe, speed))
    memory_manager.ensure_capacity(ram_gb=ram,vram_gb=12,stage='Wan A14B: antes de cargar',job=job)
    common._comfy_start_admission(job,stage='Inicio Wan A14B',planned_ram=6,planned_vram=3)
    session = common._open_resident(job,names,f'wan22-{recipe}-{quant}')
    success = False
    try:
        if image_path: fit_reference_image(image_path,session.input_dir/'reference.png',width,height)
        if end_image_path: fit_reference_image(end_image_path,session.input_dir/'end.png',width,height)
        graph = make_graph(quant=quant,names=names,image_name='reference.png' if image_path else None,
                           end_image_name='end.png' if end_image_path else None,recipe=recipe,mode=mode,prompt=prompt,negative=negative,
                           width=width,height=height,frames=frames,steps=steps,cfg=cfg,seed=seed,speed=speed)
        result = _infer(session,graph,job,prefix='studio_wan22_i2v')
        info = probe_media_info(result)
        if not info.get('has_video') or info.get('width') != width or info.get('height') != height or abs((info.get('duration_seconds') or 0)-frames/16)>.3:
            raise RuntimeError('WAN22_OUTPUT_INVALID: dimensiones o duración inesperadas.')
        job.check_cancelled(); VIDEOS_OUT_DIR.mkdir(parents=True,exist_ok=True)
        name = f'wan22_i2v_{quant}_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}.mp4'
        dest = VIDEOS_OUT_DIR/name; meta_path=dest.with_suffix('.json'); created=False
        meta={'id':dest.stem,'filename':name,'rel_path':f'outputs/videos/{name}','media_url':f'/media/outputs/videos/{name}',
              'type':'video','model_id':f'wan22-{recipe}-a14b-{quant}-experimental','model_name':f'Wan 2.2 {"Fun InP" if recipe == "fun" else "I2V"} A14B · {quant.upper()} experimental', 'video_submode':mode,
              'width':width,'height':height,'fps':16,'duration_seconds':info['duration_seconds'],'steps':steps,'guidance_scale':cfg,'seed':seed,
              'lora_speed':speed,'has_audio':False,'elapsed_seconds':round(time.monotonic()-started,1)}
        try:
            with dest.open('xb') as out, result.open('rb') as src:
                created=True
                while chunk := src.read(1024*1024): job.check_cancelled(); out.write(chunk)
                job.check_cancelled()
            meta_path.write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf8')
        except BaseException:
            if created: dest.unlink(missing_ok=True); meta_path.unlink(missing_ok=True)
            raise
        success=True
        return meta
    finally:
        common._finish_session(session,job,success)
        if success:
            # On an unconfirmed reset the shared manager closes the unsafe server.
            common._switch_to_idle(job)
