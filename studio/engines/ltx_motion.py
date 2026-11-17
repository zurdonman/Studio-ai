"""LTX 2.3 Union IC-LoRA using a Canny-like edge video (experimental)."""
from __future__ import annotations
import json
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
from studio.engines.ffmpeg_utils import get_ffmpeg_bin, probe_media_info, verify_image_file
from studio.engines.ltx_gguf_workflow import make_graph
from studio.engines.motion_transfer import _infer
from studio.engines.motion_framing import fit_reference_image, fit_video_filter
from studio.memory_manager import memory_manager, serialized_gpu_operation

UNION = ('union', 'Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control', '20a580c8606c06a0988d40375a61309e0fe5893d',
         'ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors', 654465352,
         'a1b888a87f661d27f08b394ae559e8e1050be33900bcc36a5cdf659e48f88d18')
WEIGHTS = DATA_ROOT / 'models' / 'ltx-motion'


def add_union_graph(g, guide_name, lora_name, strength=1.0):
    if any(not x or '/' in x or '\\' in x or x in ('.','..') for x in (guide_name,lora_name)):
        raise ValueError('LTX_MOTION_FILENAME_INVALID')
    if not 0 <= strength <= 2: raise ValueError('LTX_MOTION_STRENGTH_INVALID')
    def n(kind, **kw): return {'class_type':kind,'inputs':kw}
    g['30'] = n('LoraLoaderModelOnly', model=['1',0], lora_name=lora_name, strength_model=1.0)
    g['31'] = n('GetICLoRAParameters', iclora_model=['30',0])
    g['32'] = n('LoadVideo', file=guide_name)
    g['33'] = n('GetVideoComponents', video=['32',0])
    g['34'] = n('LTXVAddGuide', positive=['7',0], negative=['7',1], vae=['3',0],
                latent=g['10']['inputs']['video_latent'], image=['33',0], frame_idx=0, strength=strength, iclora_parameters=['31',0])
    g['10']['inputs']['video_latent'] = ['34',2]
    g['11']['inputs'].update(model=['30',0], positive=['34',0], negative=['34',1])
    g['35'] = n('LTXVCropGuides', positive=['34',0], negative=['34',1], latent=['16',0])
    g['17']['inputs']['samples'] = ['35',2]
    g['20']['inputs']['filename_prefix'] = 'studio_ltx_motion'
    # Native dynamic-combo encoding in the newer pinned runtime.
    g['20']['inputs'].pop('codec', None); g['20']['inputs']['format.codec'] = 'auto'
    return g


def _prepare_lora(job):
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    path = WEIGHTS / 'ltx' / UNION[3]
    if not path.exists() and shutil.disk_usage(WEIGHTS).free < UNION[4] + 1024**3:
        raise RuntimeError('LTX_MOTION_DISK_LOW: falta espacio para Union Control')
    env = os.environ.copy()
    env['PYTHONPATH'] = str(Path(__file__).resolve().parents[2]) + os.pathsep + env.get('PYTHONPATH','')
    common._command([sys.executable,'-m','studio.engines.ltx_motion_download',str(WEIGHTS)],job,
                    stage='Descargando y verificando Union Control',timeout=1800,env=env)
    dest = common.ROOT / 'comfy/models/loras' / path.name
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.is_symlink():
        if dest.resolve() != path.resolve(): raise RuntimeError('LTX_MOTION_LINK_CONFLICT')
    elif dest.exists(): raise RuntimeError('LTX_MOTION_LINK_CONFLICT')
    else: dest.symlink_to(path.resolve())
    return dest.name


@serialized_gpu_operation(slots=('video',), managed=True)
def generate(*, image_path:Path, video_path:Path, prompt, negative, width, height, frames, seed, strength, job:Any):
    start = time.monotonic()
    if width % 64 or height % 64 or min(width,height) < 256 or width*height > 768*512 or max(width,height)>768:
        raise ValueError('LTX_MOTION_SIZE_INVALID: múltiplos de 64 y máximo 768×512 de área.')
    if frames not in (25,49,81) or not 0 <= strength <= 2 or not prompt.strip() or len(prompt)>2000:
        raise ValueError('LTX_MOTION_CONFIG_INVALID')
    if image_path.stat().st_size > 50*1024**2 or video_path.stat().st_size > 200*1024**2:
        raise ValueError('LTX_MOTION_FILE_TOO_LARGE')
    w,h = verify_image_file(image_path)
    if w*h > 32_000_000: raise ValueError('LTX_MOTION_IMAGE_TOO_LARGE')
    info = probe_media_info(video_path)
    if not info['has_video'] or info['duration_seconds'] < frames / 24 - .01:
        raise ValueError('LTX_MOTION_VIDEO_TOO_SHORT: vídeo guía demasiado corto')
    if not common.query_gpu_info().get('has_gpu'): raise RuntimeError('LTX_MOTION_GPU_REQUIRED')
    job.log('Union Control experimental: guía de bordes, no poses ni clonación de identidad. Sin validación GPU.',level='warning')
    # This graph is not compatible with resident result cache; close only after obtaining GPU lock.
    if memory_manager.get_slot('video'): memory_manager.unload_slot('video')
    names = common._prepare_gguf(job, extra_ram=2, extra_vram=1)
    lora = _prepare_lora(job)
    session = common._ResidentComfy(job,{})
    try:
        fit_reference_image(image_path, session.input_dir/'frame.png', width, height)
        guide=session.input_dir/'guide.mp4'
        common._command([get_ffmpeg_bin(),'-hide_banner','-loglevel','error','-y','-i',str(video_path),'-map','0:v:0','-an',
                         '-vf',f'fps=24,{fit_video_filter(width, height)},format=gray,edgedetect=low=0.1:high=0.3:mode=wires,format=yuv420p',
                         '-frames:v',str(frames),'-c:v','libx264','-pix_fmt','yuv420p',str(guide)],job,
                        stage='Extrayendo bordes del vídeo guía',timeout=600,guard_label='FFmpeg control LTX')
        graph=make_graph(prompt=prompt,negative_prompt=negative,width=width,height=height,frames=frames,fps=24,seed=seed,
                         image_name='frame.png',model=names['dit'],text_encoder=names['text_encoder'],connectors=names['connectors'],
                         video_vae=names['video_vae'],audio_vae=names['audio_vae'])
        add_union_graph(graph,'guide.mp4',lora,strength)
        output=_infer(session,graph,job,prefix='studio_ltx_motion')
        media=probe_media_info(output)
        if not media['has_video'] or not media['has_audio']: raise RuntimeError('LTX_MOTION_OUTPUT_INVALID: falta vídeo o audio')
        job.check_cancelled(); VIDEOS_OUT_DIR.mkdir(parents=True,exist_ok=True)
        name=f'ltx_motion_{datetime.now().strftime("%Y%m%d_%H%M%S")}_{uuid.uuid4().hex[:8]}.mp4'
        dest=VIDEOS_OUT_DIR/name; meta_path=dest.with_suffix('.json')
        meta={'id':dest.stem,'filename':name,'rel_path':f'outputs/videos/{name}','media_url':f'/media/outputs/videos/{name}',
              'type':'video','model_id':'ltx23-union-control-q4','model_name':'LTX 2.3 GGUF Q4 · Union Control bordes',
              'video_submode':'Control por bordes','width':media['width'],'height':media['height'],'fps':24,
              'duration_seconds':media['duration_seconds'],'steps':8,'guidance_scale':1.0,'has_audio':True,
              'elapsed_seconds':round(time.monotonic()-start,1)}
        if meta_path.exists(): raise RuntimeError('LTX_MOTION_OUTPUT_COLLISION')
        created = False
        try:
            with dest.open('xb') as out:
                created = True
                with output.open('rb') as src: shutil.copyfileobj(src,out)
            meta_path.write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf8')
        except BaseException:
            if created:
                dest.unlink(missing_ok=True); meta_path.unlink(missing_ok=True)
            raise
        return meta
    except Exception:
        session.guard.raise_if_stopped();raise
    finally: session.unload()
