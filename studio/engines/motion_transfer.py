"""Experimental single-window character animation using private, pinned Comfy.

No GPU validation yet. Never expose an arbitrary graph or the private server.
"""
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

from PIL import Image, ImageOps
from studio.config import DATA_ROOT, VIDEOS_OUT_DIR
from studio.engines import ltx_gguf_backend as common
from studio.engines.ffmpeg_utils import get_ffmpeg_bin, probe_media_info, verify_image_file
from studio.engines.motion_manifest import files_for
from studio.memory_manager import memory_manager, serialized_gpu_operation

WEIGHTS = DATA_ROOT / 'models' / 'motion-transfer'
FOLDERS = {'dit': 'diffusion_models', 'umt5': 'text_encoders', 'vision': 'clip_vision', 'vae': 'vae', 'sam': 'checkpoints'}


def make_graph(*, engine, quant, image_name, video_name, names, prompt, negative, width, height, frames, seed, steps, cfg, person='human', strength=1.0):
    files_for(engine, quant)
    if not prompt.strip() or len(prompt) > 2000 or len(negative) > 2000:
        raise ValueError('MOTION_PROMPT_INVALID')
    if width < 256 or height < 256 or width % 32 or height % 32 or width * height > 768 * 512 or max(width, height) > 768:
        raise ValueError('MOTION_SIZE_INVALID: múltiplos de 32; máximo 768×512 de área y 768 px de lado.')
    if frames not in (25, 49, 81) or not 1 <= steps <= 50 or not 0 <= cfg <= 10 or not 0 <= strength <= 2:
        raise ValueError('MOTION_CONFIG_INVALID')
    if not 0 <= seed <= 2**32 - 1 or not person.strip() or len(person) > 120:
        raise ValueError('MOTION_CONFIG_INVALID')
    for value in [image_name, video_name, *names.values()]:
        if not value or '/' in value or '\\' in value or value in ('.', '..'):
            raise ValueError('MOTION_FILENAME_INVALID')
    def n(kind, **kw): return {'class_type': kind, 'inputs': kw}
    gguf = quant.startswith('q')
    g = {
      '1': n('StudioWanAnimate2GGUF' if engine == 'wan' and gguf else 'UnetLoaderGGUF' if gguf else 'UNETLoader',
             **({'unet_name': names['dit']} if gguf else {'unet_name': names['dit'], 'weight_dtype':'default'})),
      '2': n('CLIPLoader', clip_name=names['umt5'], type='wan', device='default'),
      '3': n('VAELoader', vae_name=names['vae']),
      '4': n('CLIPVisionLoader', clip_name=names['vision']),
      '5': n('LoadImage', image=image_name),
      '6': n('LoadVideo', file=video_name),
      '7': n('GetVideoComponents', video=['6', 0]),
      '8': n('CLIPVisionEncode', clip_vision=['4', 0], image=['5', 0], crop='none'),
      '9': n('CLIPTextEncode', text=prompt, clip=['2', 0]),
      '10': n('CLIPTextEncode', text=negative, clip=['2', 0]),
      '11': n('ModelSamplingSD3', model=['1', 0], shift=5.0),
      '12': n('BasicScheduler', model=['11', 0], scheduler='simple', steps=steps, denoise=1.0),
      '13': n('KSamplerSelect', sampler_name='euler'),
      '15': n('SamplerCustom', model=['11', 0], add_noise=True, noise_seed=seed, cfg=cfg,
              positive=['14', 0], negative=['14', 1], sampler=['13', 0], sigmas=['12', 0], latent_image=['14', 2]),
      '17': n('VAEDecode', samples=['15', 1] if engine == 'scail' else ['16', 0], vae=['3', 0]),
      '18': n('CreateVideo', images=['17', 0], fps=24.0),
      '19': n('SaveVideo', video=['18', 0], filename_prefix='studio_motion', format='auto', **{'format.codec':'auto'}),
    }
    if engine == 'wan':
        g['20'] = n('ImageFromBatch', image=['7', 0], batch_index=0, length=1)
        g['21'] = n('CLIPVisionEncode', clip_vision=['4', 0], image=['20', 0], crop='none')
        g['14'] = n('WanAnimate2ToVideo', positive=['9', 0], negative=['10', 0], vae=['3', 0],
                    reference_image=['5', 0], pose_video=['7', 0], clip_vision_output=['8', 0],
                    clip_vision_output_pose=['21', 0], width=width, height=height, length=frames, batch_size=1,
                    video_frame_offset=0, pose_strength=strength, pose_start_percent=0.0, pose_end_percent=1.0, reference_image_strength=1.0)
        g['16'] = n('TrimVideoLatent', samples=['15', 0], trim_amount=['14', 3])
    else:
        # Native SAM3 tracks independently on reference and driving image, then
        # assigns identities left-to-right. Not a guarantee when subjects cross.
        g['20'] = n('CheckpointLoaderSimple', ckpt_name=names['sam'])
        g['21'] = n('CLIPTextEncode', clip=['20', 1], text=person)
        g['22'] = n('SAM3_VideoTrack', model=['20', 0], images=['7', 0], conditioning=['21', 0], detection_threshold=.5, max_objects=4, detect_interval=1)
        g['23'] = n('SAM3_VideoTrack', model=['20', 0], images=['5', 0], conditioning=['21', 0], detection_threshold=.5, max_objects=4, detect_interval=1)
        g['24'] = n('SCAIL2ColoredMask', driving_track_data=['22', 0], ref_track_data=['23', 0], object_indices='', sort_by='left_to_right', replacement_mode=False)
        g['14'] = n('WanSCAILToVideo', positive=['9', 0], negative=['10', 0], vae=['3', 0], width=width, height=height, length=frames, batch_size=1,
                    pose_video=['7', 0], pose_video_mask=['24', 0], reference_image=['5', 0], reference_image_mask=['24', 1],
                    clip_vision_output=['8', 0], pose_strength=strength, pose_start=0.0, pose_end=1.0, video_frame_offset=0, previous_frame_count=5, replacement_mode=False)
    return g


def _install_motion_plugin():
    src = Path(__file__).with_name('comfy_motion_node') / '__init__.py'
    dest = common.ROOT / 'comfy/custom_nodes/StudioMotion'
    if dest.is_symlink() or (dest.exists() and not dest.is_dir()):
        raise RuntimeError('MOTION_PLUGIN_CONFLICT: directorio no gestionado')
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / '__init__.py'
    if target.is_symlink() or (target.exists() and target.read_bytes() != src.read_bytes()):
        raise RuntimeError('MOTION_PLUGIN_CONFLICT: nodo diferente, no se sobrescribe')
    if not target.exists(): shutil.copyfile(src, target)


def _prepare(engine, quant, job):
    components = files_for(engine, quant)
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    missing = sum(size for _, _, _, filename, size, _ in components if not (WEIGHTS / 'ltx' / filename).is_file())
    runtime_margin = (2 if (common.ROOT / 'ready.json').is_file() else 10) * 1024**3
    if shutil.disk_usage(WEIGHTS).free < missing + runtime_margin:
        raise RuntimeError(f'MOTION_DISK_LOW: se necesitan {missing / 1e9:.1f} GB de archivos faltantes y margen de runtime de {runtime_margin / 1024**3:.0f} GiB.')
    common._install(job)
    env = os.environ.copy()
    env['PYTHONPATH'] = str(Path(__file__).resolve().parents[2]) + os.pathsep + env.get('PYTHONPATH', '')
    common._command([sys.executable, '-m', 'studio.engines.motion_download', str(WEIGHTS), engine, quant], job,
                    stage='Descargando y verificando modelos de movimiento', timeout=7200, env=env)
    names = {}
    for role, _, _, filename, _, _ in components:
        src = WEIGHTS / 'ltx' / filename
        dest = common.ROOT / 'comfy/models' / FOLDERS[role] / src.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.is_symlink():
            if dest.resolve() != src.resolve(): raise RuntimeError('MOTION_LINK_CONFLICT: enlace de modelo ajeno')
        elif dest.exists(): raise RuntimeError('MOTION_LINK_CONFLICT: no se sobrescriben modelos existentes')
        else: dest.symlink_to(src.resolve())
        names[role] = src.name
    if engine == 'wan' and quant.startswith('q'):
        _install_motion_plugin()
        job.studio_motion_plugin = True
    return names


def _infer(session, graph, job, prefix="studio_motion"):
    info = common.requests.get(session.base + '/object_info', timeout=30); info.raise_for_status()
    missing = {n['class_type'] for n in graph.values()} - set(info.json())
    if missing: raise RuntimeError('MOTION_RUNTIME_MISSING_NODES: ' + ', '.join(sorted(missing)))
    reply = common.requests.post(session.base + '/prompt', json={'prompt': graph}, timeout=30)
    if not reply.ok: raise RuntimeError('MOTION_GRAPH_REJECTED: ' + reply.text[:1500])
    ident = reply.json()['prompt_id']
    deadline = time.monotonic() + 7200
    while time.monotonic() < deadline:
        job.check_cancelled(); session.guard.raise_if_stopped()
        if not session.alive(): raise RuntimeError('MOTION_ENGINE_EXITED: proceso Comfy detenido')
        response = common.requests.get(session.base + '/history/' + ident, timeout=10); response.raise_for_status()
        result = response.json().get(ident)
        if result:
            status = result.get('status') or {}
            if status.get('status_str') == 'error':
                detail = next((str(m[1].get('exception_message', '')) for m in status.get('messages', []) if len(m) > 1 and m[0] == 'execution_error' and isinstance(m[1], dict)), 'Error desconocido')
                raise RuntimeError('MOTION_INFERENCE_FAILED: ' + detail[:1000])
            if status.get('completed'):
                paths = list(session.output_dir.glob(prefix + '_*.mp4'))
                if len(paths) != 1: raise RuntimeError('MOTION_OUTPUT_MISSING')
                return paths[0]
        job.heartbeat(step='Movimiento: preparación e inferencia; progreso no disponible')
        time.sleep(1)
    raise TimeoutError('MOTION_TIMEOUT: máximo dos horas')


@serialized_gpu_operation(slots=('video',), managed=True)
def generate(*, engine, quant, image_path: Path, video_path: Path, prompt, negative, width, height, frames, seed, steps, cfg, person, strength, job: Any):
    started = time.monotonic()
    files_for(engine, quant)
    if image_path.stat().st_size > 50 * 1024**2: raise ValueError('MOTION_IMAGE_TOO_LARGE')
    w, h = verify_image_file(image_path)
    if w * h > 32_000_000: raise ValueError('MOTION_IMAGE_TOO_LARGE: máximo 32 megapíxeles')
    if not video_path.is_file() or video_path.stat().st_size > 200 * 1024**2: raise ValueError('MOTION_VIDEO_TOO_LARGE')
    info = probe_media_info(video_path)
    if not info['has_video'] or not info['duration_seconds'] or info['duration_seconds'] < frames / 24 - .01:
        raise ValueError('MOTION_VIDEO_TOO_SHORT: el clip debe cubrir los fotogramas solicitados a 24 fps')
    # Validate config before downloads/unloading any resident process.
    placeholder = {item[0]:'placeholder' for item in files_for(engine, quant)}
    make_graph(engine=engine, quant=quant, image_name='reference.png', video_name='drive.mp4', names=placeholder,
               prompt=prompt, negative=negative, width=width, height=height, frames=frames, seed=seed, steps=steps, cfg=cfg, person=person, strength=strength)
    if not common.query_gpu_info().get('has_gpu'): raise RuntimeError('MOTION_GPU_REQUIRED: se necesita GPU CUDA')
    job.log('Movimiento experimental sin validación GPU: solo un segmento, sin audio. La memoria real se vigila.', level='warning')
    if memory_manager.get_slot('video'): memory_manager.unload_slot('video')
    common._comfy_start_admission(job, stage='Inicio movimiento', planned_ram=6, planned_vram=3)
    names = _prepare(engine, quant, job)
    session = common._ResidentComfy(job, {})
    try:
        with Image.open(image_path) as im:
            im = ImageOps.exif_transpose(im).convert('RGB'); im.thumbnail((1024, 1024)); im.save(session.input_dir / 'reference.png')
        source = session.input_dir / 'drive.mp4'
        common._command([get_ffmpeg_bin(), '-hide_banner','-loglevel','error','-y','-i',str(video_path), '-map','0:v:0', '-an',
                         '-vf',f'fps=24,scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},setsar=1',
                         '-frames:v',str(frames), '-c:v','libx264','-pix_fmt','yuv420p',str(source)], job,
                        stage='Preparando segmento guía 24 fps', timeout=600, guard_label='FFmpeg movimiento')
        graph = make_graph(engine=engine, quant=quant, image_name='reference.png', video_name='drive.mp4', names=names,
                           prompt=prompt, negative=negative, width=width, height=height, frames=frames, seed=seed, steps=steps, cfg=cfg, person=person, strength=strength)
        output = _infer(session, graph, job)
        output_info = probe_media_info(output)
        if not output_info['has_video']: raise RuntimeError('MOTION_OUTPUT_INVALID')
        job.check_cancelled(); VIDEOS_OUT_DIR.mkdir(parents=True, exist_ok=True)
        name = f'{engine}_motion_{datetime.now().strftime("%Y%m%d_%H%M%S")}_{uuid.uuid4().hex[:8]}.mp4'
        dest = VIDEOS_OUT_DIR / name; meta_path = dest.with_suffix('.json')
        meta = {'id':dest.stem, 'filename':name, 'rel_path':f'outputs/videos/{name}', 'media_url':f'/media/outputs/videos/{name}',
                'type':'video','model_id':f'{engine}-motion-{quant}', 'model_name':f'{engine.upper()} movimiento · {quant}',
                'video_submode':'Transferencia de movimiento', 'width':output_info['width'], 'height':output_info['height'],
                'fps':24, 'duration_seconds':output_info['duration_seconds'], 'steps':steps, 'guidance_scale':cfg,
                'elapsed_seconds':round(time.monotonic()-started, 1), 'has_audio':False}
        if meta_path.exists(): raise RuntimeError('MOTION_OUTPUT_COLLISION')
        created = False
        try:
            with dest.open('xb') as out:
                created = True
                with output.open('rb') as src: shutil.copyfileobj(src, out)
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf8')
        except BaseException:
            if created:
                dest.unlink(missing_ok=True); meta_path.unlink(missing_ok=True)
            raise
        return meta
    except Exception:
        session.guard.raise_if_stopped(); raise
    finally:
        session.unload()
