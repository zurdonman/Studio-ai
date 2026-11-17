"""Experimental character animation using private, pinned Comfy.

A short Wan Q4 run completed on L4; long continuation and other settings remain experimental.
Never expose an arbitrary graph or the private server.
"""
from __future__ import annotations
import hashlib
import json
import math
import queue
import threading
import os
import re
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
from studio.engines.motion_manifest import files_for
from studio.engines.motion_framing import fit_reference_image, fit_video_filter
from studio.memory_manager import memory_manager, serialized_gpu_operation

WEIGHTS = DATA_ROOT / 'models' / 'motion-transfer'
FOLDERS = {'dit': 'diffusion_models', 'umt5': 'text_encoders', 'vision': 'clip_vision', 'vae': 'vae', 'sam': 'checkpoints'}
_MOTION_PLUGIN_PREVIOUS_HASHES = frozenset({'de5729c6cc7536397e6111151a44b578b3ed17c23bcf26f04d0768aa26d6d832'})


def make_graph(*, engine, quant, image_name, video_name, names, prompt, negative, width, height, frames, seed, steps, cfg, person='human', strength=1.0, continuation_name=None, prefix='studio_motion'):
    files_for(engine, quant)
    if not prompt.strip() or len(prompt) > 2000 or len(negative) > 2000:
        raise ValueError('MOTION_PROMPT_INVALID')
    if width < 256 or height < 256 or width % 32 or height % 32 or width * height > 768 * 512 or max(width, height) > 768:
        raise ValueError('MOTION_SIZE_INVALID: múltiplos de 32; máximo 768×512 de área y 768 px de lado.')
    if not isinstance(frames, int) or frames < 5 or frames > 81 or (frames - 1) % 4 or not 1 <= steps <= 50 or not 0 <= cfg <= 10 or not 0 <= strength <= 2:
        raise ValueError('MOTION_CONFIG_INVALID')
    if not 0 <= seed <= 2**32 - 1 or not person.strip() or len(person) > 120:
        raise ValueError('MOTION_CONFIG_INVALID')
    for value in [image_name, video_name, *names.values(), *([continuation_name] if continuation_name else []), prefix]:
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
      '19': n('SaveVideo', video=['18', 0], filename_prefix=prefix, format='auto', **{'format.codec':'auto'}),
    }
    if engine == 'wan':
        g['20'] = n('ImageFromBatch', image=['7', 0], batch_index=0, length=1)
        g['21'] = n('CLIPVisionEncode', clip_vision=['4', 0], image=['20', 0], crop='none')
        g['14'] = n('WanAnimate2ToVideo', positive=['9', 0], negative=['10', 0], vae=['3', 0],
                    reference_image=['5', 0], pose_video=['7', 0], clip_vision_output=['8', 0],
                    clip_vision_output_pose=['21', 0], width=width, height=height, length=frames, batch_size=1,
                    video_frame_offset=0, pose_strength=strength, pose_start_percent=0.0, pose_end_percent=1.0, reference_image_strength=1.0)
        g['16'] = n('TrimVideoLatent', samples=['15', 0], trim_amount=['14', 3])
        if continuation_name:
            g['23'] = n('LoadVideo', file=continuation_name)
            g['24'] = n('GetVideoComponents', video=['23', 0])
            g['14']['inputs']['continue_motion'] = ['24', 0]
    else:
        if continuation_name: raise ValueError('MOTION_CONTINUATION_UNSUPPORTED')
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
    if target.is_symlink() or (target.exists() and not target.is_file()):
        raise RuntimeError('MOTION_PLUGIN_CONFLICT: nodo no gestionado')
    desired = src.read_bytes()
    # Upgrade only the exact previous Studio adapter; never overwrite a user node.
    if target.exists():
        current = target.read_bytes()
        if current == desired: return
        if hashlib.sha256(current).hexdigest() not in _MOTION_PLUGIN_PREVIOUS_HASHES:
            raise RuntimeError('MOTION_PLUGIN_CONFLICT: nodo diferente, no se sobrescribe')
    temp = dest / f'.studio-motion-{uuid.uuid4().hex}.tmp'
    try:
        temp.write_bytes(desired)
        os.replace(temp, target)
    finally:
        temp.unlink(missing_ok=True)


def _motion_fingerprint(engine, quant):
    try:
        return tuple((p.stat().st_dev, p.stat().st_ino, p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ctime_ns)
                     for p in (WEIGHTS / 'ltx' / item[3] for item in files_for(engine, quant)))
    except OSError:
        return None


def _invalidate_motion_receipts(engine, quant):
    directory = WEIGHTS / '.motion-verified'
    for _, _, _, filename, _, _ in files_for(engine, quant):
        relative = str(Path('ltx') / filename)
        key = hashlib.sha256(relative.encode()).hexdigest()
        (directory / (key + '.json')).unlink(missing_ok=True)


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


def _report_comfy_error(messages, graph, job):
    """Report bounded stack locations, not node inputs, prompts or absolute paths."""
    from studio.diagnostics import redact_secrets

    error = next((item[1] for item in messages if isinstance(item, (list, tuple))
                  and len(item) > 1 and item[0] == 'execution_error' and isinstance(item[1], dict)), {}) if isinstance(messages, list) else {}
    raw_id = str(error.get('node_id', ''))[:80]
    node_id = raw_id if re.fullmatch(r'[A-Za-z0-9_.-]+', raw_id) else '?'
    # Trust our submitted graph, not arbitrary strings returned by the worker.
    node = graph.get(node_id, {})
    node_type = node.get('class_type', '?') if isinstance(node, dict) else '?'
    if not isinstance(node_type, str) or not re.fullmatch(r'[A-Za-z0-9_.-]{1,120}', node_type):
        node_type = '?'
    error_type = str(error.get('exception_type', 'unknown'))[:120]
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', error_type):
        error_type = 'unknown'
    message = redact_secrets(str(error.get('exception_message') or 'Error desconocido')).replace('\n', ' ').replace('\r', ' ')[:1000]
    job.log(f'Comfy error: nodo {node_id} ({node_type}); tipo {error_type}', level='error')
    executed = error.get('executed')
    if isinstance(executed, list):
        known = [str(item) for item in executed if str(item) in graph][:50]
        job.log('Nodos completados antes del fallo: ' + (', '.join(f'{item} ({graph[item].get("class_type", "?")})' for item in known) or 'ninguno'), level='error')
    stack = error.get('traceback', [])
    if isinstance(stack, (list, str)):
        stack_text = stack if isinstance(stack, str) else '\n'.join(x for x in stack if isinstance(x, str))
        frames = re.findall(r'File "([^"\n]+)", line (\d+), in ([\w<>.]+)', stack_text[-50000:])
        for filename, line, function in frames[-30:]:
            basename = filename.replace('\\', '/').rsplit('/', 1)[-1][:120]
            job.log(f'Comfy traceback: {basename}:{line[:10]} en {function[:120]}', level='error')
    return f'MOTION_INFERENCE_FAILED: nodo {node_id} ({node_type}), {error_type}: {message}'


def _infer(session, graph, job, prefix="studio_motion"):
    info = common.requests.get(session.base + '/object_info', timeout=30); info.raise_for_status()
    missing = {n['class_type'] for n in graph.values()} - set(info.json())
    if missing: raise RuntimeError('MOTION_RUNTIME_MISSING_NODES: ' + ', '.join(sorted(missing)))
    client_id = uuid.uuid4().hex
    events = queue.Queue()
    stop, ready = threading.Event(), threading.Event()
    thread = threading.Thread(target=common._events_listener,
                              args=(session.base, client_id, events, stop, ready), daemon=True)
    thread.start(); ready.wait(5)
    try:
        reply = common.requests.post(session.base + '/prompt', json={'prompt': graph, 'client_id': client_id}, timeout=30)
        if not reply.ok:
            from studio.diagnostics import redact_secrets
            raise RuntimeError('MOTION_GRAPH_REJECTED: ' + redact_secrets(reply.text[:1500]))
        ident = reply.json()['prompt_id']
        job.log(f'Comfy aceptó el grafo {ident}; {len(graph)} nodos. Monitorizando ejecución.')
        deadline = time.monotonic() + 7200
        current_node = None
        while time.monotonic() < deadline:
            job.check_cancelled(); session.guard.raise_if_stopped()
            if not session.alive(): raise RuntimeError('MOTION_ENGINE_EXITED: proceso Comfy detenido')
            while not events.empty():
                event = events.get_nowait()
                if not isinstance(event, dict): continue
                data = event.get('data') or {}
                if not isinstance(data, dict) or data.get('prompt_id') not in (None, ident): continue
                if event.get('type') == 'executing' and data.get('node') is not None:
                    node_id = str(data['node'])
                    if node_id in graph and node_id != current_node:
                        current_node = node_id
                        job.log(f'Comfy ejecutando nodo {node_id}: {graph[node_id]["class_type"]}',
                                step=f'Comfy: {graph[node_id]["class_type"]}')
                elif event.get('type') == 'progress':
                    try:
                        value, maximum = float(data['value']), float(data['max'])
                        if math.isfinite(value) and math.isfinite(maximum) and maximum > 0:
                            node_id = str(data.get('node') or current_node or '')
                            node_type = graph.get(node_id, {}).get('class_type', 'procesando')
                            job.heartbeat(step=f'Comfy {node_type}: {value:g}/{maximum:g}')
                    except (KeyError, ValueError, TypeError): pass
            response = common.requests.get(session.base + '/history/' + ident, timeout=10); response.raise_for_status()
            result = response.json().get(ident)
            if result:
                status = result.get('status') or {}
                if status.get('status_str') == 'error':
                    raise RuntimeError(_report_comfy_error(status.get('messages', []), graph, job))
                if status.get('completed'):
                    paths = list(session.output_dir.glob(prefix + '_*.mp4'))
                    if len(paths) != 1: raise RuntimeError('MOTION_OUTPUT_MISSING')
                    if getattr(session, 'keep_loaded', False):
                        try:
                            cleanup = common.requests.get(session.base + '/studio-resident/status', timeout=10)
                            cleanup.raise_for_status()
                            status = cleanup.json()
                            if status.get('version') != 1 or status.get('prompt_id') != ident or status.get('ok') is not True:
                                raise RuntimeError('limpieza no confirmada')
                            deleted = common.requests.post(session.base + '/history', json={'delete': [ident]}, timeout=10)
                            deleted.raise_for_status()
                            if not getattr(session, 'motion_cleanup_failed', False):
                                session.cleanup_ok = True
                            job.log(f"Comfy ha limpiado los temporales en memoria; conserva {status.get('retained_loaders', 0)} cargadores para reutilizar.")
                        except Exception:
                            session.cleanup_ok = False
                            session.motion_cleanup_failed = True
                            job.log('No se confirmó la limpieza de Comfy; se cerrará al finalizar el trabajo.', level='warning')
                    return paths[0]
            if current_node is None:
                job.heartbeat(step='Movimiento: preparando nodos Comfy')
            else:
                job.heartbeat()
            time.sleep(1)
        raise TimeoutError('MOTION_TIMEOUT: máximo dos horas por segmento')
    finally:
        stop.set(); thread.join(timeout=2)


def _duration_frames(seconds):
    try: seconds = float(seconds)
    except (TypeError, ValueError): raise ValueError('MOTION_DURATION_INVALID') from None
    if not math.isfinite(seconds) or not 1/24 <= seconds <= 86400:
        raise ValueError('MOTION_DURATION_INVALID: entre 1/24 de segundo y 24 horas.')
    return max(1, math.floor(seconds * 24 + .5))


def _wan_plan(total_frames):
    if not isinstance(total_frames, int) or not 1 <= total_frames <= 86400 * 24:
        raise ValueError('MOTION_DURATION_INVALID')
    produced = 0
    while produced < total_frames:
        overlap = 1 if produced else 0
        take = min(81 - overlap, total_frames - produced)
        needed = take + overlap
        window = max(5, 1 + 4 * math.ceil((needed - 1) / 4))
        yield {'start': produced - overlap, 'overlap': overlap, 'take': take, 'window': window}
        produced += take


def _generate_wan_segments(session, job, *, total_frames, image_path, video_path, names, quant, prompt, negative, width, height, seed, steps, cfg, person, strength):
    plan = list(_wan_plan(total_frames))
    job.log(f'Wan: {total_frames} fotogramas a 24 fps ({total_frames/24:.3f} s), {len(plan)} segmento(s); '
            'continuación experimental con un fotograma solapado, sin audio.', level='warning')
    work = session.root / 'motion-work'
    work.mkdir(exist_ok=True)
    previous = None
    clips = []
    for index, part in enumerate(plan):
        job.check_cancelled(); session.guard.raise_if_stopped()
        job.log(f'Segmento {index+1}/{len(plan)}: guía desde fotograma {part["start"]}, '
                f'ventana {part["window"]}, salida {part["take"]}, solape {part["overlap"]}.')
        guide_name = f'drive_{index:06d}.mp4'
        guide = session.input_dir / guide_name
        # Decode from the start so frame indices remain deterministic even for VFR and keyframes.
        vf = (f'fps=24,trim=start_frame={part["start"]}:end_frame={part["start"] + part["window"]},'
              f'setpts=PTS-STARTPTS,{fit_video_filter(width, height)}')
        common._command([get_ffmpeg_bin(), '-hide_banner','-loglevel','error','-y','-i',str(video_path),
                         '-map','0:v:0','-an','-vf',vf,'-c:v','libx264','-pix_fmt','yuv420p',str(guide)], job,
                        stage=f'Preparando guía {index+1}/{len(plan)}', timeout=1800, guard_label='FFmpeg movimiento')
        guide_info = probe_media_info(guide)
        # Wan pads at most the window rounding; never accept a missing required frame.
        if not guide_info['has_video'] or guide_info['duration_seconds'] + .015 < (part['take'] + part['overlap']) / 24:
            raise ValueError('MOTION_VIDEO_TOO_SHORT: faltan fotogramas en el vídeo guía convertido a 24 fps.')
        previous_name = None
        if previous is not None:
            previous_name = f'continuation_{index:06d}.mp4'
            shutil.copyfile(previous, session.input_dir / previous_name)
        prefix = f'studio_motion_{index:06d}'
        graph = make_graph(engine='wan', quant=quant, image_name='reference.png', video_name=guide_name, names=names,
                           prompt=prompt, negative=negative, width=width, height=height, frames=part['window'],
                           seed=(seed + index) % (2**32), steps=steps, cfg=cfg, person=person, strength=strength,
                           continuation_name=previous_name, prefix=prefix)
        previous = _infer(session, graph, job, prefix=prefix)
        clip = work / f'piece_{index:06d}.mp4'
        start = part['overlap']; end = start + part['take']
        common._command([get_ffmpeg_bin(),'-hide_banner','-loglevel','error','-y','-i',str(previous),
                         '-vf',f'trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS',
                         '-an','-c:v','libx264','-pix_fmt','yuv420p',str(clip)], job,
                        stage=f'Recortando segmento {index+1}/{len(plan)}',timeout=600,guard_label='FFmpeg movimiento')
        info = probe_media_info(clip)
        if not info['has_video'] or abs(info['duration_seconds'] - part['take']/24) > .02:
            raise RuntimeError('MOTION_SEGMENT_INVALID: el segmento generado no contiene los fotogramas esperados.')
        clips.append(clip)
    listing = work / 'segments.ffconcat'
    listing.write_text('ffconcat version 1.0\n' + ''.join(f"file '{p.name}'\n" for p in clips), encoding='utf8')
    result = work / 'wan_joined.mp4'
    common._command([get_ffmpeg_bin(),'-hide_banner','-loglevel','error','-y','-f','concat','-safe','1',
                     '-i',str(listing),'-map','0:v:0','-an','-c','copy','-movflags','+faststart',str(result)], job,
                    stage='Uniendo segmentos de movimiento',timeout=1800,guard_label='FFmpeg movimiento')
    info = probe_media_info(result)
    if not info['has_video'] or abs(info['duration_seconds'] - total_frames/24) > .04:
        raise RuntimeError('MOTION_OUTPUT_INVALID: duración final incorrecta.')
    return result, len(plan)


@serialized_gpu_operation(slots=('video',), managed=True)
def generate(*, engine, quant, image_path: Path, video_path: Path, prompt, negative, width, height, frames, seed, steps, cfg, person, strength, job: Any, duration_sec=None):
    started = time.monotonic()
    files_for(engine, quant)
    if image_path.stat().st_size > 50 * 1024**2: raise ValueError('MOTION_IMAGE_TOO_LARGE')
    w, h = verify_image_file(image_path)
    if w * h > 32_000_000: raise ValueError('MOTION_IMAGE_TOO_LARGE: máximo 32 megapíxeles')
    if not video_path.is_file() or video_path.stat().st_size > 200 * 1024**2: raise ValueError('MOTION_VIDEO_TOO_LARGE')
    if duration_sec is not None and engine != 'wan':
        raise ValueError('MOTION_DURATION_UNSUPPORTED: duración en segundos disponible solo para Wan.')
    total_frames = _duration_frames(duration_sec) if duration_sec is not None else frames
    info = probe_media_info(video_path)
    if not info['has_video'] or not info['duration_seconds'] or info['duration_seconds'] + .01 < total_frames / 24:
        raise ValueError(f'MOTION_VIDEO_TOO_SHORT: se necesitan {total_frames / 24:.2f} s de vídeo guía a 24 fps.')
    # Validate config before downloads/unloading any resident process.
    placeholder = {item[0]:'placeholder' for item in files_for(engine, quant)}
    make_graph(engine=engine, quant=quant, image_name='reference.png', video_name='drive.mp4', names=placeholder,
               prompt=prompt, negative=negative, width=width, height=height, frames=next(_wan_plan(total_frames))['window'] if engine == 'wan' else frames, seed=seed, steps=steps, cfg=cfg, person=person, strength=strength)
    if not common.query_gpu_info().get('has_gpu'): raise RuntimeError('MOTION_GPU_REQUIRED: se necesita GPU CUDA')
    job.log(f'Movimiento: motor {engine}, variante {quant}, {width}×{height}, {total_frames} fotogramas a 24 fps, '
            f'{steps} pasos, CFG {cfg:g}, semilla {seed}, fuerza {strength:g}.')
    job.log('Movimiento experimental; foto y guía se ajustan sin recortar y pueden tener márgenes grises. '
            'Wan Q4 ha completado una prueba corta en L4; los vídeos largos y otros formatos no están validados. Salida sin audio.', level='warning')
    # Each engine/variant has its own cache identity; never reuse incompatible loaders.
    job.comfy_keep_loaded = True
    model_id = f'{engine}-motion-{quant}'
    resident = common._resident(model_id)
    if resident and (getattr(resident, 'motion_fingerprint', None) != _motion_fingerprint(engine, quant)
                     or getattr(resident, 'options', None) != common.job_options(job)):
        job.log('Los archivos o la configuración cambiaron: reiniciando Comfy de forma segura.')
        memory_manager.unload_slot('video'); resident = None
    if resident is None:
        common._switch_to_idle(job)
        common._comfy_start_admission(job, stage='Inicio movimiento', planned_ram=6, planned_vram=3)
    try:
        names = _prepare(engine, quant, job)
        if resident:
            job.log('Comfy activo: reutilizando proceso y cargadores cuando sea posible.')
        session = common._open_resident(job, names, model_id)
        session.motion_fingerprint = _motion_fingerprint(engine, quant)
        session.motion_cleanup_failed = False
    except BaseException:
        if resident: memory_manager.unload_slot('video')
        raise
    success = False
    try:
        fit_reference_image(image_path, session.input_dir / 'reference.png', width, height)
        segment_count = 1
        if engine == 'wan':
            output, segment_count = _generate_wan_segments(session, job, total_frames=total_frames, image_path=image_path,
                video_path=video_path, names=names, quant=quant, prompt=prompt, negative=negative, width=width, height=height,
                seed=seed, steps=steps, cfg=cfg, person=person, strength=strength)
        else:
            source = session.input_dir / 'drive.mp4'
            common._command([get_ffmpeg_bin(), '-hide_banner','-loglevel','error','-y','-i',str(video_path), '-map','0:v:0', '-an',
                             '-vf',f'fps=24,{fit_video_filter(width, height)}',
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
                'elapsed_seconds':round(time.monotonic()-started, 1), 'has_audio':False,
                'requested_duration_sec': duration_sec, 'segments': segment_count}
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
        success = True
        return meta
    except Exception:
        if not getattr(job, 'cancel_requested', False):
            # An inference/load failure may indicate corruption even when metadata
            # looks unchanged. Force a full digest on the next attempt, not deletion.
            try:
                _invalidate_motion_receipts(engine, quant)
            except OSError:
                pass
        session.guard.raise_if_stopped(); raise
    finally:
        common._finish_session(session, job, success)
