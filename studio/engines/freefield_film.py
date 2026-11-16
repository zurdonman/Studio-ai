"""Freefield's FILM idea, implemented using ComfyUI core in an isolated worker.

This is a bounded, experimental 2x recipe, not the full Freefield application.
No inference result is cached; the worker is stopped after every job.
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

from studio.config import DATA_ROOT, VIDEOS_OUT_DIR
from studio.engines import ltx_gguf_backend as common
from studio.engines.ffmpeg_utils import get_ffmpeg_bin, probe_media_info
from studio.memory_manager import memory_manager, serialized_gpu_operation

MODEL = ('FILM', 'Comfy-Org/Frame-Interpolation', 'aa7009be4d1471ea0bac91d6fe4d7e4b637cbbfc',
         'frame_interpolation/film_net_fp16.safetensors', 68882302,
         'f226e51375dc839d4b40e5c3d63da560dd1ea1c962364ec78f5adf2d05db05c0')
WEIGHTS = DATA_ROOT / 'models' / 'freefield-film'
MODEL_PATH = WEIGHTS / 'ltx' / MODEL[3]


def make_graph(name: str, prefix: str) -> dict:
    if any(not v or '/' in v or '\\' in v or v in ('.', '..') for v in (name, prefix)):
        raise ValueError('Nombres de entrada o salida no válidos')
    def n(kind, **inputs): return {'class_type': kind, 'inputs': inputs}
    return {
        '1': n('LoadVideo', file=name),
        '2': n('GetVideoComponents', video=['1', 0]),
        '3': n('FrameInterpolationModelLoader', model_name=Path(MODEL[3]).name),
        '4': n('FrameInterpolate', interp_model=['3', 0], images=['2', 0], multiplier=2),
        '5': n('CreateVideo', images=['4', 0], fps=48.0),
        '6': n('SaveVideo', video=['5', 0], filename_prefix=prefix, format='auto', **{'format.codec': 'auto'}),
    }


def validate_input(path: Path) -> dict:
    if not path.is_file() or path.suffix.lower() not in ('.mp4', '.mov', '.webm', '.mkv'):
        raise ValueError('FILM_INPUT_INVALID: selecciona un archivo de vídeo.')
    if path.stat().st_size > 200 * 1024**2:
        raise ValueError('FILM_INPUT_LIMIT: máximo 200 MiB por clip.')
    info = probe_media_info(path)
    if not info['has_video'] or not info['width'] or not info['height']:
        raise ValueError('FILM_INPUT_INVALID: se necesita un vídeo con dimensiones conocidas.')
    if info['size_bytes'] > 200 * 1024**2 or info['duration_seconds'] <= 0 or info['duration_seconds'] > 10.5:
        raise ValueError('FILM_INPUT_LIMIT: máximo 10,5 segundos y 200 MiB por clip.')
    if info['width'] * info['height'] > 768 * 512 or max(info['width'], info['height']) > 768:
        raise ValueError('FILM_INPUT_LIMIT: máximo 768×512 píxeles de área y 768 px de lado; reduce primero el clip.')
    return info


def _prepare(job: Any) -> None:
    common._install(job)
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(WEIGHTS).free < 256 * 1024**2 and not MODEL_PATH.exists():
        raise RuntimeError('FILM_DISK_LOW: se necesitan 256 MiB libres para descargar el modelo.')
    env = os.environ.copy()
    env['PYTHONPATH'] = str(Path(__file__).resolve().parents[2]) + os.pathsep + env.get('PYTHONPATH', '')
    common._command([sys.executable, '-m', 'studio.engines.freefield_film_download', str(WEIGHTS)], job,
                    env=env, stage='Descargando y verificando FILM (69 MB)', timeout=1800)
    dest = common.ROOT / 'comfy' / 'models' / 'frame_interpolation' / MODEL_PATH.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_symlink():
        if dest.resolve() != MODEL_PATH.resolve():
            raise RuntimeError('FILM_LINK_CONFLICT: el enlace ya apunta a otro modelo.')
    elif dest.exists():
        raise RuntimeError('FILM_LINK_CONFLICT: no se sobrescribe un modelo existente.')
    else:
        dest.symlink_to(MODEL_PATH.resolve())


def _infer(session: Any, graph: dict, job: Any) -> Path:
    info = common.requests.get(session.base + '/object_info', timeout=30)
    info.raise_for_status()
    required = {node['class_type'] for node in graph.values()}
    missing = required - set(info.json())
    if missing:
        raise RuntimeError('FILM_RUNTIME_MISSING_NODES: ' + ', '.join(sorted(missing)))
    reply = common.requests.post(session.base + '/prompt', json={'prompt': graph}, timeout=20)
    if not reply.ok:
        raise RuntimeError('FILM_GRAPH_REJECTED: ' + reply.text[:1200])
    ident = reply.json()['prompt_id']
    deadline = time.monotonic() + 3600
    while time.monotonic() < deadline:
        job.check_cancelled()
        session.guard.raise_if_stopped()
        if not session.alive():
            raise RuntimeError('FILM_ENGINE_EXITED: el proceso Comfy se detuvo.')
        result = common.requests.get(session.base + '/history/' + ident, timeout=10)
        result.raise_for_status()
        data = result.json().get(ident)
        if data:
            status = data.get('status') or {}
            if status.get('status_str') == 'error':
                messages = status.get('messages') or []
                detail = next((str(x[1].get('exception_message', '')) for x in messages
                               if len(x) > 1 and isinstance(x[1], dict) and x[0] == 'execution_error'), 'Error desconocido')
                raise RuntimeError('FILM_INFERENCE_FAILED: ' + detail[:800])
            if status.get('completed'):
                files = list(session.output_dir.glob(graph['6']['inputs']['filename_prefix'] + '_*.mp4'))
                if len(files) != 1:
                    raise RuntimeError('FILM_OUTPUT_MISSING: no se produjo un MP4 único.')
                return files[0]
        job.heartbeat(step='FILM: interpolando fotogramas; duración y memoria dependen del clip')
        time.sleep(1)
    raise TimeoutError('FILM_TIMEOUT: se alcanzó el límite de una hora.')


@serialized_gpu_operation(slots=('video',), managed=True)
def interpolate(path: Path, *, job: Any) -> dict:
    started = time.monotonic()
    info = validate_input(path)
    gpu = common.query_gpu_info()
    if not gpu.get('has_gpu'):
        raise RuntimeError('FILM_GPU_REQUIRED: esta receta experimental necesita GPU NVIDIA/CUDA.')
    job.log('FILM experimental: el vídeo se normaliza a 24 fps y se interpola a 48 fps; el audio se recompone. No validado en GPU.', level='warning')
    # A job waits for the same serialized GPU lock as LTX. Never stop an active generation.
    if memory_manager.get_slot('video'):
        memory_manager.unload_slot('video')
    common._comfy_start_admission(job, stage='Inicio FILM', planned_ram=4, planned_vram=2)
    _prepare(job)
    session = common._ResidentComfy(job, {})
    try:
        source = session.input_dir / 'input.mp4'
        # Strictly bounded decoded output, with fixed framerate, to avoid an unbounded
        # image tensor from variable-FPS or malformed videos. Source stays untouched.
        common._command([get_ffmpeg_bin(), '-v', 'error', '-y', '-i', str(path), '-map', '0:v:0', '-map', '0:a:0?',
                         '-vf', 'fps=24,scale=trunc(iw/2)*2:trunc(ih/2)*2,setsar=1', '-frames:v', '253',
                         '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac', str(source)], job,
                        stage='Preparando vídeo temporal a 24 fps', timeout=600, guard_label='FFmpeg FILM')
        prefix = 'studio_film_' + uuid.uuid4().hex
        output = _infer(session, make_graph(source.name, prefix), job)
        job.check_cancelled()
        generated_info = probe_media_info(output)
        if not generated_info.get('has_video'):
            raise RuntimeError('FILM_OUTPUT_INVALID: falta vídeo.')
        VIDEOS_OUT_DIR.mkdir(parents=True, exist_ok=True)
        name = f'film_{datetime.now().strftime("%Y%m%d_%H%M%S")}_{uuid.uuid4().hex[:8]}.mp4'
        dest = VIDEOS_OUT_DIR / name
        temporary = session.root / 'final.mp4'
        if info['has_audio']:
            common._command([get_ffmpeg_bin(), '-v', 'error', '-y', '-i', str(output), '-i', str(source),
                             '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy', '-c:a', 'aac',
                             str(temporary)], job, stage='Añadiendo el audio al vídeo interpolado', timeout=600, guard_label='FFmpeg FILM')
        else:
            shutil.copyfile(output, temporary)
        final = probe_media_info(temporary)
        if not final.get('has_video') or (info['has_audio'] and not final.get('has_audio')):
            raise RuntimeError('FILM_OUTPUT_INVALID: faltan pistas; no se publicará el resultado.')
        job.check_cancelled()
        if dest.exists():
            raise RuntimeError('FILM_OUTPUT_COLLISION: no se sobrescribe el resultado existente.')
        meta = {'id': dest.stem, 'filename': name, 'rel_path': f'outputs/videos/{name}',
                'media_url': f'/media/outputs/videos/{name}', 'type': 'video', 'model_id': 'freefield-film',
                'model_name': 'FILM · interpolación 2x experimental', 'fps': 48, 'source_fps_normalized': 24,
                'has_audio': final['has_audio'], 'duration_seconds': final['duration_seconds'],
                'width': final['width'], 'height': final['height'], 'elapsed_seconds': round(time.monotonic() - started, 1),
                'video_submode': 'Interpolación FILM 2x'}
        metadata = VIDEOS_OUT_DIR / (dest.stem + '.json')
        try:
            shutil.copyfile(temporary, dest)
            metadata.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf8')
        except BaseException:
            dest.unlink(missing_ok=True)
            metadata.unlink(missing_ok=True)
            raise
        return meta
    except Exception:
        session.guard.raise_if_stopped()
        raise
    finally:
        session.unload()
