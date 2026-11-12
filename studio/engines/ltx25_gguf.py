"""LTX-2.5 single-pass GGUF A/V recipes, experimental and untested on GPU.

Uses the isolated, pinned Comfy runtime shared with LTX 2.3; never substitutes
its Gemma 4 encoder or silently falls back to the 2.3 graph.
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
from studio.engines.ffmpeg_utils import probe_media_info, verify_image_file
from studio.engines.ltx_gguf_workflow import make_graph
from studio.memory_manager import memory_manager, serialized_gpu_operation

REPO = 'unsloth/LTX-2.5-GGUF'
REV = 'ac3c540c6025903784336e99b2dc179385c4c3b7'
WEIGHTS = DATA_ROOT / 'models' / 'ltx25-gguf'
MODELS = {
    'ltx-2.5-distilled-gguf-ud-q2-experimental': ('UD-Q2_K_XL', 'distilled/ltx-2.5-22b-distilled-UD-Q2_K_XL.gguf', 9500270976, 'd6fdc7e117f1a9780bdbe15c898d63c12a7d747847f54b276b69a6be5ce3c57d'),
    'ltx-2.5-distilled-gguf-ud-q4-experimental': ('UD-Q4_K_XL', 'distilled/ltx-2.5-22b-distilled-UD-Q4_K_XL.gguf', 14000234880, '55442750eb0013d94fffaa4dadb577c5f4dc07107ec04389d41444f6e517d980'),
}
COMMON = (
    ('text_encoder', REPO, REV, 'text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors', 26263858182, 'ef7243612fdae7a75cb4d5cee9433e81380675fb6c213bd98ae74a9cd16561d1'),
    ('video_vae', REPO, REV, 'vae/ltx-2.5-video-vae-conv-bf16.safetensors', 1452269922, '685b06ee3d9b2039647698fc4ea33175112462fc374e2777312c907897dfce8d'),
    ('audio_vae', REPO, REV, 'vae/ltx-2.5-audio-vae-bf16.safetensors', 364866540, 'c52733d37f6a7fb7949c3dc0fb468c6cb2169e4d836983a73babb9f0d54837a5'),
)
_verified: dict[str, tuple] = {}


def files_for(model_id: str) -> tuple:
    if model_id not in MODELS:
        raise ValueError('GGUF_MODEL_UNSUPPORTED: modelo LTX 2.5 desconocido')
    _, filename, size, sha = MODELS[model_id]
    return (('dit', REPO, REV, filename, size, sha), *COMMON)


def path_for(item) -> Path:
    role, _, _, filename, _, _ = item
    return WEIGHTS / ('gemma' if role == 'text_encoder' else 'ltx') / filename


def files_present(model_id: str) -> bool:
    try:
        return (common.ROOT / 'ready.json').is_file() and all(path_for(x).is_file() and path_for(x).stat().st_size == x[4] for x in files_for(model_id))
    except OSError:
        return False


def disk_inventory(model_id: str) -> dict:
    files = files_for(model_id)
    present, complete, size = 0, 0, 0
    for item in files:
        try:
            path = path_for(item)
            if path.is_file():
                actual = path.stat().st_size
                present += 1
                complete += int(actual == item[4])
                size += actual
        except OSError:
            pass
    return {'present': present, 'complete_sizes': complete, 'total': len(files), 'size_bytes': size}


def _fingerprint(model_id: str) -> tuple | None:
    try:
        return tuple((path_for(x).stat().st_size, path_for(x).stat().st_mtime_ns, path_for(x).stat().st_ino) for x in files_for(model_id))
    except OSError:
        return None


def _links(model_id: str) -> dict[str, str]:
    folders = {'dit': 'diffusion_models', 'text_encoder': 'text_encoders', 'video_vae': 'vae', 'audio_vae': 'vae'}
    names = {}
    for item in files_for(model_id):
        role = item[0]
        source = path_for(item)
        if not source.is_file():
            raise RuntimeError(f'GGUF_MISSING_COMPONENT: {role}')
        dest = common.ROOT / 'comfy' / 'models' / folders[role] / source.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.is_symlink():
            if dest.resolve() != source.resolve():
                raise RuntimeError(f'GGUF_MODEL_LINK_CONFLICT: {dest.name}')
        elif dest.exists():
            raise RuntimeError(f'GGUF_MODEL_LINK_CONFLICT: {dest.name}')
        else:
            dest.symlink_to(source.resolve())
        names[role] = source.name
    return names


def _prepare(model_id: str, job: Any, *, extra_ram: float = 0, extra_vram: float = 0) -> dict:
    files = files_for(model_id)  # allowlist before any side effect
    gpu = common.query_gpu_info()
    if not gpu.get('has_gpu') or not gpu.get('bf16_supported') or float(gpu.get('vram_total_gb') or 0) < 16:
        raise RuntimeError('GGUF_GPU_UNSUPPORTED: LTX 2.5 requiere GPU BF16 con al menos 16 GiB; no se descargó nada')
    session = common._resident(model_id)
    if session:
        memory_manager.ensure_capacity(ram_gb=extra_ram, vram_gb=extra_vram, stage='LTX 2.5 residente: margen adicional estimado', job=job)
        job.log('Reutilizando motor LTX 2.5 y caché; Comfy puede mover o liberar pesos bajo presión.')
        return session.names
    if memory_manager.get_slot('video'):
        job.log('Liberando el modelo de vídeo anterior para cambiar de receta; archivos intactos.')
        memory_manager.unload_slot('video')
    # The 26.3 GB BF16 Gemma 4 encoder is not the quantized Gemma 3 of LTX 2.3.
    # This conservative budget is not a measured peak and can reject an L4.
    memory_manager.ensure_capacity(ram_gb=45 + extra_ram, vram_gb=16 + extra_vram,
                                   stage='LTX 2.5 GGUF (estimación preventiva, no medida)', job=job)
    common.ROOT.mkdir(parents=True, exist_ok=True)
    missing = sum(x[4] for x in files if not path_for(x).is_file() or path_for(x).stat().st_size != x[4])
    required = missing + 10 * 1024**3
    if shutil.disk_usage(common.ROOT).free < required:
        raise RuntimeError(f'GGUF_DISK_LOW: se requieren {required / 1024**3:.1f} GiB libres para archivos y entorno')
    common._install(job)
    common._probe_runtime(job, {'CLIPLoader', 'LTXVDualCFGGuider', 'VAEDecodeTiled'})
    fingerprint = _fingerprint(model_id)
    if fingerprint is None or fingerprint != _verified.get(model_id):
        from studio.catalog import catalog_manager
        env = os.environ.copy()
        token = catalog_manager.get_hf_token()
        if token:
            env['HF_TOKEN'] = token
        repo = str(Path(__file__).resolve().parents[2])
        env['PYTHONPATH'] = repo + os.pathsep + env.get('PYTHONPATH', '')
        common._command([sys.executable, '-m', 'studio.engines.ltx_gguf_download', str(WEIGHTS), model_id], job,
                        env=env, cwd=repo, stage='Descargando y verificando LTX 2.5 GGUF', timeout=7200)
        _verified[model_id] = _fingerprint(model_id)
    else:
        job.log('Componentes LTX 2.5 ya verificados en esta sesión y sin cambios.')
    return _links(model_id)


def make_graph25(*, prompt: str, negative_prompt: str, width: int, height: int, frames: int, fps: int,
                 seed: int, image_name: str | None, names: dict, image_strength: float = 1) -> dict:
    # Single-pass version of Comfy's pinned LTX-2.5 blueprint. Full 8+3 upscale is not used.
    graph = make_graph(prompt=prompt, negative_prompt=negative_prompt, width=width, height=height,
                       frames=frames, fps=fps, seed=seed, image_name=image_name, image_strength=image_strength,
                       model=names['dit'], text_encoder=names['text_encoder'], connectors='unused.safetensors',
                       video_vae=names['video_vae'], audio_vae=names['audio_vae'])
    graph['2'] = {'class_type': 'CLIPLoader', 'inputs': {'clip_name': names['text_encoder'], 'type': 'ltxv', 'device': 'default'}}
    graph['11'] = {'class_type': 'LTXVDualCFGGuider', 'inputs': {'model': ['1', 0], 'positive': ['7', 0], 'negative': ['7', 1], 'video_cfg': 1.0, 'audio_cfg': 1.0}}
    graph['13']['inputs']['sampler_name'] = 'euler_ancestral'
    graph['17'] = {'class_type': 'VAEDecodeTiled', 'inputs': {'samples': ['16', 0], 'vae': ['3', 0], 'tile_size': 512, 'overlap': 64, 'temporal_size': 64, 'temporal_overlap': 16}}
    return graph


@serialized_gpu_operation(slots=('video',), managed=True)
def prepare(model_id: str, *, job: Any) -> dict:
    _prepare(model_id, job)
    return {'prepared': True, 'note': 'LTX 2.5 preparado en disco; se cargará durante la generación.'}


@serialized_gpu_operation(slots=('video',), managed=True)
def generate(*, model_id: str, image_path: Path | None, prompt: str, negative_prompt: str, seed: int, job: Any,
             width: int, height: int, duration_sec: float, fps: int, speech_language: str, image_strength: float) -> dict:
    started = time.monotonic()
    files_for(model_id)
    if not prompt.strip() or speech_language not in ('auto', 'es'):
        raise ValueError('Prompt o idioma no válido')
    if image_path:
        verify_image_file(image_path)
    frames = common.frames_for_duration(duration_sec, fps)
    effective = ('All spoken dialogue is in Spanish. Do not translate quoted dialogue.\n' + prompt) if speech_language == 'es' else prompt
    # Validate the graph before any download; the file names below are placeholders only.
    make_graph25(prompt=effective, negative_prompt=negative_prompt, width=width, height=height, frames=frames, fps=fps,
                 seed=seed, image_name='input.png' if image_path else None, image_strength=image_strength,
                 names={'dit':'x.gguf','text_encoder':'x.safetensors','video_vae':'v.safetensors','audio_vae':'a.safetensors'})
    scale = width * height * frames / (512 * 320 * 49)
    extra_ram, extra_vram = max(0, scale - 1) * 2, max(0, scale - 1) * .5
    names = _prepare(model_id, job, extra_ram=extra_ram, extra_vram=extra_vram)
    session = common._open_resident(job, names, model_id)
    VIDEOS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    seed = int(seed) if seed else int.from_bytes(os.urandom(4), 'big')
    job.log(f'LTX 2.5: {width}×{height}, {frames} fotogramas a {fps} fps (~{frames/fps:.2f} s), una pasada, 8 pasos.')
    suffix, generated, image_name = uuid.uuid4().hex, None, None
    try:
        if image_path:
            from PIL import Image
            image_name = f'frame_{suffix}.png'
            with Image.open(image_path) as image:
                image.convert('RGB').save(session.input_dir / image_name)
        graph = make_graph25(prompt=effective, negative_prompt=negative_prompt, width=width, height=height, frames=frames, fps=fps,
                             seed=seed, image_name=image_name, names=names, image_strength=image_strength)
        graph['20']['inputs']['filename_prefix'] = f'studio_gguf_{suffix}'
        generated = common._infer(graph, session.input_dir, session.output_dir, job, model_id)
        info = probe_media_info(generated)
        if not info.get('has_video') or not info.get('has_audio'):
            raise RuntimeError('GGUF_OUTPUT_INVALID: falta vídeo o audio; no se publicará')
        actual = info.get('duration_seconds') or 0
        warning = None
        if actual and abs(actual - frames / fps) > max(.3, 2 / fps):
            warning = f'Duración medida {actual:.2f} s; esperada ~{frames/fps:.2f} s. Revisa el resultado.'
            job.log(warning, level='warning', step='Aviso de duración')
        name = f'ltx25_gguf_{datetime.now().strftime("%Y%m%d_%H%M%S")}_{seed % 100000}.mp4'
        dest = VIDEOS_OUT_DIR / name
        if dest.exists():
            raise RuntimeError('GGUF_OUTPUT_COLLISION: no se sobrescribirá el vídeo')
        shutil.copyfile(generated, dest)
    except BaseException:
        memory_manager.unload_slot('video')
        raise
    finally:
        if image_name:
            (session.input_dir / image_name).unlink(missing_ok=True)
        if generated:
            generated.unlink(missing_ok=True)
    meta = {'id': dest.stem, 'filename': name, 'rel_path': f'outputs/videos/{name}', 'media_url': f'/media/outputs/videos/{name}',
            'type': 'video', 'model_id': model_id, 'model_name': f'LTX 2.5 destilado GGUF {MODELS[model_id][0]} experimental',
            'video_submode': 'i2v' if image_path else 't2v', 'audio_native': True, 'prompt': prompt, 'effective_prompt': effective,
            'seed': seed, 'width': info.get('width'), 'height': info.get('height'), 'duration_seconds': info.get('duration_seconds'),
            'fps': fps, 'steps': 8, 'guidance_scale': 1.0, 'requested_duration_sec': duration_sec, 'frames': frames,
            'duration_warning': warning, 'speech_language_hint': speech_language, 'image_strength': image_strength if image_path else None,
            'elapsed_seconds': round(time.monotonic()-started, 2), 'created_at': datetime.now().isoformat(timespec='seconds')}
    (VIDEOS_OUT_DIR / (dest.stem + '.json')).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf8')
    return meta
