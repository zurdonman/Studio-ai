"""LTX-2.5 single-pass GGUF A/V recipes, experimental and untested on GPU.

BF16 uses the pinned Comfy runtime shared with LTX 2.3; Gemma 4 GGUF
uses a separate patched runtime. No silent fallback to the 2.3 graph.
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
Q5_ID = 'ltx-2.5-distilled-gguf-ud-q2-gemma-q5-experimental'
Q4_Q5_ID = 'ltx-2.5-distilled-gguf-q4km-gemma-q5-experimental'
UDQ4_Q5_ID = 'ltx-2.5-distilled-gguf-ud-q4-gemma-q5-experimental'
Q5_IDS = {Q5_ID, Q4_Q5_ID, UDQ4_Q5_ID}
Q5_RUNTIME = DATA_ROOT / 'runtimes' / 'ltx25-gemma4-gguf'
Q5_COMFY_COMMIT = '2f35f4a08176d993cded35dac3332be4f7287f41'
Q5_ENCODER = ('text_encoder', 'elix3r/gemma4-12b-with-proj-ltx-2.5-GGUF',
              '2a18e836d286eb0570ec9f013c3591eb8c614d57', 'gemma4-12b-with-proj-ltx-2.5-Q5_K_M.gguf',
              9514920864, '1d35d4fbfa34cca1513d8e9fdd77c0573778b21ffdcbe4ca9c906f37a8c502f9')
MODELS = {
    'ltx-2.5-distilled-gguf-ud-q2-experimental': ('UD-Q2_K_XL', 'distilled/ltx-2.5-22b-distilled-UD-Q2_K_XL.gguf', 9500270976, 'd6fdc7e117f1a9780bdbe15c898d63c12a7d747847f54b276b69a6be5ce3c57d'),
    Q5_ID: ('UD-Q2_K_XL + Gemma 4 GGUF Q5_K_M', 'distilled/ltx-2.5-22b-distilled-UD-Q2_K_XL.gguf', 9500270976, 'd6fdc7e117f1a9780bdbe15c898d63c12a7d747847f54b276b69a6be5ce3c57d'),
    Q4_Q5_ID: ('Q4_K_M + Gemma 4 GGUF Q5_K_M', 'distilled/ltx-2.5-22b-distilled-Q4_K_M.gguf', 13584703872, 'ba40d53770314e69e88361661e55c6cc3b12a363646ba2ea53c8bcd459b66747'),
    UDQ4_Q5_ID: ('UD-Q4_K_XL + Gemma 4 GGUF Q5_K_M', 'distilled/ltx-2.5-22b-distilled-UD-Q4_K_XL.gguf', 14000234880, '55442750eb0013d94fffaa4dadb577c5f4dc07107ec04389d41444f6e517d980'),
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
    components = (Q5_ENCODER, *COMMON[1:]) if model_id in Q5_IDS else COMMON
    return (('dit', REPO, REV, filename, size, sha), *components)


def path_for(item) -> Path:
    role, _, _, filename, _, _ = item
    return WEIGHTS / ('gemma' if role == 'text_encoder' else 'ltx') / filename


def files_present(model_id: str) -> bool:
    try:
        return ((Q5_RUNTIME if model_id in Q5_IDS else common.ROOT) / 'ready.json').is_file() and (model_id not in Q5_IDS or (Q5_RUNTIME / 'gemma4-patch.json').is_file()) and all(path_for(x).is_file() and path_for(x).stat().st_size == x[4] for x in files_for(model_id))
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
    runtime = Q5_RUNTIME if model_id in Q5_IDS else common.ROOT
    folders = {'dit': 'diffusion_models', 'text_encoder': 'text_encoders', 'video_vae': 'vae', 'audio_vae': 'vae'}
    names = {}
    for item in files_for(model_id):
        role = item[0]
        source = path_for(item)
        if not source.is_file():
            raise RuntimeError(f'GGUF_MISSING_COMPONENT: {role}')
        dest = runtime / 'comfy' / 'models' / folders[role] / source.name
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
    # Budgets are preventive estimates, not measured peaks for either variant.
    q5 = model_id in Q5_IDS
    runtime = Q5_RUNTIME if q5 else common.ROOT
    # The 26.3 GB BF16 Gemma 4 encoder is not the quantized Gemma 3 of LTX 2.3.
    # This conservative budget is not a measured peak and can reject an L4.
    memory_manager.ensure_capacity(ram_gb=(35 if model_id in {Q4_Q5_ID, UDQ4_Q5_ID} else 32 if q5 else 45) + extra_ram,
                                   vram_gb=(16 if model_id in {Q4_Q5_ID, UDQ4_Q5_ID} else 14 if q5 else 16) + extra_vram,
                                   stage='LTX 2.5 GGUF (estimación preventiva, no medida)', job=job)
    runtime.mkdir(parents=True, exist_ok=True)
    missing = sum(x[4] for x in files if not path_for(x).is_file() or path_for(x).stat().st_size != x[4])
    required = missing + 10 * 1024**3
    if shutil.disk_usage(runtime).free < required:
        raise RuntimeError(f'GGUF_DISK_LOW: se requieren {required / 1024**3:.1f} GiB libres para archivos y entorno')
    if q5:
        job.log('Variante Gemma 4 GGUF Q5: runtime aislado y estimaciones de memoria no medidas; requiere acceso al repositorio elix3r.')
        common._install(job, runtime, Q5_COMFY_COMMIT)
        from studio.catalog import catalog_manager
        env = os.environ.copy()
        token = catalog_manager.get_hf_token()
        if token:
            env['HF_TOKEN'] = token
        repo = str(Path(__file__).resolve().parents[2])
        env['PYTHONPATH'] = repo + os.pathsep + env.get('PYTHONPATH', '')
        common._command([sys.executable, '-m', 'studio.engines.ltx25_gguf_patch',
                         str(runtime / 'comfy' / 'custom_nodes' / 'ComfyUI-GGUF'), str(runtime / 'gemma4-patch.json')],
                        job, env=env, cwd=repo, stage='Verificando acceso y parche Gemma 4 GGUF', timeout=600)
        common._probe_runtime(job, {'CLIPLoaderGGUF', 'LTXVDualCFGGuider', 'VAEDecodeTiled'}, runtime)
    else:
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
    graph['2'] = ({'class_type': 'CLIPLoaderGGUF', 'inputs': {'clip_name': names['text_encoder'], 'type': 'ltxv'}}
                  if names['text_encoder'].endswith('.gguf') else
                  {'class_type': 'CLIPLoader', 'inputs': {'clip_name': names['text_encoder'], 'type': 'ltxv', 'device': 'default'}})
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
             width: int, height: int, duration_sec: float, fps: int, speech_language: str, image_strength: float,
             voice_mode: str = 'none', voice_path: Path | None = None) -> dict:
    from studio.engines.ltx23_voice import add_voice_graph, frames_for_audio
    started = time.monotonic()
    files_for(model_id)
    if not prompt.strip() or speech_language not in ('auto', 'es'):
        raise ValueError('Prompt o idioma no válido')
    if image_path:
        verify_image_file(image_path)
    frames = common.frames_for_duration(duration_sec, fps)
    if voice_mode not in ('none', 'audio', 'voice_reference') or (voice_mode == 'none') != (voice_path is None):
        raise ValueError('El modo de audio y el archivo deben coincidir.')
    if voice_mode == 'voice_reference' and not image_path:
        raise ValueError('MSR_IMAGE_REQUIRED: se necesita una imagen y una muestra de voz.')
    voice_info = None
    if voice_path is not None:
        voice_info = probe_media_info(voice_path)
        seconds = float(voice_info.get('duration_seconds') or 0)
        if not voice_info.get('has_audio') or voice_info.get('has_video') or not 1 <= seconds <= (20 if voice_mode == 'audio' else 15):
            raise ValueError('Se necesita audio sin vídeo: 1–20 s para audio completo o 1–15 s para voz de referencia.')
        if voice_mode == 'audio':
            frames = frames_for_audio(seconds, fps)
            job.log(f'Audio completo: {seconds:.2f} s; duración del vídeo ajustada a {frames} fotogramas. Ruta LTX 2.5 experimental sin prueba GPU.')
        else:
            job.log('MSR V2 experimental: la imagen y la muestra guían identidad y voz; no garantizan parecido ni sincronización. La referencia puede recortarse al límite de MSR.')
    effective = ('All spoken dialogue is in Spanish. Do not translate quoted dialogue.\n' + prompt) if speech_language == 'es' else prompt
    # Validate the graph before any download; the file names below are placeholders only.
    make_graph25(prompt=effective, negative_prompt=negative_prompt, width=width, height=height, frames=frames, fps=fps,
                 seed=seed, image_name='input.png' if image_path else None, image_strength=image_strength,
                 names={'dit':'x.gguf','text_encoder':'x.gguf' if model_id in Q5_IDS else 'x.safetensors','video_vae':'v.safetensors','audio_vae':'a.safetensors'})
    scale = width * height * frames / (512 * 320 * 49)
    extra_ram, extra_vram = max(0, scale - 1) * 2, max(0, scale - 1) * .5
    if voice_mode == 'audio':
        # Preventive allowances for the encoded waveform; not measured peaks.
        extra_ram += 1
        extra_vram += .5
    elif voice_mode == 'voice_reference':
        # Adapter weights and guide tokens; conservative estimates, not measured.
        extra_ram += 5
        extra_vram += 3
    names = _prepare(model_id, job, extra_ram=extra_ram, extra_vram=extra_vram)
    if voice_mode == 'voice_reference':
        from studio.engines.ltx25_msr import prepare_msr
        runtime = Q5_RUNTIME if model_id in Q5_IDS else common.ROOT
        if prepare_msr(job, runtime=runtime, weights=WEIGHTS, model_id=model_id):
            names = _prepare(model_id, job, extra_ram=extra_ram, extra_vram=extra_vram)
    session = (common._open_resident(job, names, model_id, Q5_RUNTIME) if model_id in Q5_IDS
               else common._open_resident(job, names, model_id))
    if voice_mode == 'voice_reference':
        try:
            response = common.requests.get(session.base + '/object_info', timeout=20)
            response.raise_for_status()
            missing = {'StudioStrictMSRLoader', 'ComfyUILTX25MSRMultiReferenceGuide', 'LTXVCropGuides'} - set(response.json())
            if missing:
                raise RuntimeError('MSR_RUNTIME_MISSING_NODES: ' + ', '.join(sorted(missing)))
        except Exception:
            memory_manager.unload_slot('video')
            raise
    VIDEOS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    seed = int(seed) if seed else int.from_bytes(os.urandom(4), 'big')
    job.log(f'LTX 2.5: {width}×{height}, {frames} fotogramas a {fps} fps (~{frames/fps:.2f} s), una pasada, 8 pasos.')
    suffix, generated, image_name, audio_name = uuid.uuid4().hex, None, None, None
    try:
        if image_path:
            from PIL import Image
            image_name = f'frame_{suffix}.png'
            with Image.open(image_path) as image:
                image.convert('RGB').save(session.input_dir / image_name)
        graph = make_graph25(prompt=effective, negative_prompt=negative_prompt, width=width, height=height, frames=frames, fps=fps,
                             seed=seed, image_name=image_name, names=names, image_strength=image_strength)
        if voice_path is not None:
            from studio.engines.ffmpeg_utils import get_ffmpeg_bin
            audio_name = f'voice_{suffix}.wav'
            audio_command = [get_ffmpeg_bin(), '-v', 'error', '-y', '-i', str(voice_path), '-vn', '-ac', '1', '-ar', '44100']
            if voice_mode == 'audio':
                audio_command += ['-af', f'apad,atrim=duration={frames / fps:.6f}']
            common._command(audio_command + ['-c:a', 'pcm_s16le', str(session.input_dir / audio_name)],
                            job, stage='Preparando audio LTX 2.5', timeout=120)
            if voice_mode == 'audio':
                add_voice_graph(graph, 'audio', audio_name)
            else:
                from studio.engines.ltx25_msr import add_msr_graph
                add_msr_graph(graph, image_name, audio_name)
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
        if audio_name:
            (session.input_dir / audio_name).unlink(missing_ok=True)
        if generated:
            generated.unlink(missing_ok=True)
    meta = {'id': dest.stem, 'filename': name, 'rel_path': f'outputs/videos/{name}', 'media_url': f'/media/outputs/videos/{name}',
            'type': 'video', 'model_id': model_id, 'model_name': f'LTX 2.5 destilado GGUF {MODELS[model_id][0]} experimental',
            'video_submode': 'i2v' if image_path else 't2v', 'audio_native': True, 'prompt': prompt, 'effective_prompt': effective,
            'seed': seed, 'width': info.get('width'), 'height': info.get('height'), 'duration_seconds': info.get('duration_seconds'),
            'fps': fps, 'steps': 8, 'guidance_scale': 1.0, 'requested_duration_sec': duration_sec, 'frames': frames,
            'duration_warning': warning, 'speech_language_hint': speech_language, 'image_strength': image_strength if image_path else None,
            'voice_mode': voice_mode, 'source_audio_duration_seconds': voice_info.get('duration_seconds') if voice_info else None,
            'elapsed_seconds': round(time.monotonic()-started, 2), 'created_at': datetime.now().isoformat(timespec='seconds')}
    (VIDEOS_OUT_DIR / (dest.stem + '.json')).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf8')
    return meta
