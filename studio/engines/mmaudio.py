"""Experimental local MMAudio V2 audio for short silent clips; no GPU validation yet."""
from __future__ import annotations
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from studio.config import DATA_ROOT, TEMP_DIR, VIDEOS_OUT_DIR
from studio.engines import ltx_gguf_backend as common
from studio.engines.ffmpeg_utils import get_ffmpeg_bin, probe_media_info
from studio.memory_manager import memory_manager, serialized_gpu_operation

ROOT = DATA_ROOT / 'runtimes' / 'mmaudio-v2'
WEIGHTS = DATA_ROOT / 'models' / 'mmaudio-v2'
SOURCE_REV = '974010a026c731054592d8f777218bd9d85a6c24'
MODEL_ID = 'mmaudio-v2-experimental'
_verified = None


def validate_input(path: Path, replace_audio: bool = False) -> dict:
    if not path.is_file() or path.suffix.lower() not in ('.mp4', '.mov', '.mkv', '.webm'):
        raise ValueError('MMAUDIO_INPUT_INVALID: selecciona un vídeo compatible.')
    info = probe_media_info(path)
    if not info['has_video'] or not info['width'] or not info['height']:
        raise ValueError('MMAUDIO_INPUT_INVALID: falta vídeo válido.')
    if info['size_bytes'] > 200 * 1024**2 or not 2 <= info['duration_seconds'] <= 8:
        raise ValueError('MMAUDIO_INPUT_LIMIT: duración entre 2 y 8 segundos y máximo 200 MiB.')
    if max(info['width'], info['height']) > 4096:
        raise ValueError('MMAUDIO_INPUT_LIMIT: lado máximo 4096 píxeles.')
    if info['has_audio'] and not replace_audio:
        raise ValueError('MMAUDIO_AUDIO_EXISTS: activa la sustitución explícita de audio en la copia.')
    return info


def _fingerprint():
    from studio.engines.mmaudio_download import FILES
    try:
        return tuple((p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ctime_ns, p.stat().st_ino)
                     for p in (WEIGHTS / group / name for group, _, _, name, *_ in FILES))
    except OSError:
        return None


def _prepare(job: Any) -> Path:
    global _verified
    from studio.engines.mmaudio_download import FILES
    ROOT.mkdir(parents=True, exist_ok=True); WEIGHTS.mkdir(parents=True, exist_ok=True)
    missing = sum(size for group, _, _, name, size, *_ in FILES if not (WEIGHTS / group / name).is_file())
    if shutil.disk_usage(WEIGHTS).free < missing + 5 * 1024**3:
        raise RuntimeError('MMAUDIO_DISK_LOW: falta espacio para pesos, dependencias y margen de seguridad.')
    source, py = ROOT / 'source', ROOT / 'venv/bin/python'
    marker = ROOT / 'runtime-v1.json'
    env = common._setup_env()
    if not source.exists():
        # An interrupted clone must not poison the next attempt or overwrite
        # an existing user directory.
        with tempfile.TemporaryDirectory(prefix='source-', dir=ROOT) as tmp:
            staged = Path(tmp) / 'source'
            common._command(['git', 'clone', '--no-checkout', 'https://github.com/hkchengrex/MMAudio.git', str(staged)], job,
                            stage='Preparando código MMAudio', timeout=600, env=env)
            common._command(['git', '-C', str(staged), 'checkout', '--detach', SOURCE_REV], job,
                            stage='Fijando versión MMAudio', timeout=120, env=env)
            job.check_cancelled()
            if source.exists() or source.is_symlink():
                raise RuntimeError('MMAUDIO_SOURCE_CONFLICT: apareció un directorio existente; no se sobrescribe.')
            staged.rename(source)
    if source.is_symlink() or not (source / '.git').is_dir():
        raise RuntimeError('MMAUDIO_SOURCE_CONFLICT: código existente no gestionado.')
    head = subprocess.run(['git', '-C', str(source), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True, timeout=20).stdout.strip()
    dirty = subprocess.run(['git', '-C', str(source), 'status', '--porcelain', '--untracked-files=no'], capture_output=True, text=True, check=True, timeout=20).stdout.strip()
    if head != SOURCE_REV or dirty:
        raise RuntimeError('MMAUDIO_SOURCE_CONFLICT: código modificado o revisión distinta; no se sobrescribe.')
    if not marker.is_file() or not py.is_file():
        py = common._prepare_venv(job, ROOT)
        torch_check = 'import torch,torchaudio,torchvision; assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(); assert tuple(map(int,torch.__version__.split("+")[0].split(".")[:2])) >= (2,5)'
        try:
            common._command([str(py), '-c', torch_check], job, stage='Comprobando PyTorch CUDA BF16', timeout=90, env=env)
        except RuntimeError:
            job.log('PyTorch CUDA compatible no está disponible; se instalará en el entorno aislado.', level='warning')
            common._command([str(py), '-m', 'pip', 'install', '--index-url', 'https://download.pytorch.org/whl/cu121',
                             'torch==2.5.1', 'torchvision==0.20.1', 'torchaudio==2.5.1'], job,
                            stage='Instalando PyTorch CUDA en entorno aislado', timeout=3600, env=env)
            common._command([str(py), '-c', torch_check], job, stage='Verificando PyTorch CUDA BF16', timeout=90, env=env)
        common._command([str(py), '-m', 'pip', 'install', 'open_clip_torch==2.29.0', 'numpy<2.1', 'av==14.0.1',
                         'einops>=0.8,<0.9', 'torchdiffeq==0.2.5', 'hydra-core==1.3.2',
                         'tensordict>=0.6,<0.7', 'colorlog', 'librosa>=0.10,<0.12', 'timm>=1,<2', 'huggingface_hub>=0.26,<1'],
                        job, stage='Instalando dependencias MMAudio aisladas', timeout=2400, env=env)
        check = 'import sys; sys.path.insert(0,sys.argv[1]); import torch,torchaudio,torchvision,open_clip,av; from mmaudio.eval_utils import load_video; from mmaudio.model.utils.features_utils import FeaturesUtils; assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()'
        common._command([str(py), '-c', check, str(source)], job, stage='Verificando runtime MMAudio', timeout=120, env=env)
        marker.write_text(json.dumps({'source': SOURCE_REV}), encoding='utf8')
    fp = _fingerprint()
    if fp is None or fp != _verified:
        repo = str(Path(__file__).resolve().parents[2])
        env['PYTHONPATH'] = repo
        common._command([str(py), '-m', 'studio.engines.mmaudio_download', str(WEIGHTS)], job,
                        stage='Descargando y verificando MMAudio y componentes (~10 GB)', timeout=14400, env=env, cwd=repo)
        _verified = _fingerprint()
        if _verified is None: raise RuntimeError('MMAUDIO_WEIGHTS_MISSING: faltan componentes.')
    return py


@serialized_gpu_operation(slots=('video',), managed=True)
def generate(path: Path, *, prompt: str, negative: str, seed: int, steps: int, replace_audio: bool, job: Any) -> dict:
    started = time.monotonic()
    info = validate_input(path, replace_audio)
    if not common.query_gpu_info().get('bf16_supported') or not common.query_gpu_info().get('has_gpu'):
        raise RuntimeError('MMAUDIO_GPU_UNSUPPORTED: se necesita GPU NVIDIA CUDA con BF16.')
    job.log('MMAudio local experimental: Foley, no garantiza diálogo ni sincronización. No validado en GPU.', level='warning')
    # Release any Comfy model tensors first; the compatible neutral process may survive.
    job.comfy_keep_loaded = True
    common._switch_to_idle(job)
    memory_manager.ensure_capacity(ram_gb=8, vram_gb=8, stage='MMAudio: margen preventivo no medido', job=job)
    py = _prepare(job)
    memory_manager.ensure_capacity(ram_gb=8, vram_gb=8, stage='MMAudio: carga de inferencia', job=job)
    TEMP_DIR.mkdir(parents=True, exist_ok=True); VIDEOS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='mmaudio-', dir=TEMP_DIR) as tmp:
        work = Path(tmp); video = work / 'input.mp4'; audio = work / 'audio.flac'; final = work / 'final.mp4'
        # Low-resolution conditioning copy avoids storing full-resolution decoded frames.
        common._command([get_ffmpeg_bin(), '-v', 'error', '-y', '-i', str(path), '-map', '0:v:0',
                         '-vf', 'fps=25,scale=384:384:force_original_aspect_ratio=decrease,pad=384:384:(ow-iw)/2:(oh-ih)/2',
                         '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(video)], job,
                        stage='Preparando copia temporal del vídeo', timeout=300, guard_label='MMAudio vídeo')
        env = common._runtime_env(); env['HF_HUB_OFFLINE'] = '1'; env['TRANSFORMERS_OFFLINE'] = '1'
        common._command([str(py), str(Path(__file__).with_name('mmaudio_worker.py')), str(ROOT/'source'), str(WEIGHTS),
                         str(video), str(audio), str(info['duration_seconds']), prompt, negative, str(seed), str(steps)], job,
                        stage='MMAudio: generando audio local', timeout=3600, env=env, guard_label='MMAudio inferencia')
        ainfo = probe_media_info(audio)
        if not ainfo['has_audio'] or ainfo['duration_seconds'] < info['duration_seconds'] - .5:
            raise RuntimeError('MMAUDIO_OUTPUT_INVALID: el audio falta o es demasiado corto.')
        # Always preserve the source; replace its track only in the new MP4.
        common._command([get_ffmpeg_bin(), '-v', 'error', '-y', '-i', str(path), '-i', str(audio),
                         '-map', '0:v:0', '-map', '1:a:0', '-map_metadata', '-1', '-c:v', 'libx264',
                         '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-af', 'apad', '-t', str(info['duration_seconds']),
                         '-movflags', '+faststart', str(final)], job, stage='Acoplando audio a una copia del vídeo', timeout=600,
                        guard_label='MMAudio composición')
        outinfo = probe_media_info(final)
        if not outinfo['has_video'] or not outinfo['has_audio'] or abs(outinfo['duration_seconds'] - info['duration_seconds']) > .3:
            raise RuntimeError('MMAUDIO_OUTPUT_INVALID: pistas o duración incorrectas.')
        job.check_cancelled()
        name = f'mmaudio_{datetime.now().strftime("%Y%m%d_%H%M%S")}_{uuid.uuid4().hex[:8]}.mp4'
        dest = VIDEOS_OUT_DIR / name
        if dest.exists(): raise RuntimeError('MMAUDIO_OUTPUT_COLLISION')
        shutil.copyfile(final, dest)
    meta = {'id': dest.stem, 'filename': name, 'rel_path': f'outputs/videos/{name}',
            'media_url': f'/media/outputs/videos/{name}', 'type': 'video', 'model_id': MODEL_ID,
            'model_name': 'MMAudio V2 · Foley experimental', 'duration_seconds': outinfo['duration_seconds'],
            'has_audio': True, 'original_audio_replaced': info['has_audio'], 'prompt': prompt, 'seed': seed, 'steps': steps,
            'license_note': 'MMAudio CC BY-NC 4.0; Apple CLIP solo investigación científica no comercial.',
            'elapsed_seconds': round(time.monotonic()-started, 2), 'created_at': datetime.now().isoformat(timespec='seconds')}
    (VIDEOS_OUT_DIR / (dest.stem + '.json')).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf8')
    return meta
