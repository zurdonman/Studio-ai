"""Experimental local Fish S2-Pro TTS and authorized voice cloning."""
from __future__ import annotations
import json
import shutil
import subprocess
import tempfile
import time
import uuid
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any

from studio.config import AUDIO_OUT_DIR, DATA_ROOT, TEMP_DIR
from studio.engines import ltx_gguf_backend as common
from studio.engines.ffmpeg_utils import probe_media_info
from studio.engines.voice_clone_reference import prepared_reference_audio
from studio.memory_manager import memory_manager, serialized_gpu_operation

ENGINE = 'fish-s2-pro-experimental'
SOURCE_REV = '0f8a12e8a651062fdee9c33a894bdccd3a8513f1'
ROOT = DATA_ROOT / 'runtimes' / 'fish-s2-pro'
WEIGHTS = DATA_ROOT / 'models' / 'fish-s2-pro'
_verified = None


def _fingerprint():
    from studio.engines.fish_s2_download import FILES
    try:
        return tuple((p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ctime_ns, p.stat().st_ino)
                     for p in (WEIGHTS / name for name in FILES))
    except OSError:
        return None


def prepare(job: Any) -> Path:
    global _verified
    from studio.engines.fish_s2_download import FILES
    ROOT.mkdir(parents=True, exist_ok=True)
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    missing = sum(size for name, (size, _, _) in FILES.items() if not (WEIGHTS/name).is_file())
    if shutil.disk_usage(WEIGHTS).free < missing + 10 * 1024**3:
        raise RuntimeError('FISH_DISK_LOW: falta espacio para pesos, entorno y margen de seguridad.')
    source, py, marker = ROOT/'source', ROOT/'venv/bin/python', ROOT/'runtime-v1.json'
    env = common._setup_env()
    if not source.exists() and not source.is_symlink():
        with tempfile.TemporaryDirectory(prefix='source-', dir=ROOT) as temp:
            staged = Path(temp)/'source'
            common._command(['git', 'clone', '--no-checkout', 'https://github.com/fishaudio/fish-speech.git', str(staged)], job,
                            stage='Preparando código Fish Audio', timeout=600, env=env)
            common._command(['git', '-C', str(staged), 'checkout', '--detach', SOURCE_REV], job,
                            stage='Fijando versión Fish Audio', timeout=120, env=env)
            job.check_cancelled()
            if source.exists() or source.is_symlink(): raise RuntimeError('FISH_SOURCE_CONFLICT')
            staged.rename(source)
    if source.is_symlink() or not (source/'.git').is_dir(): raise RuntimeError('FISH_SOURCE_CONFLICT: código no gestionado.')
    head = subprocess.run(['git', '-C', str(source), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True, timeout=20).stdout.strip()
    dirty = subprocess.run(['git', '-C', str(source), 'status', '--porcelain', '--untracked-files=no'], capture_output=True, text=True, check=True, timeout=20).stdout.strip()
    if head != SOURCE_REV or dirty: raise RuntimeError('FISH_SOURCE_CONFLICT: revisión distinta o código modificado.')
    if not marker.is_file() or not py.is_file():
        py = common._prepare_venv(job, ROOT)
        check_torch = 'import torch,torchaudio; assert torch.__version__.split("+")[0]=="2.8.0" and torchaudio.__version__.split("+")[0]=="2.8.0" and torch.cuda.is_available() and torch.cuda.is_bf16_supported()'
        try:
            common._command([str(py), '-c', check_torch], job, stage='Comprobando PyTorch Fish CUDA', timeout=90, env=env)
        except RuntimeError:
            job.log('Se instalará PyTorch 2.8 CUDA en el entorno aislado de Fish.', level='warning')
            common._command([str(py), '-m', 'pip', 'install', '--index-url', 'https://download.pytorch.org/whl/cu128',
                             'torch==2.8.0', 'torchaudio==2.8.0'], job, stage='Instalando PyTorch Fish CUDA', timeout=7200, env=env)
            common._command([str(py), '-c', check_torch], job, stage='Verificando PyTorch Fish CUDA', timeout=90, env=env)
        # The upstream package includes server-only pyaudio and an incompatible
        # protobuf pin in audiotools. Install the inference stack explicitly.
        common._command([str(py), '-m', 'pip', 'install', 'transformers==5.19.0', 'tokenizers==0.23.2',
                         'hydra-core==1.3.2', 'natsort>=8.4', 'einops>=0.7', 'librosa>=0.10.1,<0.12',
                         'loguru>=0.6', 'pyrootutils>=1.0.4', 'einx[torch]==0.2.2', 'tiktoken>=0.8',
                         'safetensors', 'soundfile>=0.12.1', 'argbind>=0.3.7', 'pyloudnorm', 'julius',
                         'ffmpy', 'ipython', 'rich', 'matplotlib', 'pystoi', 'torch-stoi', 'flatten-dict',
                         'markdown2', 'randomname', 'tensorboard', 'protobuf>=3.20,<6', 'click'], job,
                        stage='Instalando dependencias Fish aisladas', timeout=3600, env=env)
        common._command([str(py), '-m', 'pip', 'install', '--no-deps', 'descript-audiotools==0.7.2', 'descript-audio-codec==1.0.0'], job,
                        stage='Instalando códec Fish', timeout=1200, env=env)
        check = ('import sys; sys.path.insert(0,sys.argv[1]); import torch,torchaudio; '
                 'from fish_speech.models.text2semantic import inference; from fish_speech.models.dac import modded_dac; '
                 'assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()')
        common._command([str(py), '-c', check, str(source)], job, stage='Verificando runtime Fish', timeout=120, env=env)
        marker.write_text(json.dumps({'source': SOURCE_REV, 'torch': '2.8.0'}), encoding='utf8')
    fp = _fingerprint()
    if fp is None or fp != _verified:
        repo = str(Path(__file__).resolve().parents[2]); env['PYTHONPATH'] = repo
        common._command([str(py), '-m', 'studio.engines.fish_s2_download', str(WEIGHTS)], job,
                        stage='Descargando y verificando Fish S2-Pro (~11 GB)', timeout=21600, env=env, cwd=repo)
        _verified = _fingerprint()
        if _verified is None: raise RuntimeError('FISH_WEIGHTS_MISSING')
    return py


@serialized_gpu_operation(slots=('tts',), managed=True)
def synthesize(*, text: str, reference_path: Path | None, reference_text: str, job: Any) -> dict:
    start = time.monotonic()
    text, transcript = (text or '').strip(), (reference_text or '').strip()
    if not text or len(text) > 500: raise ValueError('FISH_TEXT_INVALID: escribe hasta 500 caracteres.')
    if len(transcript) > 2000: raise ValueError('FISH_TRANSCRIPT_INVALID: transcripción demasiado larga.')
    if reference_path and not transcript: raise ValueError('Fish necesita la transcripción exacta de la muestra.')
    if not reference_path and transcript: raise ValueError('La transcripción requiere una muestra de voz.')
    gpu = common.query_gpu_info()
    if not gpu.get('has_gpu') or not gpu.get('bf16_supported'):
        raise RuntimeError('FISH_GPU_UNSUPPORTED: se requiere GPU NVIDIA CUDA con BF16.')
    job.log('Fish S2-Pro experimental: requisitos oficiales de al menos 24 GB de VRAM; no validado en L4.', level='warning')
    job.comfy_keep_loaded = True
    common._switch_to_idle(job)
    # One-shot workers release GPU on success, cancellation and failure. Do not
    # retain the upstream model unless a future resident protocol is validated.
    slot = memory_manager.get_slot('tts')
    if slot: memory_manager.unload_slot('tts')
    memory_manager.ensure_capacity(ram_gb=18, vram_gb=19, stage='Fish S2-Pro: presupuesto preventivo no medido', job=job)
    with prepared_reference_audio(reference_path) if reference_path else nullcontext(None) as reference:
        py = prepare(job)
        memory_manager.ensure_capacity(ram_gb=18, vram_gb=19, stage='Fish S2-Pro: carga de inferencia', job=job)
        TEMP_DIR.mkdir(parents=True, exist_ok=True); AUDIO_OUT_DIR.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='fish-s2-', dir=TEMP_DIR) as tmp:
            work = Path(tmp); result = work/'audio.wav'; request = work/'request.json'
            request.write_text(json.dumps({'text': text, 'transcript': transcript, 'reference': str(reference) if reference else None,
                                           'output': str(result)}, ensure_ascii=False), encoding='utf8')
            request.chmod(0o600)
            env = common._runtime_env(); env['HF_HUB_OFFLINE']='1'; env['TRANSFORMERS_OFFLINE']='1'
            common._command([str(py), str(Path(__file__).with_name('fish_s2_worker.py')), str(ROOT/'source'), str(WEIGHTS), str(request)],
                            job, stage='Generando voz Fish S2-Pro', timeout=3600, env=env, guard_label='Fish S2-Pro')
            info = probe_media_info(result)
            if not info.get('has_audio') or info.get('has_video') or not (info.get('duration_seconds') or 0) > 0:
                raise RuntimeError('FISH_OUTPUT_INVALID: audio no verificable.')
            job.check_cancelled()
            name = f'voz_fish_{datetime.now().strftime("%Y%m%d_%H%M%S")}_{uuid.uuid4().hex[:8]}.wav'
            output = AUDIO_OUT_DIR/name
            if output.exists(): raise RuntimeError('FISH_OUTPUT_COLLISION')
            shutil.copyfile(result, output)
    meta = {'id': output.stem, 'filename': name, 'rel_path': f'outputs/audio/{name}',
            'media_url': f'/media/outputs/audio/{name}', 'type': 'audio', 'model_id': ENGINE,
            'model_name': 'Fish S2-Pro · experimental', 'duration_seconds': info['duration_seconds'],
            'elapsed_seconds': round(time.monotonic()-start, 2), 'created_at': datetime.now().isoformat(timespec='seconds')}
    (AUDIO_OUT_DIR/(output.stem+'.json')).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf8')
    return meta
