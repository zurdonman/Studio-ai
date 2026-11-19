"""Experimental FL2VA MiniMax H3 GGUF runner, separate from Comfy and LTX.

Pin every component. Generating automatically prepares missing components after
license and download authorization; this runtime has not been validated with real weights on L4.
"""
from __future__ import annotations

import json
import os
import platform
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
from studio.engines.ffmpeg_utils import get_ffmpeg_bin, probe_media_info, verify_image_file
from studio.memory_manager import memory_manager, serialized_gpu_operation
from studio.jobs import JobCancelledError
from studio.memory_policy import MemoryPressureError

RUNTIME_REV = '27b9ddca07be0c25ace3d1a560bcbad281eb660d'
REV = 'babc040eff0084465460ead58fdb8ec44296f8dd'
REPO = 'unsloth/MiniMax-H3-GGUF'
ROOT = DATA_ROOT / 'runtimes' / 'minimax-h3-sdcpp'
WEIGHTS = DATA_ROOT / 'models' / 'minimax-h3-gguf'

# role, filename, byte size, SHA-256 (Hugging Face LFS), encoder choice
VARIANTS = {
    'h3-fl2va-udq2': ('minimax_h3_fl2va_pruned-UD-Q2_K_XL.gguf', 8063029344, 'cfe0795c00ab6e6ebf8c64fe4574f45a828e8a93e0876bca704e055662a9d7b8', 'q2'),
    'h3-fl2va-q4': ('minimax_h3_fl2va_pruned-Q4_K.gguf', 11420663904, 'dd948e08ad0ba3c71bd42f368e283dd82e790f5122a63b276e22a3e0283d0c10', 'q4'),
}
ENCODERS = {
    'q2': ('qwen3vl_32b_minimax_h3-Q2_K_M.gguf', 13102161024, 'a8ccadccd57ef34c838ffb8a7da8368bb554721b2760274a1d3b0df63960b997'),
    'q4': ('qwen3vl_32b_minimax_h3-Q4_K_M.gguf', 18218065024, '11e6efe70a57ce7f4838c47bdbd1a1c4b8ce10e2b7747f1b065990b70f4b05fc'),
}
VAES = (
    ('video_vae', 'vae/minimax_h3_video_vae_fp16.safetensors', 5207808496, '7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522'),
    ('audio_vae', 'vae/minimax_h3_audio_vae_fp32.safetensors', 605254808, '8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48'),
)
_verified: dict[str, tuple] = {}


def files_for(model_id: str) -> tuple:
    if model_id not in VARIANTS:
        raise ValueError('H3_MODEL_UNSUPPORTED: variante desconocida')
    name, size, sha, enc = VARIANTS[model_id]
    ename, esize, esha = ENCODERS[enc]
    items = [('dit', name, size, sha), ('text_encoder', ename, esize, esha), *VAES]
    return tuple((role, REPO, REV, filename, size, sha) for role, filename, size, sha in items)


def path_for(item: tuple) -> Path:
    return WEIGHTS / ('gemma' if item[0] == 'text_encoder' else 'ltx') / item[3]


def fingerprint(model_id: str) -> tuple | None:
    try:
        return tuple((p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ino)
                     for p in map(path_for, files_for(model_id)))
    except OSError:
        return None


def _cuda_env() -> dict:
    """Keep host settings and add typical Colab/toolkit and NVIDIA wheel library paths."""
    env = os.environ.copy()
    paths = [x for x in env.get('LD_LIBRARY_PATH', '').split(os.pathsep) if x]
    roots = [Path('/usr/local/cuda'), Path('/usr/local/cuda-12'), Path('/usr/local/cuda-13')]
    for key in ('CUDA_HOME', 'CUDA_PATH'):
        if env.get(key):
            roots.append(Path(env[key]))
    candidates = [root / 'lib64' for root in roots]
    for entry in sys.path:
        if entry:
            candidates.extend((Path(entry) / 'nvidia' / 'cuda_runtime' / 'lib',
                               Path(entry) / 'nvidia' / 'cublas' / 'lib'))
    for path in candidates:
        if path.is_dir() and str(path) not in paths:
            paths.append(str(path))
    if paths:
        env['LD_LIBRARY_PATH'] = os.pathsep.join(paths)
    return env


def _cuda_major(env: dict) -> str | None:
    # Check actual loadable libraries and runtime version, not the maximum CUDA
    # version printed by nvidia-smi. The release was linked with CUDA 12.8 / 13.3.
    if sys.platform != 'linux' or platform.machine().lower() not in ('x86_64', 'amd64'):
        return None
    probe = '''import ctypes, sys\nm = sys.argv[1]\nr = ctypes.CDLL("libcudart.so." + m)\nctypes.CDLL("libcublas.so." + m)\nctypes.CDLL("libcublasLt.so." + m)\nv = ctypes.c_int()\nif r.cudaRuntimeGetVersion(ctypes.byref(v)) != 0 or v.value < int(sys.argv[2]): sys.exit(1)\n'''
    for major, minimum in (('12', '12080'), ('13', '13030')):
        try:
            result = subprocess.run([sys.executable, '-c', probe, major, minimum], env=env,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
            if result.returncode == 0:
                return major
        except (OSError, subprocess.TimeoutExpired):
            continue
    return None


def _install_prebuilt(job: Any) -> Path | None:
    from studio.engines.minimax_h3_prebuilt import ASSETS, REVISION
    env = _cuda_env()
    major = _cuda_major(env)
    if major is None:
        job.log('H3: bibliotecas CUDA 12.8+ o 13.3+ del precompilado no detectadas; se intentará compilar el motor.', level='warning')
        return None
    dest = ROOT / f'prebuilt-cuda{major}'
    binary = dest / 'sd-cli'
    marker = dest / 'studio-ready.json'
    expected = {'revision': REVISION, 'cuda_major': major, 'sha256': ASSETS[major][1]}
    if dest.exists():
        try:
            if dest.is_symlink() or json.loads(marker.read_text()) != expected or not binary.is_file():
                raise ValueError('carpeta sin marcador válido')
            common._command([str(binary), '--help'], job, stage='Comprobando runtime H3 precompilado', timeout=90, env=env)
            job.log(f'H3: reutilizando runtime precompilado CUDA {major}; sin compilación.')
            return binary
        except (JobCancelledError, MemoryPressureError, MemoryError):
            raise
        except (OSError, ValueError, RuntimeError, TimeoutError):
            job.check_cancelled()
            job.log('H3: el runtime precompilado guardado no está utilizable; se intentará compilar sin sobrescribirlo.', level='warning')
            return None
    phase = 'descarga o verificación'
    try:
        job.check_cancelled()
        job.log(f'H3: descargando runtime precompilado CUDA {major} de Unsloth y verificando SHA-256; no es necesario compilar.')
        with tempfile.TemporaryDirectory(prefix='.h3-prebuilt-', dir=ROOT) as temp:
            repo = str(Path(__file__).resolve().parents[2])
            child_env = env.copy()
            child_env['PYTHONPATH'] = repo + os.pathsep + child_env.get('PYTHONPATH', '')
            common._command([sys.executable, '-m', 'studio.engines.minimax_h3_prebuilt', major, temp], job,
                            stage='Descargando y verificando runtime H3 precompilado', timeout=1800, env=child_env, cwd=repo)
            staged = Path(temp) / 'bundle'
            phase = 'comprobación del ejecutable'
            common._command([str(staged / 'sd-cli'), '--help'], job, stage='Comprobando runtime H3 precompilado', timeout=90, env=env)
            job.check_cancelled()
            (staged / 'studio-ready.json').write_text(json.dumps(expected), encoding='utf8')
            os.rename(staged, dest)
        job.log(f'H3: runtime precompilado CUDA {major} preparado y verificado; sin compilación.')
        return binary
    except (JobCancelledError, MemoryPressureError, MemoryError):
        raise
    except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
        job.check_cancelled()
        detail = str(exc)
        if 'H3_PREBUILT_INVALID' in detail or 'H3_PREBUILT_ARCHIVE_INVALID' in detail:
            reason = 'integridad o contenido incorrectos'
        elif 'URLError' in detail or 'HTTPError' in detail:
            reason = 'error de red o acceso a GitHub'
        elif isinstance(exc, TimeoutError):
            reason = 'tiempo agotado'
        else:
            reason = type(exc).__name__
        job.log(f'H3: falló la {phase} del precompilado ({reason}); se intentará compilar el motor.', level='warning')
        return None


def _install(job: Any) -> Path:
    source = ROOT / 'source'
    binary = ROOT / 'build' / 'bin' / 'sd-cli'
    marker = ROOT / 'ready.json'
    try:
        if json.loads(marker.read_text()) == {'revision': RUNTIME_REV, 'cuda': True} and binary.is_file():
            return binary
    except (OSError, ValueError):
        pass
    ROOT.mkdir(parents=True, exist_ok=True)
    prebuilt = _install_prebuilt(job)
    if prebuilt is not None:
        return prebuilt
    job.log('H3: iniciando compilación CUDA local; puede tardar varios minutos.')
    for executable in ('git', 'cmake', 'nvcc'):
        if not shutil.which(executable):
            raise RuntimeError(f'H3_BUILD_TOOL_MISSING: falta {executable}; instala la herramienta en el entorno antes de preparar H3. No se han descargado pesos.')
    owner = ROOT / 'studio-source.json'
    if source.is_symlink():
        raise RuntimeError('H3_RUNTIME_CONFLICT: el directorio de código es un enlace; no se modifica')
    if not source.exists():
        owner.write_text(json.dumps({'repo': 'https://github.com/unslothai/stable-diffusion.cpp.git'}), encoding='utf8')
        common._command(['git', 'clone', '--no-checkout', 'https://github.com/unslothai/stable-diffusion.cpp.git', str(source)],
                        job, stage='Descargando runtime H3', timeout=900)
    try:
        managed = json.loads(owner.read_text()) == {'repo': 'https://github.com/unslothai/stable-diffusion.cpp.git'}
    except (OSError, ValueError):
        managed = False
    if not managed or not (source / '.git').is_dir():
        raise RuntimeError('H3_RUNTIME_CONFLICT: directorio de runtime ajeno o incompleto; no se modifica')
    # A partial --no-checkout clone has no checked-out files; otherwise refuse
    # to overwrite edits in a managed checkout. Build artifacts live elsewhere.
    current = subprocess.run(['git', '-C', str(source), 'rev-parse', '--verify', 'HEAD'], capture_output=True, text=True, timeout=30)
    if current.returncode == 0:
        status = subprocess.run(['git', '-C', str(source), 'status', '--porcelain', '--untracked-files=no'], capture_output=True, text=True, timeout=30)
        # A --no-checkout clone records every tracked file as deleted; detect that
        # state via the absence of its root CMakeLists, rather than altering edits.
        if status.returncode or (status.stdout.strip() and (source / 'CMakeLists.txt').exists()):
            raise RuntimeError('H3_RUNTIME_CONFLICT: hay cambios en el código del runtime; no se sobrescriben')
    common._command(['git', '-C', str(source), 'checkout', '--detach', RUNTIME_REV], job, stage='Fijando versión H3', timeout=120)
    common._command(['git', '-C', str(source), 'submodule', 'update', '--init', 'ggml'], job, stage='Preparando ggml H3', timeout=900)
    common._command(['cmake', '-S', str(source), '-B', str(ROOT / 'build'), '-DCMAKE_BUILD_TYPE=Release',
                     '-DSD_CUDA=ON', '-DSD_WEBP=OFF', '-DSD_WEBM=OFF', '-DSD_CUDA_CUTLASS_PATH:PATH=',
                     '-DCMAKE_CUDA_ARCHITECTURES=89'], job, stage='Configurando H3 CUDA (Ada/L4)', timeout=300)
    common._command(['cmake', '--build', str(ROOT / 'build'), '--target', 'sd-cli', '--parallel', '1'],
                    job, stage='Compilando H3 CUDA (puede tardar)', timeout=3600, guard_label='compilación H3', build_progress=True)
    if not binary.is_file():
        raise RuntimeError('H3_RUNTIME_MISSING: no se compiló sd-cli')
    common._command([str(binary), '--help'], job, stage='Comprobando ejecutable H3', timeout=90)
    marker.write_text(json.dumps({'revision': RUNTIME_REV, 'cuda': True}), encoding='utf8')
    return binary


def _prepare(model_id: str, job: Any) -> Path:
    files = files_for(model_id)  # fail before side effects
    gpu = common.query_gpu_info()
    if not gpu.get('has_gpu') or 'L4' not in str(gpu.get('gpu_name', '')).upper():
        raise RuntimeError('H3_GPU_UNSUPPORTED: esta compilación experimental está fijada para NVIDIA L4 (sm89).')
    ROOT.mkdir(parents=True, exist_ok=True)
    remaining = sum(x[4] for x in files if not path_for(x).is_file() or path_for(x).stat().st_size != x[4])
    needed = remaining + 6 * 1024**3
    if shutil.disk_usage(ROOT).free < needed:
        raise RuntimeError(f'H3_DISK_LOW: faltan {needed / 1024**3:.1f} GiB libres para componentes restantes y margen de instalación')
    memory_manager.ensure_capacity(ram_gb=4, vram_gb=2, stage='Preparación H3: mínimo real y reservas', job=job)
    binary = _install(job)
    if fingerprint(model_id) != _verified.get(model_id) or fingerprint(model_id) is None:
        from studio.catalog import catalog_manager
        env = os.environ.copy()
        token = catalog_manager.get_hf_token()
        if token:
            env['HF_TOKEN'] = token
        repo = str(Path(__file__).resolve().parents[2])
        env['PYTHONPATH'] = repo + os.pathsep + env.get('PYTHONPATH', '')
        common._command([sys.executable, '-m', 'studio.engines.minimax_h3_download', str(WEIGHTS), model_id],
                        job, stage='Descargando y verificando H3', timeout=10800, env=env, cwd=repo, guard_label='descarga H3')
        _verified[model_id] = fingerprint(model_id)
    return binary


@serialized_gpu_operation(slots=('video',), managed=True)
def prepare(model_id: str, *, job: Any) -> dict:
    binary = _prepare(model_id, job)
    return {'prepared': True, 'model_id': model_id, 'note': 'Runtime y componentes verificados en disco; sin prueba de inferencia.', 'runtime': str(binary)}


def build_command(binary: Path, model_id: str, prompt: str, width: int, height: int, frames: int,
                  steps: int, seed: int, output: Path, image: Path | None, end: Path | None,
                  offload: bool) -> list[str]:
    files = {x[0]: path_for(x) for x in files_for(model_id)}
    command = [str(binary), '--mode', 'vid_gen', '--diffusion-model', str(files['dit']),
               '--llm', str(files['text_encoder']), '--vae', str(files['video_vae']),
               '--audio-vae', str(files['audio_vae']), '--prompt', prompt,
               '--width', str(width), '--height', str(height), '--video-frames', str(frames),
               '--fps', '24', '--steps', str(steps), '--cfg-scale', '1.0', '--seed', str(seed),
               '--backend', 'te=cpu', '--diffusion-fa', '--output', str(output)]
    if offload:
        command.append('--offload-to-cpu')
    if image:
        command.extend(['--init-img', str(image)])
    if end:
        command.extend(['--end-img', str(end)])
    return command


@serialized_gpu_operation(slots=('video',), managed=True)
def generate(model_id: str, *, prompt: str, image: Path | None, end: Path | None, width: int,
             height: int, frames: int, steps: int, seed: int, offload: bool, job: Any) -> dict:
    started = time.monotonic()
    files_for(model_id)
    if not prompt.strip() or len(prompt) > 3000 or not (256 <= width <= 768 and 256 <= height <= 768) or width % 32 or height % 32 or width * height > 768 * 512:
        raise ValueError('H3_INPUT_INVALID: prompt y resolución (múltiplos de 32, máximo 768×512)')
    if frames not in (22, 39, 56, 73, 124) or steps not in (4, 8) or not 0 <= seed <= 4294967295:
        raise ValueError('H3_INPUT_INVALID: fotogramas, pasos o semilla no admitidos')
    if end and not image:
        raise ValueError('H3_INPUT_INVALID: un fotograma final requiere uno inicial')
    for item in (image, end):
        if item:
            verify_image_file(item)
    # The API requires explicit license authorization before submitting this job.
    # Preparation handles missing runtime/weights, existing files and cancellation;
    # it never changes the requested variant or bypasses integrity checks.
    job.log('H3: preparando automáticamente la variante seleccionada si faltan componentes; sigue el progreso en el monitor.')
    binary = _prepare(model_id, job)
    job.check_cancelled()
    if memory_manager.get_slot('video'):
        memory_manager.unload_slot('video')
    memory_manager.ensure_capacity(ram_gb=4, vram_gb=2, stage='Inicio H3: mínimo real y reservas', job=job)
    job.log('H3 experimental: encoder en CPU; vigilancia RAM/VRAM activa. Un pico instantáneo todavía puede causar OOM.', level='warning')
    TEMP_DIR.mkdir(parents=True, exist_ok=True); VIDEOS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    actual_seed = seed or int.from_bytes(os.urandom(4), 'big')
    with tempfile.TemporaryDirectory(prefix='h3-', dir=TEMP_DIR) as temp:
        avi = Path(temp) / 'clip.avi'
        cmd = build_command(binary, model_id, prompt, width, height, frames, steps, actual_seed, avi, image, end, offload)
        with tempfile.TemporaryFile() as log:
            env = _cuda_env() if binary.parent.name.startswith('prebuilt-cuda') else None
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)
            guard = common._ComfyMemoryGuard(proc, job, label='H3')
            inference_started = time.monotonic()
            try:
                while proc.poll() is None:
                    job.check_cancelled(); guard.raise_if_stopped()
                    if time.monotonic() - inference_started > 10800:
                        raise TimeoutError('H3_TIMEOUT: tiempo máximo superado')
                    job.heartbeat(step='Generando H3: proceso aislado en ejecución')
                    time.sleep(1)
                guard.raise_if_stopped(); job.check_cancelled()
                if proc.returncode:
                    raise RuntimeError(f'H3_INFERENCE_FAILED: código {proc.returncode}: {common._tail(log)}')
            finally:
                guard.done.set(); common._stop(proc); guard.close()
        info = probe_media_info(avi)
        if not info.get('has_video'):
            raise RuntimeError('H3_OUTPUT_INVALID: archivo sin vídeo')
        # Preserve the generated clip in the library before conversion: cancellation
        # or an FFmpeg failure must not discard an expensive successful inference.
        stem = f'h3_{datetime.now().strftime("%Y%m%d_%H%M%S")}_{uuid.uuid4().hex[:8]}'
        recovery = VIDEOS_OUT_DIR / f'{stem}_recuperacion.avi'
        shutil.move(str(avi), recovery)
        if not info.get('has_audio'):
            raise RuntimeError(f'H3_OUTPUT_INVALID: vídeo sin audio conservado como {recovery.name}')
        dest = VIDEOS_OUT_DIR / f'{stem}.mp4'
        tmp_mp4 = Path(temp) / 'final.mp4'
        try:
            common._command([get_ffmpeg_bin(), '-v', 'error', '-y', '-i', str(recovery), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-movflags', '+faststart', str(tmp_mp4)], job, stage='Convirtiendo vídeo y audio H3 a MP4', timeout=600)
            info = probe_media_info(tmp_mp4)
            if not info.get('has_video') or not info.get('has_audio'):
                raise RuntimeError('H3_OUTPUT_INVALID: MP4 sin vídeo o audio')
            shutil.move(str(tmp_mp4), dest)
        except Exception:
            job.log(f'Clip original conservado en la biblioteca: {recovery.name}', level='warning')
            raise
        recovery.unlink()
    meta = {'id': dest.stem, 'filename': dest.name, 'rel_path': f'outputs/videos/{dest.name}', 'media_url': f'/media/outputs/videos/{dest.name}',
            'type': 'video', 'model_id': model_id, 'model_name': f'MiniMax H3 GGUF {model_id} experimental',
            'video_submode': 'flf2v' if end else 'i2v' if image else 't2v', 'audio_native': True, 'prompt': prompt,
            'width': info['width'], 'height': info['height'], 'duration_seconds': info['duration_seconds'], 'fps': 24,
            'frames': frames, 'steps': steps, 'guidance_scale': 1.0, 'seed': actual_seed, 'offload': offload,
            'elapsed_seconds': round(time.monotonic() - started, 2), 'created_at': datetime.now().isoformat(timespec='seconds')}
    (dest.with_suffix('.json')).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf8')
    return meta
