"""Isolated ThinkSound light runtime for local video-to-audio (experimental)."""
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any
from studio.config import DATA_ROOT
from studio.engines import ltx_gguf_backend as common

ROOT = DATA_ROOT/'runtimes'/'thinksound-v2a'
WEIGHTS = DATA_ROOT/'models'/'thinksound-v2a'
SOURCE_REV = '4e2cfd57efaaf40c90460d8111b6a3c1e7dfb2f0'
MODEL_ID = 'thinksound-light-v2a-experimental'
_verified = None


def _fingerprint():
    from studio.engines.thinksound_download import FILES
    try:
        return tuple((p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ctime_ns, p.stat().st_ino)
                     for p in (WEIGHTS/group/name for group, _, _, name, *_ in FILES))
    except OSError:
        return None


def prepare(job: Any) -> Path:
    global _verified
    from studio.engines.thinksound_download import FILES
    ROOT.mkdir(parents=True, exist_ok=True); WEIGHTS.mkdir(parents=True, exist_ok=True)
    missing = sum(size for group, _, _, name, size, *_ in FILES if not (WEIGHTS/group/name).is_file())
    if shutil.disk_usage(WEIGHTS).free < missing + 10*1024**3:
        raise RuntimeError('THINKSOUND_DISK_LOW: falta espacio para pesos, entorno y margen.')
    source, py, marker = ROOT/'source', ROOT/'venv/bin/python', ROOT/'runtime-v1.json'
    env = common._setup_env()
    if not source.exists() and not source.is_symlink():
        with tempfile.TemporaryDirectory(prefix='source-', dir=ROOT) as tmp:
            staged = Path(tmp)/'source'
            common._command(['git', 'clone', '--no-checkout', 'https://github.com/QwenAudio/ThinkSound.git', str(staged)], job,
                            stage='Preparando código ThinkSound', timeout=600, env=env)
            common._command(['git', '-C', str(staged), 'checkout', '--detach', SOURCE_REV], job,
                            stage='Fijando versión ThinkSound', timeout=120, env=env)
            job.check_cancelled()
            if source.exists() or source.is_symlink(): raise RuntimeError('THINKSOUND_SOURCE_CONFLICT')
            staged.rename(source)
    if source.is_symlink() or not (source/'.git').is_dir(): raise RuntimeError('THINKSOUND_SOURCE_CONFLICT: código no gestionado.')
    head = subprocess.run(['git', '-C', str(source), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True, timeout=20).stdout.strip()
    dirty = subprocess.run(['git', '-C', str(source), 'status', '--porcelain', '--untracked-files=no'], capture_output=True, text=True, check=True, timeout=20).stdout.strip()
    if head != SOURCE_REV or dirty: raise RuntimeError('THINKSOUND_SOURCE_CONFLICT: código modificado o revisión distinta.')
    if not marker.is_file() or not py.is_file():
        py = common._prepare_venv(job, ROOT)
        torch_check = 'import torch,torchaudio,torchvision; assert torch.__version__.split("+")[0]=="2.6.0" and torchaudio.__version__.split("+")[0]=="2.6.0" and torch.cuda.is_available() and torch.cuda.is_bf16_supported()'
        try:
            common._command([str(py), '-c', torch_check], job, stage='Comprobando PyTorch ThinkSound', timeout=90, env=env)
        except RuntimeError:
            common._command([str(py), '-m', 'pip', 'install', '--index-url', 'https://download.pytorch.org/whl/cu124',
                             'torch==2.6.0', 'torchvision==0.21.0', 'torchaudio==2.6.0'], job,
                            stage='Instalando PyTorch CUDA ThinkSound', timeout=7200, env=env)
        # Upstream's full training/server dependency set pins incompatible
        # NumPy versions (laion-clap requires 1.23.5). The V2A inference path
        # does not import CLAP; install that path's dependencies explicitly.
        common._command([str(py), '-m', 'pip', 'install', 'transformers==4.52.3', 'numpy==1.26.4', 'av==14.0.1',
                         'pandas==2.0.2', 'lightning==2.5.1.post0', 'prefigure==0.0.9', 'wandb==0.18.7',
                         'alias-free-torch==0.0.6', 'einops-exts==0.0.4', 'ema-pytorch==0.2.3',
                         'encodec==0.1.1', 'k-diffusion==0.1.1', 'open_clip_torch==2.32.0',
                         'timm==1.0.19', 'vector-quantize-pytorch==1.9.14', 'x-transformers==1.26.6',
                         'sentencepiece==0.2.1', 'omegaconf==2.3.0', 'hydra-core==1.3.2', 'safetensors',
                         'librosa==0.10.2.post1', 'argbind', 'pyloudnorm', 'julius', 'ffmpy', 'ipython',
                         'rich', 'matplotlib', 'pystoi', 'torch-stoi', 'flatten-dict', 'markdown2',
                         'randomname', 'tensorboard'], job,
                        stage='Instalando dependencias ThinkSound aisladas', timeout=7200, env=env)
        common._command([str(py), '-m', 'pip', 'install', '--no-deps', 'descript-audiotools==0.7.2',
                         'descript-audio-codec==1.0.0'], job, stage='Instalando códec ThinkSound', timeout=1800, env=env)
        check = ('import sys; sys.path.insert(0,sys.argv[1]); import torch,torchaudio,torchvision,av,extract_latents,predict; '
                 'from ThinkSound.models import diffusion,autoencoders; '
                 'assert torch.cuda.is_available() and torch.cuda.is_bf16_supported() and torch.__version__.split("+")[0]=="2.6.0"')
        common._command([str(py), '-c', check, str(source)], job, stage='Verificando runtime ThinkSound', timeout=180, env=env)
        marker.write_text(json.dumps({'source':SOURCE_REV}), encoding='utf8')
    fp = _fingerprint()
    if fp is None or fp != _verified:
        repo = str(Path(__file__).resolve().parents[2]); env['PYTHONPATH'] = repo
        common._command([str(py), '-m', 'studio.engines.thinksound_download', str(WEIGHTS)], job,
                        stage='Descargando y verificando ThinkSound y componentes (~24,5 GB)', timeout=28800, env=env, cwd=repo)
        _verified = _fingerprint()
        if _verified is None: raise RuntimeError('THINKSOUND_WEIGHTS_MISSING')
    return py
