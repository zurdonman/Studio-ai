"""Isolated, automatically prepared ControlFoley V2A runtime; experimental."""
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any
from studio.config import DATA_ROOT
from studio.engines import ltx_gguf_backend as common

ROOT = DATA_ROOT / 'runtimes' / 'controlfoley-v2a'
WEIGHTS = DATA_ROOT / 'models' / 'controlfoley-v2a'
SOURCE_REV = 'b6c902888d45d75f022e253e3db29836360a3f82'
MODEL_ID = 'controlfoley-v2a-experimental'
_verified = None


def _fingerprint():
    from studio.engines.controlfoley_download import FILES
    try:
        return tuple((p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ctime_ns, p.stat().st_ino)
                     for p in (WEIGHTS / group / name for group, _, _, name, *_ in FILES))
    except OSError:
        return None


def prepare(job: Any) -> Path:
    global _verified
    from studio.engines.controlfoley_download import FILES
    ROOT.mkdir(parents=True, exist_ok=True); WEIGHTS.mkdir(parents=True, exist_ok=True)
    missing = sum(size for group, _, _, name, size, *_ in FILES if not (WEIGHTS/group/name).is_file())
    if shutil.disk_usage(WEIGHTS).free < missing + 6 * 1024**3:
        raise RuntimeError('CONTROLFOLEY_DISK_LOW: falta espacio para pesos, entorno y margen.')
    source, py, marker = ROOT/'source', ROOT/'venv/bin/python', ROOT/'runtime-v1.json'
    env = common._setup_env()
    if not source.exists():
        with tempfile.TemporaryDirectory(prefix='source-', dir=ROOT) as tmp:
            staged = Path(tmp)/'source'
            common._command(['git', 'clone', '--no-checkout', 'https://github.com/xiaomi-research/controlfoley.git', str(staged)], job,
                            stage='Preparando código ControlFoley', timeout=600, env=env)
            common._command(['git', '-C', str(staged), 'checkout', '--detach', SOURCE_REV], job,
                            stage='Fijando versión ControlFoley', timeout=120, env=env)
            job.check_cancelled()
            if source.exists() or source.is_symlink(): raise RuntimeError('CONTROLFOLEY_SOURCE_CONFLICT')
            staged.rename(source)
    if source.is_symlink() or not (source/'.git').is_dir(): raise RuntimeError('CONTROLFOLEY_SOURCE_CONFLICT: código no gestionado.')
    head = subprocess.run(['git', '-C', str(source), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True, timeout=20).stdout.strip()
    dirty = subprocess.run(['git', '-C', str(source), 'status', '--porcelain', '--untracked-files=no'], capture_output=True, text=True, check=True, timeout=20).stdout.strip()
    if head != SOURCE_REV or dirty: raise RuntimeError('CONTROLFOLEY_SOURCE_CONFLICT: revisión diferente o código modificado.')
    if not marker.is_file() or not py.is_file():
        py = common._prepare_venv(job, ROOT)
        torch_check = 'import torch,torchaudio,torchvision; assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(); assert tuple(map(int,torch.__version__.split("+")[0].split(".")[:2])) >= (2,5)'
        try:
            common._command([str(py), '-c', torch_check], job, stage='Comprobando PyTorch CUDA BF16', timeout=90, env=env)
        except RuntimeError:
            job.log('Instalando PyTorch CUDA compatible en el entorno aislado de ControlFoley.', level='warning')
            common._command([str(py), '-m', 'pip', 'install', '--index-url', 'https://download.pytorch.org/whl/cu121',
                             'torch==2.5.1', 'torchvision==0.20.1', 'torchaudio==2.5.1'], job,
                            stage='Instalando PyTorch CUDA ControlFoley', timeout=3600, env=env)
            common._command([str(py), '-c', torch_check], job, stage='Verificando PyTorch CUDA BF16', timeout=90, env=env)
        common._command([str(py), '-m', 'pip', 'install', 'open_clip_torch==2.29.0', 'numpy<2.1', 'av==14.0.1',
                         'einops>=0.8,<0.9', 'torchdiffeq==0.2.5', 'colorlog', 'librosa>=0.10,<0.12',
                         'timm>=1,<2', 'huggingface_hub>=0.26,<1', 'julius==0.2.7'], job,
                        stage='Instalando dependencias ControlFoley V2A aisladas', timeout=2400, env=env)
        repo = str(Path(__file__).resolve().parents[2]); env['PYTHONPATH'] = repo
        check = ('import sys; sys.path.insert(0,sys.argv[1]); sys.path.insert(0,sys.argv[1]+"/lib"); '
                 'from studio.engines.controlfoley_worker import bind_source_packages,install_unused_reference_stubs; bind_source_packages(sys.argv[1]); install_unused_reference_stubs(); '
                 'import torch,torchaudio,torchvision,open_clip,av; from controlfoley.inference_utils import load_video; '
                 'from controlfoley.feature_extractor import FeaturesUtils; assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()')
        common._command([str(py), '-c', check, str(source)], job, stage='Verificando runtime ControlFoley V2A', timeout=120, env=env)
        marker.write_text(json.dumps({'source': SOURCE_REV}), encoding='utf8')
    fp = _fingerprint()
    if fp is None or fp != _verified:
        repo = str(Path(__file__).resolve().parents[2]); env['PYTHONPATH'] = repo
        common._command([str(py), '-m', 'studio.engines.controlfoley_download', str(WEIGHTS)], job,
                        stage='Descargando y verificando ControlFoley V2A y componentes (~19 GB)', timeout=21600, env=env, cwd=repo)
        _verified = _fingerprint()
        if _verified is None: raise RuntimeError('CONTROLFOLEY_WEIGHTS_MISSING')
    return py
