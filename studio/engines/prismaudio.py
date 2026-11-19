"""Isolated PrismAudio V2A runtime. Experimental and unvalidated on GPUs."""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any
from studio.config import DATA_ROOT
from studio.engines import ltx_gguf_backend as common

ROOT = DATA_ROOT/'runtimes'/'prismaudio-v2a'
WEIGHTS = DATA_ROOT/'models'/'prismaudio-v2a'
SOURCE_REV = '86eed40e1e2eddc54787a31ae66df01eb8befd3f'
VIDEO_REV = '4f831e332b80d6e77bdd43882206f23aba2a67a7'
MODEL_ID = 'prismaudio-v2a-experimental'
_verified = None


def _fingerprint():
    from studio.engines.prismaudio_download import FILES
    try:
        files = [WEIGHTS/group/name for group, _, _, name, *_ in FILES]
        files.append(WEIGHTS/'videoprism/sentencepiece.model')
        return tuple((p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ctime_ns, p.stat().st_ino) for p in files)
    except OSError:
        return None


def _source(job, root, url, revision, env):
    if not root.exists() and not root.is_symlink():
        with tempfile.TemporaryDirectory(prefix='source-', dir=ROOT) as tmp:
            staged = Path(tmp)/'source'
            common._command(['git','clone','--no-checkout',url,str(staged)], job, stage='Preparando código PrismAudio/VideoPrism', timeout=600, env=env)
            common._command(['git','-C',str(staged),'checkout','--detach',revision], job, stage='Fijando revisión de código', timeout=120, env=env)
            job.check_cancelled()
            if root.exists() or root.is_symlink(): raise RuntimeError('PRISMAUDIO_SOURCE_CONFLICT')
            staged.rename(root)
    if root.is_symlink() or not (root/'.git').is_dir(): raise RuntimeError('PRISMAUDIO_SOURCE_CONFLICT: código no gestionado')
    head = subprocess.run(['git','-C',str(root),'rev-parse','HEAD'],capture_output=True,text=True,check=True,timeout=20).stdout.strip()
    dirty = subprocess.run(['git','-C',str(root),'status','--porcelain','--untracked-files=no'],capture_output=True,text=True,check=True,timeout=20).stdout.strip()
    if head != revision or dirty: raise RuntimeError('PRISMAUDIO_SOURCE_CONFLICT: código modificado o revisión diferente')


def prepare(job: Any) -> Path:
    global _verified
    from studio.engines.prismaudio_download import FILES, TOKENIZER_SIZE
    ROOT.mkdir(parents=True, exist_ok=True); WEIGHTS.mkdir(parents=True, exist_ok=True)
    missing = sum(size for group, _, _, name, size, *_ in FILES if not (WEIGHTS/group/name).is_file())
    if not (WEIGHTS/'videoprism/sentencepiece.model').is_file(): missing += TOKENIZER_SIZE
    if shutil.disk_usage(WEIGHTS).free < missing + 12*1024**3:
        raise RuntimeError('PRISMAUDIO_DISK_LOW: falta espacio para pesos, entorno y margen.')
    env = common._setup_env(); repo = str(Path(__file__).resolve().parents[2])
    fp = _fingerprint()
    if fp is None or fp != _verified:
        from studio.catalog import catalog_manager
        token = catalog_manager.get_hf_token()
        if token: env['HF_TOKEN'] = token
        env['PYTHONPATH'] = repo
        try:
            common._command([sys.executable,'-m','studio.engines.prismaudio_download',str(WEIGHTS)], job,
                            stage='Descargando PrismAudio, VideoPrism y T5Gemma (~11 GB)', timeout=28800, env=env, cwd=repo)
        except RuntimeError as exc:
            raise RuntimeError('PRISMAUDIO_DOWNLOAD_FAILED: revisa el acceso a T5Gemma en Hugging Face y la conexión. '+str(exc)) from exc
        _verified = _fingerprint()
        if _verified is None: raise RuntimeError('PRISMAUDIO_WEIGHTS_MISSING')
    env = common._setup_env()
    source, video_source, py, marker = ROOT/'source', ROOT/'videoprism', ROOT/'venv/bin/python', ROOT/'runtime-v1.json'
    _source(job, source, 'https://github.com/QwenAudio/ThinkSound.git', SOURCE_REV, env)
    _source(job, video_source, 'https://github.com/google-deepmind/videoprism.git', VIDEO_REV, env)
    if not marker.is_file() or not py.is_file():
        py = common._prepare_venv(job, ROOT)
        torch_check = 'import torch,torchaudio,torchvision; assert torch.__version__.split("+")[0]=="2.6.0" and torch.cuda.is_available() and torch.cuda.is_bf16_supported()'
        try:
            common._command([str(py),'-c',torch_check],job,stage='Comprobando PyTorch PrismAudio',timeout=90,env=env)
        except RuntimeError:
            common._command([str(py),'-m','pip','install','--index-url','https://download.pytorch.org/whl/cu124',
                             'torch==2.6.0','torchvision==0.21.0','torchaudio==2.6.0'],job,stage='Instalando PyTorch PrismAudio',timeout=7200,env=env)
        # Use inference dependencies, omitting training/server-only packages.
        # audiotools' obsolete protobuf pin conflicts with JAX/TensorFlow.
        common._command([str(py),'-m','pip','install','numpy==1.26.4','transformers==4.57.3','av==14.0.1',
                         'jax[cuda12]==0.5.3','flax==0.10.4','tensorflow-cpu==2.19.1','einshape',
                         'mediapy','moviepy','pandas==2.2.3','lightning==2.5.1.post0','prefigure==0.0.10',
                         'wandb==0.18.7','alias-free-torch==0.0.6','einops-exts==0.0.4','ema-pytorch==0.2.3',
                         'encodec==0.1.1','k-diffusion==0.1.1','open_clip_torch==2.31.0','timm==1.0.19',
                         'vector-quantize-pytorch==1.9.14','x-transformers==1.26.6','sentencepiece==0.2.1',
                         'omegaconf==2.3.0','hydra-core==1.3.2','safetensors','librosa==0.10.2.post1',
                         'argbind','pyloudnorm','julius','ffmpy','ipython','rich','matplotlib','pystoi',
                         'torch-stoi','flatten-dict','markdown2','randomname','tensorboard'],job,
                        stage='Instalando dependencias PrismAudio y VideoPrism',timeout=7200,env=env)
        common._command([str(py),'-m','pip','install','--no-deps','descript-audiotools==0.7.2','descript-audio-codec==1.0.0'],job,
                        stage='Instalando códec PrismAudio',timeout=1800,env=env)
        check = ('import sys; sys.path[:0]=[sys.argv[1],sys.argv[2]]; import torch,jax,tensorflow,av; '
                 'from transformers import T5GemmaForConditionalGeneration; from videoprism import models; '
                 'import data_utils.prismaudio_data_process,predict; from PrismAudio.models import diffusion,autoencoders; '
                 'assert torch.cuda.is_available() and torch.cuda.is_bf16_supported() and jax.default_backend()=="gpu"')
        common._command([str(py),'-c',check,str(source),str(video_source)],job,stage='Verificando runtime JAX/CUDA PrismAudio',timeout=180,env=env)
        marker.write_text(json.dumps({'source':SOURCE_REV,'videoprism':VIDEO_REV}),encoding='utf8')
    return py
