"""Pinned, optional LTX 2.5 MSR image + voice conditioning (experimental)."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from studio.engines import ltx_gguf_backend as common
from studio.memory_manager import memory_manager

REV = '03061c18f8a79d26f3c63931dd8e28f5e5d872a6'
LORA = ('msr_voice', 'LiconStudio/LTX-2.5-Multiple-Subject-Reference',
        'aa19b48f845995525d8ace60cdc89b842e951ce6', 'LTX-2.5-Licon-MSR-V2.safetensors',
        2215094328, '7cf982168d4f77f44b02943fb458f937f90c89335dcf7cf71b2f425e78bab713')
LINK = 'studio-ltx25-msr-v2.safetensors'
_verified = None


def add_msr_graph(graph: dict, reference_image: str, audio: str) -> dict:
    for name in (reference_image, audio):
        if not name or '/' in name or '\\' in name or name in ('.', '..'):
            raise ValueError('Nombre de referencia no válido')
    def node(kind, **inputs): return {'class_type': kind, 'inputs': inputs}
    # Guide the video latent before concatenating it with the independently
    # generated audio latent; crop reference frames before the video decoder.
    graph['40'] = node('LoadImage', image=reference_image)
    graph['41'] = node('LoadAudio', audio=audio)
    graph['42'] = node('LTXVAudioVAEEncode', audio=['41', 0], audio_vae=['4', 0])
    graph['43'] = node('StudioStrictMSRLoader', model=['1', 0], lora_name=LINK, strength_model=1.0)
    graph['44'] = node('ComfyUILTX25MSRMultiReferenceGuide', positive=['7', 0], negative=['7', 1],
                       vae=['3', 0], latent=graph['10']['inputs']['video_latent'], pic1=['40', 0],
                       msr_parameters=['43', 1], audio_ref1=['42', 0], strength=1.0,
                       reference_frames='25', use_tiled_encode=True, tile_size=256, tile_overlap=64)
    graph['10']['inputs']['video_latent'] = ['44', 2]
    graph['11']['inputs'].update(model=['43', 0], positive=['44', 0], negative=['44', 1])
    graph['45'] = node('LTXVCropGuides', positive=['44', 0], negative=['44', 1], latent=['16', 0])
    graph['17']['inputs']['samples'] = ['45', 2]
    return graph


def prepare_msr(job: Any, *, runtime: Path, weights: Path, model_id: str) -> bool:
    """Install only pinned source/weights; return whether the resident was stopped."""
    global _verified
    root = runtime / 'comfy' / 'custom_nodes'
    source = runtime / 'studio-msr-source'
    if not source.exists():
        root.mkdir(parents=True, exist_ok=True)
        common._command(['git', 'clone', '--no-checkout', 'https://github.com/liconstudio/ComfyUI-LTX2.5-MSR.git', str(source)],
                        job, stage='Preparando código MSR', timeout=600)
        common._command(['git', '-C', str(source), 'checkout', '--detach', REV], job, stage='Fijando versión MSR', timeout=120)
    if source.is_symlink() or not (source / '.git').is_dir():
        raise RuntimeError('MSR_SOURCE_CONFLICT: el código existente no está gestionado')
    status = subprocess.run(['git', '-C', str(source), 'status', '--porcelain', '--untracked-files=no'],
                            capture_output=True, text=True, timeout=20, check=True)
    head = subprocess.run(['git', '-C', str(source), 'rev-parse', 'HEAD'], capture_output=True, text=True, timeout=20, check=True)
    if status.stdout.strip() or head.stdout.strip() != REV:
        raise RuntimeError('MSR_SOURCE_CONFLICT: la revisión MSR o sus archivos han cambiado; no se sobrescriben')
    wrapper = root / 'StudioStrictMSR' / '__init__.py'
    local_wrapper = Path(__file__).with_name('comfy_msr_node') / '__init__.py'
    if wrapper.exists() and wrapper.read_bytes() != local_wrapper.read_bytes():
        raise RuntimeError('MSR_NODE_CONFLICT: nodo estricto diferente, no se sobrescribe')
    if not wrapper.exists():
        wrapper.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local_wrapper, wrapper)
    target = weights / 'ltx' / LORA[3]
    required = (0 if target.is_file() and target.stat().st_size == LORA[4] else LORA[4]) + 1024**3
    weights.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(weights).free < required:
        raise RuntimeError('MSR_DISK_LOW: no hay espacio para el adaptador y el margen de seguridad')
    def fingerprint():
        try:
            s = target.stat(); return (s.st_size, s.st_mtime_ns, s.st_ino)
        except OSError:
            return None
    if fingerprint() is None or fingerprint() != _verified:
        from studio.catalog import catalog_manager
        env = os.environ.copy()
        token = catalog_manager.get_hf_token()
        if token: env['HF_TOKEN'] = token
        repo = str(Path(__file__).resolve().parents[2])
        env['PYTHONPATH'] = repo + os.pathsep + env.get('PYTHONPATH', '')
        common._command([sys.executable, '-m', 'studio.engines.ltx25_msr_download', str(weights)], job,
                        env=env, cwd=repo, stage='Descargando y verificando adaptador MSR V2', timeout=7200)
        _verified = fingerprint()
        if _verified is None: raise RuntimeError('MSR_LORA_MISSING: adaptador no disponible')
    dest = runtime / 'comfy' / 'models' / 'loras' / LINK
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_symlink():
        if dest.resolve() != target.resolve(): raise RuntimeError('MSR_LINK_CONFLICT: enlace existente diferente')
    elif dest.exists():
        raise RuntimeError('MSR_LINK_CONFLICT: archivo existente no gestionado')
    else:
        dest.symlink_to(target.resolve())
    session = common._resident(model_id)
    if session is not None:
        try:
            info = common.requests.get(session.base + '/object_info/StudioStrictMSRLoader', timeout=10).json()
            available = 'StudioStrictMSRLoader' in info
        except (ValueError, common.requests.RequestException):
            available = False
        if not available:
            job.log('Reiniciando el motor una vez para activar el nodo MSR; los pesos en disco se conservan.')
            memory_manager.unload_slot('video')
            return True
    return False
