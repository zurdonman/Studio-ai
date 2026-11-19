"""Local, opt-in official SF3D/TripoSR CLI adapters. No model code is installed automatically."""
from __future__ import annotations

import json
import os
import struct
import tempfile
import time
import uuid
from pathlib import Path

from studio.config import TEMP_DIR, OUTPUTS_DIR
from studio.engines.ffmpeg_utils import verify_image_file
from studio.engines import ltx_gguf_backend as common
from studio.memory_manager import memory_manager, serialized_gpu_operation

MODELS_OUT_DIR = OUTPUTS_DIR / '3d'

# Only user-configured server-side paths are used; API clients cannot supply executables.
SPECS = {
    'sf3d': ('STUDIO_SF3D', 'Stable Fast 3D', ('config.yaml', 'model.safetensors')),
    'triposr': ('STUDIO_TRIPOSR', 'TripoSR', ('config.yaml', 'model.ckpt')),
}


def runtime_status(engine):
    prefix, _, files = SPECS[engine]
    repo, python, weights = [os.environ.get(prefix + suffix, '').strip() for suffix in ('_REPO', '_PYTHON', '_WEIGHTS')]
    from studio.engines.scene_setup import managed_config, AUTOMATED
    managed = managed_config(engine) if not any((repo, python, weights)) else None
    if managed: return {'configured': True, 'missing': [], 'managed': True, 'auto_setup': engine in AUTOMATED, 'manual_override': False}
    missing = []
    if not repo or not (Path(repo).expanduser() / 'run.py').is_file(): missing.append(prefix + '_REPO (run.py)')
    if not python or not Path(python).expanduser().is_file(): missing.append(prefix + '_PYTHON')
    if not weights or not all((Path(weights).expanduser() / f).is_file() for f in files): missing.append(prefix + '_WEIGHTS (' + ', '.join(files) + ')')
    return {'configured': not missing, 'missing': missing, 'managed': False, 'auto_setup': engine in AUTOMATED, 'manual_override': bool(any((repo, python, weights)))}


def validate_glb(path):
    if path.is_symlink() or not path.is_file(): raise RuntimeError('3D_OUTPUT_MISSING: no se produjo un GLB.')
    size = path.stat().st_size
    if size < 20 or size > 1024**3: raise RuntimeError('3D_OUTPUT_INVALID: tamaño GLB inválido.')
    with path.open('rb') as fh:
        magic, version, length = struct.unpack('<4sII', fh.read(12))
        chunk_len, chunk_type = struct.unpack('<I4s', fh.read(8))
        if magic != b'glTF' or version != 2 or length != size or chunk_type != b'JSON' or chunk_len > size - 20 or chunk_len > 16 * 1024**2:
            raise RuntimeError('3D_OUTPUT_INVALID: cabecera GLB inválida.')
        try:
            doc = json.loads(fh.read(chunk_len))
        except (ValueError, UnicodeError) as exc:
            raise RuntimeError('3D_OUTPUT_INVALID: JSON GLB inválido.') from exc
        if not isinstance(doc, dict) or not doc.get('meshes'):
            raise RuntimeError('3D_OUTPUT_INVALID: GLB sin malla.')


@serialized_gpu_operation(slots=(), managed=True)
def generate(engine, image_path: Path, texture_resolution: int, job):
    if engine not in SPECS: raise ValueError('3D_MODEL_UNSUPPORTED')
    if texture_resolution not in (512, 1024, 2048): raise ValueError('3D_TEXTURE_INVALID')
    if image_path.stat().st_size > 50 * 1024**2: raise ValueError('3D_IMAGE_TOO_LARGE')
    w, h = verify_image_file(image_path)
    if w * h > 32_000_000: raise ValueError('3D_IMAGE_TOO_LARGE')
    status = runtime_status(engine)
    if not status['configured']: raise RuntimeError('3D_RUNTIME_NOT_CONFIGURED: ' + ', '.join(status['missing']) + '. Consulta docs/MODELOS_3D.md.')
    prefix, label, _ = SPECS[engine]
    from studio.engines.scene_setup import managed_config
    manual = [os.environ.get(prefix + s, '').strip() for s in ('_REPO', '_PYTHON', '_WEIGHTS')]
    config = dict(zip(('repo', 'python', 'weights'), manual)) if any(manual) else managed_config(engine)
    repo, weights = [str(Path(config[k]).expanduser().resolve()) for k in ('repo', 'weights')]
    # Preserve the venv python symlink: resolving it bypasses pyvenv.cfg.
    python = os.path.abspath(os.path.expanduser(config['python']))
    started = time.monotonic()
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='studio_3d_', dir=TEMP_DIR) as temp:
        work = Path(temp)
        # A canonical image avoids exposing an arbitrary original path to the child.
        from PIL import Image, ImageOps
        with Image.open(image_path) as image:
            ImageOps.exif_transpose(image).convert('RGBA').save(work / 'input.png')
        output = work / 'output'
        if engine == 'sf3d':
            args = [python, str(Path(repo).expanduser() / 'run.py'), str(work / 'input.png'), '--pretrained-model', weights,
                    '--output-dir', str(output), '--texture-resolution', str(texture_resolution)]
            result = output / '0' / 'mesh.glb'
        else:
            args = [python, str(Path(repo).expanduser() / 'run.py'), str(work / 'input.png'), '--pretrained-model-name-or-path', weights,
                    '--output-dir', str(output), '--model-save-format', 'glb']
            result = output / '0' / 'mesh.glb'
        from studio.engines.scene_setup import scene_env
        env = scene_env()
        job.log(f'{label}: ejecutando runtime local configurado; progreso no disponible.')
        common._command(args, job, stage=f'Generando malla 3D con {label}', timeout=7200, cwd=repo, env=env, guard_label=label)
        validate_glb(result)
        job.check_cancelled()
        MODELS_OUT_DIR.mkdir(parents=True, exist_ok=True)
        name = f'{engine}_{uuid.uuid4().hex}.glb'
        dest = MODELS_OUT_DIR / name
        meta = {'id': dest.stem, 'filename': name, 'rel_path': f'outputs/3d/{name}', 'media_url': f'/media/outputs/3d/{name}',
                'type': '3d', 'model_id': engine, 'model_name': label, 'elapsed_seconds': round(time.monotonic()-started, 1),
                'description': 'Malla con UV y materiales' if engine == 'sf3d' else 'Malla con colores de vértice'}
        created = False
        try:
            with dest.open('xb') as out, result.open('rb') as src:
                created = True
                while chunk := src.read(1024 * 1024):
                    job.check_cancelled()
                    out.write(chunk)
                job.check_cancelled()
            dest.with_suffix('.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf8')
        except BaseException:
            if created:
                dest.unlink(missing_ok=True)
                dest.with_suffix('.json').unlink(missing_ok=True)
            raise
        return meta
