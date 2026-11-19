"""Experimental, fixed 3D subprocess adapters. Heavy runtimes are user-installed."""
from __future__ import annotations
import json
import os
import tempfile
import time
import uuid
import zipfile
from pathlib import Path

from PIL import Image, ImageOps
from studio.config import OUTPUTS_DIR, TEMP_DIR
from studio.engines.ffmpeg_utils import verify_image_file
from studio.engines import ltx_gguf_backend as common
from studio.engines.models_3d import MODELS_OUT_DIR, validate_glb
from studio.memory_manager import memory_manager, serialized_gpu_operation

WORLDS_OUT_DIR = OUTPUTS_DIR / 'worlds'
# Environment configuration is server-side only, never received from a client.
SPECS = {
    'hunyuan2': ('STUDIO_HUNYUAN2', 'Hunyuan3D 2.0', 'hy3dgen/shapegen/pipelines.py', 'tencent/Hunyuan3D-2'),
    'hunyuanmini': ('STUDIO_HUNYUANMINI', 'Hunyuan3D 2mini', 'hy3dgen/shapegen/pipelines.py', 'tencent/Hunyuan3D-2mini'),
    'hunyuan21': ('STUDIO_HUNYUAN21', 'Hunyuan3D 2.1', 'hy3dshape/hy3dshape/pipelines.py', 'tencent/Hunyuan3D-2.1'),
    'trellis2': ('STUDIO_TRELLIS2', 'TRELLIS.2', 'trellis2/pipelines/trellis2_image_to_3d.py', 'microsoft/TRELLIS.2-4B'),
    'idol': ('STUDIO_IDOL', 'Idol', 'idol/pipeline.py', 'speridlabs/idol'),
    'sharp': ('STUDIO_SHARP', 'Apple SHARP', 'src/sharp/cli/predict.py', ''),
    'worldgen': ('STUDIO_WORLDGEN', 'WorldGen', 'src/worldgen/worldgen.py', ''),
    'mirror': ('STUDIO_MIRROR', 'HunyuanWorld-Mirror', 'infer.py', ''),
    'mirror2': ('STUDIO_MIRROR2', 'WorldMirror 2.0', 'hyworld2/worldrecon/pipeline.py', 'tencent/HY-World-2.0'),
    'world2': ('STUDIO_WORLD2', 'HY-World 2.0 completo', 'hyworld2/worldgen/traj_generate.py', ''),
    'world1': ('STUDIO_WORLD1', 'HunyuanWorld 1.0', 'hunyuan_world_node.py', ''),
    'worldsplat': ('STUDIO_WORLDSPLAT', 'WorldSplat', 'worldsplat/pipeline.py', ''),
}
OBJECTS = {'hunyuan2', 'hunyuanmini', 'hunyuan21', 'trellis2', 'idol'}
WORLDS = set(SPECS) - OBJECTS


def runtime_status(engine, variant='base'):
    prefix, _, marker, _ = SPECS[engine]
    from studio.engines.scene_setup import managed_config, AUTOMATED, runtime_key
    if engine == 'trellis2' and variant in ('q4','q8'):
        manual = any(os.environ.get(k, '').strip() for k in ('STUDIO_TRELLIS_GGUF_EXE', 'STUDIO_TRELLIS_GGUF_'+variant.upper()+'_WEIGHTS'))
    else:
        manual = any(os.environ.get(prefix+s, '').strip() for s in ('_REPO', '_PYTHON'))
    managed = managed_config(engine, variant) if not manual else None
    if managed: return {'configured': True, 'missing': [], 'managed': True, 'auto_setup': runtime_key(engine, variant) in AUTOMATED, 'manual_override': False}
    if engine == 'trellis2' and variant in ('q4', 'q8'):
        exe = os.environ.get('STUDIO_TRELLIS_GGUF_EXE', '').strip()
        weights = os.environ.get('STUDIO_TRELLIS_GGUF_' + variant.upper() + '_WEIGHTS', '').strip()
        missing = []
        if not exe or not Path(exe).expanduser().is_file(): missing.append('STUDIO_TRELLIS_GGUF_EXE')
        if not weights or not (Path(weights).expanduser() / 'ss_flow.gguf').is_file(): missing.append('STUDIO_TRELLIS_GGUF_' + variant.upper() + '_WEIGHTS')
        return {'configured': not missing, 'missing': missing, 'managed': False, 'auto_setup': runtime_key(engine, variant) in AUTOMATED, 'manual_override': manual}
    repo = os.environ.get(prefix + '_REPO', '').strip()
    python = os.environ.get(prefix + '_PYTHON', '').strip()
    missing = []
    if not repo or not (Path(repo).expanduser() / marker).is_file(): missing.append(prefix + '_REPO (' + marker + ')')
    if not python or not Path(python).expanduser().is_file(): missing.append(prefix + '_PYTHON')
    if engine == 'world2':
        for name in ('STUDIO_WORLD2_VLM_HOST', 'STUDIO_WORLD2_VLM_PORT', 'STUDIO_WORLD2_VLM_MODEL'):
            if not os.environ.get(name, '').strip(): missing.append(name)
        port = os.environ.get('STUDIO_WORLD2_VLM_PORT', '')
        if port and (not port.isdigit() or not 1 <= int(port) <= 65535): missing.append('STUDIO_WORLD2_VLM_PORT válido')
        count = os.environ.get('STUDIO_WORLD2_GPUS', '1')
        if not count.isdigit() or not 1 <= int(count) <= 8: missing.append('STUDIO_WORLD2_GPUS (1–8)')
    return {'configured': not missing, 'missing': missing, 'managed': False, 'auto_setup': runtime_key(engine, variant) in AUTOMATED, 'manual_override': manual}


def validate_ply(path):
    if path.is_symlink() or not path.is_file() or not 32 <= path.stat().st_size <= 2 * 1024**3:
        raise RuntimeError('3D_OUTPUT_INVALID: PLY ausente o tamaño inválido')
    with path.open('rb') as f: header = f.read(65536)
    if not header.startswith(b'ply\n') and not header.startswith(b'ply\r\n'):
        raise RuntimeError('3D_OUTPUT_INVALID: cabecera PLY inválida')
    if b'end_header' not in header or b'element vertex ' not in header:
        raise RuntimeError('3D_OUTPUT_INVALID: PLY sin vértices o cabecera incompleta')


def _bundle(out, target, engine, job):
    """Copy only known output types. No symlinks, input frames or unbounded archives."""
    if engine == 'world2':
        files = sorted(p for p in (out/'training').rglob('*') if p.is_file() and p.suffix.lower() in ('.ply', '.spz', '.glb', '.json'))
        scene = out/'scene'
        files += sorted(p for p in scene.rglob('*') if p.is_file() and (p.name in ('camera.json', 'global_normal.npy') or (p.suffix.lower() == '.png' and any(part in ('depth', 'depths', 'normal', 'normals') for part in p.relative_to(scene).parts))))
    elif engine == 'world1':
        files = sorted(p for p in out.glob('mesh_layer*.ply') if p.is_file())
        if (out/'panorama.png').is_file(): files.append(out/'panorama.png')
    else:
        files = sorted(p for p in out.rglob('*') if p.is_file() and p.suffix.lower() in ('.ply', '.png', '.json', '.npy') and
                       not any(part in ('input_frames', 'images', 'images_resized') for part in (p.relative_to(out).parts[1:] if engine == 'mirror' else p.relative_to(out).parts)))
    plys = [p for p in files if p.suffix.lower() == '.ply']
    if not plys: raise RuntimeError('3D_OUTPUT_MISSING: la reconstrucción no produjo PLY.')
    for p in plys: validate_ply(p)
    if len(files) > 1000 or sum(p.stat().st_size for p in files) > 4 * 1024**3:
        raise RuntimeError('3D_OUTPUT_TOO_LARGE: paquete superior a 4 GiB o 1000 archivos.')
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        for p in files:
            job.check_cancelled()
            if p.is_symlink() or out.resolve() not in p.resolve().parents: raise RuntimeError('3D_OUTPUT_INVALID: enlace externo')
            z.write(p, str(p.relative_to(out)))


@serialized_gpu_operation(slots=(), managed=True)
def generate(engine, images, prompt, variant, texture, texture_resolution, seed, steps, low_vram, mesh, idol_mode, fp8, job):
    if engine not in SPECS: raise ValueError('3D_MODEL_UNSUPPORTED')
    if variant not in ('base', 'fast', 'turbo', 'q4', 'q8'): raise ValueError('3D_VARIANT_INVALID')
    if (variant in ('q4', 'q8') and engine != 'trellis2') or (engine in ('hunyuan21', 'idol', *WORLDS) and variant != 'base'):
        raise ValueError('3D_VARIANT_INVALID')
    if engine == 'trellis2' and variant not in ('base', 'q4', 'q8'): raise ValueError('3D_VARIANT_INVALID')
    if texture_resolution not in (512, 1024, 2048) or not 1 <= steps <= 100 or not 0 <= seed <= 2**32-1:
        raise ValueError('3D_CONFIG_INVALID')
    if len(prompt) > 2000 or len(images) > 17: raise ValueError('3D_INPUT_INVALID')
    if engine in OBJECTS | {'sharp', 'mirror', 'mirror2', 'world2'} and not images: raise ValueError('3D_IMAGE_REQUIRED')
    if engine in ('worldgen', 'world1', 'worldsplat') and not images and not prompt.strip(): raise ValueError('3D_PROMPT_REQUIRED')
    if engine == 'worldsplat' and (images or not prompt.strip()): raise ValueError('WorldSplat admite solo texto en esta receta.')
    status = runtime_status(engine, variant)
    if not status['configured']: raise RuntimeError('3D_RUNTIME_NOT_CONFIGURED: ' + ', '.join(status['missing']))
    started = time.monotonic()
    prefix, label, _, default_weights = SPECS[engine]
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='studio_scene_', dir=TEMP_DIR) as temp:
        work = Path(temp); inp = work / 'images'; inp.mkdir(); out = work/'output'
        for i, path in enumerate(images):
            if path.stat().st_size > 50*1024**2: raise ValueError('3D_IMAGE_TOO_LARGE')
            w, h = verify_image_file(path)
            if w*h > 32_000_000: raise ValueError('3D_IMAGE_TOO_LARGE')
            with Image.open(path) as im: ImageOps.exif_transpose(im).convert('RGBA').save(inp / f'image_{i:02d}.png')
        first = str(inp/'image_00.png') if images else ''
        from studio.engines.scene_setup import scene_env
        env = scene_env()
        if engine == 'trellis2' and variant in ('q4','q8'):
            from studio.engines.scene_setup import managed_config
            manual = any(os.environ.get(k, '').strip() for k in ('STUDIO_TRELLIS_GGUF_EXE', 'STUDIO_TRELLIS_GGUF_'+variant.upper()+'_WEIGHTS'))
            config = None if manual else managed_config(engine, variant)
            exe = str(Path(os.environ['STUDIO_TRELLIS_GGUF_EXE'] if manual else config['exe']).expanduser().resolve())
            weights = str(Path(os.environ['STUDIO_TRELLIS_GGUF_' + variant.upper() + '_WEIGHTS'] if manual else config['weights']).expanduser().resolve())
            out.mkdir(); args = [exe, first, str(out/'result.glb'), '--models', weights, '--seed', str(seed), '--atlas', str(texture_resolution)]
            if config: args.append('--require-gpu')
            cwd = str(Path(exe).parent)
        else:
            from studio.engines.scene_setup import managed_config
            manual = any(os.environ.get(prefix+s, '').strip() for s in ('_REPO', '_PYTHON'))
            config = None if manual else managed_config(engine, variant)
            repo = str(Path(os.environ[prefix+'_REPO'] if manual else config['repo']).expanduser().resolve())
            # Preserve the venv entrypoint symlink, which selects pyvenv.cfg.
            py = os.path.abspath(os.path.expanduser(os.environ[prefix+'_PYTHON'] if manual else config['python']))
            weights = os.environ.get(prefix+'_WEIGHTS', '').strip() or (config.get('weights', default_weights) if config else default_weights)
            if Path(weights).expanduser().exists(): weights = str(Path(weights).expanduser().resolve())
            if engine == 'mirror' and not config:
                args = [py, str(Path(repo)/'infer.py'), '--input_path', str(inp), '--output_path', str(out)]
            elif engine == 'mirror':
                config_path = work/'request.json'
                config_path.write_text(json.dumps({'engine':'mirror','repo':repo,'input_dir':str(inp),'output':str(out),'weights':weights}),encoding='utf8')
                args = [py, str(Path(__file__).with_name('scene_worker.py')), str(config_path)]
            elif engine == 'mirror2':
                args = [py, '-m', 'hyworld2.worldrecon.pipeline', '--input_path', str(inp), '--strict_output_path', str(out), '--pretrained_model_name_or_path', weights, '--no_interactive']
            else:
                paint_default = 'tencent/Hunyuan3D-2.1' if engine == 'hunyuan21' else 'tencent/Hunyuan3D-2'
                paint = os.environ.get(prefix+'_PAINT_WEIGHTS', '').strip() or (config.get('paint_weights', paint_default) if config else paint_default)
                if Path(paint).expanduser().exists(): paint = str(Path(paint).expanduser().resolve())
                cfg = {'engine':engine,'repo':repo,'output':str(out),'image':first,'weights':weights,'paint_weights':paint,
                       'variant':variant,'texture':texture,'texture_resolution':texture_resolution,'steps':steps,'seed':seed,
                       'mast3r_dir': (os.environ.get('STUDIO_WORLDSPLAT_MAST3R', '') if manual else config.get('mast3r', '')) if engine == 'worldsplat' else '',
                       'prompt':prompt,'low_vram':low_vram,'mesh':mesh,'idol_mode':idol_mode,'fp8':fp8,
                       'vlm_host':os.environ.get('STUDIO_WORLD2_VLM_HOST', ''), 'vlm_port':os.environ.get('STUDIO_WORLD2_VLM_PORT', ''),
                       'vlm_model':os.environ.get('STUDIO_WORLD2_VLM_MODEL', ''), 'gpus':int(os.environ.get('STUDIO_WORLD2_GPUS', '1')) if engine == 'world2' else 1}
                config = work/'request.json'; config.write_text(json.dumps(cfg, ensure_ascii=False), encoding='utf8')
                args = [py, str(Path(__file__).with_name('scene_worker.py')), str(config)]
            cwd = repo
        job.log(f'{label}: runtime externo experimental, progreso no disponible.')
        common._command(args, job, stage=f'Generando 3D con {label}', timeout=14400, cwd=cwd, env=env, guard_label=label)
        if engine in ('mirror','mirror2','world1','world2'):
            ext = 'zip'; result = work/'bundle.zip'; _bundle(out, result, engine, job)
            description = 'Paquete con capas de malla' if engine == 'world1' else 'Generación completa: malla y splat cuando el upstream los exporte' if engine == 'world2' else 'Reconstrucción: splat, puntos, profundidad, normales y cámaras según disponibilidad'
        else:
            ext = 'glb' if engine in OBJECTS - {'idol'} else 'ply'; result = out / f'result.{ext}'
            if ext == 'glb': validate_glb(result)
            else: validate_ply(result)
            description = ('Malla PLY' if engine == 'worldgen' and mesh else 'Gaussian splat PLY') if ext == 'ply' else ('Malla GLB con textura solicitada' if texture or engine == 'trellis2' else 'Malla GLB sin textura')
        job.check_cancelled()
        folder = MODELS_OUT_DIR if engine in OBJECTS else WORLDS_OUT_DIR
        folder.mkdir(parents=True, exist_ok=True)
        name = f'{engine}_{uuid.uuid4().hex}.{ext}'; dest = folder/name
        rel = f'outputs/{"3d" if engine in OBJECTS else "worlds"}/{name}'
        meta = {'id':dest.stem,'filename':name,'rel_path':rel,'media_url':'/media/'+rel,'type':'3d' if engine in OBJECTS else 'world',
                'model_id':engine,'model_name':label,'description':description,'variant':variant,'elapsed_seconds':round(time.monotonic()-started,1)}
        created = False
        try:
            with dest.open('xb') as target, result.open('rb') as source:
                created = True
                while chunk := source.read(1024 * 1024):
                    job.check_cancelled()
                    target.write(chunk)
                job.check_cancelled()
            dest.with_suffix('.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf8')
        except BaseException:
            if created:
                dest.unlink(missing_ok=True); dest.with_suffix('.json').unlink(missing_ok=True)
            raise
        return meta
