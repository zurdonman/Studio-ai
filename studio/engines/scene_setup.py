"""Per-engine opt-in installers. Runs only pinned, predefined recipes.

Nothing from an HTTP client is interpreted as shell or a package/source path.
The managed venv is private; an interrupted or failed install is never marked ready.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import threading
from pathlib import Path

from studio.config import DATA_ROOT
from studio.engines import ltx_gguf_backend as common

ROOT = DATA_ROOT / 'runtimes' / 'scene3d'
WEIGHTS = DATA_ROOT / 'models' / 'scene3d'
SETUP_VERSION = 2
_lock = threading.Lock()

# Fixed revisions inspected in the corresponding upstream projects.
REPOS = {
 'sf3d': ('Stability-AI/stable-fast-3d','ff21fc491b4dc5314bf6734c7c0dabd86b5f5bb2'),
 'triposr': ('VAST-AI-Research/TripoSR','107cefdc244c39106fa830359024f6a2f1c78871'),
 'idol': ('speridlabs/idol-sdk','4702a7f00850e34b6e83a2170cc123c3e9570744'),
 'sharp': ('apple-aiml-research/ml-sharp','aed6527499ef91cba3b54c18d49a870f25947190'),
 'hunyuan2': ('Tencent-Hunyuan/Hunyuan3D-2','f8db63096c8282cb27354314d896feba5ba6ff8a'),
 'hunyuanmini': ('Tencent-Hunyuan/Hunyuan3D-2','f8db63096c8282cb27354314d896feba5ba6ff8a'),
 'trellisgguf': ('pwilkin/trellis.cpp','6851891d706e2e26a8bb9dd99f28de851d68b64d'),
 'mirror2': ('Tencent-Hunyuan/HY-World-2.0','df9988efb87bfc0f4947eb3889411cf957478b06'),
 'mirror': ('Tencent-Hunyuan/HunyuanWorld-Mirror','c61300e47ce06db2f02c8e2551aae68be7ae3a0b'),
 'worldsplat': ('timscheuerai/worldsplat','075853fabdd2376ffb53fbefd5047e3a736989dc'),
 'worldgen': ('ZiYang-xie/WorldGen','7ce7b2767fdf31e2727b69a2e61e2e950e3a017f'),
 'trellis2': ('microsoft/TRELLIS.2','75fbf0183001ed9876c8dbb35de6b68552ee08bd'),
 'hunyuan21': ('Tencent-Hunyuan/Hunyuan3D-2.1','82920d643c0dc2f7bfd7255f45f62d386edfe60c'),
 'world2': ('Tencent-Hunyuan/HY-World-2.0','df9988efb87bfc0f4947eb3889411cf957478b06'),
 'world1': ('krmahil/comfyui-hunyuan-world','3e45e6e0f3e4d438d57f1e7dd18d4c8d5d95c192'),
 'sf3donnx': ('Stability-AI/stable-fast-3d','ff21fc491b4dc5314bf6734c7c0dabd86b5f5bb2'),
 'hunyuanonnx': ('Tencent-Hunyuan/Hunyuan3D-2','f8db63096c8282cb27354314d896feba5ba6ff8a'),
 'diffsplat': ('chenguolin/DiffSplat','a99d75707541009283bc5c073f1a07b5add41d00'),
}
MAST3R_REPO = ('naver/mast3r', 'f5209afc300cec36239a7ac992263f36847bbba0')
# Some stacks have external services/unresolved native/model dependencies; do not
# advertise them as automatically prepared until an end-to-end recipe exists.
AUTOMATED = {'sf3d', 'triposr', 'idol', 'sharp', 'hunyuan2', 'hunyuanmini', 'trellisgguf', 'mirror2', 'mirror', 'worldsplat', 'worldgen', 'trellis2', 'hunyuan21', 'world1', 'world2', 'sf3donnx', 'hunyuanonnx', 'diffsplat'}
HF_REVISIONS = {
 'sf3d': ('stabilityai/stable-fast-3d','f0c9a8ffd62cb1bbc8a7a53c9f87a0be1b6be778'),
 'triposr': ('stabilityai/TripoSR','5b521936b01fbe1890f6f9baed0254ab6351c04a'),
 'idol': ('speridlabs/idol','2731c2291fc65a797c5cb627a5e5526fc1490902'),
 'hunyuan2': ('tencent/Hunyuan3D-2','9cd649ba6913f7a852e3286bad86bfa9a2d83dcf'),
 'hunyuanmini': ('tencent/Hunyuan3D-2mini','f90a0f7df7d5e6f71109cf333f6a95a0ae3194a6'),
 'trellisgguf': ('ilintar/trellis2-gguf','a57397bd3d351599d9729fc144b3f87c3f87d65b'),
 'mirror2': ('tencent/HY-World-2.0','d78a16c91c7a56488894a1c8de4f5c7cc28aa8b0'),
 'mirror': ('tencent/HunyuanWorld-Mirror','5574b7b0d5ac9d80e8a92976222370a0d20a57a0'),
 'trellis2': ('microsoft/TRELLIS.2-4B','af44b45f2e35a493886929c6d786e563ec68364d'),
 'hunyuan21': ('tencent/Hunyuan3D-2.1','0b94677654c57bb9a6b6845cd7b704ccf551d327'),
 'realesrgan': ('lllyasviel/Annotators','982e7edaec38759d914a963c48c4726685de7d96'),
 'zim': ('naver-iv/zim-anything-vitl','667e2d7c233f6f1cacd12ccc64bdf6cc7b5aa16d'),
 'sf3donnx': ('needle-tools/SF3D-webgpu','56d2f58d27395b0be9c633671b439b4aac482549'),
 'hunyuanonnx': ('creador/hunyuan3d-2mini-turbo-onnx','a7ba7383016f58d2552876244864bf5a91700eb9'),
 'diffsplat': ('chenguolin/DiffSplat','8c698c39ad577e66bb28ae8e2cbb0a02d94743e7'),
}


def runtime_key(engine, variant='base'):
    if engine == 'worldsplat' and variant == 'fast': return 'diffsplat'
    if variant == 'onnx' and engine in ('sf3d','hunyuanmini'): return 'sf3donnx' if engine == 'sf3d' else 'hunyuanonnx'
    return 'trellisgguf' if engine == 'trellis2' and variant in ('q4', 'q8') else engine


ONNX_FILES = {
 'sf3donnx': {'onnx/image_tokenizer_single.onnx':'cc8c76276a9cf86ece8a854827ed570bdbf71e458df09bde5664af46af29c21e', 'onnx/backbone_fp16.onnx':'8bcde6d22589e8bbb753c4ca1a91f2c800f27a794b75405ef0dbee6b07b0da12', 'onnx/decoder_single.onnx':'ec4655df567128cc86b18b8a86b7b79d092d5223c8521a6d9993d08209e2785d', 'features_mlp_weights.json':'b493e908039ebb70bf99f451b3ff85dc70cb2a62b29cefd17453ebdf27b8b4d7'},
 'hunyuanonnx': {'dinov2_giant_1022_int8.onnx':'8429fb4bde0bccf4ed87429cd457f18993bcbc88a090512ea93192c8931c0de2', 'dit_turbo.onnx':'c86df6cba75bb91874d521004a650d3d5ac7ce6b2406955e62ca6d54a9b6be7a', 'vae_kv.onnx':'e508ca1b332394f0c996fd0157f4f0b4a4f76178e8fcbaa13e3e4a51b9de217e', 'vae_query.onnx':'22c85052811c07ebac315b55a2c94a767723105767ed257832167d1a5838f310'}
}

GGUF_FILES = ('ss_flow.gguf', 'ss_dec.gguf', 'shape_flow_512.gguf', 'shape_flow_1024.gguf', 'shape_dec.gguf', 'tex_flow_512.gguf', 'tex_flow_1024.gguf', 'tex_dec.gguf', 'dinov3.gguf', 'birefnet.gguf')
TRELLIS_MODELS = ('ss_flow_img_dit_1_3B_64_bf16', 'shape_dec_next_dc_f16c32_fp16',
                  'slat_flow_img2shape_dit_1_3B_512_bf16', 'slat_flow_img2shape_dit_1_3B_1024_bf16',
                  'tex_dec_next_dc_f16c32_fp16', 'slat_flow_imgshape2tex_dit_1_3B_512_bf16',
                  'slat_flow_imgshape2tex_dit_1_3B_1024_bf16')

def _shape_folder(engine, variant):
    base = 'hunyuan3d-dit-v2-0' if engine == 'hunyuan2' else 'hunyuan3d-dit-v2-mini'
    return base + {'base':'', 'fast':'-fast', 'turbo':'-turbo'}[variant]

def managed_config(engine, variant='base'):
    """Read only a matching ready marker. Never trust paths from marker contents."""
    key = runtime_key(engine, variant)
    if key not in AUTOMATED: return None
    root = ROOT / key
    marker = root / 'ready.json'
    try:
        data = json.loads(marker.read_text(encoding='utf8'))
        if data != {'version': SETUP_VERSION, 'revision': REPOS[key][1]}: return None
        if root.is_symlink() or marker.is_symlink(): return None
        if key == 'trellisgguf':
            exe = root/'source/build/trellis-cli'; weights = WEIGHTS/key/variant
            if variant not in ('q4','q8') or not exe.is_file() or not all((weights/f).is_file() for f in GGUF_FILES): return None
            return {'exe':str(exe), 'weights':str(weights)}
        py = root/'venv/bin/python'; repo = root/'source'
        source_files = {'sf3d':'run.py','triposr':'run.py','idol':'idol/pipeline.py','sharp':'src/sharp/cli/predict.py', 'hunyuan2':'hy3dgen/shapegen/pipelines.py', 'hunyuanmini':'hy3dgen/shapegen/pipelines.py', 'mirror2':'hyworld2/worldrecon/pipeline.py', 'mirror':'infer.py', 'worldsplat':'worldsplat/pipeline.py', 'worldgen':'src/worldgen/worldgen.py', 'trellis2':'trellis2/pipelines/trellis2_image_to_3d.py', 'hunyuan21':'hy3dshape/hy3dshape/pipelines.py', 'world1':'hunyuan_world_node.py', 'world2':'hyworld2/worldgen/traj_generate.py', 'sf3donnx':'run.py', 'hunyuanonnx':'hy3dgen/shapegen/pipelines.py', 'diffsplat':'src/infer_gsdiff_sd.py'}
        if not py.is_file() or not (repo/source_files[key]).is_file(): return None
        cfg = {'repo':str(repo), 'python':str(py)}
        if key == 'diffsplat':
            path=WEIGHTS/key
            required=('gsrecon_gobj265k_cnp_even4/checkpoints/124320/model.safetensors', 'gsvae_gobj265k_sd/checkpoints/039030/model.safetensors', 'gsdiff_gobj83k_sd15__render/checkpoints/013020/unet_ema/diffusion_pytorch_model.safetensors')
            if not all((path/f).is_file() for f in required): return None
            cfg['weights']=str(path)
        if key == 'worldsplat':
            mast3r = root/'mast3r'
            if not (mast3r/'mast3r/model.py').is_file() or not (mast3r/'dust3r/dust3r/model.py').is_file(): return None
            cfg['mast3r'] = str(mast3r)
        if key in ONNX_FILES:
            path=WEIGHTS/key
            if not all((path/f).is_file() for f in ONNX_FILES[key]): return None
            cfg['weights']=str(path)
            return cfg
        if engine in ('sf3d','triposr'):
            files = ('config.yaml','model.safetensors' if engine=='sf3d' else 'model.ckpt')
            path = WEIGHTS/engine
            if not all((path/f).is_file() for f in files): return None
            cfg['weights'] = str(path)
        if engine == 'trellis2':
            path = WEIGHTS/engine
            if not (path/'pipeline.json').is_file() or not all((path/'ckpts'/f'{name}.{ext}').is_file() for name in TRELLIS_MODELS for ext in ('json', 'safetensors')): return None
            cfg['weights'] = str(path)
        if engine == 'hunyuan21':
            path = WEIGHTS/key
            required = ('hunyuan3d-dit-v2-1/config.yaml', 'hunyuan3d-dit-v2-1/model.fp16.ckpt',
                        'hunyuan3d-vae-v2-1/config.yaml', 'hunyuan3d-vae-v2-1/model.fp16.ckpt',
                        'hunyuan3d-paintpbr-v2-1/model_index.json', 'hunyuan3d-paintpbr-v2-1/unet/diffusion_pytorch_model.bin', 'RealESRGAN_x4plus.pth')
            if not all((path/f).is_file() for f in required): return None
            cfg.update(weights=str(path), paint_weights=str(path), realesrgan=str(path/'RealESRGAN_x4plus.pth'))
        if key == 'world1':
            hw = root/'hunyuanworld'
            if not (hw/'hy3dworld/__init__.py').is_file() or not all((hw/'ZIM/zim_vit_l_2092'/f).is_file() for f in ('encoder.onnx','decoder.onnx')): return None
            cfg['hunyuanworld'] = str(hw)
        if engine == 'idol':
            path = WEIGHTS/engine
            if not all((path/f).is_file() for f in ('config.json','denoiser.safetensors','decoder.safetensors','decoder_gs.safetensors','birefnet.safetensors')): return None
            cfg['weights'] = str(path)
        if key in ('hunyuan2', 'hunyuanmini'):
            if variant not in ('base','fast','turbo'): return None
            path = WEIGHTS/key
            sub = path/_shape_folder(key,variant)
            paint = WEIGHTS/'hunyuan2-paint'
            required = [sub/'config.yaml', sub/'model.fp16.safetensors', paint/'hunyuan3d-delight-v2-0/model_index.json', paint/'hunyuan3d-paint-v2-0-turbo/unet/diffusion_pytorch_model.bin']
            if not all(p.is_file() for p in required): return None
            cfg.update(weights=str(path), paint_weights=str(paint))
        if key == 'mirror':
            path=WEIGHTS/key
            if not all((path/f).is_file() for f in ('config.json','model.safetensors')): return None
            cfg['weights']=str(path)
        if key == 'mirror2':
            path=WEIGHTS/key
            if not all((path/'HY-WorldMirror-2.0'/f).is_file() for f in ('config.json','model.safetensors')): return None
            cfg['weights']=str(path)
        return cfg
    except (OSError, ValueError, KeyError, TypeError): return None


def _clone(job, dest, repo, revision, recursive=False):
    if dest.is_symlink(): raise RuntimeError('3D_SETUP_CONFLICT: enlace de repositorio no permitido')
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        common._command(['git','clone','https://github.com/'+repo+'.git',str(dest)],job,stage='Descargando código '+repo,timeout=1800)
    if not (dest/'.git').is_dir(): raise RuntimeError('3D_SETUP_CONFLICT: repositorio existente no gestionado; no se modifica')
    # Refuse changes to prevent checkout from destroying user work.
    import subprocess
    check = subprocess.run(['git','-C',str(dest),'status','--porcelain','--untracked-files=no'],capture_output=True,text=True,check=True)
    if check.stdout.strip(): raise RuntimeError('3D_SETUP_CONFLICT: código modificado; no se sobrescribe')
    common._command(['git','-C',str(dest),'checkout','--detach',revision],job,stage='Fijando versión '+repo,timeout=300)
    if recursive:
        common._command(['git','-C',str(dest),'-c','url.https://github.com/.insteadOf=git@github.com:','submodule','update','--init','--recursive'],job,stage='Preparando submódulos '+repo,timeout=1800)


def _pip(job, py, repo, *args, stage='Instalando dependencias 3D', timeout=5400):
    env = common._setup_env()
    env['PATH'] = str(py.parent) + os.pathsep + env.get('PATH','')
    env.setdefault('MAX_JOBS', '2')
    common._command([str(py),'-m','pip','install','--disable-pip-version-check',*args],job,stage=stage,timeout=timeout,cwd=str(repo),env=env)


def _install_env(job, key, root, repo):
    venv = root/'venv'; py = venv/'bin/python'
    if venv.is_symlink() or (venv.exists() and not (venv/'pyvenv.cfg').is_file()):
        raise RuntimeError('3D_SETUP_CONFLICT: entorno existente no gestionado; no se borra')
    env = common._setup_env()
    if not py.exists():
        version = '3.13' if key == 'sharp' else '3.11' if key in ('mirror2', 'worldgen', 'world2') else '3.10' if key in ('mirror', 'trellis2', 'hunyuan21', 'world1', 'diffsplat') else None
        executable = shutil.which('python'+version) if version else sys.executable
        if not executable: raise RuntimeError(f'3D_PYTHON_REQUIRED: este motor requiere python{version}; instálalo antes de preparar el motor.')
        common._command([executable,'-m','venv',str(venv)],job,stage='Creando entorno Python aislado',timeout=300,env=env)
    _pip(job,py,repo,'--upgrade','pip==24.0' if key == 'world1' else 'pip','wheel','setuptools==69.5.1','ninja','packaging', stage='Preparando herramientas de compilación')
    if key in ONNX_FILES:
        _pip(job,py,repo,'onnxruntime-gpu[cuda,cudnn]==1.22.0','numpy<2','Pillow','scikit-image','trimesh',stage='Instalando runtime ONNX CUDA')
        return py
    torch = ('torch==2.5.1','torchvision==0.20.1')
    if key == 'sharp': torch = ('torch==2.8.0','torchvision==0.23.0')
    elif key in ('mirror2', 'worldgen', 'world2'): torch = ('torch==2.7.1','torchvision==0.22.1')
    elif key == 'mirror': torch = ('torch==2.4.0','torchvision==0.19.0')
    elif key == 'diffsplat': torch = ('torch==2.3.1','torchvision==0.18.1')
    elif key == 'trellis2': torch = ('torch==2.6.0','torchvision==0.21.0')
    cuda_wheels = 'cu121' if key == 'diffsplat' else 'cu128' if key in ('mirror2', 'sharp', 'worldgen', 'world2') else 'cu124'
    _pip(job,py,repo,*torch,'--index-url','https://download.pytorch.org/whl/'+cuda_wheels,stage='Instalando PyTorch en el entorno aislado')
    if key == 'sf3d':
        _pip(job,py,repo,'--no-build-isolation','-r','requirements.txt',stage='Dependencias y extensiones de Stable Fast 3D')
    elif key == 'triposr':
        # The upstream requirements use a moving torchmcubes git HEAD; pin it instead.
        _pip(job,py,repo,'--no-build-isolation','git+https://github.com/tatsy/torchmcubes.git@a9ac95aa329957ba646e7bb057f825f5dfa77f38',stage='Compilando torchmcubes')
        # Remove the moving dependency without altering upstream source.
        req = root/'requirements-studio.txt'
        req.write_text('\n'.join(x for x in (repo/'requirements.txt').read_text().splitlines() if 'torchmcubes.git' not in x)+'\n')
        _pip(job,py,repo,'-r',str(req),stage='Dependencias TripoSR')
    elif key in ('hunyuan2','hunyuanmini'):
        _pip(job,py,repo,'-r','requirements.txt','-e','.',stage='Instalando Hunyuan 3D')
        for sub in ('hy3dgen/texgen/custom_rasterizer','hy3dgen/texgen/differentiable_renderer'):
            _pip(job,py,repo,'--no-build-isolation',str(repo/sub),stage='Compilando Hunyuan Paint')
    elif key == 'hunyuan21':
        _pip(job,py,repo,'--no-build-isolation','-r','requirements.txt',stage='Dependencias Hunyuan 2.1')
        _pip(job,py,repo,'--no-build-isolation',str(repo/'hy3dpaint/custom_rasterizer'),stage='Compilando rasterizador Hunyuan')
        renderer = repo/'hy3dpaint/DifferentiableRenderer'
        script = ("import sys,sysconfig,subprocess,pybind11; "
                  "subprocess.run(['c++','-O3','-shared','-std=c++11','-fPIC',"
                  "'-I'+pybind11.get_include(),'-I'+sysconfig.get_path('include'),"
                  "'mesh_inpaint_processor.cpp','-o','mesh_inpaint_processor'+sysconfig.get_config_var('EXT_SUFFIX')],check=True)")
        common._command([str(py),'-c',script],job,stage='Compilando renderer Hunyuan',timeout=1800,cwd=str(renderer),env=common._setup_env())
    elif key == 'mirror':
        _pip(job,py,repo,'-r','requirements.txt','onnxruntime','uniception','gsplat==1.5.3',stage='Instalando WorldMirror')
    elif key == 'world1':
        hw=root/'hunyuanworld'
        _clone(job,hw,'Tencent-Hunyuan/HunyuanWorld-1.0','57fa9f3a79eae1079c279968cdb9d82c75fa7c86')
        _clone(job,hw/'ZIM','naver-ai/ZIM','392e613c8a3d395ce4dda3b51edb6dabc1fd0004')
        _pip(job,py,repo,'-r','requirements.txt',stage='Instalando dependencias HunyuanWorld 1.0')
        _pip(job,py,repo,'basicsr-fixed','facexlib','gfpgan','git+https://github.com/xinntao/Real-ESRGAN.git@a4abfb2979a7bbff3f69f58f58ae324608821e27', stage='Instalando Real-ESRGAN')
        _pip(job,py,repo,'-e',str(hw/'ZIM'),stage='Instalando ZIM')
        _pip(job,py,repo,'--no-build-isolation','git+https://github.com/facebookresearch/pytorch3d.git@ddfc55127fed1a74514cd24db08a7c2e62e76b08', 'git+https://github.com/microsoft/MoGe.git@0286b495230a074aadf1c76cc5c679e943e5d1c6',stage='Compilando PyTorch3D y MoGe')
    elif key == 'world2':
        _pip(job,py,repo,'-r','requirements.txt',stage='Dependencias HY-World 2.0')
        _pip(job,py,repo,'--no-build-isolation','-e',str(repo/'hyworld2/worldgen/third_party/gsplat_maskgaussian'),stage='Compilando gsplat')
        _pip(job,py,repo,'--no-build-isolation','flash-attn',stage='Compilando FlashAttention')
        # All except PyTorch3D are pinned by upstream; pin PyTorch3D as well.
        gitreq=root/'requirements-git-studio.txt'
        gitreq.write_text((repo/'requirements_git.txt').read_text().replace('git+https://github.com/facebookresearch/pytorch3d.git', 'git+https://github.com/facebookresearch/pytorch3d.git@ddfc55127fed1a74514cd24db08a7c2e62e76b08'))
        _pip(job,py,repo,'--no-build-isolation','-r',str(gitreq),stage='Dependencias nativas HY-World')
        _pip(job,py,repo,'--no-build-isolation',str(repo/'hyworld2/worldgen/third_party/navmesh'),stage='Compilando navmesh')
    elif key == 'mirror2':
        _pip(job,py,repo,'-r','requirements.txt',stage='Dependencias de WorldMirror 2.0')
        _pip(job,py,repo,'--no-build-isolation','-e',str(repo/'hyworld2/worldgen/third_party/gsplat_maskgaussian'),stage='Compilando gsplat')
        _pip(job,py,repo,'--no-build-isolation','flash-attn',stage='Compilando FlashAttention')
    elif key == 'trellis2':
        _pip(job,py,repo,'imageio','imageio-ffmpeg','tqdm','easydict','opencv-python-headless',
             'ninja','trimesh','transformers','tensorboard','pandas','lpips','zstandard',
             'pillow','kornia','timm','scipy','huggingface-hub','safetensors',
             'git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8',
             stage='Dependencias TRELLIS.2')
        _pip(job,py,repo,'--no-build-isolation','flash-attn==2.7.3',stage='Compilando FlashAttention')
        extensions = (
            ('nvdiffrast','NVlabs/nvdiffrast','253ac4fcea7de5f396371124af597e6cc957bfae'),
            ('nvdiffrec','JeffreyXiang/nvdiffrec','b296927cc7fd01c2ac1087c8065c4d7248f72da4'),
            ('cumesh','JeffreyXiang/CuMesh','12289e1062f0603f2f0d0771b02e1395d247f26f'),
            ('flexgemm','JeffreyXiang/FlexGEMM','6dd94a859c26ee8246888502eada3dd8ad85532e'),
        )
        for name, project, revision in extensions:
            dest = root/'extensions'/name
            _clone(job, dest, project, revision, recursive=True)
            _pip(job,py,repo,'--no-build-isolation',str(dest),stage='Compilando '+name)
        _pip(job,py,repo,'--no-build-isolation',str(repo/'o-voxel'),stage='Compilando o-voxel')
    elif key == 'worldgen':
        # Avoid DA-2's conflicting torch pins and upstream's moving git refs.
        _pip(job,py,repo,'diffusers>=0.33.1','transformers>=4.48.3','xformers==0.0.31.post1',
             'py360convert','einops','pillow','scikit-image','sentencepiece','opencv-python',
             'peft','open3d','trimesh','timm','accelerate','omegaconf','tabulate',
             'git+https://github.com/EasternJournalist/utils3d.git@3913c65d81e05e47b9f367250cf8c0f7462a0900',
             stage='Dependencias WorldGen')
        _pip(job,py,repo,'https://github.com/mit-han-lab/nunchaku/releases/download/v0.2.0/nunchaku-0.2.0+torch2.7-cp311-cp311-linux_x86_64.whl',
             stage='Instalando Nunchaku para WorldGen')
        _pip(job,py,repo,'--no-deps','git+https://github.com/EnVision-Research/DA-2.git@d659838585f2bc9967c7e9367af573795271dbbb#subdirectory=src',
             stage='Instalando DA-2')
        _pip(job,py,repo,'--no-build-isolation','git+https://github.com/facebookresearch/pytorch3d.git@ddfc55127fed1a74514cd24db08a7c2e62e76b08',
             stage='Compilando PyTorch3D')
        _pip(job,py,repo,'--no-deps','-e','.',stage='Instalando WorldGen')
    elif key == 'diffsplat':
        rade = root/'rade'
        _clone(job,rade,'BaowenZ/RaDe-GS','d72f20792005ae1d6555a82aa2d15345f247604e',recursive=True)
        _pip(job,py,repo,'xformers==0.0.27',stage='Instalando xFormers DiffSplat')
        _pip(job,py,repo,'--no-build-isolation',str(rade/'submodules/diff-gaussian-rasterization'),stage='Compilando rasterizador DiffSplat')
        req=root/'diffsplat-requirements.txt'
        req.write_text((repo/'settings/requirements.txt').read_text().replace('git+https://github.com/openai/CLIP.git','git+https://github.com/openai/CLIP.git@d05afc436d78f1c48dc0dbf8e5980a9d471f35f6'))
        _pip(job,py,repo,'-r',str(req),stage='Dependencias DiffSplat')
    elif key == 'worldsplat':
        # MASt3R and DUSt3R are source trees, not pip-installable packages.
        mast3r = root/'mast3r'
        _clone(job, mast3r, *MAST3R_REPO, recursive=True)
        _pip(job,py,repo,'-e','.[all]',stage='Instalando WorldSplat y gsplat')
        _pip(job,py,repo,'-r',str(mast3r/'requirements.txt'),'-r',str(mast3r/'dust3r/requirements.txt'),stage='Dependencias MASt3R y DUSt3R')
    elif key == 'idol': _pip(job,py,repo,'-e','.',stage='Instalando Idol SDK')
    elif key == 'sharp': _pip(job,py,repo,'-r','requirements.txt',stage='Instalando Apple SHARP')
    return py


def scene_env():
    """Give subprocesses Studio's configured HF token without putting it in argv."""
    from studio.catalog import catalog_manager
    env = common._setup_env()
    token = catalog_manager.get_hf_token()
    if token:
        env['HF_TOKEN'] = token
    return env


def _snapshot(job, key, allow_patterns=None, local_dir=None):
    repo, revision = HF_REVISIONS[key]
    code = ("from huggingface_hub import snapshot_download; import sys,json; "
            "snapshot_download(repo_id=sys.argv[1],revision=sys.argv[2],allow_patterns=json.loads(sys.argv[3]),"
            "local_dir=(sys.argv[4] or None),max_workers=4)")
    env = scene_env()
    common._command([sys.executable,'-c',code,repo,revision,json.dumps(allow_patterns),str(local_dir or '')],job,
                    stage='Descargando pesos '+repo,timeout=14400,env=env)


def _get_weights(job, engine, key, variant, texture):
    if key in ONNX_FILES:
        _snapshot(job,key,list(ONNX_FILES[key]),WEIGHTS/key)
        import hashlib
        for filename, expected in ONNX_FILES[key].items():
            digest=hashlib.sha256()
            with (WEIGHTS/key/filename).open('rb') as f:
                for chunk in iter(lambda:f.read(1024*1024),b''): job.check_cancelled(); digest.update(chunk)
            if digest.hexdigest() != expected: raise RuntimeError('3D_WEIGHTS_INVALID: SHA256 no coincide: '+filename)
    elif key in ('sf3d','triposr'):
        _snapshot(job,key,['config.yaml', 'model.safetensors' if key=='sf3d' else 'model.ckpt'],WEIGHTS/key)
    elif key == 'idol': _snapshot(job,key,['config.json','*.safetensors'],WEIGHTS/key)
    elif key in ('hunyuan2','hunyuanmini'):
        sub = _shape_folder(key,variant)
        _snapshot(job,key,[sub+'/config.yaml',sub+'/model.fp16.safetensors'],WEIGHTS/key)
        _snapshot(job,'hunyuan2',['hunyuan3d-delight-v2-0/*','hunyuan3d-paint-v2-0-turbo/*'],WEIGHTS/'hunyuan2-paint')
    elif key == 'trellisgguf': _snapshot(job,key,[variant+'/*.gguf'],WEIGHTS/key)
    elif key == 'diffsplat': _snapshot(job,key,['gsrecon_gobj265k_cnp_even4/*','gsvae_gobj265k_sd/*','gsdiff_gobj83k_sd15__render/*'],WEIGHTS/key)
    elif key == 'trellis2': _snapshot(job,key,['pipeline.json','ckpts/*'],WEIGHTS/key)
    elif key == 'hunyuan21':
        _snapshot(job,key,['hunyuan3d-dit-v2-1/*','hunyuan3d-vae-v2-1/*','hunyuan3d-paintpbr-v2-1/**'],WEIGHTS/key)
        _snapshot(job,'realesrgan',['RealESRGAN_x4plus.pth'],WEIGHTS/key)
        import hashlib
        digest=hashlib.sha256()
        with (WEIGHTS/key/'RealESRGAN_x4plus.pth').open('rb') as f:
            for chunk in iter(lambda:f.read(1024*1024),b''): job.check_cancelled(); digest.update(chunk)
        if digest.hexdigest() != '4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1':
            raise RuntimeError('3D_WEIGHTS_INVALID: SHA256 de RealESRGAN no coincide.')
    elif key == 'world1': _snapshot(job,'zim',['zim_vit_l_2092/*.onnx'],ROOT/key/'hunyuanworld/ZIM')
    elif key == 'mirror2': _snapshot(job,key,['HY-WorldMirror-2.0/config.json','HY-WorldMirror-2.0/model.safetensors'],WEIGHTS/key)
    elif key == 'mirror': _snapshot(job,key,['config.json','model.safetensors'],WEIGHTS/key)
    # SHARP downloads its upstream checkpoint on first execution.


def prepare(engine, variant, texture, job):
    key = runtime_key(engine, variant)
    if key not in REPOS: raise ValueError('3D_ENGINE_UNKNOWN')
    if key not in AUTOMATED: raise RuntimeError('3D_SETUP_MANUAL: esta receta todavía requiere preparación externa; consulta la documentación.')
    valid = ('fast',) if key == 'diffsplat' else ('onnx',) if key in ONNX_FILES else ('base','fast','turbo') if engine in ('hunyuan2','hunyuanmini') else ('q4','q8') if key == 'trellisgguf' else ('base',)
    if variant not in valid: raise ValueError('3D_VARIANT_INVALID')
    while not _lock.acquire(timeout=1):
        job.check_cancelled(); job.heartbeat(step='Esperando turno de instalación 3D')
    try:
        root = ROOT/key
        if root.is_symlink(): raise RuntimeError('3D_SETUP_CONFLICT: runtime es un enlace')
        root.mkdir(parents=True,exist_ok=True); WEIGHTS.mkdir(parents=True,exist_ok=True)
        minimum_gib = 32 if key in ('trellis2', 'hunyuan21', 'world1', 'world2') else 12
        if shutil.disk_usage(root).free < minimum_gib*1024**3 or shutil.disk_usage(WEIGHTS).free < minimum_gib*1024**3:
            raise RuntimeError(f'3D_DISK_LOW: se requieren al menos {minimum_gib} GiB libres en cada ubicación para iniciar; el uso real puede ser mayor.')
        repo = root/'source'; marker=root/'ready.json'
        wanted = {'version':SETUP_VERSION,'revision':REPOS[key][1]}
        try: existing=json.loads(marker.read_text())
        except (OSError,ValueError): existing=None
        runtime_missing = not ((repo/'build/trellis-cli').is_file() if key == 'trellisgguf' else (root/'venv/bin/python').is_file())
        if key == 'worldsplat':
            runtime_missing |= not (root/'mast3r/mast3r/model.py').is_file() or not (root/'mast3r/dust3r/dust3r/model.py').is_file()
        if existing != wanted or not repo.is_dir() or runtime_missing:
            marker.unlink(missing_ok=True)
            job.log('Preparando entorno exclusivo. La descarga y compilación pueden tardar mucho.')
            _clone(job,repo,*REPOS[key],recursive=key in ('trellisgguf', 'world2'))
            if key == 'trellisgguf':
                common._command(['cmake','-S',str(repo),'-B',str(repo/'build'),'-DGGML_CUDA=ON'],job,stage='Configurando trellis.cpp CUDA',timeout=1200)
                common._command(['cmake','--build',str(repo/'build'),'--target','trellis-cli','--parallel','2'],job,stage='Compilando trellis.cpp CUDA',timeout=7200)
                if not (repo/'build/trellis-cli').is_file(): raise RuntimeError('3D_SETUP_FAILED: falta trellis-cli')
            else:
                py=_install_env(job,key,root,repo)
                checks={'sf3d':'import sf3d.system', 'triposr':'import tsr.system', 'idol':'import idol.pipeline', 'sharp':'import sharp.cli', 'hunyuan2':'import hy3dgen.shapegen; import hy3dgen.texgen', 'hunyuanmini':'import hy3dgen.shapegen; import hy3dgen.texgen', 'mirror2':'import hyworld2.worldrecon.pipeline', 'mirror':'import infer', 'worldsplat':f"import sys; sys.path[:0] = [{str(root/'mast3r')!r}, {str(root/'mast3r/dust3r')!r}]; import worldsplat.pipeline, mast3r.model, dust3r.model, gsplat", 'worldgen':'import worldgen, da2, pytorch3d, nunchaku', 'trellis2':'import trellis2.pipelines, o_voxel, cumesh, flex_gemm, nvdiffrast.torch', 'hunyuan21':"import sys; sys.path[:0]=['hy3dshape','hy3dpaint']; import hy3dshape.pipelines, textureGenPipeline", 'world1':f"import sys; sys.path.insert(0,{str(root/'hunyuanworld')!r}); import hy3dworld, open3d", 'world2':'import hyworld2.worldrecon.pipeline, pytorch3d, flash_attn', 'diffsplat':'import torch, diffusers, diff_gaussian_rasterization', 'sf3donnx':'import onnxruntime, skimage, trimesh', 'hunyuanonnx':'import onnxruntime, skimage, trimesh'}
                common._command([str(py),'-c',checks[key]],job,stage='Comprobando imports del runtime',timeout=180,cwd=str(repo),env=common._setup_env())
        _get_weights(job,engine,key,variant,texture)
        job.check_cancelled()
        expected = {'sf3d': ('config.yaml', 'model.safetensors'), 'triposr': ('config.yaml', 'model.ckpt'), 'idol': ('config.json','denoiser.safetensors','decoder.safetensors','decoder_gs.safetensors','birefnet.safetensors')}
        if key in expected and not all((WEIGHTS/key/f).is_file() for f in expected[key]):
            raise RuntimeError('3D_WEIGHTS_MISSING: faltan archivos de pesos tras la descarga.')
        tmp=marker.with_suffix('.tmp'); tmp.write_text(json.dumps(wanted),encoding='utf8'); os.replace(tmp,marker)
        if not managed_config(engine, variant):
            if existing != wanted: marker.unlink(missing_ok=True)
            raise RuntimeError('3D_WEIGHTS_MISSING: preparación incompleta; faltan archivos del modelo o runtime.')
        job.log('Entorno preparado. SHARP y algunas dependencias descargan pesos al generar; inferencia GPU no probada.')
        return {'prepared':True,'model_id':'scene3d-'+engine,'engine':engine,'variant':variant,'texture':texture}
    finally:
        _lock.release()
