"""Experimental ONNX CUDA image-to-mesh workers.

Hunyuan graph contract follows the community model card; SF3D uses its exported
three graphs and a sampled occupancy grid. Neither path has been run with the
real model weights in Studio. Execute only in an isolated ONNX environment.
"""
from pathlib import Path


def _image(path, size, background):
    import numpy as np
    from PIL import Image, ImageOps
    im = ImageOps.exif_transpose(Image.open(path)).convert('RGBA')
    alpha = im.getchannel('A')
    bounds = alpha.getbbox()
    if bounds: im = im.crop(bounds)
    im.thumbnail((int(size*.85), int(size*.85)), Image.Resampling.LANCZOS)
    out = Image.new('RGBA', (size,size), (*background,255))
    out.alpha_composite(im, ((size-im.width)//2, (size-im.height)//2))
    return np.asarray(out.convert('RGB'), dtype=np.float32)/255.0


def _sessions(folder, paths):
    import onnxruntime as ort
    ort.preload_dlls()
    if 'CUDAExecutionProvider' not in ort.get_available_providers():
        raise RuntimeError('ONNX_CUDA_UNAVAILABLE: onnxruntime CUDA no está disponible; revisa CUDA/cuDNN.')
    sessions = [ort.InferenceSession(str(folder/name), providers=['CUDAExecutionProvider','CPUExecutionProvider']) for name in paths]
    if any('CUDAExecutionProvider' not in s.get_providers() for s in sessions):
        raise RuntimeError('ONNX_CUDA_UNAVAILABLE: ONNX Runtime no pudo cargar el proveedor CUDA.')
    return sessions


def _surface(query, n=96, bound=1.01, level=0.0):
    import numpy as np
    from skimage.measure import marching_cubes
    axis = np.linspace(-bound,bound,n,dtype=np.float32)
    volume = np.empty(n*n*n, dtype=np.float32)
    for start in range(0,len(volume),4096):
        ix=np.arange(start,min(start+4096,len(volume)))
        points=np.stack((axis[ix//(n*n)],axis[(ix//n)%n],axis[ix%n]),axis=-1)
        volume[start:start+len(ix)] = np.asarray(query(points)).reshape(-1)
        if start % (4096*64) == 0: print('ONNX: decodificando geometría', flush=True)
    if not volume.min() < level < volume.max(): raise RuntimeError('ONNX_EMPTY_MESH: no se encontró una superficie en el nivel indicado.')
    verts, faces, _, _ = marching_cubes(volume.reshape(n,n,n), level=level, spacing=(2*bound/(n-1),)*3)
    return verts-bound, faces


def run(cfg):
    import numpy as np
    import trimesh
    folder=Path(cfg['weights']); image=cfg['image']; output=Path(cfg['output']); output.mkdir(parents=True,exist_ok=True)
    if cfg['engine'] == 'hunyuanmini':
        conditioner, dit, kv, query = _sessions(folder, ['dinov2_giant_1022_int8.onnx','dit_turbo.onnx','vae_kv.onnx','vae_query.onnx'])
        rgb=_image(image,1022,(255,255,255)).transpose(2,0,1)[None]
        cond=conditioner.run(None, {'image':rgb})[0]
        x=np.random.default_rng(cfg['seed']).standard_normal((1,3072,64),dtype=np.float32)
        sigmas=(0,199/999,399/999,599/999,799/999,1)
        for a,b in zip(sigmas,sigmas[1:]):
            v=dit.run(None,{'x':x,'t':np.array([a],dtype=np.float32),'cond':cond,'guidance':np.array([5],dtype=np.float32)})[0]
            x += (b-a)*v
        k,v=kv.run(None,{'latents':x})
        verts,faces=_surface(lambda p:query.run(None,{'points':p[None], 'k':k, 'v':v})[0])
    elif cfg['engine'] == 'sf3d':
        tokenizer, backbone, decoder = _sessions(folder,['onnx/image_tokenizer_single.onnx','onnx/backbone_fp16.onnx','onnx/decoder_single.onnx'])
        rgb=_image(image,512,(128,128,128))[None]
        c2w=np.array([[0,0,1,1.6],[1,0,0,0],[0,1,0,0],[0,0,0,1]],dtype=np.float32)[None]
        f=np.float32(.5/np.tan(np.deg2rad(20)))
        intr=np.array([[f,0,.5],[0,f,.5],[0,0,1]],dtype=np.float32)[None]
        tokens=tokenizer.run(None,{'rgb':rgb,'c2w':c2w,'intrinsic_normed':intr})[0]
        triplane=np.asarray(backbone.run(None,{'image_tokens':tokens})[0],dtype=np.float32)
        names=[o.name for o in decoder.get_outputs()]
        def density(points):
            outputs=decoder.run(None,{'triplane':triplane,'positions':points[None]})
            return outputs[names.index('density')]
        verts,faces=_surface(density,n=96,bound=1.0,level=10.0)
    else: raise ValueError('ONNX_ENGINE_INVALID')
    mesh=trimesh.Trimesh(vertices=verts,faces=faces,process=False)
    mesh.export(str(output/'result.glb'))
