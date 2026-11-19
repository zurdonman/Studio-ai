"""Fixed subprocess entrypoints for optional upstream 3D runtimes.

Runs in each administrator-prepared venv. Input JSON is created by Studio, never
accepted as a command or script from an API client. No weights are bundled.
"""
import importlib.util
import json
import os
import sys
from pathlib import Path


def main(cfg):
    engine = cfg['engine']
    out = Path(cfg['output'])
    out.mkdir(parents=True, exist_ok=True)
    repo = cfg['repo']
    sys.path.insert(0, repo)
    image_path = cfg.get('image')
    if cfg.get('variant') == 'onnx':
        from scene_onnx_worker import run
        run(cfg)
    elif engine in ('hunyuan2', 'hunyuanmini', 'hunyuan21'):
        import torch
        from PIL import Image
        image = Image.open(image_path).convert('RGBA')
        if engine == 'hunyuan21':
            sys.path.insert(0, str(Path(repo) / 'hy3dshape'))
            sys.path.insert(0, str(Path(repo) / 'hy3dpaint'))
            from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline as Shape
            from hy3dshape.rembg import BackgroundRemover
        else:
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline as Shape
            from hy3dgen.rembg import BackgroundRemover
        # Only remove when alpha is fully opaque; preserve user-supplied masks.
        if image.getextrema()[3][0] == 255:
            image = BackgroundRemover()(image.convert('RGB'))
        subfolder = {
            'hunyuan2': 'hunyuan3d-dit-v2-0' + ({'base':'', 'fast':'-fast', 'turbo':'-turbo'}[cfg['variant']]),
            'hunyuanmini': 'hunyuan3d-dit-v2-mini' + ({'base':'', 'fast':'-fast', 'turbo':'-turbo'}[cfg['variant']]),
            'hunyuan21': 'hunyuan3d-dit-v2-1',
        }[engine]
        model = Shape.from_pretrained(cfg['weights'], subfolder=subfolder, use_safetensors=engine != 'hunyuan21')
        mesh = model(image=image, num_inference_steps=cfg['steps'], generator=torch.Generator(device='cuda').manual_seed(cfg['seed']))[0]
        del model
        torch.cuda.empty_cache()
        shape = out / 'shape.glb'
        mesh.export(str(shape))
        if cfg['texture']:
            if engine == 'hunyuan21':
                from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
                conf = Hunyuan3DPaintConfig(6, 512)
                conf.multiview_pretrained_path = cfg['paint_weights']
                conf.realesrgan_ckpt_path = cfg.get('realesrgan_path') or str(Path(repo) / 'hy3dpaint/ckpt/RealESRGAN_x4plus.pth')
                conf.multiview_cfg_path = str(Path(repo) / 'hy3dpaint/cfgs/hunyuan-paint-pbr.yaml')
                conf.custom_pipeline = str(Path(repo) / 'hy3dpaint/hunyuanpaintpbr')
                painter = Hunyuan3DPaintPipeline(conf)
                painter(mesh_path=str(shape), image_path=image, output_mesh_path=str(out / 'result.obj'))
            else:
                from hy3dgen.texgen import Hunyuan3DPaintPipeline
                painter = Hunyuan3DPaintPipeline.from_pretrained(cfg['paint_weights'])
                mesh = painter(mesh, image=image)
                mesh.export(str(out / 'result.glb'))
        else:
            shape.rename(out / 'result.glb')
    elif engine == 'trellis2':
        from PIL import Image
        import o_voxel
        from trellis2.pipelines import Trellis2ImageTo3DPipeline
        pipeline = Trellis2ImageTo3DPipeline.from_pretrained(cfg['weights'])
        pipeline.cuda()
        mesh = pipeline.run(Image.open(image_path), seed=cfg['seed'])[0]
        mesh.simplify(16777216)
        glb = o_voxel.postprocess.to_glb(vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs,
            coords=mesh.coords, attr_layout=mesh.layout, voxel_size=mesh.voxel_size,
            aabb=[[-0.5,-0.5,-0.5],[0.5,0.5,0.5]], decimation_target=1000000,
            texture_size=cfg['texture_resolution'], remesh=True, remesh_band=1, remesh_project=0, verbose=True)
        glb.export(str(out / 'result.glb'), extension_webp=True)
    elif engine == 'idol':
        from idol.pipeline import IdolPipeline
        pipeline = IdolPipeline.from_pretrained(cfg['weights'])
        result = pipeline(image_path, mode=cfg['idol_mode'], seed=cfg['seed'], steps=cfg['steps'])
        result.gaussian.save_ply(str(out / 'result.ply'))
    elif engine == 'sharp':
        from sharp.cli import main_cli
        args = ['predict', '-i', image_path, '-o', str(out), '--device', 'cuda']
        if cfg['weights']: args.extend(['-c', cfg['weights']])
        main_cli.main(args=args, standalone_mode=False)
        (out / (Path(image_path).stem + '.ply')).rename(out / 'result.ply')
    elif engine == 'mirror':
        # Upstream CLI hardcodes the Hub ID; override only this process so the
        # pinned files prepared by Studio are actually consumed.
        import runpy
        from src.models.models.worldmirror import WorldMirror
        original = WorldMirror.from_pretrained
        WorldMirror.from_pretrained = classmethod(lambda cls, *args, **kwargs: original(cfg['weights'], **kwargs))
        sys.argv = [str(Path(repo)/'infer.py'), '--input_path', cfg['input_dir'], '--output_path', str(out)]
        runpy.run_path(str(Path(repo)/'infer.py'), run_name='__main__')
    elif engine == 'worldgen':
        import torch
        from PIL import Image
        from worldgen import WorldGen
        torch.manual_seed(cfg['seed'])
        pipeline = WorldGen(mode='i2s' if image_path else 't2s', device='cuda', low_vram=cfg['low_vram'])
        image = Image.open(image_path).convert('RGB') if image_path else None
        result = pipeline.generate_world(prompt=cfg['prompt'], image=image, return_mesh=cfg['mesh'])
        if cfg['mesh']:
            import open3d as o3d
            if not o3d.io.write_triangle_mesh(str(out / 'result.ply'), result): raise RuntimeError('No se pudo guardar la malla PLY')
        else:
            result.save(str(out / 'result.ply'))
    elif engine == 'worldsplat' and cfg.get('variant') == 'fast':
        # DiffSplat is an object generator, not WorldSplat's navigable world.
        import shutil
        import subprocess
        run_dir=out/'diffsplat'; run_dir.mkdir()
        for name in ('gsrecon_gobj265k_cnp_even4','gsvae_gobj265k_sd','gsdiff_gobj83k_sd15__render'):
            dest=run_dir/name; dest.mkdir()
            (dest/'checkpoints').symlink_to(Path(cfg['weights'])/name/'checkpoints',target_is_directory=True)
        prompt_file=out/'prompt.txt'
        prompt_file.write_text(' '.join(cfg['prompt'].split())+'\n',encoding='utf8')
        args=[sys.executable,str(Path(repo)/'src/infer_gsdiff_sd.py'),'--config_file',str(Path(repo)/'configs/gsdiff_sd15.yaml'),
              '--tag','gsdiff_gobj83k_sd15__render','--output_dir',str(run_dir),'--prompt_file',str(prompt_file),
              '--name_by_id','--save_ply','--half_precision','--seed',str(cfg['seed'])]
        env=os.environ.copy()
        env['PYTHONPATH']=repo + os.pathsep + env.get('PYTHONPATH','')
        subprocess.run(args,cwd=repo,env=env,check=True)
        generated=list((run_dir/'gsdiff_gobj83k_sd15__render/inference').glob('*.ply'))
        if len(generated)!=1: raise RuntimeError('DIFFSPLAT_OUTPUT_MISSING: se esperaba un PLY.')
        shutil.copyfile(generated[0],out/'result.ply')
    elif engine == 'worldsplat':
        if cfg.get('mast3r_dir'):
            sys.path[:0] = [cfg['mast3r_dir'], str(Path(cfg['mast3r_dir'])/'dust3r')]
        from worldsplat.video_gen import generate_video
        from worldsplat.pose_estimation import estimate_poses
        from worldsplat.splat_builder import build_splats
        # Do not call upstream run_pipeline: it launches a blocking viewer.
        frames = generate_video(prompt=cfg['prompt'], output_dir=out, seed=cfg['seed'])
        poses = estimate_poses(frames, out)
        result = Path(build_splats(frames, poses, out))
        if result.resolve() != (out / 'result.ply').resolve():
            import shutil
            shutil.copyfile(result, out / 'result.ply')
    elif engine == 'world2':
        # Official five-stage WorldNav/WorldStereo pipeline; an independently
        # configured vLLM service and compatible multi-GPU runtime are required.
        import shutil
        import subprocess
        world_dir = Path(repo) / 'hyworld2/worldgen'
        scene = out / 'scene'
        scene.mkdir(parents=True)
        shutil.copyfile(image_path, scene / 'panorama.png')
        training = out / 'training'
        common = ['--llm_addr', cfg['vlm_host'], '--llm_port', cfg['vlm_port'], '--llm_name', cfg['vlm_model']]
        dist = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node', str(cfg['gpus'])]
        epochs = {1:8000, 2:4000, 3:3000, 4:2000, 5:1800, 6:1700, 7:1600, 8:1500}[cfg['gpus']]
        stages = [
            [sys.executable, 'traj_generate.py', '--target_path', str(scene), *common, '--apply_nav_traj', '--apply_up_route', '--apply_recon_iteration', '--force_vlm'],
            [*dist, 'traj_render.py', '--target_path', str(scene), *common],
            [*dist, 'video_gen.py', '--target_path', str(scene), '--fsdp'],
            [*dist, 'gen_gs_data.py', '--root_path', str(scene), '--save_normal', '--split_sky'],
            [sys.executable, '-m', 'world_gs_trainer', 'default', '--data_dir', str(scene/'gs_data'), '--result_dir', str(training),
             '--max_steps', str(epochs), '--save_steps', str(epochs), '--eval_steps', str(epochs), '--ply_steps', str(epochs),
             '--save_ply', '--convert_to_spz', '--disable_video', '--use_scale_regularization', '--antialiased', '--depth_loss', '--normal_loss',
             '--sky_depth_from_pcd', '--use_mask_gaussian', '--mask_export_stochastic', '--no-mask-export-anchor-protection', '--use_anchor_protection', '--export_mesh',
             '--strategy.refine-start-iter', '150', '--strategy.refine-stop-iter', '750', '--strategy.refine-every', '100',
             '--strategy.refine-scale2d-stop-iter', '750', '--strategy.reset-every', '99990', '--strategy.grow-grad2d', '0.0001', '--strategy.prune-scale3d', '0.1'],
        ]
        for index, args in enumerate(stages, 1):
            print(f'HY-World 2.0: etapa {index}/5', flush=True)
            subprocess.run(args, cwd=str(world_dir), check=True)
    elif engine == 'world1':
        if cfg.get('hunyuanworld_dir'):
            sys.path.insert(0, cfg['hunyuanworld_dir'])
            os.chdir(cfg['hunyuanworld_dir'])
        import numpy as np
        import torch
        from PIL import Image
        spec = importlib.util.spec_from_file_location('studio_world1_node', str(Path(repo) / 'hunyuan_world_node.py'))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        stage = out / 'stage'
        stage.mkdir()
        mod._get_output_dir = lambda: str(stage)
        image = None
        if image_path:
            image = torch.from_numpy(np.asarray(Image.open(image_path).convert('RGB')).astype('float32') / 255.0).unsqueeze(0)
        mod.HYWorldFullPipelineNode().run_pipeline(prompt=cfg['prompt'], image=image, seed=cfg['seed'],
            target_resolution=1920, fp8_quantization=cfg['fp8'], output_dir=str(out))
    else:
        raise ValueError('Unknown 3D engine')


if __name__ == '__main__':
    with open(sys.argv[1], encoding='utf-8') as f:
        main(json.load(f))
