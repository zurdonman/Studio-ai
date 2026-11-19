"""ControlFoley video+text worker; no reference-audio mode or network."""
import importlib.machinery
import importlib.util
import os
import sys
import types
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'


def bind_source_packages(source):
    # Both upstream directories are namespace packages. In particular this
    # worker's directory also has a controlfoley.py, which would otherwise win.
    for name in ('controlfoley', 'lib'):
        spec = importlib.machinery.ModuleSpec(name, None, is_package=True)
        spec.submodule_search_locations = [str(Path(source) / name)]
        sys.modules[name] = importlib.util.module_from_spec(spec)


def install_unused_reference_stubs():
    # Public FeaturesUtils imports the large optional reference-audio stack and
    # eagerly creates MusicGen even when no reference audio is supplied. This
    # V2A-only recipe never calls either feature; fail loudly if that changes.
    clap = types.ModuleType('laion_clap')
    def unsupported(*args, **kwargs):
        raise RuntimeError('CONTROLFOLEY_REFERENCE_AUDIO_UNSUPPORTED')
    clap.CLAP_Module = unsupported
    sys.modules['laion_clap'] = clap
    models = types.ModuleType('lib.audiocraft.audiocraft.models')
    class NoReferenceMusicGen:
        @staticmethod
        def get_pretrained(*args, **kwargs): return None
    models.MusicGen = NoReferenceMusicGen
    sys.modules['lib.audiocraft.audiocraft.models'] = models


def main():
    source, weights, video, out, duration, prompt, negative, seed, steps = sys.argv[1:]
    sys.path.insert(0, source)
    sys.path.insert(0, str(Path(source) / 'lib'))
    bind_source_packages(source)
    install_unused_reference_stubs()
    import torch
    import torchaudio
    import open_clip.factory as factory
    from lib.autoencoder import autoencoder
    from lib.flow_matching import FlowMatching
    from controlfoley.inference_utils import all_model_cfg, generate, load_video
    from controlfoley.audio_model import create_audio_generation_model
    from controlfoley.feature_extractor import FeaturesUtils
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CONTROLFOLEY_GPU_UNSUPPORTED: se requiere CUDA y BF16')
    root = Path(weights); clip = root / 'clip'; ext = root / 'control/ext_weights'
    def local_clip(model_id, filename=None, cache_dir=None):
        if model_id not in ('apple/DFN5B-CLIP-ViT-H-14-384', 'apple/DFN5B-CLIP-ViT-H-14-378') or filename not in (None, 'open_clip_config.json'):
            raise RuntimeError('CONTROLFOLEY_UNEXPECTED_CLIP')
        path = clip / (filename or 'open_clip_pytorch_model.bin')
        if not path.is_file(): raise RuntimeError('CONTROLFOLEY_CLIP_MISSING')
        return str(path)
    factory.download_pretrained_from_hf = local_clip
    bigvgan = autoencoder.BigVGANv2
    original_loader = bigvgan._from_pretrained
    def compat_loader(*args, **kwargs):
        kwargs.setdefault('proxies', None)
        kwargs.setdefault('resume_download', False)
        return original_loader(*args, **kwargs)
    bigvgan._from_pretrained = compat_loader
    original_vocoder = bigvgan.from_pretrained
    def local_vocoder(model_id, **kwargs):
        if model_id != 'nvidia/bigvgan_v2_44khz_128band_512x': raise RuntimeError('CONTROLFOLEY_UNEXPECTED_VOCODER')
        return original_vocoder(str(root / 'vocoder'), **kwargs)
    bigvgan.from_pretrained = local_vocoder
    original_load = torch.load
    def safe_load(*args, **kwargs):
        kwargs['weights_only'] = True
        return original_load(*args, **kwargs)
    torch.load = safe_load
    cfg = all_model_cfg['large_44k']
    print('Cargando ControlFoley V2A y componentes locales', flush=True)
    with torch.inference_mode():
        net = create_audio_generation_model(cfg.model_name).to('cuda', torch.bfloat16).eval()
        net.load_weights(torch.load(root / 'control/weights/controlfoley.pth', map_location='cpu', weights_only=True))
        features = FeaturesUtils(tod_vae_ckpt=str(ext/'v1-44.pth'), synchformer_ckpt=str(ext/'synchformer_state_dict.pth'),
                                 cav_mae_ckpt=str(ext/'cav_mae_st.pth'), clap_ckpt=None,
                                 enable_conditions=True, mode='44k', need_vae_encoder=False).to('cuda', torch.bfloat16).eval()
        info = load_video(Path(video), float(duration), load_all_frames=False)
        if info.total_duration < float(duration) - .5:
            raise RuntimeError('CONTROLFOLEY_VIDEO_SHORT: no se decodificó el clip completo')
        seq = cfg.seq_cfg
        seq.total_time_seconds = info.total_duration
        net.update_seq_lengths(seq.latent_sequence_length, seq.clip_sequence_length, seq.visual_sequence_length, seq.sync_sequence_length)
        rng = torch.Generator(device='cuda').manual_seed(int(seed))
        print('Generando efectos de sonido ControlFoley', flush=True)
        audio = generate(info.clip_embeddings.unsqueeze(0), info.visual_features.unsqueeze(0), info.sync_embeddings.unsqueeze(0),
                         None, None, 0.0, [prompt], negative_text=[negative], feature_utils=features, net=net,
                         fm=FlowMatching(min_sigma=0, inference_mode='euler', num_steps=int(steps)), rng=rng,
                         cfg_strength=4.5, clip_batch_size_multiplier=4, sync_batch_size_multiplier=4).float().cpu()[0]
        if audio.numel() == 0: raise RuntimeError('CONTROLFOLEY_EMPTY_AUDIO')
        torchaudio.save(out, audio, seq.audio_sample_rate)


if __name__ == '__main__':
    main()
