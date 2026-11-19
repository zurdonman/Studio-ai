"""One-shot MMAudio worker; no network or automatic downloads during inference."""
import os
import sys
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'


def main():
    source, weights, video, out, duration, prompt, negative, seed, steps = sys.argv[1:]
    sys.path.insert(0, source)
    import torch
    import torchaudio
    import open_clip.factory as factory
    from mmaudio.ext.autoencoder import autoencoder
    from mmaudio.eval_utils import all_model_cfg, generate, load_video
    from mmaudio.model.flow_matching import FlowMatching
    from mmaudio.model.networks import get_my_mmaudio
    from mmaudio.model.utils.features_utils import FeaturesUtils

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('MMAUDIO_GPU_UNSUPPORTED: se requiere CUDA y BF16')
    root = Path(weights)
    clip = root / 'clip'
    # open_clip 2.29 fetches both of these via this function. Restrict it to the
    # verified, pinned files; unexpected dependencies fail instead of downloading.
    def local_clip(model_id, filename=None, cache_dir=None):
        if model_id not in ('apple/DFN5B-CLIP-ViT-H-14-384', 'apple/DFN5B-CLIP-ViT-H-14-378'):
            raise RuntimeError('MMAUDIO_UNEXPECTED_CLIP')
        if filename not in (None, 'open_clip_config.json'):
            raise RuntimeError('MMAUDIO_UNEXPECTED_CLIP_FILE')
        path = clip / (filename or 'open_clip_pytorch_model.bin')
        if not path.is_file(): raise RuntimeError('MMAUDIO_CLIP_MISSING')
        return str(path)
    factory.download_pretrained_from_hf = local_clip
    original_vocoder = autoencoder.BigVGANv2.from_pretrained
    def local_vocoder(model_id, **kwargs):
        if model_id != 'nvidia/bigvgan_v2_44khz_128band_512x':
            raise RuntimeError('MMAUDIO_UNEXPECTED_VOCODER')
        return original_vocoder(str(root / 'vocoder'), **kwargs)
    autoencoder.BigVGANv2.from_pretrained = local_vocoder
    model = all_model_cfg['large_44k_v2']
    model.model_path = root / 'mmaudio/weights/mmaudio_large_44k_v2.pth'
    model.vae_path = root / 'mmaudio/ext_weights/v1-44.pth'
    model.synchformer_ckpt = root / 'mmaudio/ext_weights/synchformer_state_dict.pth'
    print('Cargando MMAudio y componentes locales', flush=True)
    with torch.inference_mode():
        net = get_my_mmaudio(model.model_name).to('cuda', torch.bfloat16).eval()
        net.load_weights(torch.load(model.model_path, map_location='cuda', weights_only=True))
        features = FeaturesUtils(tod_vae_ckpt=model.vae_path, synchformer_ckpt=model.synchformer_ckpt,
                                 enable_conditions=True, mode=model.mode, need_vae_encoder=False).to('cuda', torch.bfloat16).eval()
        info = load_video(Path(video), float(duration), load_all_frames=False)
        if info.duration_sec < float(duration) - .3:
            raise RuntimeError('MMAUDIO_VIDEO_SHORT: no se decodificó el clip completo')
        seq = model.seq_cfg
        seq.duration = info.duration_sec
        net.update_seq_lengths(seq.latent_seq_len, seq.clip_seq_len, seq.sync_seq_len)
        rng = torch.Generator(device='cuda').manual_seed(int(seed))
        print('Generando efectos de sonido', flush=True)
        result = generate(info.clip_frames.unsqueeze(0), info.sync_frames.unsqueeze(0), [prompt],
                          negative_text=[negative], feature_utils=features, net=net,
                          fm=FlowMatching(min_sigma=0, inference_mode='euler', num_steps=int(steps)),
                          rng=rng, cfg_strength=4.5)
        audio = result.float().cpu()[0]
        if audio.numel() == 0: raise RuntimeError('MMAUDIO_EMPTY_AUDIO')
        torchaudio.save(out, audio, seq.sampling_rate)


if __name__ == '__main__':
    main()
