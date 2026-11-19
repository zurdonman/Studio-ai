"""One-shot PrismAudio feature and diffusion worker, local artifacts only."""
import csv
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'


def main():
    mode, source_s, weights_s, request_s = sys.argv[1:]
    source, weights, request = Path(source_s), Path(weights_s), Path(request_s)
    work = request.parent
    data = json.loads(request.read_text(encoding='utf8'))
    sys.path[:0] = [str(source), str(source.parent/'videoprism')]
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(): raise RuntimeError('PRISMAUDIO_GPU_UNSUPPORTED')
    if mode == 'extract':
        import jax
        if jax.default_backend() != 'gpu': raise RuntimeError('PRISMAUDIO_JAX_GPU_UNAVAILABLE')
        import transformers
        for cls in (transformers.AutoModelForSeq2SeqLM, transformers.AutoTokenizer):
            original = cls.from_pretrained
            def local(_cls, name, *args, _original=original, **kwargs):
                if name != 'google/t5gemma-l-l-ul2-it': raise RuntimeError('PRISMAUDIO_UNEXPECTED_MODEL')
                kwargs['local_files_only'] = True
                return _original(str(weights/'gemma'), *args, **kwargs)
            cls.from_pretrained = classmethod(local)
        from videoprism import models as vp
        load_vp = vp.load_pretrained_weights
        vp.load_pretrained_weights = lambda name: load_vp(name, checkpoint_path=str(weights/'videoprism/flax_lvt_large_f8r288_repeated.npz'))
        vp.TEXT_TOKENIZERS['c4_en']['model_path'] = str(weights/'videoprism/sentencepiece.model')
        from thinksound_worker import AvVideoReader
        import data_utils.v2a_utils.thinksound_288_al as dataset
        dataset.StreamingMediaDecoder = AvVideoReader
        dataset.get_video_duration = lambda _: 8.0
        from data_utils.prismaudio_data_process import process_batch, save_outputs
        from data_utils.v2a_utils.feature_utils_288 import FeaturesUtils
        from torch.utils.data.dataloader import default_collate
        csv_path = work/'captions.csv'
        with csv_path.open('w', encoding='utf8', newline='') as f:
            writer = csv.writer(f); writer.writerow(['id','caption_cot'])
            writer.writerow(['demo', data['prompt'] or 'Sounds matching the video'])
        features_dir = work/'features'; features_dir.mkdir(exist_ok=True)
        ds = dataset.VGGSound(root=work/'videos', tsv_path=csv_path, sample_rate=44100, save_dir=str(features_dir), inference_mode=True)
        if len(ds) != 1: raise RuntimeError('PRISMAUDIO_VIDEO_MISSING')
        sample = ds[0]
        if sample is None: raise RuntimeError('PRISMAUDIO_VIDEO_DECODE_FAILED')
        batch = default_collate([sample])
        model = FeaturesUtils(vae_ckpt=None, enable_conditions=True, synchformer_ckpt=str(weights/'model/synchformer_state_dict.pth')).eval().cuda(0)
        output = process_batch(batch, SimpleNamespace(module=model), 0, inference_mode=True)
        save_outputs(output, batch['id'], str(features_dir))
        if not (features_dir/'demo.npz').is_file(): raise RuntimeError('PRISMAUDIO_FEATURES_MISSING')
    elif mode == 'infer':
        import torchaudio
        import predict
        from PrismAudio.models import create_model_from_config
        config = json.loads((source/'PrismAudio/configs/model_configs/prismaudio.json').read_text())
        duration = 8
        config['sample_size'] = duration * config['sample_rate']
        cfg = config['model']['diffusion']['config']
        if 'sync_seq_len' in cfg: cfg['sync_seq_len'] = 24*duration
        if 'clip_seq_len' in cfg: cfg['clip_seq_len'] = 8*duration
        if 'latent_seq_len' in cfg: cfg['latent_seq_len'] = round(44100/64/32*duration)
        torch.manual_seed(data['seed']); torch.cuda.manual_seed(data['seed'])
        with torch.inference_mode():
            model = create_model_from_config(config)
            model.load_state_dict(torch.load(weights/'model/prismaudio.ckpt', map_location='cpu', weights_only=True))
            state = torch.load(weights/'model/vae.ckpt', map_location='cpu', weights_only=True)['state_dict']
            model.pretransform.load_state_dict({k.removeprefix('autoencoder.'): v for k, v in state.items() if k.startswith('autoencoder.')})
            del state
            audio, meta = predict.load(str(work/'features/demo.npz'), duration)
            for key, value in meta.items():
                if isinstance(value, torch.Tensor): meta[key] = value.to('cuda:0')
            audio = predict.predict_step(model, batch=[audio,(meta,)], diffusion_objective=config['model']['diffusion']['diffusion_objective'], device='cuda:0')
            torchaudio.save(data['output'], audio[0], 44100)
    else: raise RuntimeError('PRISMAUDIO_UNKNOWN_STAGE')


if __name__ == '__main__':
    saved_err = os.dup(2)
    with tempfile.TemporaryFile() as private_log:
        os.dup2(private_log.fileno(), 1); os.dup2(private_log.fileno(), 2)
        try:
            main()
        except Exception as exc:
            os.write(saved_err, f'PRISMAUDIO_INFERENCE_FAILED: {type(exc).__name__}; ejecución local fallida.\n'.encode())
            sys.exit(1)
