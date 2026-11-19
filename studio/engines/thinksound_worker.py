"""Isolated per-segment ThinkSound feature extraction and synthesis."""
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


def local_transformers(weights):
    import transformers
    mapping = {'facebook/metaclip-h14-fullcc2.5b': str(weights/'clip'),
               'google/t5-v1_1-xl': str(weights/'t5')}
    for cls in (transformers.AutoModel, transformers.AutoProcessor, transformers.AutoTokenizer, transformers.T5EncoderModel):
        original = cls.from_pretrained
        def load(_cls, name, *args, _original=original, **kwargs):
            if name not in mapping: raise RuntimeError('THINKSOUND_UNEXPECTED_PRETRAINED')
            kwargs['local_files_only'] = True
            return _original(mapping[name], *args, **kwargs)
        cls.from_pretrained = classmethod(load)


class AvVideoReader:
    """Subset of torio's reader for Studio's constant 25 fps, eight-second clips.

    PyAV ships FFmpeg libraries in its wheel, avoiding a dependency on system
    libav ABI versions. Returns T,C,H,W uint8 tensors like torio rgb24.
    """
    def __init__(self, path):
        self.path = str(path)
        self.streams = []
        self.chunks = []

    def add_basic_video_stream(self, *, frames_per_chunk, frame_rate, format):
        if format != 'rgb24' or frame_rate not in (4.0, 8.0, 25.0): raise ValueError('VIDEO_FEATURE_FRAME_RATE')
        self.streams.append((int(frames_per_chunk), float(frame_rate)))

    def fill_buffer(self):
        import av
        import torch
        needed = [{min(199, round(i * 25.0 / fps)): i for i in range(count)} for count, fps in self.streams]
        # Fixed rates 8 and 25 fps have unique frame indexes for 8 seconds.
        chunks = [[None]*count for count, _ in self.streams]
        with av.open(self.path) as container:
            for index, frame in enumerate(container.decode(video=0)):
                if index >= 200: break
                targets = [(n, indexes[index]) for n, indexes in enumerate(needed) if index in indexes]
                if targets:
                    tensor = torch.from_numpy(frame.to_ndarray(format='rgb24')).permute(2, 0, 1).contiguous()
                    for n, pos in targets: chunks[n][pos] = tensor
        if any(any(x is None for x in chunk) for chunk in chunks):
            raise RuntimeError('THINKSOUND_VIDEO_SHORT: el segmento no contiene todos los fotogramas.')
        self.chunks = [torch.stack(chunk) for chunk in chunks]

    def pop_chunks(self):
        return self.chunks


def main():
    mode, source_str, weights_str, request_str = sys.argv[1:]
    source, weights, request = Path(source_str), Path(weights_str), Path(request_str)
    data = json.loads(request.read_text(encoding='utf8'))
    work = request.parent
    sys.path.insert(0, str(source))
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('THINKSOUND_GPU_UNSUPPORTED')
    if mode == 'extract':
        local_transformers(weights)
        from extract_latents import main as extract
        import data_utils.v2a_utils.vggsound_224_no_audio as dataset
        from data_utils.v2a_utils.feature_utils_224 import FeaturesUtils
        dataset.StreamingMediaDecoder = AvVideoReader
        original_encode = FeaturesUtils.encode_video_with_clip
        def half_clip(self, x, batch_size=-1):
            return original_encode(self, x.to(dtype=self.clip_model.dtype), batch_size)
        FeaturesUtils.encode_video_with_clip = half_clip
        video_dir = work/'videos'; video_dir.mkdir(exist_ok=True)
        # The caller prepared this exact isolated segment as demo.mp4.
        csv_path = work/'captions.csv'
        with csv_path.open('w', encoding='utf8', newline='') as f:
            writer = csv.writer(f); writer.writerow(['id', 'caption', 'caption_cot'])
            writer.writerow(['demo', data['prompt'] or 'Sound of the video', data['prompt'] or 'Sounds matching the video'])
        extract(SimpleNamespace(root=str(video_dir), tsv_path=str(csv_path), save_dir=str(work/'features'),
                                sample_rate=44100, duration_sec=8.0, audio_samples=352800,
                                start_row=0, end_row=None, synchformer_ckpt=str(weights/'model/synchformer_state_dict.pth'), use_half=True))
        if not (work/'features/demo.npz').is_file(): raise RuntimeError('THINKSOUND_FEATURES_MISSING')
    elif mode == 'infer':
        import predict
        import torchaudio
        from ThinkSound.models import create_model_from_config
        config = json.loads((source/'ThinkSound/configs/model_configs/thinksound.json').read_text())
        duration = 8
        config['sample_size'] = duration * config['sample_rate']
        cfg = config['model']['diffusion']['config']
        cfg['sync_seq_len'] = 24*duration; cfg['clip_seq_len'] = 8*duration
        cfg['latent_seq_len'] = round(44100/64/32*duration)
        torch.manual_seed(data['seed']); torch.cuda.manual_seed(data['seed'])
        with torch.inference_mode():
            model = create_model_from_config(config)
            model.load_state_dict(torch.load(weights/'model/thinksound_light.ckpt', map_location='cpu', weights_only=True))
            vae_state = torch.load(weights/'model/vae.ckpt', map_location='cpu', weights_only=True)['state_dict']
            model.pretransform.load_state_dict({k.removeprefix('autoencoder.'): v for k, v in vae_state.items() if k.startswith('autoencoder.')})
            del vae_state
            audio, meta = predict.load(str(work/'features/demo.npz'), duration)
            for key, value in meta.items():
                if isinstance(value, torch.Tensor): meta[key] = value.to('cuda:0')
            audio = predict.predict_step(model, batch=[audio, (meta,)],
                                         diffusion_objective=config['model']['diffusion']['diffusion_objective'], device='cuda:0')
            torchaudio.save(data['output'], audio[0], 44100)
    else:
        raise RuntimeError('THINKSOUND_UNKNOWN_STAGE')


if __name__ == '__main__':
    # Upstream can log prompts; keep its logs private even on failure.
    saved_err = os.dup(2)
    with tempfile.TemporaryFile() as private_log:
        os.dup2(private_log.fileno(), 1); os.dup2(private_log.fileno(), 2)
        try:
            main()
        except Exception as exc:
            os.write(saved_err, f'THINKSOUND_INFERENCE_FAILED: {type(exc).__name__}; ejecución local fallida.\n'.encode())
            sys.exit(1)
