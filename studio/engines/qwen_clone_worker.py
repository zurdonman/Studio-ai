"""Private long-lived Qwen worker. JSON requests on stdin, replies in private files."""
import json
import os
import sys
import traceback
from pathlib import Path


def write_reply(path, result):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(result, ensure_ascii=False), encoding='utf8')
    os.replace(tmp, path)


def main():
    source, model_path, directory = map(Path, sys.argv[1:4])
    sys.path.insert(0, str(source))
    ready = directory / 'ready.json'
    try:
        import torch
        import soundfile as sf
        from qwen_tts import Qwen3TTSModel
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError('QWEN_GPU_UNSUPPORTED: se necesita GPU CUDA compatible con BF16')
        model = Qwen3TTSModel.from_pretrained(str(model_path), device_map='cuda:0', dtype=torch.bfloat16, attn_implementation='sdpa')
        write_reply(ready, {'ok': True})
    except BaseException as exc:
        traceback.print_exc()
        write_reply(ready, {'ok': False, 'error': str(exc)[:1000]})
        return
    for line in sys.stdin:
        try:
            req = json.loads(line)
            reply = directory / (req['id'] + '.json')
            with torch.inference_mode():
                wavs, sr = model.generate_voice_clone(text=req['text'], language='Spanish', ref_audio=req['reference'],
                    ref_text=req['transcript'] or None, x_vector_only_mode=not bool(req['transcript']), max_new_tokens=2048)
            if len(wavs) != 1 or len(wavs[0]) == 0:
                raise RuntimeError('QWEN_EMPTY_AUDIO: el modelo no devolvió audio')
            sf.write(req['output'], wavs[0], int(sr), subtype='PCM_16')
            write_reply(reply, {'ok': True})
        except BaseException as exc:
            traceback.print_exc()
            if 'reply' in locals():
                write_reply(reply, {'ok': False, 'error': str(exc)[:1000]})


if __name__ == '__main__':
    main()
