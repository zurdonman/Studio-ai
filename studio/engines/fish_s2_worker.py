"""One-shot Fish S2-Pro inference using only the pinned local checkpoint."""
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def main():
    source, weights, request_file = map(Path, sys.argv[1:])
    request = json.loads(request_file.read_text(encoding='utf8'))
    sys.path.insert(0, str(source))
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA_BF16_REQUIRED')
    from fish_speech.models.text2semantic.inference import main as cli
    args = ['--text', request['text'], '--checkpoint-path', str(weights), '--device', 'cuda',
            '--output', request['output'], '--output-dir', str(request_file.parent), '--seed', '42']
    if request.get('reference'):
        args += ['--prompt-audio', request['reference'], '--prompt-text', request['transcript']]
    cli.main(args=args, standalone_mode=False)


if __name__ == '__main__':
    # Upstream can log sampled user text. Never expose that output or the prompt
    # through the task monitor, including when the model fails mid-generation.
    saved_err = os.dup(2)
    with tempfile.TemporaryFile() as private_log:
        os.dup2(private_log.fileno(), 1)
        os.dup2(private_log.fileno(), 2)
        try:
            main()
        except Exception as exc:
            os.write(saved_err, f'FISH_INFERENCE_FAILED: {type(exc).__name__}; ejecución local fallida.\n'.encode())
            sys.exit(1)
