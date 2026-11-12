"""Download the fixed Qwen3-TTS Base snapshot and verify every file."""
from pathlib import Path
import hashlib
import sys
from huggingface_hub import hf_hub_download

REPO = 'Qwen/Qwen3-TTS-12Hz-0.6B-Base'
REVISION = '5d83992436eae1d760afd27aff78a71d676296fc'
# Git blob IDs for small files, LFS SHA-256 for safetensors.
FILES = {
    'config.json': (4494, 'git', '4e923f500f5c3eeb78d44072e7049b28817c2e04'),
    'generation_config.json': (245, 'git', '1872b16d4535564abf2db5d6debe6cddd82b7f2e'),
    'merges.txt': (1671839, 'git', '20024bfe7c83998e9aeaf98a0cd6a2ce6306c2f0'),
    'model.safetensors': (1829344272, 'sha256', '180b3b10eb1c9f1b4db7806d5475bae3071c0243c299d49926bab1da3b6946f6'),
    'preprocessor_config.json': (127, 'git', '0525dd953bb9241912f7147666f0d535165d5d4f'),
    'speech_tokenizer/config.json': (2336, 'git', '06cc8dc4c5ec8a1929086b71b98c313020d9268b'),
    'speech_tokenizer/configuration.json': (76, 'git', 'ab58e2eaf53cd14a1a2a7527d9261ceea93a24cd'),
    'speech_tokenizer/model.safetensors': (682293092, 'sha256', '836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258'),
    'speech_tokenizer/preprocessor_config.json': (234, 'git', 'ba40914f4f49ab98a8ca545d4892ef7291a39592'),
    'tokenizer_config.json': (7344, 'git', '6ff9fd60cc623bb54bbd603cbd418c97a11528d7'),
    'vocab.json': (2776833, 'git', '4783fe10ac3adce15ac8f358ef5462739852c569'),
}


def verify(path, size, algorithm, expected):
    if not path.is_file() or path.stat().st_size != size:
        return False
    h = hashlib.sha1() if algorithm == 'git' else hashlib.sha256()
    if algorithm == 'git':
        h.update(f'blob {size}\0'.encode())
    with path.open('rb') as f:
        for part in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest() == expected


def main(root):
    for name, (size, algorithm, checksum) in FILES.items():
        path = root / name
        print(f'[Qwen] Preparando {name}', flush=True)
        if not path.exists():
            hf_hub_download(REPO, filename=name, revision=REVISION, local_dir=str(root))
        if not verify(path, size, algorithm, checksum):
            raise RuntimeError(f'QWEN_FILE_MISMATCH: {name} no coincide con la revisión fijada; no se sobrescribe')
        print(f'[Qwen] Verificado {name}', flush=True)


if __name__ == '__main__':
    main(Path(sys.argv[1]))
