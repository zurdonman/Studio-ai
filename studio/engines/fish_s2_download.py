"""Preparatory Fish S2-Pro manifest; this is NOT a complete or usable runtime.

Not wired to the UI or a job. A future integration must obtain explicit download
and license consent, check resources and verify the isolated engine as well.
"""
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download
from studio.engines.qwen_clone_download import verify

REPO = 'fishaudio/s2-pro'
REVISION = '1de9996b6be38b745688de084d87a5633f714e4e'
FILES = {
    'LICENSE.md': (10360, 'git', 'b469a1a983b3f961e839881fc0ea5c4eb82ec4be'),
    'chat_template.jinja': (4116, 'git', '699ff8df401fe4788525e9c1f9b86a99eadd6230'),
    'codec.pth': (1871099728, 'sha256', '74fc41c5a7151c6f350af8bd7e5d6e3accfcc7f3dfbfac23afd35af07052bb2f'),
    'config.json': (1863, 'git', '5214fbf56c44b20d00b1d5654d68b3b1f9ec9ea5'),
    'model-00001-of-00002.safetensors': (4986872984, 'sha256', 'c4218e8ac93be83b35eee30b4f94cb2e9b5ecff40f3e21611438d2f4f8804aad'),
    'model-00002-of-00002.safetensors': (4136876104, 'sha256', '76738d23465deaac431433232c0762908cc99a6eddc3d49f67307d92680827be'),
    'model.safetensors.index.json': (32686, 'git', '4a355d7d68ae0885db2c3ed3e2d9846420eef9a1'),
    'special_tokens_map.json': (101864, 'git', '7989e2cfbbfec23cfe75e17cb5494c2354a1769e'),
    'tokenizer.json': (12217872, 'sha256', 'f24e08099d45a8adf3f52f5f0b03276e433bb9d689bb15fcbcc48ce58744588b'),
    'tokenizer_config.json': (860832, 'git', '47a7c4ee0e88f50904020eec33f723cec6933c61'),
}


def main(root: Path):
    root = Path(root)
    if root.is_symlink():
        raise RuntimeError('FISH_ROOT_CONFLICT: el directorio no puede ser un enlace simbólico')
    root.mkdir(parents=True, exist_ok=True)
    for i, (name, (size, kind, sha)) in enumerate(FILES.items(), 1):
        path = root / name
        print(f'[{i}/{len(FILES)}] Preparando Fish S2-Pro {name} ({size/1024**3:.2f} GiB)', flush=True)
        if path.is_symlink():
            raise RuntimeError(f'FISH_FILE_MISMATCH: {name} es un enlace; no se sobrescribe')
        if not path.exists():
            hf_hub_download(REPO, filename=name, revision=REVISION, local_dir=str(root))
        if path.is_symlink() or not verify(path, size, kind, sha):
            raise RuntimeError(f'FISH_FILE_MISMATCH: {name}; no se sobrescribe')
        print(f'[{i}/{len(FILES)}] Verificado {name}', flush=True)


if __name__ == '__main__':
    main(Path(sys.argv[1]))
