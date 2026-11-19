"""Verified PrismAudio, VideoPrism and gated T5Gemma downloads."""
import base64
import hashlib
import os
import sys
import tempfile
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download
from studio.engines.qwen_clone_download import verify

# For gated LFS files the public API hides SHA256. The pinned Git blob ID of
# each LFS pointer still authenticates the SHA256 computed from downloaded data.
FILES = (
    ('model','FunAudioLLM/PrismAudio','d1a33add3f936ba35bf1bd6a86d34ef991ba604f','prismaudio.ckpt',2697789718,'sha256','9f85d728697f18eafc030952c40ea946e8e4d15f0f884fad94a1061acdf2f51d'),
    ('model','FunAudioLLM/PrismAudio','d1a33add3f936ba35bf1bd6a86d34ef991ba604f','synchformer_state_dict.pth',950058171,'sha256','8aff082f2df5c3bc52759db0c865c7ee772ae6400b860d1b7e90413f2defb67c'),
    ('model','FunAudioLLM/PrismAudio','d1a33add3f936ba35bf1bd6a86d34ef991ba604f','vae.ckpt',2521255758,'sha256','8132c32ec65bbba2f1457f2659136c5f2cb31f9902af1c659d78c9f7d3aa619e'),
    ('videoprism','google/videoprism-lvt-large-f8r288','1eda021e2bae000eaae7f763bb9b50d65bdda6af','flax_lvt_large_f8r288_repeated.npz',2319542188,'sha256','eb951c65a5f6b1f793a40deb3a7492a3a68b28b61e1358a893086e5e4300936b'),
    ('gemma','google/t5gemma-l-l-ul2-it','ddd0c8a68e3b95e069b17c15e36e7fab3ae48f05','config.json',3155,'git','e0774cd843043b75e6bbb9a1ffca8d7d87a8c8f5'),
    ('gemma','google/t5gemma-l-l-ul2-it','ddd0c8a68e3b95e069b17c15e36e7fab3ae48f05','generation_config.json',156,'git','a1afbdafc25667679ab591fc9aca56d4a0b91c3d'),
    ('gemma','google/t5gemma-l-l-ul2-it','ddd0c8a68e3b95e069b17c15e36e7fab3ae48f05','model.safetensors',2483608368,'lfs_git','dc62826433a3622bc0593f560f67ea0a3ce6e380'),
    ('gemma','google/t5gemma-l-l-ul2-it','ddd0c8a68e3b95e069b17c15e36e7fab3ae48f05','special_tokens_map.json',636,'git','8d6368f7e735fbe4781bf6e956b7c6ad0586df80'),
    ('gemma','google/t5gemma-l-l-ul2-it','ddd0c8a68e3b95e069b17c15e36e7fab3ae48f05','tokenizer.json',34362429,'lfs_git','55fbda7411a5eb9973350fb98651bf5b8e037bfc'),
    ('gemma','google/t5gemma-l-l-ul2-it','ddd0c8a68e3b95e069b17c15e36e7fab3ae48f05','tokenizer.model',4241003,'lfs_git','796efe9ab515c15e146ce7588e6d7b9b8134dbf8'),
    ('gemma','google/t5gemma-l-l-ul2-it','ddd0c8a68e3b95e069b17c15e36e7fab3ae48f05','tokenizer_config.json',46437,'git','4d625a54ce31c6629f25dadcc7d878e771b40ef7'),
    ('gemma','google/t5gemma-l-l-ul2-it','ddd0c8a68e3b95e069b17c15e36e7fab3ae48f05','chat_template.jinja',577,'git','f278a366f2a229b3f0612a5772d089af6c3a8c40'),
)
TOKENIZER_URL = 'https://storage.googleapis.com/t5-data/vocabs/cc_en.32000/sentencepiece.model'
TOKENIZER_SIZE = 798330
TOKENIZER_MD5 = 'VYeXvRJFMbsE/c4HVSd7KQ=='


def verify_lfs_pointer(path: Path, size: int, expected_blob: str) -> bool:
    if not path.is_file() or path.stat().st_size != size: return False
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024), b''): digest.update(chunk)
    pointer = f'version https://git-lfs.github.com/spec/v1\noid sha256:{digest.hexdigest()}\nsize {size}\n'.encode()
    return hashlib.sha1(f'blob {len(pointer)}\0'.encode()+pointer).hexdigest() == expected_blob


def tokenizer_valid(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size != TOKENIZER_SIZE: return False
    return base64.b64encode(hashlib.md5(path.read_bytes()).digest()).decode() == TOKENIZER_MD5


def main(root: Path):
    root = Path(root)
    # Check the gated model first so denied access fails before public multi-GB downloads.
    ordered = sorted(FILES, key=lambda row: row[0] != 'gemma')
    for i, (group, repo, revision, name, size, kind, digest) in enumerate(ordered, 1):
        folder = root/group; folder.mkdir(parents=True, exist_ok=True)
        target = folder/name
        print(f'[{i}/{len(FILES)+1}] Preparando {group}/{name} ({size/1024**3:.2f} GiB)', flush=True)
        if target.is_symlink(): raise RuntimeError(f'PRISMAUDIO_FILE_MISMATCH: {name} es un enlace')
        if not target.exists(): hf_hub_download(repo, filename=name, revision=revision, local_dir=str(folder))
        okay = verify_lfs_pointer(target, size, digest) if kind == 'lfs_git' else verify(target, size, kind, digest)
        if target.is_symlink() or not okay: raise RuntimeError(f'PRISMAUDIO_FILE_MISMATCH: {name}; no se sobrescribe')
        print(f'[{i}/{len(FILES)+1}] Verificado {group}/{name}', flush=True)
    target = root/'videoprism'/'sentencepiece.model'
    print(f'[{len(FILES)+1}/{len(FILES)+1}] Preparando tokenizer VideoPrism', flush=True)
    if target.is_symlink(): raise RuntimeError('PRISMAUDIO_TOKENIZER_CONFLICT')
    if not target.exists():
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix='tokenizer-', delete=False) as tmp:
            temp = Path(tmp.name)
            try:
                with requests.get(TOKENIZER_URL, stream=True, timeout=(30, 120)) as response:
                    response.raise_for_status()
                    for part in response.iter_content(1024*1024): tmp.write(part)
                tmp.flush()
                if not tokenizer_valid(temp): raise RuntimeError('PRISMAUDIO_TOKENIZER_MISMATCH')
                if target.exists() or target.is_symlink(): raise RuntimeError('PRISMAUDIO_TOKENIZER_CONFLICT')
                os.replace(temp, target)
            finally:
                temp.unlink(missing_ok=True)
    if not tokenizer_valid(target): raise RuntimeError('PRISMAUDIO_TOKENIZER_MISMATCH')
    print(f'[{len(FILES)+1}/{len(FILES)+1}] Verificado tokenizer VideoPrism', flush=True)


if __name__ == '__main__': main(Path(sys.argv[1]))
