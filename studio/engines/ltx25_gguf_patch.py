"""Download and verify the author's gated patch in an isolated plugin checkout."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = 'elix3r/gemma4-12b-with-proj-ltx-2.5-GGUF'
REV = '2a18e836d286eb0570ec9f013c3591eb8c614d57'
PATCH_SHA256 = '1c36aa38fa9c86ec503926bc6b35226e2575b3bfd27b352bff50deccb1cd958e'
GGUF_COMMIT = '6ea2651e7df66d7585f6ffee804b20e92fb38b8a'


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(plugin: Path, stamp: Path) -> None:
    try:
        patch = Path(hf_hub_download(repo_id=REPO, revision=REV,
                     filename='patches/ComfyUI-GGUF-ltx25-gemma4.patch', token=os.environ.get('HF_TOKEN') or None))
    except Exception as exc:
        raise RuntimeError('GGUF_GEMMA4_ACCESS: acepta las condiciones en https://huggingface.co/' + REPO +
                           ' y configura tu acceso a Hugging Face en Studio; no se descargaron los pesos') from exc
    if digest(patch) != PATCH_SHA256:
        raise RuntimeError('GGUF_GEMMA4_PATCH_HASH: parche no coincide con SHA-256; no se aplica')
    def git(*args, check=True):
        return subprocess.run(['git', '-C', str(plugin), *args], capture_output=True, text=True, check=check)
    if git('rev-parse', 'HEAD').stdout.strip() != GGUF_COMMIT:
        raise RuntimeError('GGUF_GEMMA4_PATCH_REVISION: plugin no coincide con la versión esperada')
    status = git('status', '--porcelain', '--untracked-files=no').stdout.strip()
    loader = plugin / 'loader.py'
    if stamp.is_file():
        try:
            state = json.loads(stamp.read_text())
            valid = (state == {'patch_sha256': PATCH_SHA256, 'loader_sha256': digest(loader)} and
                     status == 'M loader.py' and git('apply', '--reverse', '--check', str(patch), check=False).returncode == 0)
        except (OSError, ValueError):
            valid = False
        if not valid:
            raise RuntimeError('GGUF_GEMMA4_PATCH_CONFLICT: el plugin aislado ha cambiado; no se modifica')
        print('Parche GGUF Gemma 4 existente y verificado.', flush=True)
        return
    if status:
        raise RuntimeError('GGUF_GEMMA4_PATCH_CONFLICT: plugin aislado modificado; no se modifica')
    if git('apply', '--check', str(patch), check=False).returncode:
        raise RuntimeError('GGUF_GEMMA4_PATCH_INCOMPATIBLE: no se puede aplicar el parche sin forzarlo')
    git('apply', str(patch))
    if git('status', '--porcelain', '--untracked-files=no').stdout.strip() != 'M loader.py':
        raise RuntimeError('GGUF_GEMMA4_PATCH_CONFLICT: cambios inesperados en el plugin aislado')
    stamp.write_text(json.dumps({'patch_sha256': PATCH_SHA256, 'loader_sha256': digest(loader)}), encoding='utf-8')
    print('Parche GGUF Gemma 4 verificado y aplicado al runtime aislado.', flush=True)


if __name__ == '__main__':
    run(Path(sys.argv[1]), Path(sys.argv[2]))
