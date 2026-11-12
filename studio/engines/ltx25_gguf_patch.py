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


def _download_error(exc: Exception, token_supplied: bool) -> str:
    # Do not put exception messages, URLs with query strings or credentials in logs.
    # HF Hub may wrap the original HTTP/network error in a cache exception.
    chain = []
    seen = set()
    current = exc
    while current is not None and id(current) not in seen and len(chain) < 8:
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__ or current.__context__
    names = {type(item).__name__ for item in chain}
    statuses = {getattr(getattr(item, 'response', None), 'status_code', None) for item in chain}
    token_state = 'sí' if token_supplied else 'no (podría existir una sesión local de Hugging Face)'
    context = f"Tipo: {type(exc).__name__}; token de Studio en el proceso: {token_state}."
    if 'GatedRepoError' in names or 401 in statuses or 403 in statuses:
        return ('GGUF_GEMMA4_ACCESS: Hugging Face ha rechazado el acceso. Abre https://huggingface.co/' + REPO +
                ', acepta las condiciones y configura en Studio el acceso de esa misma cuenta. ' + context)
    if 'RevisionNotFoundError' in names or 'EntryNotFoundError' in names and 'LocalEntryNotFoundError' not in names:
        return 'GGUF_GEMMA4_RESOURCE: no se encontró la revisión o el parche fijado en Hugging Face. ' + context
    if 'RepositoryNotFoundError' in names or 404 in statuses:
        return ('GGUF_GEMMA4_REPOSITORY: Hugging Face no encontró el repositorio o no autorizó el acceso. '
                'Comprueba la cuenta y sus permisos. ' + context)
    if names.intersection({'LocalEntryNotFoundError', 'OfflineModeIsEnabled', 'ConnectError', 'ConnectTimeout',
                           'ReadTimeout', 'ConnectionError', 'Timeout', 'SSLError'}):
        return ('GGUF_GEMMA4_NETWORK: no se pudo recuperar el parche; revisa la conexión y si Hugging Face está en modo sin conexión. '
                'Si vuelve a fallar, comparte el log. ' + context)
    status_text = ', '.join(str(x) for x in sorted(x for x in statuses if isinstance(x, int)))
    return ('GGUF_GEMMA4_DOWNLOAD: no se pudo recuperar el parche; comparte el log. ' + context +
            (f' HTTP: {status_text}.' if status_text else ''))


def run(plugin: Path, stamp: Path) -> None:
    try:
        patch = Path(hf_hub_download(repo_id=REPO, revision=REV,
                     filename='patches/ComfyUI-GGUF-ltx25-gemma4.patch', token=os.environ.get('HF_TOKEN') or None))
    except Exception as exc:
        raise RuntimeError(_download_error(exc, bool(os.environ.get('HF_TOKEN'))) + ' No se descargaron los pesos.') from None
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
    try:
        run(Path(sys.argv[1]), Path(sys.argv[2]))
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr, flush=True)
        sys.exit(1)
