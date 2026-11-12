"""Download pinned GGUF files and emit measured byte progress to the parent."""
import hashlib
import inspect
import json
import os
import sys
import time
from pathlib import Path

from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm
from studio.engines.ltx_gguf_manifest import FILES


def _emit(role: str, phase: str, done: int, total: int, before: int, all_bytes: int) -> None:
    fraction = min(1.0, max(0.0, done / total)) if total else 0.0
    # Each file reserves half of its weight for transfer and half for hashing.
    fraction = fraction / 2 if phase == 'descarga' else (0.5 + fraction / 2)
    print('GGUF_PROGRESS:' + json.dumps({'role': role, 'phase': phase, 'done': done, 'total': total,
          'overall': (before + total * fraction) / all_bytes}), flush=True)


def verify(path: Path, size: int, sha: str, progress=None) -> bool:
    if not path.is_file() or path.stat().st_size != size:
        return False
    digest = hashlib.sha256()
    read, last = 0, 0.0
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
            read += len(chunk)
            if progress and time.monotonic() - last >= .5:
                progress(read)
                last = time.monotonic()
    if progress:
        progress(read)
    return digest.hexdigest() == sha


def main(root: Path) -> None:
    total_bytes = sum(item[4] for item in FILES)
    before = 0
    for index, (role, repo, revision, filename, size, sha) in enumerate(FILES, 1):
        target_dir = root / ('ltx' if role != 'text_encoder' else 'gemma')
        target = target_dir / filename
        print(f'[{index}/{len(FILES)}] Preparando {role} ({size / 1024**3:.2f} GiB)', flush=True)
        if not target.exists():
            target_dir.mkdir(parents=True, exist_ok=True)
            kwargs = dict(repo_id=repo, revision=revision, filename=filename,
                          local_dir=str(target_dir), token=os.environ.get('HF_TOKEN') or None)
            if 'tqdm_class' in inspect.signature(hf_hub_download).parameters:
                # HF HTTP and Xet report actual transferred bytes; no time-based fake percentage.
                class DownloadProgress(tqdm):
                    def __init__(self, *args, **kw):
                        kw['file'] = open(os.devnull, 'w')
                        super().__init__(*args, **kw)
                        self._studio_last = 0.0
                    def update(self, n=1):
                        result = super().update(n)
                        now = time.monotonic()
                        if now - self._studio_last >= .5:
                            _emit(role, 'descarga', min(size, int(self.n)), size, before, total_bytes)
                            self._studio_last = now
                        return result
                    # Newer Xet exposes transfer and reconstruction as separate
                    # streams. Track reconstructed file bytes only, not two
                    # independent bars that jump backwards when interleaved.
                    def update_transfer(self, n=1):
                        pass
                    def set_transfer_postfix_str(self, text, refresh=False):
                        pass
                    def close(self):
                        super().close()
                        try:
                            self.fp.close()
                        except Exception:
                            pass
                kwargs['tqdm_class'] = DownloadProgress
            else:
                print('GGUF_NOTE:Descarga en curso; porcentaje no disponible con esta versión de huggingface_hub.', flush=True)
            hf_hub_download(**kwargs)
        _emit(role, 'verificación', 0, size, before, total_bytes)
        if not verify(target, size, sha, lambda n: _emit(role, 'verificación', n, size, before, total_bytes)):
            raise RuntimeError(f'GGUF_CORRUPT_EXISTING: {role} no coincide con SHA-256; no se modifica. Revisa el archivo antes de reintentar.')
        before += size
        print(f'[{index}/{len(FILES)}] Verificado {role}', flush=True)


if __name__ == '__main__':
    main(Path(sys.argv[1]))
