"""Download and safely unpack the pinned Unsloth H3 runtime in a child process."""
from __future__ import annotations

import hashlib
import os
import shutil
import stat
import sys
import time
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

TAG = 'master-982-0518b34-u27b9ddc'
REVISION = '27b9ddca07be0c25ace3d1a560bcbad281eb660d'
ASSETS = {
    '12': (357978823, '4d39c7a385dc2af47d43b9d0e2c17c7fae2d57605c0ad75d084e92d41fbb47a6'),
    '13': (315257931, '5129f05af93cac94f6e6182007ff9b18284fbea71f930f4b93360bc5a35d50a1'),
}


def unpack(archive: Path, dest: Path, stem: str) -> None:
    """Only a flat bundle of regular files, no symlinks, traversal or bombs."""
    with zipfile.ZipFile(archive) as zf:
        infos = zf.infolist()
        if len(infos) > 100 or sum(i.file_size for i in infos) > 2 * 1024**3:
            raise ValueError('H3_PREBUILT_ARCHIVE_INVALID: tamaño inesperado')
        names = set()
        for info in infos:
            p = PurePosixPath(info.filename)
            mode = info.external_attr >> 16
            if (len(p.parts) != 2 or p.parts[0] != stem or p.parts[1] in ('.', '..')
                    or '\\' in info.filename or info.is_dir() or stat.S_ISLNK(mode)
                    or p.parts[1] in names):
                raise ValueError('H3_PREBUILT_ARCHIVE_INVALID: ruta no segura')
            names.add(p.parts[1])
        if 'sd-cli' not in names or 'UNSLOTH_BUILD.txt' not in names:
            raise ValueError('H3_PREBUILT_ARCHIVE_INVALID: faltan componentes')
        dest.mkdir()
        for info in infos:
            path = dest / PurePosixPath(info.filename).name
            with zf.open(info) as src, path.open('xb') as out:
                shutil.copyfileobj(src, out, 1024 * 1024)
        if f'source_commit: {REVISION}' not in (dest / 'UNSLOTH_BUILD.txt').read_text(encoding='utf8'):
            raise ValueError('H3_PREBUILT_ARCHIVE_INVALID: revisión distinta')
        (dest / 'sd-cli').chmod(0o755)


def download(major: str, temporary: Path) -> None:
    size, expected_sha = ASSETS[major]
    stem = f'sd-{TAG}-bin-Linux-Ubuntu-22.04-x86_64-cuda{major}'
    url = f'https://github.com/unslothai/stable-diffusion.cpp/releases/download/{TAG}/{stem}.zip'
    archive = temporary / 'runtime.zip'
    digest, count, last = hashlib.sha256(), 0, 0.0
    req = urllib.request.Request(url, headers={'User-Agent': 'Studio-IA-H3/1.0'})
    with urllib.request.urlopen(req, timeout=60) as response, archive.open('xb') as out:
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            count += len(block)
            if count > size:
                raise ValueError('H3_PREBUILT_INVALID: tamaño excedido')
            digest.update(block)
            out.write(block)
            if time.monotonic() - last >= 5:
                print(f'[Preparando runtime H3 precompilado: {count / 1024**2:.0f} / {size / 1024**2:.0f} MiB]', flush=True)
                last = time.monotonic()
    if count != size or digest.hexdigest() != expected_sha:
        raise ValueError('H3_PREBUILT_INVALID: tamaño o SHA-256 incorrecto')
    print('[Verificado SHA-256 del runtime H3 precompilado]', flush=True)
    unpack(archive, temporary / 'bundle', stem)
    os.unlink(archive)


if __name__ == '__main__':
    download(sys.argv[1], Path(sys.argv[2]))
