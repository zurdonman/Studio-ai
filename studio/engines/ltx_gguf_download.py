"""Download the exact pinned artifacts; run in a cancelable child process."""
import hashlib
import os
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download
from studio.engines.ltx_gguf_manifest import FILES


def verify(path: Path, size: int, sha: str) -> bool:
    if not path.is_file() or path.stat().st_size != size:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == sha


def main(root: Path) -> None:
    for index, (role, repo, revision, filename, size, sha) in enumerate(FILES, 1):
        target_dir = root / ("ltx" if role != "text_encoder" else "gemma")
        target = target_dir / filename
        print(f"[{index}/{len(FILES)}] Comprobando {role} ({size / 1024**3:.2f} GiB)", flush=True)
        if not verify(target, size, sha):
            if target.exists():
                raise RuntimeError(f"GGUF_CORRUPT_EXISTING: {role} no coincide con SHA-256; no se modifica. Revisa el archivo antes de reintentar.")
            target_dir.mkdir(parents=True, exist_ok=True)
            hf_hub_download(repo_id=repo, revision=revision, filename=filename,
                            local_dir=str(target_dir), token=os.environ.get("HF_TOKEN") or None)
            if not verify(target, size, sha):
                raise RuntimeError(f"GGUF_CHECKSUM_MISMATCH: {role}; no se usará el archivo")
        print(f"[{index}/{len(FILES)}] Verificado {role}", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
