"""Pinned weights for ControlFoley's video+text-only inference recipe."""
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download
from studio.engines.qwen_clone_download import verify
from studio.engines.mmaudio_download import FILES as MMAUDIO_FILES

REPO = 'YJX-Xiaomi/ControlFoley'
REV = 'cde633f5c3f5ccbedf07994ad6dc124b761df22f'
FILES = (
    ('control', REPO, REV, 'weights/controlfoley.pth', 11292123018, 'sha256', '8e2359cdf412b07381282c07e53205ce3d778a4a8be1edbc5c00d373e5186ed6'),
    ('control', REPO, REV, 'ext_weights/cav_mae_st.pth', 762301703, 'sha256', 'de9f7dff5a643690ae9e98bb686abf4f259456a9f3be8ef1fa403384eeefc6d8'),
    ('control', REPO, REV, 'ext_weights/v1-44.pth', 1221942998, 'sha256', 'ab6cc15dc31947675f75c950c41f4dcfd0d6d1817555ac871f809ec388e4651a'),
    ('control', REPO, REV, 'ext_weights/synchformer_state_dict.pth', 950058171, 'sha256', '8aff082f2df5c3bc52759db0c865c7ee772ae6400b860d1b7e90413f2defb67c'),
) + tuple(f for f in MMAUDIO_FILES if f[0] in ('clip', 'vocoder'))


def main(root: Path):
    for i, (group, repo, revision, filename, size, kind, checksum) in enumerate(FILES, 1):
        target = root / group / filename
        print(f'[{i}/{len(FILES)}] Preparando {group}/{filename} ({size / 1024**3:.2f} GiB)', flush=True)
        if not target.exists():
            hf_hub_download(repo, filename=filename, revision=revision, local_dir=str(root / group))
        if not verify(target, size, kind, checksum):
            raise RuntimeError(f'CONTROLFOLEY_FILE_MISMATCH: {group}/{filename}; no se sobrescribe el archivo')
        print(f'[{i}/{len(FILES)}] Verificado {group}/{filename}', flush=True)


if __name__ == '__main__':
    main(Path(sys.argv[1]))
