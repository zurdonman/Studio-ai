"""Pinned MMAudio and auxiliary weights, downloaded and verified before inference."""
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download
from studio.engines.qwen_clone_download import verify

# Git object IDs for small configs; LFS SHA-256 for large checkpoints.
FILES = (
    ('mmaudio', 'hkchengrex/MMAudio', 'eb13a1a98fdbec91753775c57b074ccdfc60587c', 'weights/mmaudio_large_44k_v2.pth', 4122474715, 'sha256', 'a6bf693424fbd4ce0244fff8c412347714d5ac586e28dbeffadfa0f2b647af74'),
    ('mmaudio', 'hkchengrex/MMAudio', 'eb13a1a98fdbec91753775c57b074ccdfc60587c', 'ext_weights/v1-44.pth', 1221942998, 'sha256', 'ab6cc15dc31947675f75c950c41f4dcfd0d6d1817555ac871f809ec388e4651a'),
    ('mmaudio', 'hkchengrex/MMAudio', 'eb13a1a98fdbec91753775c57b074ccdfc60587c', 'ext_weights/synchformer_state_dict.pth', 950058171, 'sha256', '8aff082f2df5c3bc52759db0c865c7ee772ae6400b860d1b7e90413f2defb67c'),
    ('clip', 'apple/DFN5B-CLIP-ViT-H-14-378', '01b771ed0d1395ca5ffdd279897d665ebe00dfd2', 'open_clip_config.json', 735, 'git', '56abccffc4875142ac955f96cec904a1bd4434d6'),
    ('clip', 'apple/DFN5B-CLIP-ViT-H-14-378', '01b771ed0d1395ca5ffdd279897d665ebe00dfd2', 'open_clip_pytorch_model.bin', 3947081637, 'sha256', 'c07a17b547d461c60a3cce5062b26bf8545b13de602c4c59d8490361eb716033'),
    ('vocoder', 'nvidia/bigvgan_v2_44khz_128band_512x', '95a9d1dcb12906c03edd938d77b9333d6ded7dfb', 'config.json', 1403, 'git', '624a661dcef98677775b0b16d36dc9adb02a74bd'),
    ('vocoder', 'nvidia/bigvgan_v2_44khz_128band_512x', '95a9d1dcb12906c03edd938d77b9333d6ded7dfb', 'bigvgan_generator.pt', 489041291, 'sha256', 'd9fe7ec6bd0b44ed9d66973d5012d8181c1570b01e5c72df51973e241dccd357'),
)


def main(root: Path):
    for i, (group, repo, revision, filename, size, kind, checksum) in enumerate(FILES, 1):
        target = root / group / filename
        print(f'[{i}/{len(FILES)}] Preparando {group}/{filename} ({size / 1024**3:.2f} GiB)', flush=True)
        if not target.exists():
            hf_hub_download(repo, filename=filename, revision=revision, local_dir=str(root / group))
        if not verify(target, size, kind, checksum):
            raise RuntimeError(f'MMAUDIO_FILE_MISMATCH: {group}/{filename}; no se sobrescribe el archivo')
        print(f'[{i}/{len(FILES)}] Verificado {group}/{filename}', flush=True)


if __name__ == '__main__':
    main(Path(sys.argv[1]))
