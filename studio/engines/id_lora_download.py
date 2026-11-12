"""Download the exact upstream one-stage components into Studio's data directory.

Run as a cancellable child process; HF_TOKEN is inherited from the parent only.
"""
from pathlib import Path
import json
import shutil
import sys

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

BASE_REPO = "Lightricks/LTX-2.3"
BASE_NAME = "ltx-2.3-22b-dev.safetensors"
GEMMA_REPO = "google/gemma-3-12b-it-qat-q4_0-unquantized"
LORA_REPO = "AviadDahan/LTX-2.3-ID-LoRA-TalkVid-3K"
LORA_NAME = "lora_weights.safetensors"


def main(root: Path, plan_only: bool = False):
    root.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    planned = []
    for repo, name, subdir in [(BASE_REPO, BASE_NAME, "base"), (GEMMA_REPO, None, "gemma"), (LORA_REPO, LORA_NAME, "lora")]:
        info = api.model_info(repo_id=repo, files_metadata=True)
        relevant = [x for x in (info.siblings or []) if (x.rfilename == name if name else not x.rfilename.startswith('.') and not x.rfilename.lower().endswith((".bin", ".pt", ".onnx", ".gguf")))]
        if not relevant:
            raise RuntimeError(f"No se encontraron los archivos requeridos de {repo}; revisa acceso y licencia.")
        for item in relevant:
            size = int(getattr(item, "size", None) or 0)
            if item.rfilename.endswith(".safetensors") and size <= 0:
                raise RuntimeError(f"HF no ha informado el tamaño de {item.rfilename}; descarga detenida.")
            path = root / subdir / item.rfilename
            existing = path.stat().st_size if path.is_file() else 0
            planned.append((path, size, 0 if existing == size else size))
    total = sum(size for _, size, _ in planned)
    missing = sum(size for _, _, size in planned)
    if total <= 0:
        raise RuntimeError("HF no ha informado tamaños; no se inicia la descarga sin comprobar disco.")
    free = shutil.disk_usage(root).free
    if free < missing + 12 * 1024**3:
        raise RuntimeError(f"Disco insuficiente: faltan {missing / 1024**3:.1f} GiB de modelos más 12 GiB de margen/runtime; disponibles {free / 1024**3:.1f} GiB.")
    print(f"Plan HF: {total / 1024**3:.1f} GiB de modelos, faltan {missing / 1024**3:.1f} GiB, más entorno separado.", flush=True)
    # Gemma is gated. Test access with a tiny file before installing large packages.
    hf_hub_download(GEMMA_REPO, "config.json", local_dir=root / "gemma")
    if plan_only:
        return
    hf_hub_download(BASE_REPO, BASE_NAME, local_dir=root / "base")
    snapshot_download(GEMMA_REPO, local_dir=root / "gemma", ignore_patterns=["*.bin", "*.pt", "*.onnx", "*.gguf"], max_workers=4)
    hf_hub_download(LORA_REPO, LORA_NAME, local_dir=root / "lora")
    base = root / "base" / BASE_NAME
    lora = root / "lora" / LORA_NAME
    gemma = root / "gemma"
    index = gemma / "model.safetensors.index.json"
    if not base.is_file() or not lora.is_file() or not (gemma / "config.json").is_file():
        raise RuntimeError("La descarga está incompleta.")
    if index.is_file():
        data = json.loads(index.read_text())
        shards = set(data.get("weight_map", {}).values())
        if not shards or any(not (gemma / f).is_file() or (gemma / f).stat().st_size == 0 for f in shards):
            raise RuntimeError("Faltan fragmentos del codificador Gemma.")
    elif not list(gemma.glob("model*.safetensors")):
        raise RuntimeError("Faltan pesos del codificador Gemma.")
    expected = {str(path.relative_to(root)): size for path, size, _ in planned if size > 0}
    if any(not path.is_file() or path.stat().st_size != size for path, size, _ in planned if size > 0):
        raise RuntimeError("La descarga está incompleta o tiene tamaños incorrectos.")
    (root / "complete.json").write_text(json.dumps({"base": BASE_REPO, "gemma": GEMMA_REPO, "lora": LORA_REPO, "files": expected}), encoding="utf-8")
    print("Componentes ID-LoRA descargados y presentes; no implica validación GPU.", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]), plan_only="--plan" in sys.argv[2:])
