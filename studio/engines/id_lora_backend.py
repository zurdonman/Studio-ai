"""Experimental adapter to the upstream LTX-2.3 ID-LoRA one-stage CLI.

The separate interpreter avoids replacing Studio's Diffusers/Transformers stack.
This is NOT a proven L4 recipe, and has not been tested with model weights on GPU.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from studio.config import DATA_ROOT, TEMP_DIR, VIDEOS_OUT_DIR
from studio.engines.ffmpeg_utils import get_ffmpeg_bin, probe_media_info, verify_image_file
from studio.memory_manager import memory_manager, serialized_gpu_operation
from studio.hardware import query_gpu_info

COMMIT = "13dc9c286a2d3c1a21a7781f9ac0fe7727ec56cd"
REPO = "https://github.com/ID-LoRA/ID-LoRA.git"
RUNTIME_ROOT = DATA_ROOT / "runtimes" / "id-lora-23"
WEIGHTS_ROOT = DATA_ROOT / "models" / "id-lora-23-official"


def _run_child(cmd: list[str], job: Any, *, env: dict | None = None, timeout: int = 3600, stage: str) -> None:
    """Capture logs without blocking, and kill the entire process group on cancellation/OOM."""
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="id-lora-log-", dir=TEMP_DIR) as tmp:
        log_path = Path(tmp) / "child.log"
        with log_path.open("wb") as log:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            start = time.monotonic()
            offset, last_log = 0, 0.0
            try:
                while proc.poll() is None:
                    job.check_cancelled()
                    if time.monotonic() - start > timeout:
                        raise TimeoutError(f"{stage} superó el límite de {timeout // 60} min.")
                    if time.monotonic() - last_log >= 5:
                        with log_path.open("rb") as reader:
                            reader.seek(offset)
                            data = reader.read(8192)
                            offset = reader.tell()
                        lines = data.decode("utf-8", "replace").replace("\r", "\n").splitlines()
                        # Only inference milestones, never child command lines or credentials.
                        visible = [x.strip() for x in lines if x.strip().startswith(("Step ", "Loading ", "Creating ", "Generating ", "Plan HF:", "Componentes ID-LoRA", "Fetching ", "Downloading "))]
                        if visible:
                            job.log(visible[-1][:200], step=stage)
                        else:
                            job.heartbeat(step=stage)
                        last_log = time.monotonic()
                    time.sleep(.4)
                job.check_cancelled()
                if proc.returncode:
                    with log_path.open("rb") as reader:
                        reader.seek(max(0, log_path.stat().st_size - 6000))
                        tail = reader.read().decode("utf-8", "replace")
                    if env and env.get("HF_TOKEN"):
                        tail = tail.replace(env["HF_TOKEN"], "[TOKEN OCULTO]")
                    lines = [line.strip()[:500] for line in tail.splitlines() if line.strip()]
                    diagnostic = " | ".join(lines[-4:])
                    raise RuntimeError(f"ID_LORA_CHILD_FAILED: {stage} terminó con código {proc.returncode}. {diagnostic}")
            except BaseException:
                if proc.poll() is None:
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
                    except ProcessLookupError:
                        pass
                raise


def _runtime_ready() -> bool:
    marker = RUNTIME_ROOT / "ready.json"
    try:
        return json.loads(marker.read_text(encoding="utf-8"))["commit"] == COMMIT and (RUNTIME_ROOT / "venv/bin/python").is_file() and (RUNTIME_ROOT / "source/ID-LoRA-2.3/scripts/inference_one_stage.py").is_file()
    except (OSError, ValueError, KeyError):
        return False


def _weights_ready() -> bool:
    root = WEIGHTS_ROOT
    try:
        from studio.engines.id_lora_download import BASE_NAME, LORA_NAME
        if not (root / "complete.json").is_file() or not (root / "base" / BASE_NAME).is_file() or not (root / "lora" / LORA_NAME).is_file():
            return False
        if min((root / "base" / BASE_NAME).stat().st_size, (root / "lora" / LORA_NAME).stat().st_size) <= 0:
            return False
        marker = json.loads((root / "complete.json").read_text(encoding="utf-8"))
        files = marker.get("files")
        if not isinstance(files, dict) or not files:
            return False
        resolved_root = root.resolve()
        for name, size in files.items():
            target = (root / name).resolve()
            if resolved_root not in target.parents or not target.is_file() or target.stat().st_size != size:
                return False
        gemma = root / "gemma"
        if not (gemma / "config.json").is_file():
            return False
        idx = gemma / "model.safetensors.index.json"
        if idx.is_file():
            shards = set(json.loads(idx.read_text(encoding="utf-8")).get("weight_map", {}).values())
            return bool(shards) and all((gemma / x).is_file() and (gemma / x).stat().st_size > 0 for x in shards)
        return any(x.is_file() and x.stat().st_size > 0 for x in gemma.glob("model*.safetensors"))
    except (OSError, ValueError, TypeError):
        return False


def _hardware_admission(job: Any) -> None:
    gpu = query_gpu_info()
    if not gpu.get("has_gpu") or gpu.get("bf16_supported") is not True or float(gpu.get("vram_total_gb") or 0) < 60:
        raise RuntimeError("ID_LORA_GPU_NOT_VALIDATED: esta integración BF16 experimental exige una GPU BF16 de al menos 60 GiB de VRAM total. L4/T4 no están admitidas; no se descargó nada.")
    # Conservative planning budget, not a measured peak or success guarantee.
    memory_manager.ensure_capacity(ram_gb=72, vram_gb=48, stage="ID-LoRA LTX-2.3 BF16 (estimación conservadora)", job=job)


def _prepare_runtime(job: Any) -> None:
    if _runtime_ready():
        return
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(RUNTIME_ROOT).free < 12 * 1024**3:
        raise RuntimeError("ID_LORA_DISK_LOW: faltan 12 GiB libres para instalar el entorno aislado.")
    source = RUNTIME_ROOT / "source"
    if not source.exists():
        with tempfile.TemporaryDirectory(prefix="source-", dir=RUNTIME_ROOT) as tmp:
            checkout = Path(tmp) / "repo"
            _run_child(["git", "clone", "--no-checkout", REPO, str(checkout)], job, timeout=600, stage="Descargando código oficial ID-LoRA")
            _run_child(["git", "-C", str(checkout), "checkout", "--detach", COMMIT], job, timeout=120, stage="Fijando revisión oficial")
            checkout.rename(source)
    elif not (source / ".git").exists():
        raise RuntimeError("ID_LORA_RUNTIME_INCOMPLETE: existe un directorio source no gestionado; no se elimina automáticamente.")
    else:
        _run_child(["git", "-C", str(source), "checkout", "--detach", COMMIT], job, timeout=120, stage="Fijando revisión oficial")
    py = RUNTIME_ROOT / "venv/bin/python"
    if not py.exists():
        _run_child([sys.executable, "-m", "venv", str(RUNTIME_ROOT / "venv")], job, timeout=300, stage="Creando entorno aislado")
    packages = source / "ID-LoRA-2.3/packages"
    _run_child([str(py), "-m", "pip", "install", "--disable-pip-version-check", "--no-cache-dir", "-q",
                "torch==2.7.1", "torchvision==0.22.1", "torchaudio==2.7.1", "soundfile>=0.12,<1", "transformers>=4.52,<4.58",
                str(packages / "ltx-core"), str(packages / "ltx-pipelines"), str(packages / "ltx-trainer")],
               job, timeout=2400, stage="Instalando dependencias aisladas ID-LoRA")
    script = source / "ID-LoRA-2.3/scripts/inference_one_stage.py"
    _run_child([str(py), str(script), "--help"], job, timeout=120, stage="Comprobando importaciones ID-LoRA")
    _run_child([str(py), "-c", "import torchaudio; assert 'soundfile' in torchaudio.list_audio_backends(), 'Falta backend de lectura WAV'"], job, timeout=120, stage="Comprobando lectura de voz WAV")
    (RUNTIME_ROOT / "ready.json").write_text(json.dumps({"commit": COMMIT}), encoding="utf-8")


def _prepare_weights(job: Any, *, plan_only: bool = False) -> None:
    if _weights_ready():
        return
    from studio.catalog import catalog_manager
    env = os.environ.copy()
    token = catalog_manager.get_hf_token()
    if token:
        env["HF_TOKEN"] = token
    script = Path(__file__).with_name("id_lora_download.py")
    cmd = [sys.executable, str(script), str(WEIGHTS_ROOT)] + (["--plan"] if plan_only else [])
    _run_child(cmd, job, env=env, timeout=180 if plan_only else 1800, stage="Comprobando acceso y disco ID-LoRA" if plan_only else "Descargando componentes ID-LoRA")
    if not plan_only and not _weights_ready():
        raise RuntimeError("ID_LORA_WEIGHTS_INCOMPLETE: no están todos los componentes requeridos.")


def validate_inputs(image_path: Path, voice_path: Path, prompt: str) -> dict:
    if not prompt.strip():
        raise ValueError("Describe la acción y el diálogo que debe generar el vídeo.")
    verify_image_file(image_path)
    info = probe_media_info(voice_path)
    if not info.get("has_audio") or not 2 <= info.get("duration_seconds", 0) <= 15:
        raise ValueError("La referencia debe contener entre 2 y 15 segundos de audio; se recomienda una sola voz limpia de unos 5 segundos.")
    return info


@serialized_gpu_operation(slots=(), managed=True)
def generate_id_lora_clip(image_path: Path, voice_path: Path, prompt: str, seed: int, job: Any) -> dict:
    from studio.engines.id_lora_download import BASE_NAME, LORA_NAME
    validate_inputs(image_path, voice_path, prompt)
    _hardware_admission(job)  # Must precede any download, setup, or model loading.
    _prepare_weights(job, plan_only=True)
    _prepare_runtime(job)
    _prepare_weights(job)
    _hardware_admission(job)
    job.log("Receta upstream experimental: una voz, un primer fotograma, 121 fotogramas a 25 fps, 30 pasos BF16. Sin garantía de identidad ni encaje de memoria.", progress=30, step="Preparando referencia")
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    VIDEOS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="id-lora-job-", dir=TEMP_DIR) as tmp:
        tmp = Path(tmp)
        audio = tmp / "reference.wav"
        _run_child([get_ffmpeg_bin(), "-v", "error", "-y", "-i", str(voice_path), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(audio)], job, timeout=120, stage="Preparando voz de referencia")
        outdir = tmp / "output"
        cmd = [str(RUNTIME_ROOT / "venv/bin/python"), str(RUNTIME_ROOT / "source/ID-LoRA-2.3/scripts/inference_one_stage.py"),
               "--checkpoint", str(WEIGHTS_ROOT / "base" / BASE_NAME), "--text-encoder-path", str(WEIGHTS_ROOT / "gemma"),
               "--lora-path", str(WEIGHTS_ROOT / "lora" / LORA_NAME), "--first-frame", str(image_path),
               "--reference-audio", str(audio), "--prompt", prompt, "--output-dir", str(outdir),
               "--seed", str(int(seed)), "--num-frames", "121", "--num-inference-steps", "30"]
        _run_child(cmd, job, timeout=1200, stage="Generando vídeo y voz con ID-LoRA")
        generated = outdir / "output_0000.mp4"
        info = probe_media_info(generated)
        if not info["has_video"] or not info["has_audio"]:
            raise RuntimeError("ID_LORA_AUDIO_OR_VIDEO_MISSING: no se acepta un clip sin vídeo y audio nativos.")
        job.check_cancelled()
        out = VIDEOS_OUT_DIR / f"id_lora_{datetime.now():%Y%m%d_%H%M%S}_{job.id[:6]}.mp4"
        shutil.move(str(generated), str(out))
        rel = f"outputs/videos/{out.name}"
        meta = {"id": out.stem, "filename": out.name, "rel_path": rel, "media_url": f"/media/{rel}", "type": "video",
                "model_id": "ltx23-id-lora-official-experimental", "model_name": "LTX 2.3 ID-LoRA one-stage experimental",
                "video_submode": "i2v", "prompt": prompt, "reference_voice_rel": str(voice_path.relative_to(DATA_ROOT)),
                "duration_seconds": info["duration_seconds"], "width": info["width"], "height": info["height"], "fps": 25,
                "steps": 30, "seed": int(seed), "audio_native": True, "gpu_tested_recipe": False,
                "created_at": datetime.now().isoformat(timespec="seconds")}
        try:
            out.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        except BaseException:
            out.unlink(missing_ok=True)
            raise
        return meta
