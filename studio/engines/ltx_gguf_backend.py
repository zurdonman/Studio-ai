"""Experimental, single-pass LTX-2.3 GGUF runner using isolated ComfyUI.

Not validated with real model weights or GPU. No claim that guardrails measure
peak memory. A private child process holds GPU only for the duration of a job.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

from studio.config import DATA_ROOT, TEMP_DIR, VIDEOS_OUT_DIR
from studio.engines.ffmpeg_utils import probe_media_info, verify_image_file
from studio.engines.ltx_gguf_manifest import FILES
from studio.engines.ltx_gguf_workflow import make_graph
from studio.hardware import hardware_tracker, query_gpu_info
from studio.memory_manager import memory_manager, serialized_gpu_operation
from studio.memory_policy import pressure

COMFY_COMMIT = "7ad7adb38f6cbb6c2b0713d3bd44c3fb2b9c8ebe"
GGUF_COMMIT = "6ea2651e7df66d7585f6ffee804b20e92fb38b8a"
ROOT = DATA_ROOT / "runtimes" / "ltx23-gguf"
WEIGHTS = DATA_ROOT / "models" / "ltx23-gguf-q4"


def _stop(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        except ProcessLookupError:
            pass


def _command(args: list[str], job: Any, *, stage: str, timeout: int, env: dict | None = None, cwd: str | None = None) -> None:
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile() as log:
        proc = subprocess.Popen(args, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        start, offset, last = time.monotonic(), 0, 0.0
        try:
            while proc.poll() is None:
                job.check_cancelled()
                if time.monotonic() - start > timeout:
                    raise TimeoutError(f"{stage}: tiempo máximo superado")
                if time.monotonic() - last >= 5:
                    log.flush(); log.seek(offset)
                    lines = log.read(8192).decode("utf-8", "replace").replace("\r", "\n").splitlines()
                    offset = log.tell()
                    visible = [line for line in lines if line.startswith("[") and ("Comprobando" in line or "Verificado" in line)]
                    if visible:
                        job.log(visible[-1][:180], step=stage)
                    else:
                        job.heartbeat(step=stage)
                    last = time.monotonic()
                time.sleep(.4)
            job.check_cancelled()
            if proc.returncode:
                log.flush(); log.seek(0, 2); end = log.tell(); log.seek(max(0, end - 3500))
                tail = log.read().decode("utf-8", "replace")
                for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
                    token = (env or os.environ).get(key)
                    if token:
                        tail = tail.replace(token, "[TOKEN OCULTO]")
                raise RuntimeError(f"GGUF_SETUP_FAILED: {stage} (código {proc.returncode}): " + " | ".join(tail.splitlines()[-5:])[:1200])
        finally:
            _stop(proc)


def _install(job: Any) -> None:
    marker = ROOT / "ready.json"
    source = ROOT / "comfy"
    plugin = source / "custom_nodes" / "ComfyUI-GGUF"
    py = ROOT / "venv" / "bin" / "python"
    try:
        ready = json.loads(marker.read_text())
        if ready == {"comfy": COMFY_COMMIT, "gguf": GGUF_COMMIT} and py.exists() and (source / "main.py").exists() and (plugin / "nodes.py").exists():
            return
    except (OSError, ValueError):
        pass
    ROOT.mkdir(parents=True, exist_ok=True)
    for dest, repo, revision in [(source, "https://github.com/Comfy-Org/ComfyUI.git", COMFY_COMMIT),
                                 (plugin, "https://github.com/city96/ComfyUI-GGUF.git", GGUF_COMMIT)]:
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            _command(["git", "clone", "--no-checkout", repo, str(dest)], job, stage="Preparando código GGUF", timeout=600)
        if not (dest / ".git").is_dir():
            raise RuntimeError("GGUF_RUNTIME_CONFLICT: directorio de runtime existente no gestionado; no se borra")
        _command(["git", "-C", str(dest), "checkout", "--detach", revision], job, stage="Fijando versión de GGUF", timeout=120)
    if not py.exists():
        _command([sys.executable, "-m", "venv", "--system-site-packages", str(ROOT / "venv")], job, stage="Creando entorno separado", timeout=120)
    _command([str(py), "-m", "pip", "install", "--disable-pip-version-check", "--no-cache-dir", "-r", str(source / "requirements.txt"), "-r", str(plugin / "requirements.txt")], job, stage="Instalando dependencias GGUF", timeout=2400)
    _command([str(py), str(source / "main.py"), "--help"], job, stage="Comprobando entorno GGUF", timeout=90)
    marker.write_text(json.dumps({"comfy": COMFY_COMMIT, "gguf": GGUF_COMMIT}), encoding="utf8")


def _artifact_path(role: str) -> Path:
    for item_role, _, _, name, _, _ in FILES:
        if item_role == role:
            return WEIGHTS / ("gemma" if role == "text_encoder" else "ltx") / name
    raise KeyError(role)


def _prepare_models(job: Any) -> None:
    from studio.catalog import catalog_manager
    env = os.environ.copy()
    token = catalog_manager.get_hf_token()
    if token:
        env["HF_TOKEN"] = token
    repo = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = repo + os.pathsep + env.get("PYTHONPATH", "")
    _command([sys.executable, "-m", "studio.engines.ltx_gguf_download", str(WEIGHTS)], job, env=env, cwd=repo,
             stage="Descargando y verificando GGUF y componentes", timeout=7200)


def _make_model_links(source: Path) -> dict[str, str]:
    roles = {"dit": "diffusion_models", "text_encoder": "text_encoders", "connectors": "text_encoders", "video_vae": "vae", "audio_vae": "vae"}
    names = {}
    for role, folder in roles.items():
        file = _artifact_path(role)
        if not file.is_file():
            raise RuntimeError(f"GGUF_MISSING_COMPONENT: {role}")
        dst = source / "models" / folder / file.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.is_symlink():
            if dst.resolve() != file.resolve():
                raise RuntimeError(f"GGUF_MODEL_LINK_CONFLICT: {dst.name}")
        elif dst.exists():
            raise RuntimeError(f"GGUF_MODEL_LINK_CONFLICT: {dst.name}")
        else:
            dst.symlink_to(file.resolve())
        names[role] = file.name
    return names


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _check_pressure() -> None:
    state = pressure(hardware_tracker.get_snapshot())
    if state["level"] == "critical":
        raise RuntimeError("GGUF_MEMORY_CRITICAL: memoria libre crítica; se detuvo el proceso experimental sin borrar pesos")


def _tail(logfile: Any) -> str:
    logfile.flush(); logfile.seek(0, 2); end = logfile.tell(); logfile.seek(max(0, end - 3000))
    return " | ".join(logfile.read().decode("utf-8", "replace").splitlines()[-8:])[:1500]


def _runtime_env() -> dict:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        env.pop(name, None)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    return env


def _server_command(port: int, input_dir: Path, output_dir: Path) -> list[str]:
    source = ROOT / "comfy"
    return [str(ROOT / "venv/bin/python"), str(source / "main.py"), "--listen", "127.0.0.1", "--port", str(port),
            "--disable-auto-launch", "--offline", "--disable-metadata", "--disable-all-custom-nodes", "--whitelist-custom-nodes", "ComfyUI-GGUF",
            "--input-directory", str(input_dir), "--output-directory", str(output_dir), "--reserve-vram", "1.8"]


def _wait_server(proc: subprocess.Popen, base: str, logfile: Any, job: Any) -> None:
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        job.check_cancelled(); _check_pressure()
        if proc.poll() is not None:
            raise RuntimeError("GGUF_SERVER_START_FAILED: " + _tail(logfile))
        try:
            r = requests.get(base + "/system_stats", timeout=2)
            if r.ok:
                return
        except requests.RequestException:
            pass
        job.heartbeat(step="Arrancando motor GGUF aislado")
        time.sleep(1)
    raise TimeoutError("GGUF_SERVER_TIMEOUT: ComfyUI no arrancó en 3 minutos")


def _probe_runtime(job: Any) -> None:
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="gguf-probe-", dir=TEMP_DIR) as tmp, tempfile.TemporaryFile() as log:
        inp, out = Path(tmp) / "in", Path(tmp) / "out"
        inp.mkdir(); out.mkdir()
        port = _free_port(); base = f"http://127.0.0.1:{port}"
        proc = subprocess.Popen(_server_command(port, inp, out), cwd=ROOT / "comfy", env=_runtime_env(), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            _wait_server(proc, base, log, job)
            info = requests.get(base + "/object_info", timeout=20).json()
            required = {"UnetLoaderGGUF", "DualCLIPLoaderGGUF", "LTXVAudioVAEDecode", "LTXVEmptyLatentAudio", "LTXVConcatAVLatent", "LTXVSeparateAVLatent", "LTXVImgToVideoInplace", "CreateVideo", "SaveVideo"}
            missing = sorted(required - set(info))
            if missing:
                raise RuntimeError("GGUF_RUNTIME_MISSING_NODES: " + ", ".join(missing))
        finally:
            _stop(proc)


def _infer(graph: dict, input_dir: Path, output_dir: Path, job: Any) -> Path:
    source = ROOT / "comfy"
    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    with tempfile.TemporaryFile() as logfile:
        proc = subprocess.Popen(_server_command(port, input_dir, output_dir), cwd=source, env=_runtime_env(), stdout=logfile, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            _wait_server(proc, base, logfile, job)
            reply = requests.post(base + "/prompt", json={"prompt": graph}, timeout=20)
            if not reply.ok:
                raise RuntimeError(f"GGUF_GRAPH_REJECTED: {reply.text[:1200]}")
            ident = reply.json()["prompt_id"]
            job.log("Grafo aceptado: generando vídeo y audio en un proceso aislado. Primera prueba sin validar en GPU.", step="Generando GGUF")
            deadline = time.monotonic() + 3600
            while time.monotonic() < deadline:
                job.check_cancelled(); _check_pressure()
                if proc.poll() is not None:
                    raise RuntimeError("GGUF_ENGINE_EXITED: ComfyUI finalizó durante la generación")
                data = requests.get(base + "/history/" + ident, timeout=10).json().get(ident)
                if data:
                    status = data.get("status") or {}
                    if status.get("status_str") == "error":
                        messages = status.get("messages") or []
                        detail = next((str(x[1].get("exception_message", "")) for x in messages if len(x) > 1 and isinstance(x[1], dict) and x[0] == "execution_error"), "Error desconocido")
                        raise RuntimeError("GGUF_INFERENCE_FAILED: " + detail[:800])
                    if status.get("completed"):
                        paths = list(output_dir.glob("studio_gguf_*.mp4"))
                        if len(paths) != 1:
                            raise RuntimeError("GGUF_OUTPUT_MISSING: no se produjo un MP4 único")
                        return paths[0]
                job.heartbeat(step="Generando vídeo y audio GGUF; puede tardar varios minutos")
                time.sleep(2)
            raise TimeoutError("GGUF_INFERENCE_TIMEOUT: límite de 60 minutos")
        finally:
            _stop(proc)


def _prepare_gguf(job: Any) -> dict[str, str]:
    gpu = query_gpu_info()
    if not gpu.get("has_gpu") or not gpu.get("bf16_supported") or float(gpu.get("vram_total_gb") or 0) < 16:
        raise RuntimeError("GGUF_GPU_UNSUPPORTED: se necesita una GPU BF16 con al menos 16 GiB; no se descargó nada")
    # Lower bounds are guardrails, not a measured memory budget or compatibility guarantee.
    for slot in ("video", "image", "vision"):
        if memory_manager.get_slot(slot):
            job.log(f"Liberando {slot} de memoria; sus archivos en disco se conservan")
            memory_manager.unload_slot(slot)
    memory_manager.ensure_capacity(ram_gb=32, vram_gb=14, stage="Prueba GGUF experimental (mínimos preventivos, no medición)", job=job)
    ROOT.mkdir(parents=True, exist_ok=True)
    remaining = sum(size for role, _, _, _, size, _ in FILES if not _artifact_path(role).is_file() or _artifact_path(role).stat().st_size != size)
    required = remaining + 10 * 1024**3
    if shutil.disk_usage(ROOT).free < required:
        raise RuntimeError(f"GGUF_DISK_LOW: se requieren {required / 1024**3:.1f} GiB libres para archivos restantes y entorno")
    job.log("Experimental: descarga GGUF y componentes, CPU offload y control de memoria. No se ha probado en L4.")
    _install(job)
    job.log("Comprobando nodos GGUF antes de descargar los pesos.")
    _probe_runtime(job)
    _prepare_models(job)
    names = _make_model_links(ROOT / "comfy")
    return names


def gguf_files_present() -> bool:
    # File sizes are a cheap UI hint only; _prepare_models always verifies hashes before use.
    try:
        return (ROOT / "ready.json").is_file() and all(
            _artifact_path(role).is_file() and _artifact_path(role).stat().st_size == size
            for role, _, _, _, size, _ in FILES
        )
    except OSError:
        return False


@serialized_gpu_operation(slots=(), managed=True)
def prepare_gguf(*, job: Any) -> dict:
    _prepare_gguf(job)
    return {"prepared": True, "note": "Archivos y entorno preparados; el modelo se carga durante la generación, no queda en VRAM."}


@serialized_gpu_operation(slots=(), managed=True)
def generate_gguf(*, image_path: Path | None, prompt: str, negative_prompt: str, seed: int, job: Any) -> dict:
    started = time.monotonic()
    if not prompt.strip():
        raise ValueError("Describe la escena")
    if image_path is not None:
        verify_image_file(image_path)
    names = _prepare_gguf(job)
    memory_manager.ensure_capacity(ram_gb=32, vram_gb=14, stage="Inicio GGUF experimental", job=job)
    VIDEOS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    seed = int(seed) if seed else int.from_bytes(os.urandom(4), "big")
    with tempfile.TemporaryDirectory(prefix="ltx-gguf-", dir=TEMP_DIR) as tmp:
        tmp = Path(tmp); inp, out = tmp / "input", tmp / "output"; inp.mkdir(); out.mkdir()
        image_name = None
        if image_path is not None:
            image_name = "frame.png"
            from PIL import Image
            with Image.open(image_path) as im:
                im.convert("RGB").save(inp / image_name)
        graph = make_graph(prompt=prompt, negative_prompt=negative_prompt, width=512, height=320, frames=49, fps=24,
                           seed=seed, image_name=image_name, model=names["dit"], text_encoder=names["text_encoder"],
                           connectors=names["connectors"], video_vae=names["video_vae"], audio_vae=names["audio_vae"])
        generated = _infer(graph, inp, out, job)
        info = probe_media_info(generated)
        if not info.get("has_video") or not info.get("has_audio"):
            raise RuntimeError("GGUF_OUTPUT_INVALID: falta vídeo o audio en el resultado; no se publicará")
        name = f"ltx_gguf_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{seed % 100000}.mp4"
        dest = VIDEOS_OUT_DIR / name
        if dest.exists():
            raise RuntimeError("GGUF_OUTPUT_COLLISION: no se sobrescribirá el vídeo existente")
        shutil.copyfile(generated, dest)
    meta = {"id": dest.stem, "filename": name, "rel_path": f"outputs/videos/{name}", "media_url": f"/media/outputs/videos/{name}",
            "type": "video", "model_id": "ltx-2.3-distilled-gguf-q4-experimental", "model_name": "LTX 2.3 destilado GGUF Q4 experimental",
            "video_submode": "i2v" if image_path else "t2v", "audio_native": True, "prompt": prompt, "seed": seed,
            "width": info.get("width"), "height": info.get("height"), "duration_seconds": info.get("duration_seconds"), "fps": 24, "steps": 8,
            "guidance_scale": 1.0, "elapsed_seconds": round(time.monotonic() - started, 2),
            "created_at": datetime.now().isoformat(timespec="seconds")}
    (VIDEOS_OUT_DIR / (dest.stem + ".json")).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf8")
    return meta
