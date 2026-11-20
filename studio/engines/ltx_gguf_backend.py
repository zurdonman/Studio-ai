"""Experimental, single-pass LTX-2.3 GGUF runner using isolated ComfyUI.

I2V with real weights was observed on an L4 at 512×320, including 10 s and
20 s with a resident cache. Guardrails do not measure peak memory or guarantee
other configurations. Compatible recipes can share the private resident process.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import queue
import re
import threading
import uuid
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
from studio.memory_policy import host_memory, pressure, reserves, number, MemoryPressureError
from studio.comfy_options import flags as comfy_flags, job_options

COMFY_COMMIT = "7ad7adb38f6cbb6c2b0713d3bd44c3fb2b9c8ebe"
GGUF_COMMIT = "6ea2651e7df66d7585f6ffee804b20e92fb38b8a"
ROOT = DATA_ROOT / "runtimes" / "ltx23-gguf"
WEIGHTS = DATA_ROOT / "models" / "ltx23-gguf-q4"
MODEL_ID = "ltx-2.3-distilled-gguf-q4-experimental"
_verified_fingerprint = None


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


def _command(args: list[str], job: Any, *, stage: str, timeout: int, env: dict | None = None, cwd: str | None = None, guard_label: str | None = None, build_progress: bool = False) -> None:
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile() as log:
        proc = subprocess.Popen(args, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        guard = _ComfyMemoryGuard(proc, job, label=guard_label) if guard_label else None
        start, offset, last = time.monotonic(), 0, 0.0
        try:
            while proc.poll() is None:
                if guard:
                    guard.raise_if_stopped()
                job.check_cancelled()
                if time.monotonic() - start > timeout:
                    raise TimeoutError(f"{stage}: tiempo máximo superado")
                if time.monotonic() - last >= 5:
                    log.flush(); log.seek(offset)
                    lines = log.read(8192).decode("utf-8", "replace").replace("\r", "\n").splitlines()
                    offset = log.tell()
                    progress_lines = [line for line in lines if line.startswith("GGUF_PROGRESS:")]
                    visible = [line for line in lines if line.startswith("[") and ("Preparando" in line or "Verificado" in line)]
                    build_lines = [line for line in lines if re.match(r'^\[\s*\d+%\] ', line)] if build_progress else []
                    if build_lines:
                        match = re.match(r'^\[\s*(\d+)%\]', build_lines[-1])
                        job.heartbeat(step=f'{stage}: compilación {match.group(1)}% (no es el progreso total)')
                    elif progress_lines:
                        try:
                            p = json.loads(progress_lines[-1].split(":", 1)[1])
                            pct = 100 * p["done"] / p["total"] if p["total"] else 0
                            progress_span = 85 if job.kind == "download" else 30
                            job.heartbeat(progress=5 + progress_span * min(1, max(0, p["overall"])),
                                          step=f"{p['phase'].capitalize()} {p['role']}: {pct:.1f}%",
                                          speed_text=f"{p['done'] / 1024**3:.2f} / {p['total'] / 1024**3:.2f} GiB")
                        except (ValueError, KeyError, TypeError):
                            job.heartbeat(step=stage)
                    elif visible:
                        job.log(visible[-1][:180], step=stage)
                    else:
                        job.heartbeat(step=stage)
                    last = time.monotonic()
                time.sleep(.4)
            if guard:
                guard.raise_if_stopped()
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
            if guard:
                guard.done.set()
            _stop(proc)
            if guard:
                guard.close()


def _setup_env() -> dict:
    """Prevent Colab/user Python and pip settings from redirecting a venv install."""
    env = os.environ.copy()
    for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE", "VIRTUAL_ENV", "__PYVENV_LAUNCHER__",
                 "PIP_USER", "PIP_TARGET", "PIP_PREFIX", "PIP_ROOT", "PIP_REQUIRE_VIRTUALENV"):
        env.pop(name, None)
    env["PYTHONNOUSERSITE"] = "1"
    env["PIP_CONFIG_FILE"] = os.devnull
    return env


def _prepare_venv(job: Any, root: Path | None = None) -> Path:
    root = root or ROOT
    venv = root / "venv"
    py = venv / "bin" / "python"
    if venv.is_symlink() or (venv.exists() and not venv.is_dir()):
        raise RuntimeError("GGUF_VENV_CONFLICT: el entorno existente no es un directorio gestionado; no se modifica")
    if venv.is_dir() and not (venv / "pyvenv.cfg").is_file() and any(venv.iterdir()):
        raise RuntimeError("GGUF_VENV_CONFLICT: entorno existente sin pyvenv.cfg; no se modifica")
    env = _setup_env()
    # venv/ensurepip can leave bin/python behind after failing. Re-run without
    # ensurepip: this repairs pyvenv.cfg and never removes the partial directory.
    _command([sys.executable, "-m", "venv", "--without-pip", "--system-site-packages", str(venv)],
             job, stage="Creando o reparando entorno GGUF", timeout=120, env=env)
    verify_pip = ("import pathlib,sys,pip; "
                  "p=pathlib.Path(sys.prefix).resolve(); "
                  "assert p == pathlib.Path(sys.argv[1]).resolve(), 'Python no usa el entorno GGUF'; "
                  "assert pathlib.Path(pip.__file__).resolve().is_relative_to(p), 'pip no está instalado dentro del entorno GGUF'")
    # Call ensurepip separately so its real error, normally hidden by venv, is logged.
    ensure_error = None
    try:
        _command([str(py), "-m", "ensurepip", "--upgrade", "--default-pip"], job,
                 stage="Preparando pip en entorno GGUF", timeout=120, env=env)
        _command([str(py), "-c", verify_pip, str(venv)], job,
                 stage="Verificando pip aislado", timeout=30, env=env)
    except RuntimeError as exc:
        ensure_error = str(exc)
        job.log("El pip integrado no pudo prepararse; se intenta instalar pip solo en el entorno GGUF.")
    if ensure_error:
        try:
            # --python targets the venv even when pip is not installed there.
            _command([sys.executable, "-m", "pip", "--python", str(py), "install", "--ignore-installed", "--no-deps", "pip"],
                     job, stage="Recuperando pip aislado GGUF", timeout=300, env=env)
            _command([str(py), "-c", verify_pip, str(venv)], job,
                     stage="Verificando pip aislado", timeout=30, env=env)
        except RuntimeError as exc:
            raise RuntimeError(f"GGUF_VENV_BOOTSTRAP_FAILED: {ensure_error} | Recuperación: {exc}") from exc
    return py


def _install(job: Any, root: Path | None = None, comfy_commit: str | None = None) -> None:
    root = root or ROOT
    comfy_commit = comfy_commit or COMFY_COMMIT
    marker = root / "ready.json"
    source = root / "comfy"
    plugin = source / "custom_nodes" / "ComfyUI-GGUF"
    py = root / "venv" / "bin" / "python"
    try:
        ready = json.loads(marker.read_text())
        if ready == {"comfy": comfy_commit, "gguf": GGUF_COMMIT} and py.exists() and (source / "main.py").exists() and (plugin / "nodes.py").exists():
            return
    except (OSError, ValueError):
        pass
    root.mkdir(parents=True, exist_ok=True)
    for dest, repo, revision in [(source, "https://github.com/Comfy-Org/ComfyUI.git", comfy_commit),
                                 (plugin, "https://github.com/city96/ComfyUI-GGUF.git", GGUF_COMMIT)]:
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            _command(["git", "clone", "--no-checkout", repo, str(dest)], job, stage="Preparando código GGUF", timeout=600)
        if not (dest / ".git").is_dir():
            raise RuntimeError("GGUF_RUNTIME_CONFLICT: directorio de runtime existente no gestionado; no se borra")
        _command(["git", "-C", str(dest), "checkout", "--detach", revision], job, stage="Fijando versión de GGUF", timeout=120)
    py = _prepare_venv(job) if root == ROOT else _prepare_venv(job, root)
    env = _setup_env()
    _command([str(py), "-m", "pip", "install", "--disable-pip-version-check", "--no-cache-dir", "-r", str(source / "requirements.txt"), "-r", str(plugin / "requirements.txt")], job, stage="Instalando dependencias GGUF", timeout=2400, env=env)
    _command([str(py), str(source / "main.py"), "--help"], job, stage="Comprobando entorno GGUF", timeout=90, env=env)
    marker.write_text(json.dumps({"comfy": comfy_commit, "gguf": GGUF_COMMIT}), encoding="utf8")


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
    if state["level"] in ("critical", "unknown"):
        raise RuntimeError("GGUF_MEMORY_CRITICAL: memoria libre crítica; se detuvo el proceso experimental sin borrar pesos")


def _tail(logfile: Any) -> str:
    logfile.flush(); logfile.seek(0, 2); end = logfile.tell(); logfile.seek(max(0, end - 3000))
    return " | ".join(logfile.read().decode("utf-8", "replace").splitlines()[-8:])[:1500]


def _runtime_env() -> dict:
    env = _setup_env()
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        env.pop(name, None)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    return env


def _server_command(port: int, input_dir: Path, output_dir: Path, root: Path | None = None,
                    config: tuple[str, bool] = ("auto", False), keep_loaded: bool = False) -> list[str]:
    root = root or ROOT
    source = root / "comfy"
    ram_reserve = max(2.0, host_memory()["total_gb"] * 0.1)
    return [str(root / "venv/bin/python"), str(source / "main.py"), "--listen", "127.0.0.1", "--port", str(port),
            # Supported by both pinned Comfy versions: in newer versions this is
            # an alias of --offline; in 0.33 it disables API nodes and frontend internet.
            "--disable-auto-launch", "--disable-api-nodes", "--disable-metadata", "--disable-all-custom-nodes", "--whitelist-custom-nodes", "ComfyUI-GGUF", "StudioVoice", "StudioStrictMSR",
            *(["StudioResident"] if keep_loaded else []), "--cache-ram", f"{ram_reserve:.2f}", f"{ram_reserve:.2f}",
            "--input-directory", str(input_dir), "--output-directory", str(output_dir), "--temp-directory", str(input_dir.parent / "temp"), "--reserve-vram", "1.8"] + comfy_flags(config)


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


def _probe_runtime(job: Any, extra_required: set[str] | None = None, root: Path | None = None) -> None:
    root = root or ROOT
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="gguf-probe-", dir=TEMP_DIR) as tmp, tempfile.TemporaryFile() as log:
        inp, out = Path(tmp) / "in", Path(tmp) / "out"
        inp.mkdir(); out.mkdir()
        port = _free_port(); base = f"http://127.0.0.1:{port}"
        proc = subprocess.Popen(_server_command(port, inp, out, root, job_options(job)), cwd=root / "comfy", env=_runtime_env(), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        guard = _ComfyMemoryGuard(proc, job)
        try:
            _wait_server(proc, base, log, job)
            info = requests.get(base + "/object_info", timeout=20).json()
            required = {"UnetLoaderGGUF", "DualCLIPLoaderGGUF", "LTXVAudioVAEDecode", "LTXVEmptyLatentAudio", "LTXVConcatAVLatent", "LTXVSeparateAVLatent", "LTXVImgToVideoInplace", "CreateVideo", "SaveVideo"}
            required |= extra_required or set()
            missing = sorted(required - set(info))
            if missing:
                raise RuntimeError("GGUF_RUNTIME_MISSING_NODES: " + ", ".join(missing))
        except Exception:
            guard.raise_if_stopped()
            raise
        finally:
            guard.done.set()
            _stop(proc)
            guard.close()


class _ComfyMemoryGuard:
    """Monitor the isolated worker independently of the inference HTTP loop.

    On measured pressure, kill only this child process group. Sampling cannot
    prevent an allocation that outruns the monitor or an external process OOM.
    """
    def __init__(self, proc: subprocess.Popen, job: Any, interval: float = .25, label: str = "Comfy"):
        self.proc, self.job, self.interval, self.label = proc, job, interval, label
        self.reason: str | None = None
        self.cancelled = False
        self.done = threading.Event()
        self.last_gpu_sample = time.monotonic()
        self.threads = [threading.Thread(target=target, name=name, daemon=True) for target, name in
                        ((self._watch_ram, 'comfy-ram-guard'), (self._watch_gpu, 'comfy-vram-guard'))]
        for thread in self.threads:
            thread.start()

    def _watch_ram(self) -> None:
        failures = 0
        while not self.done.is_set() and self.proc.poll() is None:
            if getattr(self.job, 'cancel_requested', False):
                self.cancelled = True
                self._kill()
                return
            if time.monotonic() - self.last_gpu_sample > 5:
                self.reason = f'GGUF_LIVE_MEMORY_STOP: medición de VRAM bloqueada; se detuvo {self.label}.'
                self._kill()
                return
            try:
                ram = host_memory()
                total, free = number(ram.get('total_gb')), number(ram.get('available_gb'))
                if total is None or total <= 0 or free is None:
                    raise ValueError('RAM no disponible')
                reserve = reserves(total)['ram_gb']
                if free < reserve:
                    self._pressure(f'RAM disponible {free:.2f} GiB por debajo de la reserva {reserve:.2f} GiB')
                    return
                failures = 0
            except Exception:
                failures += 1
                if failures >= 3:
                    self._pressure('no se puede medir RAM de forma fiable')
                    return
            self.done.wait(self.interval)

    def _watch_gpu(self) -> None:
        failures = 0
        while not self.done.is_set() and self.proc.poll() is None:
            try:
                gpu = query_gpu_info()
                free, total = number(gpu.get('vram_free_gb')), number(gpu.get('vram_total_gb'))
                if not gpu.get('has_gpu') or free is None or total is None or total <= 0:
                    raise ValueError('VRAM no disponible')
                critical = reserves(0, total)['critical_vram_gb']
                if free < critical:
                    self._pressure(f'VRAM disponible {free:.2f} GiB por debajo del margen crítico {critical:.2f} GiB')
                    return
                failures = 0
            except Exception:
                failures += 1
                if failures >= 3:
                    self._pressure('no se puede medir VRAM de forma fiable')
                    return
            self.last_gpu_sample = time.monotonic()
            self.done.wait(self.interval)

    def _pressure(self, detail: str) -> None:
        self.reason = ('GGUF_LIVE_MEMORY_STOP: ' + detail +
                       f'. Se detuvo el proceso {self.label} para proteger el entorno; pesos en disco intactos.')
        self._kill()

    def _kill(self) -> None:
        if self.done.is_set() or self.proc.poll() is not None:
            return
        try:
            os.killpg(self.proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def raise_if_stopped(self) -> None:
        if self.cancelled:
            from studio.jobs import JobCancelledError
            raise JobCancelledError('Cancelado manualmente por el usuario.')
        if self.reason:
            raise MemoryPressureError(self.reason)

    def close(self) -> None:
        self.done.set()
        for thread in self.threads:
            thread.join(timeout=1)


def _comfy_start_admission(job: Any, *, stage: str, planned_ram: float, planned_vram: float) -> None:
    # Start floor is not a predicted peak: a separate monitor stops the isolated
    # process at measured RAM/VRAM thresholds while Comfy manages its offload.
    job.log(f'{stage}: estimación orientativa no medida {planned_ram:.2f} GiB RAM y '
            f'{planned_vram:.2f} GiB VRAM; no bloquea por sí sola. Se comprueba memoria real '
            'y se vigila el proceso Comfy durante la ejecución.', level='warning')
    memory_manager.ensure_capacity(ram_gb=4, vram_gb=2,
                                   stage='Inicio Comfy: espacio mínimo real y reservas', job=job)


def _install_resident_plugin(root: Path) -> None:
    source = Path(__file__).with_name('comfy_resident_node')
    destination = root / 'comfy' / 'custom_nodes' / 'StudioResident'
    if destination.is_symlink() or (destination.exists() and not destination.is_dir()):
        raise RuntimeError('GGUF_RESIDENT_PLUGIN_CONFLICT: directorio ajeno, no se modifica')
    old_versions = {'core.py': {'4ed20ad1bc24102e7dc47321044a8e6701fcae115d167cc3b66b22394ded50b4',
                                'f00e4157aea663dbf2c0c03beb154263d5352de7bdbd5f958932b45a9234d5db',
                                '07682439076a008995ff4a80735f7aef3800fb596fbd820c5f5cadfadf52044e',
                                '0145c73e2b9ead7dada4cd0b3a625ae8f6202edc88f3c1abe93438670469879a'},
                    '__init__.py': {'9e6d8d11e127ad676cd38441ce303309613e7ef7ee0108281638eeb9d61d323c'}}
    for name in ('__init__.py', 'core.py'):
        target = destination / name
        if target.is_symlink() or (target.exists() and not target.is_file()):
            raise RuntimeError('GGUF_RESIDENT_PLUGIN_CONFLICT: nodo diferente, no se sobrescribe')
        if target.exists() and target.read_bytes() != (source / name).read_bytes():
            if hashlib.sha256(target.read_bytes()).hexdigest() not in old_versions.get(name, set()):
                raise RuntimeError('GGUF_RESIDENT_PLUGIN_CONFLICT: nodo diferente, no se sobrescribe')
    destination.mkdir(parents=True, exist_ok=True)
    for name in ('__init__.py', 'core.py'):
        target = destination / name
        if not target.exists() or target.read_bytes() != (source / name).read_bytes():
            temp = destination / ('.' + name + '.' + uuid.uuid4().hex)
            try:
                shutil.copyfile(source / name, temp)
                os.replace(temp, target)
            finally:
                temp.unlink(missing_ok=True)


def _keep_loaded(job: Any) -> bool:
    return getattr(job, 'comfy_keep_loaded', False) is True


class _ResidentComfy:
    """Comfy cache owned by the video slot. Unload stops the process and frees its GPU."""
    def __init__(self, job: Any, names: dict[str, str], runtime_root: Path | None = None):
        runtime_root = runtime_root or ROOT
        self.runtime_root = Path(runtime_root).resolve()
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="ltx-gguf-session-", dir=TEMP_DIR)
        self.root = Path(self.temp.name)
        self.input_dir, self.output_dir = self.root / "input", self.root / "output"
        self.input_dir.mkdir(); self.output_dir.mkdir()
        self.names = names
        self.options = job_options(job)
        self.keep_loaded = _keep_loaded(job)
        self.cleanup_ok = False
        job.log(f"Comfy: política VRAM {self.options[0]}, descarga agresiva entre modelos: {'sí' if self.options[1] else 'no'}; memoria real vigilada.")
        self.port = _free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        self.log = tempfile.TemporaryFile()
        self.proc = None
        self.guard = None
        try:
            if self.keep_loaded:
                _install_resident_plugin(runtime_root)
            command = (_server_command(self.port, self.input_dir, self.output_dir, runtime_root, self.options, True)
                       if self.keep_loaded else _server_command(self.port, self.input_dir, self.output_dir, runtime_root, self.options))
            motion_node = runtime_root / 'comfy/custom_nodes/StudioMotion/__init__.py'
            motion_source = Path(__file__).with_name('comfy_motion_node') / '__init__.py'
            # Once installed, load the verified private node on every compatible
            # server so a later Wan request need not restart this process.
            motion_ready = (not motion_node.is_symlink() and not motion_node.parent.is_symlink()
                            and motion_node.is_file() and motion_node.read_bytes() == motion_source.read_bytes())
            if getattr(job, "studio_motion_plugin", False) or motion_ready:
                at = command.index("--whitelist-custom-nodes") + 1
                command.insert(at, "StudioMotion")
            self.proc = subprocess.Popen(command, cwd=runtime_root / "comfy",
                                         env=_runtime_env(), stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)
            job.log(f'Comfy privado iniciado: proceso {self.proc.pid}, runtime {self.runtime_root.name}.')
            self.guard = _ComfyMemoryGuard(self.proc, job)
            _wait_server(self.proc, self.base, self.log, job)
            if self.keep_loaded:
                info = requests.get(self.base + '/object_info/StudioResidentCleanup', timeout=10)
                info.raise_for_status()
                if 'StudioResidentCleanup' not in info.json():
                    raise RuntimeError('GGUF_RESIDENT_UNAVAILABLE: no se cargó el limpiador selectivo; no se conservan pesos')
        except BaseException:
            guard = self.guard
            self.unload()
            if guard:
                guard.raise_if_stopped()
            raise

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def unload(self) -> None:
        if self.guard is not None:
            self.guard.done.set()
        if self.proc is not None:
            _stop(self.proc)
            self.proc = None
        if self.guard is not None:
            self.guard.close()
            self.guard = None
        self.log.close()
        self.temp.cleanup()


def _resident(model_id: str = MODEL_ID) -> _ResidentComfy | None:
    slot = memory_manager.get_slot("video")
    if not slot or slot.get("model_id") != model_id:
        return None
    session = slot.get("handle")
    if not isinstance(session, _ResidentComfy) or not session.alive():
        guard = getattr(session, "guard", None)
        memory_manager.unload_slot("video")
        if guard:
            guard.raise_if_stopped()
        return None
    return session


_IDLE_COMFY = 'studio-comfy-idle'


def _switch_to_idle(job: Any, runtime_root: Path | None = None) -> bool:
    """Release cached tensors on Comfy's worker thread, retaining a compatible process.

    The GPU-operation lock serializes this with inference and memory-manager changes.
    A failed or unsupported reset closes the process; it never preserves unknown state.
    """
    slot = memory_manager.get_slot('video')
    if not slot:
        return False
    session = slot.get('handle')
    runtime = Path(runtime_root or ROOT).resolve()
    if (not isinstance(session, _ResidentComfy) or not session.alive()
        or getattr(session, 'runtime_root', None) != runtime
        or not getattr(session, 'keep_loaded', False) or not _keep_loaded(job)
        or getattr(session, 'options', None) != job_options(job)):
        job.log('Comfy anterior no es compatible con el runtime o política solicitados; se iniciará otro proceso.', level='warning')
        memory_manager.unload_slot('video')
        return False
    if slot.get('model_id') == _IDLE_COMFY:
        return True
    try:
        if not session.cleanup_ok:
            raise RuntimeError('limpieza anterior no confirmada')
        if session.guard: session.guard.job = job
        job.check_cancelled()
        reply = requests.post(session.base + '/prompt', json={'prompt': {'1': {'class_type': 'StudioResidentReset', 'inputs': {}}}}, timeout=20)
        reply.raise_for_status()
        ident = reply.json()['prompt_id']
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            job.check_cancelled()
            session.guard.raise_if_stopped()
            if not session.alive(): raise RuntimeError('proceso detenido')
            response = requests.get(session.base + '/history/' + ident, timeout=10)
            response.raise_for_status()
            result = response.json().get(ident)
            if result:
                if not (result.get('status') or {}).get('completed'):
                    raise RuntimeError('reset rechazado')
                status = requests.get(session.base + '/studio-resident/status', timeout=10)
                status.raise_for_status()
                state = status.json()
                if state.get('version') != 1 or state.get('prompt_id') != ident or state.get('ok') is not True or state.get('total_loaders') != 0:
                    raise RuntimeError('reset no confirmado')
                deleted = requests.post(session.base + '/history', json={'delete': [ident]}, timeout=10)
                deleted.raise_for_status()
                session.names = {}
                session.cleanup_ok = True
                if session.guard: session.guard.job = None
                memory_manager.register_loaded('video', _IDLE_COMFY, 'Comfy · proceso compartido', session)
                job.log(f'Comfy compartido: proceso {session.proc.pid} sigue activo; caché del modelo anterior liberada.')
                return True
            job.heartbeat(step='Liberando pesos del modelo anterior en Comfy')
            time.sleep(.25)
        raise TimeoutError('reset agotó el tiempo')
    except Exception as exc:
        memory_manager.unload_slot('video')
        job.check_cancelled()
        if isinstance(exc, (MemoryPressureError, MemoryError)):
            raise
        job.log('No se pudo confirmar el cambio seguro de modelo; se cerró Comfy y se iniciará un proceso aislado nuevo.', level='warning')
        return False


def _open_resident(job: Any, names: dict[str, str], model_id: str = MODEL_ID, runtime_root: Path | None = None) -> _ResidentComfy:
    session = _resident(model_id)
    if session is not None and (getattr(session, "options", ("auto", False)) != job_options(job)
                                or getattr(session, "keep_loaded", False) != _keep_loaded(job)):
        memory_manager.unload_slot("video")
        session = None
    if session is None:
        idle = _resident(_IDLE_COMFY)
        if idle and (getattr(idle, 'runtime_root', None) != Path(runtime_root or ROOT).resolve()
                     or getattr(idle, 'options', None) != job_options(job)
                     or getattr(idle, 'keep_loaded', False) != _keep_loaded(job)):
            memory_manager.unload_slot('video')
            idle = None
        if idle:
            required = set(getattr(job, 'comfy_required_nodes', set()))
            if getattr(job, 'studio_motion_plugin', False): required.add('StudioWanAnimate2GGUF')
            try:
                if required:
                    response = requests.get(idle.base + '/object_info', timeout=30)
                    response.raise_for_status()
                    missing = required - set(response.json())
                    if missing: raise RuntimeError('faltan nodos para la nueva receta')
            except Exception:
                job.log('Se instalaron nodos nuevos o no pudieron comprobarse; se reinicia Comfy para activarlos.', level='warning')
                memory_manager.unload_slot('video')
                idle = None
        if idle:
            memory_manager.register_loaded('video', model_id, 'Comfy · motor activo', idle)
            idle.names = names
            session = idle
            job.log(f'Comfy compartido: reutilizando proceso {session.proc.pid} para la nueva receta.')
    if session is not None:
        if getattr(session, 'guard', None):
            session.guard.job = job
        session.cleanup_ok = False
        return session
    session = _ResidentComfy(job, names, runtime_root) if runtime_root else _ResidentComfy(job, names)
    try:
        memory_manager.register_loaded("video", model_id, "LTX GGUF · motor activo", session)
    except BaseException:
        session.unload()
        raise
    return session


def _events_listener(base: str, client_id: str, events: queue.Queue, stop: threading.Event, ready: threading.Event) -> None:
    async def listen():
        import aiohttp
        async with aiohttp.ClientSession() as client:
            async with client.ws_connect(base + "/ws", params={"clientId": client_id}, timeout=5) as ws:
                ready.set()
                while not stop.is_set():
                    try:
                        msg = await asyncio.wait_for(ws.receive(), timeout=1)
                    except asyncio.TimeoutError:
                        continue
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        events.put(msg.json())
                    elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        break
    try:
        asyncio.run(listen())
    except Exception:
        ready.set()  # History polling remains available if websocket is unavailable.


def _infer(graph: dict, input_dir: Path, output_dir: Path, job: Any, model_id: str = MODEL_ID) -> Path:
    session = _resident(model_id)
    if session is None:
        raise RuntimeError("GGUF_ENGINE_EXITED: el motor aislado no está activo")
    client_id = uuid.uuid4().hex
    events: queue.Queue = queue.Queue()
    stop, ready = threading.Event(), threading.Event()
    thread = threading.Thread(target=_events_listener, args=(session.base, client_id, events, stop, ready), daemon=True)
    thread.start()
    ready.wait(6)
    try:
        reply = requests.post(session.base + "/prompt", json={"prompt": graph, "client_id": client_id}, timeout=20)
        if not reply.ok:
            raise RuntimeError(f"GGUF_GRAPH_REJECTED: {reply.text[:1200]}")
        ident = reply.json()["prompt_id"]
        job.log("Grafo aceptado. Comfy gestiona la carga y el offload durante la generación; " +
                ("se intentará conservar los pesos al terminar." if getattr(session, 'keep_loaded', False) else "el motor se liberará al terminar."),
                progress=40, step="Cargando componentes LTX")
        deadline = time.monotonic() + 3600
        current_node = None
        names = {"1":"transformer", "2":"codificador de texto", "3":"VAE vídeo", "4":"VAE audio",
                 "15":"difusión", "17":"decodificación de vídeo", "18":"decodificación de audio"}
        while time.monotonic() < deadline:
            job.check_cancelled(); _check_pressure()
            if not session.alive():
                guard = getattr(session, "guard", None)
                if guard:
                    guard.raise_if_stopped()
                raise RuntimeError("GGUF_ENGINE_EXITED: ComfyUI finalizó durante la generación")
            while not events.empty():
                event = events.get_nowait()
                data = event.get("data") or {}
                if data.get("prompt_id") not in (None, ident):
                    continue
                if event.get("type") == "execution_cached":
                    cached = [names[str(node)] for node in data.get("nodes", []) if str(node) in ("1", "2", "3", "4")]
                    if cached:
                        job.log("Comfy reutiliza de caché: " + ", ".join(cached), step="Reutilizando componentes GGUF")
                elif event.get("type") == "executing":
                    current_node = str(data.get("node") or "")
                    if current_node in names:
                        job.heartbeat(step=f"GGUF: {names[current_node]} · progreso no disponible", speed_text="")
                elif event.get("type") == "progress" and data.get("max"):
                    value, maximum = float(data.get("value", 0)), float(data["max"])
                    pct = max(0, min(100, value / maximum * 100))
                    node = str(data.get("node") or current_node or "")
                    step = names.get(node, "procesando")
                    total_progress = 45 + 45 * pct / 100 if node == "15" else None
                    job.heartbeat(progress=total_progress, step=f"GGUF {step}: {pct:.1f}%", speed_text=f"{value:g}/{maximum:g}")
            data = requests.get(session.base + "/history/" + ident, timeout=10).json().get(ident)
            if data:
                status = data.get("status") or {}
                if status.get("status_str") == "error":
                    messages = status.get("messages") or []
                    detail = next((str(x[1].get("exception_message", "")) for x in messages if len(x) > 1 and isinstance(x[1], dict) and x[0] == "execution_error"), "Error desconocido")
                    raise RuntimeError("GGUF_INFERENCE_FAILED: " + detail[:800])
                if status.get("completed"):
                    prefix = graph["20"]["inputs"]["filename_prefix"]
                    paths = list(output_dir.glob(prefix + "_*.mp4"))
                    if len(paths) != 1:
                        raise RuntimeError("GGUF_OUTPUT_MISSING: no se produjo un MP4 único")
                    if getattr(session, 'keep_loaded', False):
                        try:
                            response = requests.get(session.base + '/studio-resident/status', timeout=10)
                            response.raise_for_status()
                            cleaned = response.json()
                            if cleaned.get('version') != 1 or cleaned.get('prompt_id') != ident or cleaned.get('ok') is not True:
                                raise RuntimeError('limpieza selectiva no confirmada')
                            response = requests.post(session.base + '/history', json={'delete': [ident]}, timeout=10)
                            response.raise_for_status()
                            session.cleanup_ok = True
                            kept = cleaned.get('retained_loaders', 0)
                            job.log(f'Temporales de ejecución retirados; {kept}/4 cargadores conservados. Comfy puede mover o descargar pesos bajo presión.')
                        except Exception as exc:
                            session.cleanup_ok = False
                            job.log(f'No se confirmó la limpieza selectiva ({exc}); se cerrará Comfy tras guardar el vídeo.', level='warning')
                    job.heartbeat(progress=96, step="Verificando audio y vídeo")
                    return paths[0]
            job.heartbeat()
            time.sleep(1)
        raise TimeoutError("GGUF_INFERENCE_TIMEOUT: límite de 60 minutos")
    except Exception:
        guard = getattr(session, "guard", None)
        if guard:
            guard.raise_if_stopped()
        raise
    finally:
        stop.set()
        thread.join(timeout=3)


def _finish_session(session: Any, job: Any, success: bool) -> None:
    keep = (success and _keep_loaded(job) and getattr(session, 'keep_loaded', False)
            and getattr(session, 'cleanup_ok', False) and not getattr(job, 'cancel_requested', False)
            and session.alive())
    if keep:
        try:
            # All paths are private to this process, never the user's result directory.
            for folder in (session.input_dir, session.output_dir, session.root / 'temp', session.root / 'motion-work'):
                if folder.is_symlink():
                    raise RuntimeError('directorio temporal inesperado')
                if folder.exists():
                    shutil.rmtree(folder)
                folder.mkdir(parents=True, exist_ok=True)
            # Comfy appends its own 'temp' subdirectory to --temp-directory.
            (session.root / 'temp' / 'temp').mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            job.log(f'No se pudieron retirar todos los temporales ({exc}); se cerrará el motor.', level='warning')
            keep = False
    if keep:
        if getattr(session, 'guard', None):
            session.guard.job = None  # idle watchdog remains active, without an old job cancellation
        job.log('Comfy permanece activo con los cargadores disponibles; se liberará al vaciar memoria o bajo presión. Los pesos no se garantizan en VRAM.')
    else:
        memory_manager.unload_slot('video')
        job.log('Motor Comfy liberado; los archivos descargados permanecen en disco.')


def _fingerprint():
    try:
        return tuple((_artifact_path(role).stat().st_size, _artifact_path(role).stat().st_mtime_ns,
                      _artifact_path(role).stat().st_ino) for role, _, _, _, _, _ in FILES)
    except OSError:
        return None


def _prepare_gguf(job: Any, *, extra_ram: float = 0, extra_vram: float = 0, model_id: str = MODEL_ID) -> dict[str, str]:
    global _verified_fingerprint
    gpu = query_gpu_info()
    if not gpu.get("has_gpu") or not gpu.get("bf16_supported") or float(gpu.get("vram_total_gb") or 0) < 16:
        raise RuntimeError("GGUF_GPU_UNSUPPORTED: se necesita una GPU BF16 con al menos 16 GiB; no se descargó nada")
    session = _resident() if model_id == MODEL_ID else _resident(model_id)
    if session is not None and (getattr(session, "options", ("auto", False)) != job_options(job)
                                or getattr(session, "keep_loaded", False) != _keep_loaded(job)):
        memory_manager.unload_slot("video")
        session = None
    if session is not None:
        job.log("Comfy activo: estimaciones orientativas; continúa la vigilancia de memoria real.", level="warning")
        job.log("Reutilizando motor GGUF y cargadores; Comfy puede mover o liberar pesos según la presión de memoria.")
        return session.names
    _switch_to_idle(job)
    _comfy_start_admission(job, stage="LTX 2.3 GGUF", planned_ram=32 + extra_ram, planned_vram=14 + extra_vram)
    ROOT.mkdir(parents=True, exist_ok=True)
    remaining = sum(size for role, _, _, _, size, _ in FILES if not _artifact_path(role).is_file() or _artifact_path(role).stat().st_size != size)
    required = remaining + 10 * 1024**3
    if shutil.disk_usage(ROOT).free < required:
        raise RuntimeError(f"GGUF_DISK_LOW: se requieren {required / 1024**3:.1f} GiB libres para archivos restantes y entorno")
    _install(job)
    fingerprint = _fingerprint()
    if fingerprint is None or fingerprint != _verified_fingerprint:
        job.log("Comprobando nodos GGUF antes de descargar/verificar pesos.")
        _probe_runtime(job)
        _prepare_models(job)
        _verified_fingerprint = _fingerprint()
    else:
        job.log("Componentes GGUF ya verificados en esta sesión y sin cambios; no se recalculan los hashes.")
    return _make_model_links(ROOT / "comfy")


def gguf_disk_inventory() -> dict:
    """Report final component files without implying that their hashes were checked."""
    total, complete, present, size = len(FILES), 0, 0, 0
    for role, _, _, _, expected, _ in FILES:
        try:
            path = _artifact_path(role)
            if path.is_file():
                actual = path.stat().st_size
                present += 1
                size += actual
                complete += int(actual == expected)
        except OSError:
            continue
    return {"present": present, "complete_sizes": complete, "total": total, "size_bytes": size}


def gguf_files_present() -> bool:
    # File sizes are a cheap UI hint only; _prepare_models always verifies hashes before use.
    try:
        return (ROOT / "ready.json").is_file() and all(
            _artifact_path(role).is_file() and _artifact_path(role).stat().st_size == size
            for role, _, _, _, size, _ in FILES
        )
    except OSError:
        return False


@serialized_gpu_operation(slots=("video",), managed=True)
def prepare_gguf(*, job: Any) -> dict:
    _prepare_gguf(job)
    return {"prepared": True, "note": "Archivos y entorno preparados; el modelo se carga durante la generación, no queda en VRAM."}


def frames_for_duration(duration_sec: float, fps: int) -> int:
    if not math.isfinite(duration_sec) or not 1 <= duration_sec <= 20 or not 8 <= fps <= 30:
        raise ValueError("Duración GGUF de 1 a 20 segundos y fps de 8 a 30")
    # LTX requires 8n+1 frames. The closest valid length can differ slightly.
    return max(9, min(601, 8 * round((duration_sec * fps - 1) / 8) + 1))


_voice_lora_fingerprint = None


def _prepare_voice_lora(job: Any) -> None:
    """Verify a pinned adapter and install the small local strict loader, never overwrite foreign files."""
    global _voice_lora_fingerprint
    from studio.engines.ltx23_voice import LORA, LORA_LINK
    path = WEIGHTS / 'ltx' / LORA[3]
    needed = (0 if path.is_file() and path.stat().st_size == LORA[4] else LORA[4]) + 1024**3
    if shutil.disk_usage(WEIGHTS).free < needed:
        raise RuntimeError('GGUF_DISK_LOW: falta espacio para ID-LoRA y margen')
    def fingerprint():
        try:
            st = path.stat()
            return (st.st_size, st.st_mtime_ns, st.st_ino)
        except OSError:
            return None
    if fingerprint() is None or fingerprint() != _voice_lora_fingerprint:
        from studio.catalog import catalog_manager
        env = os.environ.copy()
        token = catalog_manager.get_hf_token()
        if token:
            env['HF_TOKEN'] = token
        repo = str(Path(__file__).resolve().parents[2])
        env['PYTHONPATH'] = repo + os.pathsep + env.get('PYTHONPATH', '')
        _command([sys.executable, '-m', 'studio.engines.ltx23_voice_download', str(WEIGHTS)], job,
                 env=env, cwd=repo, stage='Descargando y verificando ID-LoRA TalkVid', timeout=7200)
        _voice_lora_fingerprint = fingerprint()
        if _voice_lora_fingerprint is None:
            raise RuntimeError('GGUF_VOICE_LORA_MISSING: falta el adaptador verificado')
    source = Path(__file__).with_name('comfy_voice_node') / '__init__.py'
    plugin = ROOT / 'comfy' / 'custom_nodes' / 'StudioVoice' / '__init__.py'
    if plugin.exists() and plugin.read_bytes() != source.read_bytes():
        raise RuntimeError('GGUF_VOICE_NODE_CONFLICT: nodo local diferente; no se sobrescribe')
    created = not plugin.exists()
    if created:
        plugin.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, plugin)
    dest = ROOT / 'comfy' / 'models' / 'loras' / LORA_LINK
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_symlink():
        if dest.resolve() != path.resolve():
            raise RuntimeError('GGUF_VOICE_LINK_CONFLICT: enlace ID-LoRA diferente')
    elif dest.exists():
        raise RuntimeError('GGUF_VOICE_LINK_CONFLICT: archivo ID-LoRA existente no gestionado')
    else:
        dest.symlink_to(path.resolve())
    session = _resident()
    if session is not None:
        if created:
            job.log('Reiniciando Comfy una vez para activar el nodo estricto de voz; pesos en disco intactos.')
            memory_manager.unload_slot('video')
        else:
            try:
                info = requests.get(session.base + '/object_info/StudioStrictVoiceLoRA', timeout=10).json()
                if 'StudioStrictVoiceLoRA' not in info:
                    memory_manager.unload_slot('video')
            except requests.RequestException:
                memory_manager.unload_slot('video')


@serialized_gpu_operation(slots=("video",), managed=True)
def generate_gguf(*, image_path: Path | None, prompt: str, negative_prompt: str, seed: int, job: Any,
                  width: int = 512, height: int = 320, duration_sec: float = 2.0, fps: int = 24,
                  speech_language: str = "auto", image_strength: float = 1.0,
                  voice_mode: str = "none", voice_path: Path | None = None) -> dict:
    from studio.engines.ltx23_voice import add_voice_graph, frames_for_audio
    started = time.monotonic()
    if not prompt.strip():
        raise ValueError("Describe la escena")
    if image_path is not None:
        verify_image_file(image_path)
    frames = frames_for_duration(duration_sec, fps)
    if voice_mode not in ('none', 'audio', 'voice_reference') or (voice_mode == 'none') != (voice_path is None):
        raise ValueError('Modo de voz y archivo no coinciden')
    voice_info = None
    if voice_path is not None:
        voice_info = probe_media_info(voice_path)
        audio_seconds = float(voice_info.get('duration_seconds') or 0)
        if not voice_info.get('has_audio') or voice_info.get('has_video') or not 1 <= audio_seconds <= (20 if voice_mode == 'audio' else 15):
            raise ValueError('Se necesita un archivo de audio de 1 a 20 s (referencia ID-LoRA: 1 a 15 s)')
        if voice_mode == 'audio':
            frames = frames_for_audio(audio_seconds, fps)
            job.log(f'Audio completo: duración {audio_seconds:.2f} s; vídeo ajustado a {frames} fotogramas (8n+1).')
        else:
            job.log('ID-LoRA GGUF experimental, no validado en L4; usa una muestra autorizada de una sola voz (~5 s).')
    if speech_language not in ("auto", "es"):
        raise ValueError("Idioma de diálogo GGUF no soportado")
    if not math.isfinite(image_strength) or not .5 <= image_strength <= 1:
        raise ValueError("La fidelidad del fotograma debe estar entre 0.5 y 1")
    effective_prompt = prompt
    if speech_language == "es":
        effective_prompt = ("All spoken dialogue is in Spanish. Characters speak the exact quoted lines in Spanish, "
                            "without translating them into English.\n" + prompt)
    if width < 256 or height < 256 or width % 32 or height % 32 or width * height > 768 * 512:
        raise ValueError("GGUF: dimensiones múltiplos de 32 y área máxima 768×512")
    if duration_sec > 10:
        job.log("Más de 10 s es experimental; un clip de 20 s se probó en L4 con caché, pero otras condiciones pueden superar la memoria disponible.", level="warning")
    # Extra workload is a guardrail, not a measured peak. A longer clip may still OOM.
    scale = width * height * frames / (512 * 320 * 49)
    extra_ram = max(0.0, scale - 1) * 2
    extra_vram = max(0.0, scale - 1) * 0.5
    if voice_mode == 'voice_reference':
        extra_ram += 4
        extra_vram += 2
    elif voice_mode == 'audio':
        extra_ram += 1
        extra_vram += .5
    names = _prepare_gguf(job, extra_ram=extra_ram, extra_vram=extra_vram)
    if voice_mode == 'voice_reference':
        _prepare_voice_lora(job)
    resident = _resident()
    if resident is None:
        _comfy_start_admission(job, stage="LTX 2.3 GGUF", planned_ram=32 + extra_ram, planned_vram=14 + extra_vram)
    else:
        job.log("Comfy activo: estimaciones orientativas; continúa la vigilancia de memoria real.", level="warning")
    VIDEOS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    seed = int(seed) if seed else int.from_bytes(os.urandom(4), "big")
    job.log(f"GGUF: {width}×{height}, {frames} fotogramas a {fps} fps (~{frames / fps:.2f} s), 8 pasos fijos. La duración se ajusta a 8n+1.")
    suffix = uuid.uuid4().hex
    image_name = None
    audio_name = None
    generated = None
    if voice_mode == 'voice_reference':
        job.comfy_required_nodes = {'StudioStrictVoiceLoRA'}
    session = _open_resident(job, names)
    completed = False
    try:
        if image_path is not None:
            image_name = f"frame_{suffix}.png"
            from PIL import Image
            with Image.open(image_path) as im:
                im.convert("RGB").save(session.input_dir / image_name)
        graph = make_graph(prompt=effective_prompt, negative_prompt=negative_prompt, width=width, height=height, frames=frames, fps=fps,
                           seed=seed, image_name=image_name, model=names["dit"], text_encoder=names["text_encoder"],
                           connectors=names["connectors"], video_vae=names["video_vae"], audio_vae=names["audio_vae"], image_strength=image_strength)
        if voice_path is not None:
            from studio.engines.ffmpeg_utils import get_ffmpeg_bin
            audio_name = f'voice_{suffix}.wav'
            args = [get_ffmpeg_bin(), '-v', 'error', '-y', '-i', str(voice_path), '-vn', '-ac', '1', '-ar', '44100']
            if voice_mode == 'audio':
                # Pad, never truncate, to the 8n+1 duration selected above.
                args += ['-af', f'apad,atrim=duration={frames / fps:.6f}']
            args += ['-c:a', 'pcm_s16le', str(session.input_dir / audio_name)]
            _command(args, job, stage='Preparando audio GGUF', timeout=120)
            add_voice_graph(graph, voice_mode, audio_name)
        graph["20"]["inputs"]["filename_prefix"] = f"studio_gguf_{suffix}"
        generated = _infer(graph, session.input_dir, session.output_dir, job)
        info = probe_media_info(generated)
        if not info.get("has_video") or not info.get("has_audio"):
            raise RuntimeError("GGUF_OUTPUT_INVALID: falta vídeo o audio en el resultado; no se publicará")
        actual_duration = info.get("duration_seconds") or 0
        duration_warning = None
        if actual_duration and abs(actual_duration - frames / fps) > max(.3, 2 / fps):
            duration_warning = f"Duración medida {actual_duration:.2f} s; esperada ~{frames / fps:.2f} s. Revisa el resultado."
            job.log(duration_warning, level="warning", step="Aviso de duración")
        name = f"ltx_gguf_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{seed % 100000}.mp4"
        dest = VIDEOS_OUT_DIR / name
        if dest.exists():
            raise RuntimeError("GGUF_OUTPUT_COLLISION: no se sobrescribirá el vídeo existente")
        shutil.copyfile(generated, dest)
        completed = True
    except Exception:
        guard = getattr(session, "guard", None)
        if guard:
            guard.raise_if_stopped()
        raise
    finally:
        try:
            if image_name:
                (session.input_dir / image_name).unlink(missing_ok=True)
            if audio_name:
                (session.input_dir / audio_name).unlink(missing_ok=True)
            if generated:
                generated.unlink(missing_ok=True)
        except Exception:
            completed = False
            raise
        finally:
            _finish_session(session, job, completed)
    meta = {"id": dest.stem, "filename": name, "rel_path": f"outputs/videos/{name}", "media_url": f"/media/outputs/videos/{name}",
            "type": "video", "model_id": MODEL_ID, "model_name": "LTX 2.3 destilado GGUF Q4 experimental",
            "video_submode": "i2v" if image_path else "t2v", "audio_native": True, "prompt": prompt, "seed": seed,
            "width": info.get("width"), "height": info.get("height"), "duration_seconds": info.get("duration_seconds"), "fps": fps, "steps": 8,
            "requested_duration_sec": duration_sec, "frames": frames, "duration_warning": duration_warning,
            "voice_mode": voice_mode, "source_audio_duration_seconds": voice_info.get('duration_seconds') if voice_info else None,
            "speech_language_hint": speech_language, "image_strength": image_strength if image_path else None,
            "comfy_vram_mode": job_options(job)[0], "comfy_evict_models": job_options(job)[1],
            "effective_prompt": effective_prompt, "guidance_scale": 1.0, "elapsed_seconds": round(time.monotonic() - started, 2),
            "created_at": datetime.now().isoformat(timespec="seconds")}
    try:
        (VIDEOS_OUT_DIR / (dest.stem + ".json")).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf8")
    except BaseException:
        memory_manager.unload_slot("video")
        raise
    return meta
