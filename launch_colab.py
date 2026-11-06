#!/usr/bin/env python3
"""
Lanzador todo-en-uno para Google Colab:
1. Instala dependencias en limpio sin importar diffusers/bitsandbytes antes de tiempo.
2. Arranca app.py en un PROCESO PYTHON NUEVO con salida sin buffer (PYTHONUNBUFFERED=1)
   y redirige cada línea en tiempo real a la celda de Google Colab (ipykernel OutStream).
3. Abre enlace directo https://... con cloudflared efímero (sin cuentas ni claves) + proxy nativo de Colab.
4. Escucha el botón «Liberar GPU» de la interfaz para desconectar la máquina de Colab al instante.
"""

import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
PORT = int(os.environ.get("PORT", "7860"))


def _print_box(lines: list[str]) -> None:
    width = max(len(ln) for ln in lines) + 4
    print("\n" + "═" * width, flush=True)
    for ln in lines:
        print(f"  {ln}", flush=True)
    print("═" * width + "\n", flush=True)


def _remove_incompatible_preinstalled_packages() -> None:
    """
    Google Colab trae preinstalado torchao==0.10.0 sin usar, y peft>=0.20 lanza ImportError
    al cargar cualquier LoRA si detecta torchao < 0.16.0. Como nuestros modelos 4-bit usan
    bitsandbytes (no torchao), desinstalar un torchao obsoleto evita ese choque en 0.2s.
    """
    check_code = (
        "import importlib.metadata as m\n"
        "try:\n"
        "    v = m.version('torchao')\n"
        "    parts = [int(x) for x in v.split('.')[:2] if x.isdigit()]\n"
        "    print(v if tuple(parts) < (0, 16) else 'OK')\n"
        "except Exception:\n"
        "    print('NONE')\n"
    )
    res = subprocess.run([sys.executable, "-c", check_code], capture_output=True, text=True)
    out = (res.stdout or "").strip()
    if out and out not in ("OK", "NONE"):
        print(f"🔧 Quitando paquete antiguo preinstalado de Colab (torchao {out}) incompatible con LoRAs...", flush=True)
        subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"])


def ensure_dependencies() -> None:
    print("📦 [1/3] Verificando dependencias de Studio IA...", flush=True)
    if not shutil.which("ffmpeg") and Path("/usr/bin/apt-get").exists():
        subprocess.run(["apt-get", "update", "-qq"], check=False)
        subprocess.run(["apt-get", "install", "-y", "-qq", "ffmpeg"], check=False)

    req_file = ROOT_DIR / (
        "requirements-colab.txt"
        if (Path("/var/colab").exists() or os.environ.get("COLAB_RELEASE_TAG"))
        else "requirements.txt"
    )
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--upgrade-strategy",
            "only-if-needed",
            "-r",
            str(req_file),
        ]
    )
    _remove_incompatible_preinstalled_packages()
    print("✅ Dependencias listas.", flush=True)


def start_cloudflared_tunnel(port: int) -> tuple[subprocess.Popen | None, str | None]:
    if platform.system().lower() != "linux":
        return None, None

    bin_path = ROOT_DIR / "cloudflared"
    if not bin_path.exists():
        arch = platform.machine().lower()
        cf_arch = "arm64" if "aarch64" in arch or "arm64" in arch else "amd64"
        url = f"https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-{cf_arch}"
        try:
            print("🌐 Descargando conector ligero cloudflared (sin cuenta)...", flush=True)
            urllib.request.urlretrieve(url, str(bin_path))
            bin_path.chmod(0o755)
        except Exception as exc:
            print(f"⚠️ No se pudo descargar cloudflared ({exc}). Se usará el enlace proxy de Colab.", flush=True)
            return None, None

    proc = subprocess.Popen(
        [str(bin_path), "tunnel", "--url", f"http://127.0.0.1:{port}", "--no-autoupdate"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    public_url: list[str] = []

    def _reader() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            m = re.search(r"https://[a-zA-Z0-9\-]+\.trycloudflare\.com", line)
            if m and not public_url:
                public_url.append(m.group(0))

    t = threading.Thread(target=_reader, daemon=True)
    t.start()

    deadline = time.time() + 18
    while time.time() < deadline and not public_url:
        if proc.poll() is not None:
            break
        time.sleep(0.3)

    return proc, (public_url[0] if public_url else None)


def wait_for_local_server(port: int, timeout: float = 35.0) -> dict | None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status", timeout=2) as resp:
                if resp.status == 200:
                    return json.loads(resp.read().decode("utf-8"))
        except Exception:
            time.sleep(0.4)
    return None


def try_colab_proxy_url(port: int) -> str | None:
    try:
        from google.colab import output  # type: ignore

        proxy = output.eval_js(f"google.colab.kernel.proxyPort({port})")
        return str(proxy) if proxy else None
    except Exception:
        return None


def disconnect_colab_if_possible() -> None:
    try:
        from google.colab import runtime  # type: ignore

        print("🔌 Desconectando entorno de Google Colab y liberando GPU...", flush=True)
        runtime.unassign()
    except Exception:
        pass


def main() -> None:
    ensure_dependencies()

    data_dir = Path("/content/studio_data") if Path("/var/colab").exists() else (ROOT_DIR / "studio_data")
    shutdown_flag = data_dir / "state" / "shutdown_requested.flag"
    if shutdown_flag.exists():
        shutdown_flag.unlink()

    print("🚀 [2/3] Arrancando servidor Studio IA en proceso limpio...", flush=True)
    env = os.environ.copy()
    env["PORT"] = str(PORT)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # Capturamos stdout/stderr en un hilo lector hacia print(..., flush=True) para que
    # Google Colab (ipykernel OutStream) muestre todos los logs en tiempo real en la celda.
    server_proc = subprocess.Popen(
        [sys.executable, "-u", str(ROOT_DIR / "app.py")],
        cwd=str(ROOT_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    def _stream_server_logs() -> None:
        assert server_proc.stdout is not None
        for line in iter(server_proc.stdout.readline, ""):
            if line:
                print(line, end="", flush=True)

    log_thread = threading.Thread(target=_stream_server_logs, daemon=True)
    log_thread.start()

    initial_status = wait_for_local_server(PORT, timeout=35.0)
    if not initial_status:
        print("❌ El servidor tardó demasiado en responder.", flush=True)
        server_proc.terminate()
        sys.exit(1)

    print("🔗 [3/3] Abriendo enlace de acceso...", flush=True)
    cf_proc, cf_url = start_cloudflared_tunnel(PORT)
    colab_url = try_colab_proxy_url(PORT)

    hw = initial_status.get("hardware", {})
    gpu = hw.get("gpu", {})
    disk = hw.get("disk", {})
    cred = hw.get("credits", {})

    msg_lines = ["✨ STUDIO IA ACTIVO Y LISTO ✨"]
    if cf_url:
        msg_lines.append(f"👉 Abre este enlace (PC o móvil):  {cf_url}")
    if colab_url:
        msg_lines.append(f"🔹 Enlace alternativo Colab:       {colab_url}")
    if not cf_url and not colab_url:
        msg_lines.append(f"👉 Enlace local:                   http://127.0.0.1:{PORT}")
    msg_lines.append(
        f"🖥️  Hardware: {gpu.get('gpu_name', 'CPU')} · Disco libre: {disk.get('free_gb', '?')} GB · Tarifa: {cred.get('hourly_rate', 0)} u/h"
    )
    msg_lines.append("📡 Debajo verás EN TIEMPO REAL cada descarga (0-100%), carga de modelo y proceso que lances:")
    _print_box(msg_lines)

    try:
        while server_proc.poll() is None:
            if shutdown_flag.exists():
                print("\n🛑 Solicitud de «Liberar GPU» recibida desde la interfaz.", flush=True)
                shutdown_flag.unlink(missing_ok=True)
                server_proc.terminate()
                if cf_proc:
                    cf_proc.terminate()
                disconnect_colab_if_possible()
                return
            time.sleep(0.8)
    except KeyboardInterrupt:
        print("\n⏹️ Deteniendo Studio IA...", flush=True)
    finally:
        try:
            if server_proc.poll() is None:
                server_proc.terminate()
        except BaseException:
            pass
        try:
            if cf_proc and cf_proc.poll() is None:
                cf_proc.terminate()
        except BaseException:
            pass
        print("✅ Servidor Studio IA detenido limpiamente.", flush=True)


if __name__ == "__main__":
    main()
