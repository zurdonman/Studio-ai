#!/usr/bin/env python3
"""
Lanzador todo-en-uno para Google Colab:
1. Instala dependencias en limpio sin importar diffusers/bitsandbytes antes de tiempo.
2. Arranca app.py en un PROCESO PYTHON NUEVO (evita el bug de @lru_cache en bitsandbytes).
3. Abre enlace directo https://... con cloudflared efímero (sin cuentas ni claves) + proxy nativo de Colab.
4. Escucha el botón «Liberar GPU» de la interfaz para desconectar la máquina de Colab al instante.
"""

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
    print("\n" + "═" * width)
    for ln in lines:
        print(f"  {ln}")
    print("═" * width + "\n", flush=True)


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
        [sys.executable, "-m", "pip", "install", "-q", "--upgrade", "-r", str(req_file)]
    )
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


def wait_for_local_server(port: int, timeout: float = 30.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status", timeout=2) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.4)
    return False


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

    # Limpiar bandera previa de apagado si existía
    data_dir = Path("/content/studio_data") if Path("/var/colab").exists() else (ROOT_DIR / "studio_data")
    shutdown_flag = data_dir / "state" / "shutdown_requested.flag"
    if shutdown_flag.exists():
        shutdown_flag.unlink()

    print("🚀 [2/3] Arrancando servidor Studio IA en proceso limpio...", flush=True)
    env = os.environ.copy()
    env["PORT"] = str(PORT)
    server_proc = subprocess.Popen(
        [sys.executable, str(ROOT_DIR / "app.py")],
        cwd=str(ROOT_DIR),
        env=env,
    )

    if not wait_for_local_server(PORT, timeout=35.0):
        print("❌ El servidor tardó demasiado en responder.", flush=True)
        server_proc.terminate()
        sys.exit(1)

    print("🔗 [3/3] Abriendo enlace de acceso...", flush=True)
    cf_proc, cf_url = start_cloudflared_tunnel(PORT)
    colab_url = try_colab_proxy_url(PORT)

    msg_lines = ["✨ STUDIO IA ACTIVO Y LISTO ✨"]
    if cf_url:
        msg_lines.append(f"👉 Abre este enlace (PC o móvil):  {cf_url}")
    if colab_url:
        msg_lines.append(f"🔹 Enlace alternativo Colab:       {colab_url}")
    if not cf_url and not colab_url:
        msg_lines.append(f"👉 Enlace local:                   http://127.0.0.1:{PORT}")
    msg_lines.append("💡 Ningún modelo se descarga solo. Elige qué bajar dentro de la app.")
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
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n⏹️ Deteniendo Studio IA...", flush=True)
    finally:
        if server_proc.poll() is None:
            server_proc.terminate()
        if cf_proc and cf_proc.poll() is None:
            cf_proc.terminate()


if __name__ == "__main__":
    main()
