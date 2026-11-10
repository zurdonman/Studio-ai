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
import secrets
import shutil
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
from pathlib import Path

from studio.diagnostics import DiagnosticSession, data_root, describe_exit, export_diagnostic_bundle, memory_snapshot, redact_secrets

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
    if req_file.name == "requirements.txt":
        has_ort = subprocess.run([sys.executable, "-c", "import onnxruntime"], capture_output=True).returncode == 0
        if not has_ort:
            subprocess.call([sys.executable, "-m", "pip", "install", "-q", "onnxruntime>=1.18.0"])
    _remove_incompatible_preinstalled_packages()
    print("✅ Dependencias listas.", flush=True)


def start_cloudflared_tunnel(port: int, diagnostics: DiagnosticSession | None = None) -> tuple[subprocess.Popen | None, str | None]:
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
        errors="replace",
    )

    public_url: list[str] = []

    def _reader() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            if diagnostics:
                diagnostics.write("tunnel.log", line)
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


def _stop_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=3)
    except ProcessLookupError:
        pass


def main() -> None:
    diagnostics = DiagnosticSession()
    print(f"📄 Diagnóstico persistente de esta ejecución: {diagnostics.folder}", flush=True)
    try:
        ensure_dependencies()
    except Exception:
        diagnostics.write("server.log", traceback.format_exc())
        diagnostics.event("dependency_setup_failed", status="failed")
        raise

    data_dir = data_root()
    shutdown_flag = data_dir / "state" / "shutdown_requested.flag"
    if shutdown_flag.exists():
        shutdown_flag.unlink()

    print("🚀 [2/3] Arrancando servidor Studio IA en proceso limpio...", flush=True)
    # Token NUEVO en cada arranque: si uno se filtra (captura, chat, log), muere al reiniciar.
    # Para fijarlo, exporta STUDIO_MCP_TOKEN antes de lanzar.
    mcp_token = os.environ.get("STUDIO_MCP_TOKEN", "").strip() or f"st_mcp_{secrets.token_hex(12)}"

    # Token de GitHub para el canal de control (solo se usa para leer/escribir 2 ficheros
    # del propio repo). Se acepta por entorno o por fichero para no dejarlo en el historial.
    channel_token_file = data_dir / "state" / "github_channel_token.txt"
    gh_token = (os.environ.get("STUDIO_CHANNEL_GH_TOKEN") or "").strip()
    if not gh_token and channel_token_file.exists():
        try:
            gh_token = channel_token_file.read_text(encoding="utf-8").strip()
        except OSError:
            gh_token = ""
    if gh_token:
        try:
            channel_token_file.parent.mkdir(parents=True, exist_ok=True)
            channel_token_file.write_text(gh_token, encoding="utf-8")
            os.chmod(channel_token_file, 0o600)
        except OSError:
            pass
    env = os.environ.copy()
    env["PORT"] = str(PORT)
    env["PYTHONUNBUFFERED"] = "1"
    env["STUDIO_MCP_TOKEN"] = mcp_token
    if gh_token:
        env["STUDIO_CHANNEL_GH_TOKEN"] = gh_token
    env["PYTHONFAULTHANDLER"] = "1"
    env["STUDIO_DIAGNOSTIC_DIR"] = str(diagnostics.folder)
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
        errors="replace",
        bufsize=1,
    )
    diagnostics.event("server_started", status="starting", server_pid=server_proc.pid)

    def _stream_server_logs() -> None:
        assert server_proc.stdout is not None
        for line in iter(server_proc.stdout.readline, ""):
            if line:
                diagnostics.write("server.log", line)
                print(redact_secrets(line), end="", flush=True)

    log_thread = threading.Thread(target=_stream_server_logs, daemon=True)
    log_thread.start()

    initial_status = wait_for_local_server(PORT, timeout=35.0)
    if not initial_status:
        print("❌ El servidor tardó demasiado en responder.", flush=True)
        _stop_process(server_proc)
        log_thread.join(timeout=2)
        diagnostics.event("startup_failed", status="failed", exit=describe_exit(server_proc.returncode, diagnostics.before, memory_snapshot()))
        print(f"📄 Consulta el diagnóstico local: {diagnostics.folder}", flush=True)
        raise RuntimeError("Studio no pudo arrancar; ver server.log. Los pesos no se han borrado.")

    print("🔗 [3/3] Abriendo enlace de acceso y conector MCP...", flush=True)
    cf_proc, cf_url = start_cloudflared_tunnel(PORT, diagnostics=diagnostics)
    colab_url = try_colab_proxy_url(PORT)

    if cf_url:
        try:
            reg_req = urllib.request.Request(
                f"http://127.0.0.1:{PORT}/api/mcp/tunnel-info",
                data=json.dumps({"public_url": cf_url}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(reg_req, timeout=4).close()
        except Exception:
            pass

    # Canal de control por GitHub (para agentes que no alcanzan el túnel, como arena.ai).
    # NO se lanza aquí: se enciende desde el panel MCP con un botón, para que el estado
    # viva en el proceso del servidor y la interfaz pueda mostrarlo y pararlo.
    gh_token = (os.environ.get("STUDIO_CHANNEL_GH_TOKEN") or "").strip()
    if not gh_token:
        for candidate in (data_dir / "state" / "github_channel.json",
                          data_dir / "state" / "github_channel_token.txt"):
            try:
                if not candidate.exists():
                    continue
                raw = candidate.read_text(encoding="utf-8").strip()
                if candidate.suffix == ".json":
                    import json as _json
                    gh_token = str((_json.loads(raw) or {}).get("gh_token", "")).strip()
                else:
                    gh_token = raw
                if gh_token:
                    break
            except (OSError, ValueError):
                continue
    if gh_token:
        env["STUDIO_CHANNEL_GH_TOKEN"] = gh_token
    msg_channel = (
        "🔗 CANAL PARA AGENTE arena.ai: listo · enciéndelo en el panel MCP (botón «Encender canal»)"
        if gh_token else
        "🔗 CANAL PARA AGENTE arena.ai: pega el token de GitHub en el panel MCP y pulsa «Encender»"
    )

    hw = initial_status.get("hardware", {})
    gpu = hw.get("gpu", {})
    disk = hw.get("disk", {})
    cred = hw.get("credits", {})

    base_ext = cf_url or f"http://127.0.0.1:{PORT}"
    msg_lines = ["✨ STUDIO IA ACTIVO Y LISTO (WEB + SERVIDOR MCP) ✨"]
    if cf_url:
        msg_lines.append(f"👉 Interfaz Web (PC o móvil):      {cf_url}")
    if colab_url:
        msg_lines.append(f"🔹 Enlace alternativo Colab:       {colab_url}")
    if not cf_url and not colab_url:
        msg_lines.append(f"👉 Enlace local:                   http://127.0.0.1:{PORT}")
    msg_lines.append(
        f"🖥️  Hardware: {gpu.get('gpu_name', 'CPU')} · Disco libre: {disk.get('free_gb', '?')} GB · Tarifa: {cred.get('hourly_rate', 0)} u/h"
    )
    msg_lines.append("─" * 74)
    msg_lines.append("🤖 CONEXIÓN PARA TU AGENTE (Termux Android / DeepSeek / Qwen / Claude / Codex):")
    msg_lines.append("   📱 [OPCIÓN MÁS FÁCIL - TERMUX / DEEPSEEK / QWEN] Copia este enlace único con token")
    msg_lines.append("      y dáselo a tu agente en el chat («Lee con curl este enlace y conéctate»):")
    msg_lines.append(f"      👉 {base_ext}/mcp/guia?token={mcp_token}")
    msg_lines.append(f"   🔗 Endpoint MCP HTTP directo (con token en URL): {base_ext}/mcp?token={mcp_token}")
    msg_lines.append(f"   🔑 Token suelto (si tu cliente lo pide aparte):   {mcp_token}")
    msg_lines.append(
        "   🔒 Este token cambia en cada arranque y NO se publica en /api/mcp/info: si se filtra, "
        "reinicia la instancia y quedará invalidado."
    )
    msg_lines.append("   💻 Si usas Claude Code en PC (Mac/Linux/WSL):")
    msg_lines.append(
        f'     curl -fsSL "{base_ext}/mcp/bridge.py?token={mcp_token}" -o studio_mcp.py && (claude mcp remove studio-ia >/dev/null 2>&1 || true) && claude mcp add studio-ia -- python3 "$(pwd)/studio_mcp.py"'
    )
    msg_lines.append("─" * 74)
    msg_lines.append(msg_channel)
    msg_lines.append(
        "   El agente escribe órdenes en channel/ordenes.json del repo y lee state/channel/resultados.json."
    )
    if gh_token:
        msg_lines.append("   🔑 Token de GitHub ya guardado en studio_data/state (nunca en el repo).")
    else:
        msg_lines.append("   Solo necesitas un PAT de GitHub con permiso 'gist' y pulsar el botón en el panel MCP.")
    msg_lines.append("─" * 74)
    msg_lines.append("📡 Debajo verás EN TIEMPO REAL todo lo que hagas tú en la Web o tu Agente vía MCP:")
    _print_box(msg_lines)

    diagnostics.event("ready", status="running", public_url=cf_url, tunnel_pid=cf_proc.pid if cf_proc else None)
    requested_stop = False
    tunnel_exit_reported = False
    last_sample = 0.0
    try:
        while server_proc.poll() is None:
            if time.monotonic() - last_sample >= 3:
                diagnostics.sample(server_proc.pid, cf_proc.pid if cf_proc else None)
                last_sample = time.monotonic()
            if cf_proc and cf_proc.poll() is not None and not tunnel_exit_reported:
                tunnel_exit_reported = True
                diagnostics.event("tunnel_exited", tunnel_returncode=cf_proc.returncode)
                print("⚠️ El túnel terminó, pero el servidor local sigue activo. No repitas la generación; conserva el job_id y consulta el diagnóstico de Colab.", flush=True)
            if shutdown_flag.exists():
                requested_stop = True
                print("\n🛑 Solicitud de «Liberar GPU» recibida desde la interfaz.", flush=True)
                shutdown_flag.unlink(missing_ok=True)
                _stop_process(server_proc)
                if cf_proc:
                    _stop_process(cf_proc)
                log_thread.join(timeout=2)
                diagnostics.event("user_shutdown", status="stopped")
                disconnect_colab_if_possible()
                return
            time.sleep(0.8)
    except KeyboardInterrupt:
        requested_stop = True
        print("\n⏹️ Deteniendo Studio IA por solicitud del usuario...", flush=True)
    finally:
        already_exited = server_proc.poll() is not None
        _stop_process(server_proc)
        if cf_proc:
            _stop_process(cf_proc)
        log_thread.join(timeout=2)
        exit_info = describe_exit(server_proc.returncode, diagnostics.before, memory_snapshot())
        unexpected = already_exited and not requested_stop
        diagnostics.event("server_exited", status="unexpected_exit" if unexpected else "stopped", exit=exit_info)
        if unexpected:
            print(f"❌ Studio IA terminó: {exit_info['explanation']}", flush=True)
            print("El lanzador cierra entonces su túnel; un 530 posterior no confirma por sí solo que se haya perdido la VM.", flush=True)
            try:
                bundle = export_diagnostic_bundle(data_dir)
                print(f"📦 Diagnóstico guardado sin pesos ni tokens: {bundle}", flush=True)
            except Exception as exc:
                print(f"⚠️ No se pudo crear el ZIP: {exc}. Los logs están en {diagnostics.folder}", flush=True)
        else:
            print("✅ Servidor Studio IA detenido por solicitud del usuario; archivos en disco intactos.", flush=True)


if __name__ == "__main__":
    main()
