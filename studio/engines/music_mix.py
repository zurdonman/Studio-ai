"""Simple, non-destructive multitrack audio bounce (no dialogue mastering)."""
import json
import math
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from studio.config import AUDIO_OUT_DIR
from studio.engines.ffmpeg_utils import get_ffmpeg_bin, probe_media_info


def mix_music_tracks(tracks: list[dict[str, Any]], job: Any) -> dict[str, Any]:
    if not 1 <= len(tracks) <= 16:
        raise ValueError("La mezcla requiere entre 1 y 16 pistas.")
    cmd = [get_ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-y"]
    filters = []
    for i, track in enumerate(tracks):
        job.check_cancelled()
        start, volume = float(track["start_sec"]), float(track["volume"])
        if not math.isfinite(start) or not 0 <= start <= 3600 or not math.isfinite(volume) or not 0 <= volume <= 2:
            raise ValueError("Inicio debe estar entre 0 y 3600 s y volumen entre 0 y 2.")
        info = probe_media_info(track["path"])
        if not info["has_audio"] or not 0 < info["duration_seconds"] <= 3600:
            raise ValueError("Cada pista debe tener audio y duración comprobable de hasta una hora.")
        trim_start = float(track.get("trim_start", 0))
        trim_end = track.get("trim_end")
        trim_end = float(trim_end) if trim_end is not None else info["duration_seconds"]
        if not math.isfinite(trim_start) or not math.isfinite(trim_end) or not 0 <= trim_start < trim_end or trim_end > info["duration_seconds"] + .05:
            raise ValueError("El recorte debe estar dentro de la duración de la pista y tener fin posterior al inicio.")
        if start + trim_end - trim_start > 600:
            raise ValueError("La mezcla no puede superar 600 segundos.")
        cmd += ["-i", str(track["path"])]
        delay = round(start * 1000)
        filters.append(f"[{i}:a]atrim=start={trim_start:.3f}:end={trim_end:.3f},aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,asetpts=PTS-STARTPTS,volume={volume:.6f},adelay={delay}|{delay}[a{i}]")
    filters.append("".join(f"[a{i}]" for i in range(len(tracks))) + f"amix=inputs={len(tracks)}:duration=longest:dropout_transition=0:normalize=0[mix]")
    AUDIO_OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = AUDIO_OUT_DIR / f"music_mix_{datetime.now():%Y%m%d_%H%M%S}_{job.id[:6]}.wav"
    cmd += ["-filter_complex", ";".join(filters), "-map", "[mix]", "-c:a", "pcm_s24le", str(out)]
    job.check_cancelled()
    job.log("Mezclando pistas en WAV; reduce volúmenes si la suma satura.", progress=20, step="Exportando mezcla")
    proc = None
    error_log = tempfile.TemporaryFile()
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=error_log)
        started = time.monotonic()
        while proc.poll() is None:
            job.check_cancelled()
            if time.monotonic() - started > 300:
                raise TimeoutError("La mezcla superó cinco minutos.")
            time.sleep(0.2)
        error_log.seek(0)
        err = error_log.read()[-1200:].decode("utf-8", "replace")
        job.check_cancelled()
        if proc.returncode:
            raise RuntimeError(f"FFmpeg no pudo mezclar las pistas: {err[-1200:]}")
        info = probe_media_info(out)
        rel = f"outputs/audio/{out.name}"
        meta = {"id": out.stem, "filename": out.name, "rel_path": rel, "media_url": f"/media/{rel}",
                "type": "audio", "duration_seconds": info["duration_seconds"], "created_at": datetime.now().isoformat(timespec="seconds"),
                "engine": "music_multitrack_ffmpeg"}
        (AUDIO_OUT_DIR / f"{out.stem}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        job.log("Mezcla terminada.", progress=100, step="Audio completado")
        return meta
    except BaseException:
        if proc and proc.poll() is None:
            proc.kill()
            proc.communicate()
        out.unlink(missing_ok=True)
        raise
    finally:
        error_log.close()
