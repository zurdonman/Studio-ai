"""Safe, temporary reference-audio preparation for voice-cloning engines.

Loading and inference are handled separately by the engine runtime.
"""
from __future__ import annotations

import math
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from studio.config import TEMP_DIR
from studio.engines.ffmpeg_utils import get_ffmpeg_bin, probe_media_info

# Limits are Studio's conservative input policy, not upstream model limits.
MAX_REFERENCE_BYTES = 50 * 1024 * 1024
MIN_REFERENCE_SECONDS = 3.0
MAX_REFERENCE_SECONDS = 15.0
MAX_REFERENCE_TEXT_CHARS = 2000
CLONE_ENGINES = frozenset({"qwen3-tts-0.6b-base", "fish-speech-1.5"})


def validate_clone_request(engine_id: str, text: str, reference_text: Optional[str]) -> str:
    """Validate inputs before any heavyweight download or model allocation."""
    if engine_id not in CLONE_ENGINES:
        raise ValueError("Motor de clonación desconocido.")
    if not (text or "").strip():
        raise ValueError("Escribe el texto que quieres convertir en voz.")
    transcript = (reference_text or "").strip()
    if len(transcript) > MAX_REFERENCE_TEXT_CHARS:
        raise ValueError("La transcripción de la muestra es demasiado larga.")
    if engine_id == "fish-speech-1.5" and not transcript:
        raise ValueError("Fish Speech necesita la transcripción exacta de la muestra de voz.")
    return transcript


@contextmanager
def prepared_reference_audio(path: Path) -> Iterator[Path]:
    """Validate and convert a reference to a private, temporary mono PCM WAV.

    The caller must supply a path already resolved and authorized by the API;
    this function does not accept URLs or enforce the media root. The output is
    removed on success, failure, or cancellation.
    """
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError("Selecciona un archivo de muestra de voz válido.")
    if path.stat().st_size > MAX_REFERENCE_BYTES:
        raise ValueError("La muestra de voz supera el límite de 50 MiB.")
    info = probe_media_info(path)
    duration = info.get("duration_seconds")
    if not info.get("has_audio") or info.get("has_video") or not isinstance(duration, (int, float)) or not math.isfinite(duration):
        raise ValueError("La muestra debe contener audio con duración verificable.")
    if not MIN_REFERENCE_SECONDS <= duration <= MAX_REFERENCE_SECONDS:
        raise ValueError("La muestra debe durar entre 3 y 15 segundos; se recomiendan unos 5 segundos.")

    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="voice-ref-", dir=str(TEMP_DIR)) as temp:
        output = Path(temp) / "reference.wav"
        result = subprocess.run(
            [get_ffmpeg_bin(), "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(path),
             "-map", "0:a:0", "-vn", "-map_metadata", "-1", "-t", "15.1", "-ac", "1", "-ar", "24000",
             "-c:a", "pcm_s16le", "-y", str(output)],
            capture_output=True, text=True, timeout=60, check=False,
        )
        if result.returncode != 0 or not output.is_file() or output.stat().st_size <= 44:
            raise ValueError("No se pudo decodificar la muestra de voz como audio PCM.")
        yield output
