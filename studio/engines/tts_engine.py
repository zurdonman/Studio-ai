import asyncio
import json
import time
import wave
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from studio.catalog import catalog_manager
from studio.config import AUDIO_OUT_DIR
from studio.engines.ffmpeg_utils import probe_media_info
from studio.jobs import JobRecord
from studio.memory_manager import (
    flush_ram_and_vram,
    memory_manager,
    run_with_live_memory_progress,
)

EDGE_TTS_VOICES: List[Dict[str, str]] = [
    {"id": "es-ES-AlvaroNeural", "label": "Álvaro (España · Masculina)", "lang": "es-ES"},
    {"id": "es-ES-ElviraNeural", "label": "Elvira (España · Femenina)", "lang": "es-ES"},
    {"id": "es-ES-XimenaNeural", "label": "Ximena (España · Femenina cálida)", "lang": "es-ES"},
    {"id": "es-MX-JorgeNeural", "label": "Jorge (México · Masculina)", "lang": "es-MX"},
    {"id": "es-MX-DaliaNeural", "label": "Dalia (México · Femenina)", "lang": "es-MX"},
    {"id": "es-AR-TomasNeural", "label": "Tomás (Argentina · Masculina)", "lang": "es-AR"},
    {"id": "es-AR-ElenaNeural", "label": "Elena (Argentina · Femenina)", "lang": "es-AR"},
    {"id": "en-US-AndrewMultilingualNeural", "label": "Andrew Multilingual (EN/ES · Masculina)", "lang": "en-US"},
    {"id": "en-US-AvaMultilingualNeural", "label": "Ava Multilingual (EN/ES · Femenina)", "lang": "en-US"},
]


def _write_pcm16_wav(samples: Any, sample_rate: int, out_path: Path) -> None:
    import numpy as np  # type: ignore

    arr = np.squeeze(np.asarray(samples, dtype=np.float32))
    pcm16 = (np.clip(arr, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm16.tobytes())


def load_tts_model_into_memory(
    model_id: str,
    auto_download: bool = False,
    job: Optional[JobRecord] = None,
) -> Dict[str, Any]:
    """Carga un modelo TTS local de Hugging Face (MMS-TTS Español, Bark Small o Kokoro-82M) en RAM/VRAM."""
    defn = catalog_manager.get_definition(model_id)
    if not defn:
        raise ValueError(f"Modelo TTS desconocido: {model_id}")

    if defn.get("builtin_no_download"):
        if job:
            job.log("Edge-TTS está listo al instante sin ocupar RAM ni VRAM.", progress=100.0, step="Listo")
        return {"model_id": model_id, "vram_allocated_gb": 0.0}

    cur_tts = memory_manager.get_slot("tts")
    if cur_tts and cur_tts.get("model_id") == model_id:
        if job:
            job.log(f"«{defn['name']}» ya está cargado en memoria.", level="success", progress=100.0, step="Listo")
        return {"model_id": model_id}

    if not catalog_manager.is_downloaded(model_id):
        if auto_download and job is not None:
            catalog_manager.download_model(model_id, job, progress_range=(1.0, 65.0))
            p_start = 65.0
        else:
            raise RuntimeError(f"«{defn['name']}» no está descargado todavía en disco.")
    else:
        p_start = 8.0

    if cur_tts:
        memory_manager.unload_slot("tts")

    art = defn["artifacts"][0]
    base_dir = catalog_manager.model_dir(model_id) / art["subdir"]

    if defn.get("engine") == "kokoro_tts":
        try:
            from kokoro import KPipeline  # type: ignore

            if job:
                job.log("Inicializando Kokoro-82M...", progress=p_start, step="Cargando Kokoro-82M...")
            kpipe = run_with_live_memory_progress(
                job=job,
                start_pct=p_start,
                end_pct=96.0,
                step_label="Cargando Kokoro-82M",
                fn=lambda: KPipeline(lang_code="e"),
                expected_seconds=10.0,
            )
            memory_manager.register_loaded(
                slot="tts",
                model_id=model_id,
                name=defn["name"],
                handle={"type": "kokoro", "pipe": kpipe},
            )
            return {"model_id": model_id}
        except ImportError:
            pass

    try:
        import torch  # type: ignore
        from transformers import pipeline  # type: ignore

        device = 0 if torch.cuda.is_available() else -1
        if job:
            job.log(f"Cargando pipeline local TTS «{defn['name']}»...", progress=p_start, step="Cargando modelo TTS...")
        tts_pipe = run_with_live_memory_progress(
            job=job,
            start_pct=p_start,
            end_pct=96.0,
            step_label=f"Cargando {defn['name']}",
            fn=lambda: pipeline("text-to-speech", model=str(base_dir), device=device),
            expected_seconds=12.0,
        )
        memory_manager.register_loaded(
            slot="tts",
            model_id=model_id,
            name=defn["name"],
            handle={"type": "transformers_tts", "pipe": tts_pipe},
        )
        return {"model_id": model_id}
    except Exception as exc:
        raise RuntimeError(f"No se pudo inicializar el modelo local TTS ({exc}).") from exc


def synthesize_voice(
    text: str,
    engine_id: str = "edge-tts-cloud",
    voice_id: str = "es-ES-AlvaroNeural",
    rate_pct: int = 0,
    pitch_hz: int = 0,
    job: Optional[JobRecord] = None,
) -> Dict[str, Any]:
    """Genera un archivo de voz verificado (.mp3 o .wav) usando Edge-TTS o modelo HF local."""
    clean_text = (text or "").strip()
    if not clean_text:
        raise ValueError("Escribe el texto que quieres convertir en voz.")

    t0 = time.time()
    AUDIO_OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts_slug = datetime.now().strftime("%Y%m%d_%H%M%S")

    if engine_id == "edge-tts-cloud":
        try:
            import edge_tts  # type: ignore
        except ImportError as exc:
            raise RuntimeError(f"edge-tts no está instalado: {exc}") from exc

        rate_str = f"{rate_pct:+d}%"
        pitch_str = f"{pitch_hz:+d}Hz"
        out_name = f"voz_{ts_slug}.mp3"
        out_path = AUDIO_OUT_DIR / out_name

        if job:
            job.log(
                f"Generando voz con Edge-TTS ({voice_id}, velocidad {rate_str}, tono {pitch_str})...",
                progress=30.0,
                step=f"Sintetizando con {voice_id}...",
            )

        async def _run_edge() -> None:
            comm = edge_tts.Communicate(clean_text, voice=voice_id, rate=rate_str, pitch=pitch_str)
            await comm.save(str(out_path))

        try:
            asyncio.run(_run_edge())
        except Exception as exc:
            raise RuntimeError(
                f"No se pudo conectar al servicio Edge-TTS ({exc}). "
                "En Google Colab funciona directamente con conexión a internet."
            ) from exc

        info = probe_media_info(out_path)
        meta = {
            "id": out_path.stem,
            "filename": out_name,
            "media_url": f"/media/outputs/audio/{out_name}",
            "type": "audio",
            "model_id": "edge-tts-cloud",
            "model_name": f"Edge-TTS ({voice_id})",
            "voice_id": voice_id,
            "text": clean_text,
            "duration_seconds": info["duration_seconds"],
            "elapsed_seconds": round(time.time() - t0, 2),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        (AUDIO_OUT_DIR / f"{out_path.stem}.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return meta

    slot = memory_manager.get_slot("tts")
    if not slot or slot["model_id"] != engine_id:
        if job:
            job.log("Cargando modelo TTS local antes de sintetizar...", progress=5.0, step="Preparando TTS local...")
        load_tts_model_into_memory(engine_id, auto_download=True, job=job)
        slot = memory_manager.get_slot("tts")
        if not slot:
            raise RuntimeError("No se pudo cargar el modelo TTS local seleccionado.")

    out_name = f"voz_local_{ts_slug}.wav"
    out_path = AUDIO_OUT_DIR / out_name
    handle = slot["handle"]

    if job:
        job.log(f"Sintetizando audio con «{slot['name']}»...", progress=70.0, step="Generando forma de onda...")

    if handle["type"] == "kokoro":
        import numpy as np  # type: ignore

        chunks = []
        for _, _, audio in handle["pipe"](clean_text, voice="ef_dora", speed=1.0 + (rate_pct / 100.0)):
            chunks.append(audio)
        if not chunks:
            raise RuntimeError("Kokoro no devolvió muestras de audio.")
        full_audio = np.concatenate(chunks)
        _write_pcm16_wav(full_audio, 24000, out_path)
    else:
        res = handle["pipe"](clean_text)
        _write_pcm16_wav(res["audio"], int(res["sampling_rate"]), out_path)

    flush_ram_and_vram()
    info = probe_media_info(out_path)
    meta = {
        "id": out_path.stem,
        "filename": out_name,
        "media_url": f"/media/outputs/audio/{out_name}",
        "type": "audio",
        "model_id": slot["model_id"],
        "model_name": slot["name"],
        "voice_id": voice_id,
        "text": clean_text,
        "duration_seconds": info["duration_seconds"],
        "elapsed_seconds": round(time.time() - t0, 2),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    (AUDIO_OUT_DIR / f"{out_path.stem}.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return meta
