"""Estudio de Sonido IA y Mezclador Multipista FFmpeg (Studio IA).

Soluciona de raíz la producción sonora en escenas con diálogo, ambiente y música:
1. Generación de Música Instrumental IA (`facebook/musicgen-small` en Transformers).
2. Generación de Efectos Foley y Paisajes Sonoros IA (`cvssp/audioldm2` en Diffusers).
3. Sintetizador Acústico Estéreo DSP 0 GB (`procedural-ambience-cpu`): ambiente cálido de restaurante/cafetería,
   lluvia suave, brisa exterior, pad armónico cinematográfico y room-tone limpio en 0.2 s sin VRAM.
4. Mezclador Multipista Profesional FFmpeg (`mix_and_master_media`):
   - Posiciona múltiples voces de diálogo al milisegundo (`start_sec` con `adelay`) sin perder -6 dB (`normalize=0`).
   - Aplica **Auto-Ducking dinámico (`sidechaincompress`)** sobre la música y el sonido de fondo cada vez que habla
     un personaje, recuperando el ambiente suavemente en las pausas.
   - Si las voces duran más que el vídeo, extiende el vídeo con fluidez (`setpts` suave o bucle ping-pong continuo)
     para que ningún personaje se quede congelado como estatua.
"""

from __future__ import annotations

import gc
import json
import subprocess
import tempfile
import threading
import wave
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from studio.catalog import CatalogManager, catalog_manager
from studio.config import AUDIO_OUT_DIR, VIDEOS_OUT_DIR
from studio.engines.ffmpeg_utils import get_ffmpeg_bin, probe_media_info
from studio.jobs import JobRecord
from studio.memory_manager import memory_manager


AMBIENCE_PRESETS: Dict[str, str] = {
    "restaurant_warm_murmur": "Ambiente cálido de restaurante/cafetería (murmullo lejano suave y micro-acústica de sala)",
    "rain_cozy": "Lluvia suave detrás del cristal / calle húmeda en estéreo",
    "cinematic_warm_pad": "Pad armónico cinematográfico cálido y envolvente",
    "wind_nature": "Brisa natural exterior / aire abierto suave",
    "room_tone_clean": "Room-tone cálido de sala interior (evita silencios muertos entre frases)",
}


def infer_ambience_preset(prompt: str, preset: str = "auto") -> str:
    p = (preset or "").strip().lower()
    if p in AMBIENCE_PRESETS:
        return p
    low = (prompt or "").lower()
    if any(k in low for k in ("restaur", "cafe", "café", "bar", "cena", "comiendo", "comida", "charla", "murmur", "chatter", "diner", "table")):
        return "restaurant_warm_murmur"
    if any(k in low for k in ("lluvia", "rain", "tormenta", "storm", "agua", "water")):
        return "rain_cozy"
    if any(k in low for k in ("viento", "wind", "bosque", "forest", "calle", "exterior", "nature", "parque", "mar", "playa")):
        return "wind_nature"
    if any(k in low for k in ("music", "música", "piano", "jazz", "pad", "cine", "ambient", "emotiv", "cuerdas", "chord", "lofi")):
        return "cinematic_warm_pad"
    return "room_tone_clean"


def _one_pole_lowpass(x: np.ndarray, cutoff_hz: float, sr: int) -> np.ndarray:
    """Filtro paso-bajo IIR vectorizado por bloques rápidos sin requerir scipy."""
    rc = 1.0 / (2.0 * np.pi * max(20.0, cutoff_hz))
    dt = 1.0 / float(sr)
    alpha = float(dt / (rc + dt))
    # Aproximación FIR exponencial truncada rápida por convolución FFT
    n_ir = min(len(x), max(64, int(4.5 / max(1e-4, alpha))))
    idx = np.arange(n_ir, dtype=np.float32)
    h = alpha * ((1.0 - alpha) ** idx)
    h /= max(1e-8, float(h.sum()))
    n_fft = 1 << (len(x) + n_ir - 1).bit_length()
    y = np.fft.irfft(np.fft.rfft(x, n=n_fft) * np.fft.rfft(h, n=n_fft), n=n_fft)[: len(x)]
    return y.astype(np.float32)


def _bandpass_fft(x: np.ndarray, low_hz: float, high_hz: float, sr: int) -> np.ndarray:
    n = len(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / float(sr))
    spec = np.fft.rfft(x)
    # Ventana suave tipo Butterworth de 2º orden en frecuencia
    hp = 1.0 / np.sqrt(1.0 + (max(10.0, low_hz) / np.maximum(freqs, 1e-3)) ** 4)
    lp = 1.0 / np.sqrt(1.0 + (freqs / max(low_hz + 20.0, high_hz)) ** 4)
    y = np.fft.irfft(spec * (hp * lp), n=n)
    return y.astype(np.float32)


def synthesize_procedural_ambience_wav(
    output_wav: Path,
    preset: str = "restaurant_warm_murmur",
    duration_seconds: float = 10.0,
    seed: int = 42,
    sample_rate: int = 48000,
) -> Dict[str, Any]:
    """Sintetiza paisajes acústicos estéreo limpios y orgánicos en CPU (0 GB VRAM)."""
    dur = max(1.5, min(90.0, float(duration_seconds)))
    n_samples = int(round(dur * sample_rate))
    rng = np.random.default_rng(int(seed) if seed >= 0 else 42)
    t = np.linspace(0.0, dur, n_samples, endpoint=False, dtype=np.float32)

    preset_id = preset if preset in AMBIENCE_PRESETS else "restaurant_warm_murmur"

    if preset_id == "restaurant_warm_murmur":
        # Dos canales estéreo descorrelacionados con bandas formánticas cálidas (240-1350 Hz) + aire de sala
        layers_l = np.zeros(n_samples, dtype=np.float32)
        layers_r = np.zeros(n_samples, dtype=np.float32)
        for voice_idx, (f_lo, f_hi, mod_hz) in enumerate(
            [(190.0, 580.0, 2.3), (320.0, 980.0, 3.1), (450.0, 1450.0, 2.7)]
        ):
            nl = rng.standard_normal(n_samples).astype(np.float32)
            nr = rng.standard_normal(n_samples).astype(np.float32)
            bl = _bandpass_fft(nl, f_lo, f_hi, sample_rate)
            br = _bandpass_fft(nr, f_lo, f_hi, sample_rate)
            env_l = 0.55 + 0.45 * np.sin(2.0 * np.pi * mod_hz * t + voice_idx * 1.3) * np.sin(2.0 * np.pi * 0.37 * t + voice_idx)
            env_r = 0.55 + 0.45 * np.sin(2.0 * np.pi * (mod_hz * 0.93) * t + voice_idx * 2.1) * np.sin(2.0 * np.pi * 0.41 * t + 0.8)
            layers_l += bl * env_l.astype(np.float32) * 0.35
            layers_r += br * env_r.astype(np.float32) * 0.35

        # Micro-toques acústicos muy suaves de vajilla/copas lejanas
        n_clinks = max(2, int(dur * 1.4))
        for _ in range(n_clinks):
            pos = int(rng.integers(int(sample_rate * 0.2), max(int(sample_rate * 0.3), n_samples - int(sample_rate * 0.25))))
            clink_len = int(sample_rate * 0.14)
            ct = np.linspace(0.0, 0.14, clink_len, endpoint=False, dtype=np.float32)
            freq = float(rng.uniform(2100.0, 3900.0))
            tone = (
                np.sin(2.0 * np.pi * freq * ct) * np.exp(-ct * 34.0)
                + 0.45 * np.sin(2.0 * np.pi * (freq * 1.52) * ct) * np.exp(-ct * 48.0)
            ).astype(np.float32) * float(rng.uniform(0.018, 0.042))
            pan = float(rng.uniform(0.25, 0.75))
            end_p = min(n_samples, pos + clink_len)
            seg_n = end_p - pos
            layers_l[pos:end_p] += tone[:seg_n] * pan
            layers_r[pos:end_p] += tone[:seg_n] * (1.0 - pan)

        room_l = _one_pole_lowpass(rng.standard_normal(n_samples).astype(np.float32), 420.0, sample_rate) * 0.12
        room_r = _one_pole_lowpass(rng.standard_normal(n_samples).astype(np.float32), 420.0, sample_rate) * 0.12
        left = layers_l + room_l
        right = layers_r + room_r
        target_peak = 0.34

    elif preset_id == "rain_cozy":
        nl = rng.standard_normal(n_samples).astype(np.float32)
        nr = rng.standard_normal(n_samples).astype(np.float32)
        left = _bandpass_fft(nl, 300.0, 4800.0, sample_rate) * 0.6 + _one_pole_lowpass(nl, 500.0, sample_rate) * 0.4
        right = _bandpass_fft(nr, 300.0, 4800.0, sample_rate) * 0.6 + _one_pole_lowpass(nr, 500.0, sample_rate) * 0.4
        swell = (0.82 + 0.18 * np.sin(2.0 * np.pi * 0.14 * t)).astype(np.float32)
        left *= swell
        right *= swell
        target_peak = 0.32

    elif preset_id == "cinematic_warm_pad":
        # Acorde cálido Am9 / Cmaj7 con ligero desafinado analógico estéreo
        freqs_hz = [110.0, 164.81, 220.0, 261.63, 329.63, 392.00]
        left = np.zeros(n_samples, dtype=np.float32)
        right = np.zeros(n_samples, dtype=np.float32)
        for i, f0 in enumerate(freqs_hz):
            amp = 0.38 / (1.0 + i * 0.28)
            lfo_l = 0.78 + 0.22 * np.sin(2.0 * np.pi * (0.11 + 0.03 * i) * t + i)
            lfo_r = 0.78 + 0.22 * np.sin(2.0 * np.pi * (0.12 + 0.025 * i) * t + i + 1.1)
            left += (amp * np.sin(2.0 * np.pi * (f0 * 0.9985) * t) * lfo_l).astype(np.float32)
            right += (amp * np.sin(2.0 * np.pi * (f0 * 1.0015) * t) * lfo_r).astype(np.float32)
        left = _one_pole_lowpass(left, 1600.0, sample_rate)
        right = _one_pole_lowpass(right, 1600.0, sample_rate)
        target_peak = 0.42

    elif preset_id == "wind_nature":
        nl = rng.standard_normal(n_samples).astype(np.float32)
        nr = rng.standard_normal(n_samples).astype(np.float32)
        swell_l = (0.55 + 0.45 * np.sin(2.0 * np.pi * 0.19 * t)).astype(np.float32)
        swell_r = (0.55 + 0.45 * np.sin(2.0 * np.pi * 0.17 * t + 1.2)).astype(np.float32)
        left = _bandpass_fft(nl, 120.0, 920.0, sample_rate) * swell_l
        right = _bandpass_fft(nr, 120.0, 920.0, sample_rate) * swell_r
        target_peak = 0.30

    else:
        # room_tone_clean
        nl = rng.standard_normal(n_samples).astype(np.float32)
        nr = rng.standard_normal(n_samples).astype(np.float32)
        left = _bandpass_fft(nl, 90.0, 1100.0, sample_rate)
        right = _bandpass_fft(nr, 90.0, 1100.0, sample_rate)
        target_peak = 0.12

    stereo = np.stack([left, right], axis=1)
    peak = float(np.max(np.abs(stereo))) if stereo.size else 1.0
    if peak > 1e-6:
        stereo = stereo * (target_peak / peak)

    # Fundido suave de entrada y salida para evitar clics
    fade_in_n = min(int(sample_rate * 0.35), n_samples // 4)
    fade_out_n = min(int(sample_rate * 0.65), n_samples // 4)
    if fade_in_n > 1:
        stereo[:fade_in_n] *= np.linspace(0.0, 1.0, fade_in_n, dtype=np.float32)[:, None]
    if fade_out_n > 1:
        stereo[-fade_out_n:] *= np.linspace(1.0, 0.0, fade_out_n, dtype=np.float32)[:, None]

    pcm16 = np.clip(stereo * 32767.0, -32768.0, 32767.0).astype(np.int16)
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_wav), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())

    return {
        "preset": preset_id,
        "preset_label": AMBIENCE_PRESETS[preset_id],
        "duration_seconds": round(dur, 2),
        "sample_rate": sample_rate,
    }


def _write_float_wav(output_wav: Path, audio_np: np.ndarray, sample_rate: int, target_peak: float = 0.88) -> float:
    arr = np.asarray(audio_np, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim == 1:
        arr = np.stack([arr, arr], axis=1)
    elif arr.ndim == 2 and arr.shape[0] in (1, 2) and arr.shape[1] > arr.shape[0] * 4:
        arr = arr.T
        if arr.shape[1] == 1:
            arr = np.repeat(arr, 2, axis=1)
    peak = float(np.max(np.abs(arr))) if arr.size else 1.0
    if peak > 1e-6:
        arr = arr * (target_peak / peak)
    n_samples = arr.shape[0]
    fade_in = min(int(sample_rate * 0.18), max(1, n_samples // 8))
    fade_out = min(int(sample_rate * 0.45), max(1, n_samples // 6))
    if fade_in > 1:
        arr[:fade_in] *= np.linspace(0.0, 1.0, fade_in, dtype=np.float32)[:, None]
    if fade_out > 1:
        arr[-fade_out:] *= np.linspace(1.0, 0.0, fade_out, dtype=np.float32)[:, None]

    pcm16 = np.clip(arr * 32767.0, -32768.0, 32767.0).astype(np.int16)
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_wav), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm16.tobytes())
    return round(n_samples / float(max(1, sample_rate)), 2)


class AudioStudioEngine:
    """Motor de generación musical (MusicGen), efectos/ambiente (AudioLDM2 / DSP) y mezcla multipista FFmpeg."""

    def __init__(self, catalog: CatalogManager) -> None:
        self.catalog = catalog
        self._lock = threading.Lock()
        self.loaded_model_id: Optional[str] = None
        self._pipe: Any = None
        self._processor: Any = None

    def unload(self) -> None:
        with self._lock:
            self._pipe = None
            self._processor = None
            self.loaded_model_id = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def generate_music_or_ambience(
        self,
        job: JobRecord,
        prompt: str,
        model_id: str = "auto",
        preset: str = "auto",
        duration_seconds: float = 10.0,
        seed: int = 42,
    ) -> Dict[str, Any]:
        dur = max(2.0, min(30.0, float(duration_seconds)))
        chosen_id = (model_id or "auto").strip()
        if chosen_id == "auto":
            if self.catalog.is_downloaded("musicgen-small-bgm"):
                chosen_id = "musicgen-small-bgm"
            elif self.catalog.is_downloaded("audioldm2-sfx"):
                chosen_id = "audioldm2-sfx"
            else:
                chosen_id = "procedural-ambience-cpu"

        defn = self.catalog.get_definition(chosen_id)
        if not defn:
            raise ValueError(f"Modelo de audio/música desconocido: {chosen_id}")

        AUDIO_OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_name = f"bgm_{job.id[:8]}.wav"
        out_path = AUDIO_OUT_DIR / out_name

        engine = defn.get("engine")
        if engine == "procedural_dsp":
            chosen_preset = infer_ambience_preset(prompt=prompt, preset=preset)
            job.log(
                f"Sintetizando paisaje acústico estéreo «{chosen_preset}» ({dur:.1f}s)...",
                progress=35.0,
                step="Sintetizando ambiente estéreo...",
            )
            meta = synthesize_procedural_ambience_wav(
                output_wav=out_path,
                preset=chosen_preset,
                duration_seconds=dur,
                seed=seed,
            )
            info = probe_media_info(out_path)
            rel_path = f"outputs/audio/{out_name}"
            meta_doc = {
                "id": out_path.stem,
                "filename": out_name,
                "rel_path": rel_path,
                "media_url": f"/media/{rel_path}",
                "type": "audio",
                "model_id": chosen_id,
                "model_name": defn["name"],
                "engine": "procedural_dsp",
                "preset": meta["preset"],
                "preset_label": meta["preset_label"],
                "prompt": prompt or meta["preset_label"],
                "duration_seconds": info["duration_seconds"],
                "size_mb": round(out_path.stat().st_size / (1024**2), 2),
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }
            (AUDIO_OUT_DIR / f"{out_path.stem}.json").write_text(
                json.dumps(meta_doc, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            job.log(
                f"Ambiente sonoro listo: {out_name} ({info['duration_seconds']:.2f}s).",
                level="success",
                progress=100.0,
                step="Ambiente completado",
            )
            return meta_doc

        if not self.catalog.is_downloaded(chosen_id):
            raise RuntimeError(
                f"El modelo «{defn['name']}» ({chosen_id}) no está descargado aún en disco. "
                f"Descárgalo primero con studio_download_model('{chosen_id}') o usa model_id='procedural-ambience-cpu' (0 GB instantáneo)."
            )

        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        base_dir = self.catalog.model_dir(chosen_id) / "base"

        if engine == "musicgen":
            from transformers import AutoProcessor, MusicgenForConditionalGeneration

            with self._lock:
                if self.loaded_model_id != chosen_id or self._pipe is None:
                    self.unload()
                    job.log(
                        f"Cargando «{defn['name']}» en {device.upper()}...",
                        progress=15.0,
                        step="Cargando MusicGen...",
                    )
                    dtype = torch.float16 if device == "cuda" else torch.float32
                    self._processor = AutoProcessor.from_pretrained(str(base_dir), local_files_only=True)
                    self._pipe = MusicgenForConditionalGeneration.from_pretrained(
                        str(base_dir),
                        torch_dtype=dtype,
                        local_files_only=True,
                    ).to(device)
                    self.loaded_model_id = chosen_id
                    memory_manager.register_loaded("audio_bgm", chosen_id, defn["name"], self)

                processor = self._processor
                model = self._pipe

            clean_prompt = (prompt or "").strip() or "warm smooth acoustic jazz piano restaurant background music"
            max_tokens = max(64, min(1500, int(round(dur * 51.2))))
            job.log(
                f"Componiendo música instrumental ({dur:.1f}s · {max_tokens} tokens): «{clean_prompt[:90]}»...",
                progress=45.0,
                step="Generando música con MusicGen...",
            )
            if seed >= 0:
                torch.manual_seed(int(seed))
            inputs = processor(text=[clean_prompt], padding=True, return_tensors="pt").to(device)
            with torch.inference_mode():
                audio_values = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=True, guidance_scale=3.0)
            sr = int(getattr(model.config.audio_encoder, "sampling_rate", 32000))
            wav_np = audio_values[0].detach().float().cpu().numpy()
            _write_float_wav(out_path, wav_np, sample_rate=sr)

        elif engine == "audioldm2":
            from diffusers import AudioLDM2Pipeline

            with self._lock:
                if self.loaded_model_id != chosen_id or self._pipe is None:
                    self.unload()
                    job.log(
                        f"Cargando «{defn['name']}» en {device.upper()}...",
                        progress=15.0,
                        step="Cargando AudioLDM 2...",
                    )
                    dtype = torch.float16 if device == "cuda" else torch.float32
                    pipe = AudioLDM2Pipeline.from_pretrained(str(base_dir), torch_dtype=dtype, local_files_only=True)
                    if device == "cuda":
                        pipe.enable_model_cpu_offload()
                    self._pipe = pipe
                    self.loaded_model_id = chosen_id
                    memory_manager.register_loaded("audio_bgm", chosen_id, defn["name"], self)
                pipe = self._pipe

            clean_prompt = (prompt or "").strip() or "cozy restaurant ambience, soft background conversation and subtle cutlery"
            gen = torch.Generator(device="cpu").manual_seed(int(seed) if seed >= 0 else 42)
            job.log(
                f"Generando efecto/paisaje sonoro con AudioLDM 2 ({dur:.1f}s): «{clean_prompt[:90]}»...",
                progress=45.0,
                step="Sintetizando audio con AudioLDM 2...",
            )
            with torch.inference_mode():
                out = pipe(
                    clean_prompt,
                    negative_prompt="low quality, harsh distortion, clipping, metallic noise",
                    num_inference_steps=60,
                    audio_length_in_s=dur,
                    generator=gen,
                )
            wav_np = out.audios[0]
            _write_float_wav(out_path, wav_np, sample_rate=16000)
        else:
            raise ValueError(f"Engine de audio no soportado: {engine}")

        info = probe_media_info(out_path)
        rel_path = f"outputs/audio/{out_name}"
        meta_doc = {
            "id": out_path.stem,
            "filename": out_name,
            "rel_path": rel_path,
            "media_url": f"/media/{rel_path}",
            "type": "audio",
            "model_id": chosen_id,
            "model_name": defn["name"],
            "engine": engine,
            "prompt": prompt,
            "duration_seconds": info["duration_seconds"],
            "size_mb": round(out_path.stat().st_size / (1024**2), 2),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        (AUDIO_OUT_DIR / f"{out_path.stem}.json").write_text(
            json.dumps(meta_doc, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        job.log(
            f"Pista generada y normalizada: {out_name} ({info['duration_seconds']:.2f}s).",
            level="success",
            progress=100.0,
            step="Audio completado",
        )
        return meta_doc


def mix_and_master_media(
    output_path: Path,
    video_path: Optional[Path] = None,
    voice_tracks: Optional[List[Dict[str, Any]]] = None,
    music_path: Optional[Path] = None,
    music_volume: float = 0.24,
    ambient_path: Optional[Path] = None,
    ambient_volume: float = 0.18,
    keep_original_audio: bool = False,
    original_audio_volume: float = 0.35,
    auto_duck_bg: bool = True,
    extend_video_to_audio: bool = True,
    fade_out_seconds: float = 0.80,
    job: Optional[JobRecord] = None,
) -> Dict[str, Any]:
    """Mezcla profesional multipista FFmpeg con posicionamiento al milisegundo y Auto-Ducking sobre diálogo.

    - Nunca divide el volumen de las voces (`amix=...:normalize=0`).
    - Posiciona 1 o varias intervenciones de voz con `start_sec` (`adelay`) en un bus limpio a 48 kHz estéreo.
    - Atenúa suavemente música y ambiente (`sidechaincompress`) mientras habla cualquier personaje.
    - Si las voces superan la duración del vídeo y `extend_video_to_audio=True`, extiende el vídeo sin congelar
      fotogramas (`setpts` suave hasta 1.4x o bucle ping-pong fluido para duraciones mayores).
    """
    ffmpeg_bin = get_ffmpeg_bin()
    v_tracks = list(voice_tracks or [])

    if video_path is not None and not video_path.exists():
        raise FileNotFoundError(f"No existe el vídeo base: {video_path}")
    for vt in v_tracks:
        vp = Path(vt["path"])
        if not vp.exists():
            raise FileNotFoundError(f"No existe la pista de voz: {vp}")
    if music_path is not None and not music_path.exists():
        raise FileNotFoundError(f"No existe la pista de música: {music_path}")
    if ambient_path is not None and not ambient_path.exists():
        raise FileNotFoundError(f"No existe la pista de ambiente: {ambient_path}")

    if video_path is None and not v_tracks and music_path is None and ambient_path is None:
        raise ValueError("Debes indicar al menos un vídeo o una pista de voz/música/ambiente para mezclar.")

    # Calcular duración total objetivo
    orig_vid_dur = 0.0
    orig_has_audio = False
    vid_w, vid_h, vid_fps = 832, 480, 24
    if video_path is not None:
        vi = probe_media_info(video_path)
        orig_vid_dur = max(0.3, float(vi["duration_seconds"] or 4.0))
        orig_has_audio = bool(vi["has_audio"])
        vid_w = int(vi["width"] or 832)
        vid_h = int(vi["height"] or 480)

    max_voice_end = 0.0
    enriched_voices: List[Dict[str, Any]] = []
    for vt in v_tracks:
        vp = Path(vt["path"])
        ai = probe_media_info(vp)
        vdur = max(0.1, float(ai["duration_seconds"] or 1.0))
        st_sec = max(0.0, float(vt.get("start_sec") or 0.0))
        vol = max(0.05, min(2.5, float(vt.get("volume", 1.0))))
        max_voice_end = max(max_voice_end, st_sec + vdur + 0.15)
        enriched_voices.append({"path": vp, "start_sec": st_sec, "volume": vol, "duration": vdur})

    if video_path is not None:
        target_dur = max(orig_vid_dur, max_voice_end) if extend_video_to_audio else orig_vid_dur
    else:
        bg_dur = 0.0
        if music_path is not None:
            bg_dur = max(bg_dur, float(probe_media_info(music_path)["duration_seconds"] or 0.0))
        if ambient_path is not None:
            bg_dur = max(bg_dur, float(probe_media_info(ambient_path)["duration_seconds"] or 0.0))
        target_dur = max(1.0, max_voice_end if max_voice_end > 0 else bg_dur)

    cmd: List[str] = [ffmpeg_bin, "-hide_banner", "-loglevel", "error"]
    input_idx = 0
    fc_parts: List[str] = []

    video_map_spec: Optional[str] = None
    reencode_video = False
    orig_audio_label: Optional[str] = None

    if video_path is not None:
        need_extend = target_dur > orig_vid_dur + 0.08
        use_pingpong = need_extend and (target_dur > orig_vid_dur * 1.40)
        if use_pingpong:
            cycles = max(1, int((target_dur / max(0.3, orig_vid_dur * 2.0)) + 1.5))
            if cycles > 1:
                cmd += ["-stream_loop", str(cycles)]
        cmd += ["-i", str(video_path)]
        vid_in = input_idx
        input_idx += 1

        if need_extend:
            reencode_video = True
            if use_pingpong:
                fc_parts.append(
                    f"[{vid_in}:v]fps={vid_fps},split=2[vf][vr];"
                    f"[vr]reverse[vrev];"
                    f"[vf][vrev]concat=n=2:v=1:a=0,trim=0:{target_dur:.3f},setpts=PTS-STARTPTS,format=yuv420p[vout]"
                )
            else:
                ratio = min(1.40, target_dur / orig_vid_dur)
                fc_parts.append(
                    f"[{vid_in}:v]setpts={ratio:.5f}*PTS,fps={vid_fps},"
                    f"trim=0:{target_dur:.3f},setpts=PTS-STARTPTS,format=yuv420p[vout]"
                )
            video_map_spec = "[vout]"
        else:
            video_map_spec = f"{vid_in}:v:0"

        if keep_original_audio and orig_has_audio:
            ovol = max(0.03, min(2.0, float(original_audio_volume)))
            fc_parts.append(
                f"[{vid_in}:a]aresample=48000,aformat=channel_layouts=stereo,"
                f"volume={ovol:.3f},apad=whole_dur={target_dur:.3f},atrim=0:{target_dur:.3f}[a_orig]"
            )
            orig_audio_label = "[a_orig]"

    # Construir bus de voces (claridad íntegra, filtro paso-alto suave anti-retumbo 75 Hz)
    voice_labels: List[str] = []
    if orig_audio_label is not None and not enriched_voices:
        # Si el vídeo ya trae la voz LipSync y no se pasan voces nuevas, tratar el audio original como bus de diálogo
        voice_labels.append(orig_audio_label)
        orig_audio_label = None

    for idx_v, vt in enumerate(enriched_voices):
        cmd += ["-i", str(vt["path"])]
        in_i = input_idx
        input_idx += 1
        delay_ms = int(round(vt["start_sec"] * 1000.0))
        delay_flt = f",adelay={delay_ms}|{delay_ms}" if delay_ms > 0 else ""
        lbl = f"[v_{idx_v}]"
        fc_parts.append(
            f"[{in_i}:a]aresample=48000,aformat=channel_layouts=stereo,"
            f"highpass=f=75,volume={vt['volume']:.3f}{delay_flt},"
            f"apad=whole_dur={target_dur:.3f},atrim=0:{target_dur:.3f}{lbl}"
        )
        voice_labels.append(lbl)

    has_dialogue_bus = len(voice_labels) > 0
    if len(voice_labels) == 1:
        fc_parts.append(f"{voice_labels[0]}anull[dialogue_bus]")
    elif len(voice_labels) > 1:
        joined_v = "".join(voice_labels)
        fc_parts.append(
            f"{joined_v}amix=inputs={len(voice_labels)}:duration=longest:dropout_transition=0:normalize=0[dialogue_bus]"
        )

    # Construir bus de fondo (música + ambiente + audio original secundario si aplica)
    bg_labels: List[str] = []
    if orig_audio_label is not None:
        bg_labels.append(orig_audio_label)

    in_fade = min(0.35, target_dur * 0.15)
    out_fade = max(0.0, min(float(fade_out_seconds), target_dur * 0.35))
    fade_out_st = max(0.0, target_dur - out_fade)
    fade_clause = f",afade=t=in:st=0:d={in_fade:.2f}"
    if out_fade > 0.05:
        fade_clause += f",afade=t=out:st={fade_out_st:.3f}:d={out_fade:.3f}"

    if music_path is not None:
        cmd += ["-stream_loop", "-1", "-i", str(music_path)]
        m_in = input_idx
        input_idx += 1
        mvol = max(0.02, min(2.0, float(music_volume)))
        fc_parts.append(
            f"[{m_in}:a]aresample=48000,aformat=channel_layouts=stereo,"
            f"atrim=0:{target_dur:.3f},asetpts=PTS-STARTPTS,volume={mvol:.3f}{fade_clause}[bg_music]"
        )
        bg_labels.append("[bg_music]")

    if ambient_path is not None:
        cmd += ["-stream_loop", "-1", "-i", str(ambient_path)]
        a_in = input_idx
        input_idx += 1
        avol = max(0.02, min(2.0, float(ambient_volume)))
        fc_parts.append(
            f"[{a_in}:a]aresample=48000,aformat=channel_layouts=stereo,"
            f"atrim=0:{target_dur:.3f},asetpts=PTS-STARTPTS,volume={avol:.3f}{fade_clause}[bg_amb]"
        )
        bg_labels.append("[bg_amb]")

    has_bg_bus = len(bg_labels) > 0
    if len(bg_labels) == 1:
        fc_parts.append(f"{bg_labels[0]}anull[bg_bus]")
    elif len(bg_labels) > 1:
        fc_parts.append(
            f"{''.join(bg_labels)}amix=inputs={len(bg_labels)}:duration=first:dropout_transition=0:normalize=0[bg_bus]"
        )

    # Combinación maestra con Auto-Ducking sobre diálogo
    if has_dialogue_bus and has_bg_bus:
        if auto_duck_bg:
            fc_parts.append(
                "[dialogue_bus]asplit=2[dlg_main][dlg_sc];"
                "[bg_bus][dlg_sc]sidechaincompress=threshold=0.022:ratio=5:attack=30:release=400:makeup=1.0[bg_ducked];"
                "[dlg_main][bg_ducked]amix=inputs=2:duration=first:dropout_transition=0:normalize=0,alimiter=limit=0.96[aout]"
            )
        else:
            fc_parts.append(
                "[dialogue_bus][bg_bus]amix=inputs=2:duration=first:dropout_transition=0:normalize=0,alimiter=limit=0.96[aout]"
            )
    elif has_dialogue_bus:
        fc_parts.append("[dialogue_bus]alimiter=limit=0.96[aout]")
    elif has_bg_bus:
        fc_parts.append("[bg_bus]alimiter=limit=0.96[aout]")
    else:
        cmd += ["-f", "lavfi", "-t", f"{target_dur:.3f}", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000"]
        s_in = input_idx
        fc_parts.append(f"[{s_in}:a]atrim=0:{target_dur:.3f}[aout]")

    if job:
        job.log(
            f"Masterizando sonido multipista ({len(enriched_voices)} voz/voces, "
            f"música={'sí' if music_path else 'no'}, ambiente={'sí' if ambient_path else 'no'}, "
            f"auto-ducking={'activo' if (auto_duck_bg and has_dialogue_bus and has_bg_bus) else 'no'})...",
            progress=65.0,
            step="Mezclando pistas y aplicando Auto-Ducking...",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd += ["-filter_complex", ";".join(fc_parts)]

    if video_map_spec is not None:
        cmd += ["-map", video_map_spec, "-map", "[aout]", "-t", f"{target_dur:.3f}"]
        if reencode_video:
            cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "19", "-colorspace", "bt709"]
        else:
            cmd += ["-c:v", "copy"]
        cmd += ["-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2", "-movflags", "+faststart", "-y", str(output_path)]
    else:
        ext = output_path.suffix.lower()
        cmd += ["-map", "[aout]", "-t", f"{target_dur:.3f}"]
        if ext == ".mp3":
            cmd += ["-c:a", "libmp3lame", "-b:a", "192k", "-ar", "48000", "-ac", "2", "-y", str(output_path)]
        else:
            cmd += ["-c:a", "pcm_s16le", "-ar", "48000", "-ac", "2", "-y", str(output_path)]

    res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if res.returncode != 0:
        raise RuntimeError(f"Error en mezcla multipista FFmpeg: {res.stderr.strip()}")

    final_info = probe_media_info(output_path)
    return {
        "duration_seconds": final_info["duration_seconds"],
        "width": vid_w if video_path else 0,
        "height": vid_h if video_path else 0,
        "voice_tracks_count": len(enriched_voices),
        "auto_ducking_applied": bool(auto_duck_bg and has_dialogue_bus and has_bg_bus),
        "video_extended": bool(video_path is not None and reencode_video),
        "size_mb": round(output_path.stat().st_size / (1024**2), 2),
    }


audio_studio_engine = AudioStudioEngine(catalog_manager)

