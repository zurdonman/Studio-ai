"""Small CPU synthesizer for importing Standard MIDI, not a sampled orchestra."""
import io
import json
import wave
from collections import defaultdict, deque
from datetime import datetime

import numpy as np

from studio.config import AUDIO_OUT_DIR

INSTRUMENTS = {"piano", "bass", "strings", "synth"}
MAX_MIDI_BYTES = 2 * 1024 * 1024


def parse_notes(data: bytes):
    import mido
    if not data or len(data) > MAX_MIDI_BYTES:
        raise ValueError("El MIDI debe ocupar entre 1 byte y 2 MiB.")
    try:
        midi = mido.MidiFile(file=io.BytesIO(data))
        if midi.type == 2 or midi.ticks_per_beat <= 0:
            raise ValueError("Solo se admiten MIDI tipo 0 o 1 con tiempos musicales.")
        merged = mido.merge_tracks(midi.tracks)
    except (EOFError, OSError, ValueError, KeyError, IndexError) as exc:
        raise ValueError(f"MIDI no válido o no compatible: {exc}") from exc
    tempo = 500000
    now = 0.0
    active = defaultdict(deque)
    notes = []
    for index, msg in enumerate(merged):
        if index > 100000:
            raise ValueError("El MIDI tiene demasiados eventos (máximo 100000).")
        now += mido.tick2second(msg.time, midi.ticks_per_beat, tempo)
        if now > 120:
            raise ValueError("El MIDI supera 120 segundos.")
        if msg.type == "set_tempo":
            tempo = msg.tempo
        elif msg.type == "note_on" and msg.velocity > 0:
            if msg.channel != 9:  # no drum kit in this oscillator synth
                active[(msg.channel, msg.note)].append((now, msg.velocity))
        elif msg.type in ("note_off", "note_on") and msg.channel != 9:
            key = (msg.channel, msg.note)
            if active[key]:
                start, velocity = active[key].popleft()
                if now > start:
                    notes.append((start, now, msg.note, velocity))
                    if len(notes) > 2000:
                        raise ValueError("El MIDI supera 2000 notas.")
    for (_, pitch), starts in active.items():
        for start, velocity in starts:
            if now > start:
                notes.append((start, now, pitch, velocity))
    if len(notes) > 2000 or sum(end - start for start, end, _, _ in notes) > 300:
        raise ValueError("Demasiadas notas o duración acumulada (máximo 300 s).")
    if not notes:
        raise ValueError("No hay notas melódicas: la percusión MIDI no está soportada.")
    return notes


def synthesize_midi(data: bytes, instrument: str, job):
    if instrument not in INSTRUMENTS:
        raise ValueError("Instrumento sintético desconocido.")
    job.log("Leyendo MIDI y mapa de tempo; percusión y pedal de sustain no se interpretan.", progress=5, step="Importando MIDI")
    notes = parse_notes(data)
    sr = 44100
    length = max(end for _, end, _, _ in notes) + .15
    audio = np.zeros(int(length * sr) + 1, dtype=np.float32)
    for idx, (start, end, pitch, velocity) in enumerate(notes):
        if idx % 8 == 0:
            job.check_cancelled()
        n = max(1, int((end - start) * sr))
        t = np.arange(n, dtype=np.float32) / sr
        freq = 440.0 * 2 ** ((pitch - 69) / 12)
        phase = 2 * np.pi * freq * t
        attack = np.minimum(1, t / (.04 if instrument == "strings" else .008))
        release = np.minimum(1, (n / sr - t) / .06)
        if instrument == "piano":
            wave_data = np.sin(phase) + .22 * np.sin(2 * phase) + .08 * np.sin(3 * phase)
            envelope = np.exp(-2.3 * t) * attack * release
        elif instrument == "bass":
            wave_data = np.sin(phase) + .35 * np.sin(2 * phase)
            envelope = attack * release * np.exp(-.6 * t)
        elif instrument == "strings":
            wave_data = np.sin(phase) + .3 * np.sin(2 * phase) + .15 * np.sin(3 * phase)
            envelope = attack * release
        else:
            wave_data = np.sin(phase) + .3 * np.sin(3 * phase)
            envelope = attack * release
        start_sample = int(start * sr)
        audio[start_sample:start_sample+n] += (wave_data * envelope * (velocity / 127) * .16).astype(np.float32)
    job.check_cancelled()
    peak = float(np.max(np.abs(audio)))
    if peak > .95:
        audio *= .95 / peak
    AUDIO_OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = AUDIO_OUT_DIR / f"midi_{instrument}_{datetime.now():%Y%m%d_%H%M%S}_{job.id[:6]}.wav"
    try:
        with wave.open(str(out), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes((np.clip(audio, -1, 1) * 32767).astype('<i2').tobytes())
        job.check_cancelled()
        rel = f"outputs/audio/{out.name}"
        meta = {"id": out.stem, "filename": out.name, "rel_path": rel, "media_url": f"/media/{rel}", "type": "audio",
                "engine": "midi_synth_cpu", "instrument": instrument, "notes": len(notes),
                "duration_seconds": round(len(audio) / sr, 2), "created_at": datetime.now().isoformat(timespec="seconds")}
        out.with_suffix('.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
        job.log("MIDI sintetizado en WAV.", progress=100, step="Audio completado")
        return meta
    except BaseException:
        out.unlink(missing_ok=True)
        raise
