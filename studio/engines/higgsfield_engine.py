"""Motor «Modo Higgsfield» 100 % Local / Google Colab (0 € APIs externas) para Studio IA.

Adapta las mejores herramientas creativas estilo Higgsfield / Cinema Studio ejecutándose íntegramente
sobre los motores propios de Studio IA en Google Colab (Qwen-Image-Edit 2511, Wan 2.1/2.2 14B FP8,
Coherencia OpenCV+Qwen2.5-VL, Edge/Kokoro TTS, LipSync HD EDTalk-256+GFPGAN y FFmpeg Auto-Ducking):

1. **Cinema Studio Óptico (`compile_cinema_prompt`):**
   Cámaras de cine reales (IMAX 70mm, ARRI 8K, Super 35, 16mm), lentes (Anamórfica 2x, 70s Prime,
   Halation Diffusion, Swirl Bokeh, Tilt-Shift, Macro), focales (8–85mm), aperturas (f/1.4–f/11)
   y etalonaje/iluminación cinematográfica.
2. **Presets de Movimiento de Cámara «Higgsfield Motion» (`HIGGSFIELD_MOTIONS`):**
   14 movimientos icónicos (Crash Zoom, Dolly Zoom Vértigo, Bullet Time, FPV Drone, SnorriCam,
   Whip Pan, Cámara al Hombro, Grúa, Cenital, Sobre el Hombro...) calibrados tanto para modelos IA 14B
   como para previsualización óptica instantánea en FFmpeg CPU.
3. **Casting / AI Character Vault (`CharacterVault`):**
   Fichas persistentes de personajes (foto ancla, rasgos físicos estructurados, voz TTS asignada y
   posición habitual en plano) para reutilizarlos en cualquier escena sin perder identidad.
4. **Auto-Director de Escena en 1 Clic (`run_higgsfield_scene_director`) y Smart Clipper 9:16 (`smart_clip_and_reframe`):**
   Encadena Fotograma Cine → Movimiento Higgsfield → Voces → LipSync HD → Ambiente + Auto-Ducking,
   más recorte inteligente centrado en rostros con OpenCV para pasar de 16:9 a 9:16 (Reels/Shorts/TikTok).
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from studio.catalog import catalog_manager
from studio.config import IMAGES_OUT_DIR, STATE_DIR, VIDEOS_OUT_DIR
from studio.engines.audio_studio_engine import audio_studio_engine, mix_and_master_media
from studio.engines.coherence_engine import analyze_pair_fast_cv
from studio.engines.ffmpeg_utils import (
    create_motion_clip_from_image,
    get_ffmpeg_bin,
    probe_media_info,
)
from studio.engines.image_engine import generate_or_edit_image
from studio.engines.lipsync_engine import lipsync_engine
from studio.engines.tts_engine import synthesize_voice
from studio.engines.video_engine import generate_video_clip
from studio.jobs import JobRecord


# ==================== 1. CATÁLOGO ÓPTICO CINEMA STUDIO ====================

CINEMA_CAMERAS: Dict[str, Dict[str, str]] = {
    "arri_full_frame_8k": {
        "label": "🎥 ARRI Full-Frame Cine Digital 8K",
        "prompt": "shot on an ARRI Alexa LF full-frame digital cinema camera, 14+ stops dynamic range, organic skin tones",
    },
    "imax_70mm_film": {
        "label": "🎞️ IMAX Grand Format 70mm Film",
        "prompt": "shot on IMAX 15/70mm grand format film camera, Kodak Vision3 65mm film stock, monumental scale and fine photochemical grain",
    },
    "modular_8k_red": {
        "label": "📹 Modular 8K Large Format (RED V-Raptor)",
        "prompt": "shot on a modular 8K large-format cinema camera, razor-sharp micro-contrast, crisp motion rendering",
    },
    "super35_studio": {
        "label": "🎬 Studio Digital Super 35",
        "prompt": "shot on a Super 35 narrative studio cinema camera, classic Hollywood framing and balanced texture",
    },
    "classic_16mm_film": {
        "label": "📽️ Classic 16mm Indie Film (Arriflex)",
        "prompt": "shot on classic 16mm Kodachrome/Vision3 film camera, authentic organic halation, tactile analog film grain",
    },
    "sony_venice_anamorphic": {
        "label": "🌌 Cinema Large-Sensor Night Rig",
        "prompt": "shot on dual-native ISO large-format cinema sensor, pristine low-light shadow detail, rich highlights",
    },
}

CINEMA_LENSES: Dict[str, Dict[str, str]] = {
    "classic_anamorphic": {
        "label": "🔭 Anamórfica Clásica 2x (Bokeh Oval + Flare)",
        "prompt": "classic 2x anamorphic cinema lens, subtle horizontal blue lens flare, oval bokeh, widescreen cinematic breathing",
    },
    "warm_70s_prime": {
        "label": "🟠 Cooke / Prime Cálida Años 70",
        "prompt": "vintage 1970s warm-coated cinema prime lens, gentle contrast roll-off, flattering warm skin rendering",
    },
    "halation_diffusion": {
        "label": "✨ Halation Black Pro-Mist Diffusion",
        "prompt": "cinema prime lens with 1/4 Black Pro-Mist halation diffusion filter, dreamy glowing practical lights",
    },
    "premium_modern_prime": {
        "label": "💎 Master Prime Moderna Ultra-Limpia",
        "prompt": "Zeiss Master Prime modern cinema lens, edge-to-edge optical perfection, zero chromatic aberration",
    },
    "swirl_bokeh_portrait": {
        "label": "🌀 Retrato Swirl Bokeh (Helios 58mm)",
        "prompt": "vintage Helios swirl-bokeh portrait lens, expressive rotational background bokeh, sharp center subject",
    },
    "compact_anamorphic": {
        "label": "📐 Compact Anamorphic Indie",
        "prompt": "compact anamorphic lens, subtle barrel character and cinematic depth separation",
    },
    "extreme_macro": {
        "label": "🔬 Macro Extremo Cinematográfico",
        "prompt": "probe extreme macro cinema lens, microscopic surface texture detail, ultra-shallow focal plane",
    },
    "creative_tilt_shift": {
        "label": "🎛️ Lente Creativa Tilt-Shift",
        "prompt": "creative tilt-shift cinema lens, selective oblique plane of focus, artistic miniature-like blur falloff",
    },
}

FOCAL_LENGTHS: Dict[str, Dict[str, str]] = {
    "14": {"label": "14mm Ultra-Angular Dramático", "prompt": "14mm ultra-wide cinema focal length, expansive spatial depth and dramatic perspective"},
    "24": {"label": "24mm Angular Dinámico", "prompt": "24mm wide-angle dynamic cinema perspective, immersive environment"},
    "35": {"label": "35mm Narrativo Clásico", "prompt": "35mm natural narrative cinema focal length, balanced subject and environment"},
    "50": {"label": "50mm Ojo Humano Natural", "prompt": "50mm standard cinema perspective, natural proportions and intimate presence"},
    "85": {"label": "85mm Primer Plano / Retrato", "prompt": "85mm classic portrait telephoto focal length, flattering facial compression"},
}

APERTURES: Dict[str, Dict[str, str]] = {
    "f/1.4": {"label": "f/1.4 Bokeh Cremoso (Fondo Desenfocado)", "prompt": "wide-open T1.4 aperture, razor-thin depth of field, creamy background bokeh"},
    "f/2.8": {"label": "f/2.8 Separación Cinematográfica", "prompt": "T2.8 cinema aperture, natural subject-to-background separation"},
    "f/4": {"label": "f/4 Equilibrado (Dos Personajes)", "prompt": "T4 balanced aperture, keeping two conversing subjects crisply in focus"},
    "f/11": {"label": "f/11 Foco Profundo Total", "prompt": "T11 deep-focus cinematography, sharp clarity from foreground to background"},
}

LIGHTING_MOODS: Dict[str, Dict[str, str]] = {
    "natural_cinema": {
        "label": "🌤️ Luz Cinematográfica Natural (BT.709)",
        "prompt": "motivated natural cinematic lighting, soft key light, subtle rim light, true-to-life BT.709 color science",
    },
    "warm_restaurant": {
        "label": "🕯️ Restaurante Íntimo / Tungsteno Cálido",
        "prompt": "warm intimate restaurant practical tungsten and candlelight, cozy amber table glow, soft shadow falloff",
    },
    "golden_hour": {
        "label": "🌅 Hora Dorada (Atardecer Cálido)",
        "prompt": "magic golden hour sunlight, warm directional backlight, atmospheric dust motes, rich cinematic shadows",
    },
    "teal_orange_blockbuster": {
        "label": "🎬 Blockbuster Teal & Orange",
        "prompt": "modern blockbuster color grade, cool slate-teal shadows contrasting warm natural skin tones, high contrast",
    },
    "neon_night": {
        "label": "🌃 Noche Neón / Urbano Lluvioso",
        "prompt": "moody nocturnal neon reflections on wet asphalt, cyan and magenta practical rim lighting, atmospheric mist",
    },
    "noir_chiaroscuro": {
        "label": "🌑 Chiaroscuro / Thriller Dramático",
        "prompt": "high-contrast chiaroscuro dramatic lighting, sculpted facial shadows, volumetric light beams",
    },
}


# ==================== 2. PRESETS DE CÁMARA «HIGGSFIELD MOTION» ====================

HIGGSFIELD_MOTIONS: Dict[str, Dict[str, str]] = {
    "dolly_in_dramatic": {
        "label": "🎯 Slow Push-In Dramático (Dolly In)",
        "video_prompt": "smooth cinematic dolly push-in camera movement slowly approaching the subject, natural subtle motion, steady gimbal",
        "ffmpeg_motion": "zoom_in",
    },
    "crash_zoom": {
        "label": "⚡ Crash Zoom (Zoom Rápido de Impacto)",
        "video_prompt": "fast energetic crash zoom whipping rapidly toward the subject's face, dramatic action cinema lens zoom, dynamic energy",
        "ffmpeg_motion": "crash_zoom",
    },
    "dolly_zoom_vertigo": {
        "label": "🌀 Dolly Zoom / Efecto Vértigo",
        "video_prompt": "cinematic Hitchcock dolly-zoom vertigo effect, camera pulls back while lens zooms in, background perspective warps dramatically around the subject",
        "ffmpeg_motion": "zoom_out",
    },
    "bullet_time_orbit": {
        "label": "🕶️ Bullet Time / Órbita 360°",
        "video_prompt": "smooth sweeping orbital arc camera circling around the subject in slow motion, parallax depth movement in background, cinema rig",
        "ffmpeg_motion": "pan_right",
    },
    "fpv_drone_dive": {
        "label": "🦅 FPV Drone Cinematográfico",
        "video_prompt": "fluid FPV cinema drone gliding forward through the space with gentle banking roll and immersive spatial depth",
        "ffmpeg_motion": "zoom_in",
    },
    "handheld_docu": {
        "label": "📹 Cámara al Hombro Realista (Handheld)",
        "video_prompt": "organic handheld cinema camera with subtle natural human breathing micro-movement, authentic documentary intimacy",
        "ffmpeg_motion": "handheld",
    },
    "snorricam_chest": {
        "label": "🎭 SnorriCam (Fijada al Actor)",
        "video_prompt": "SnorriCam body-mounted camera locked onto the main character while the surrounding environment moves dynamically behind them",
        "ffmpeg_motion": "handheld",
    },
    "whip_pan": {
        "label": "💨 Whip Pan / Barrido Dinámico",
        "video_prompt": "energetic lateral whip-pan camera sweep revealing the scene with natural motion blur settling into a crisp framing",
        "ffmpeg_motion": "pan_right",
    },
    "crane_up_reveal": {
        "label": "🏗️ Grúa Ascendente (Crane Up Reveal)",
        "video_prompt": "majestic cinema crane jib rising smoothly upward while tilting gently to reveal the full environment",
        "ffmpeg_motion": "zoom_out",
    },
    "tracking_side": {
        "label": "🛤️ Travelling Lateral Paralelo",
        "video_prompt": "smooth lateral dolly tracking shot gliding sideways parallel to the subjects with rich foreground-background parallax",
        "ffmpeg_motion": "pan_right",
    },
    "over_the_shoulder": {
        "label": "🗣️ Plano Conversación / Mesa (Ideal Diálogo)",
        "video_prompt": "steady intimate cinema framing of the characters conversing naturally with lifelike gestures, subtle head nods, and eye contact",
        "ffmpeg_motion": "zoom_in",
    },
    "subtle_breathing": {
        "label": "👄 Micro-Movimiento Retrato (Ideal LipSync)",
        "video_prompt": "ultra-clean steady portrait cinema shot, natural blinking, subtle breathing and gentle head posture, sharp facial clarity",
        "ffmpeg_motion": "zoom_in",
    },
}


def compile_cinema_prompt(
    user_prompt: str,
    camera_id: str = "arri_full_frame_8k",
    lens_id: str = "classic_anamorphic",
    focal_mm: str = "35",
    aperture: str = "f/2.8",
    lighting_id: str = "natural_cinema",
    characters: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Construye el prompt óptico completo estilo Cinema Studio respetando personajes del Casting Vault."""
    parts: List[str] = []
    clean_base = (user_prompt or "").strip()
    if clean_base:
        parts.append(clean_base)

    if characters:
        char_descs = []
        for ch in characters:
            cname = ch.get("name") or "Personaje"
            ctraits = (ch.get("appearance_prompt") or "").strip()
            if ctraits:
                char_descs.append(f"{cname} ({ctraits})")
        if char_descs:
            parts.append("featuring " + " and ".join(char_descs))

    cam = CINEMA_CAMERAS.get(camera_id) or CINEMA_CAMERAS["arri_full_frame_8k"]
    lens = CINEMA_LENSES.get(lens_id) or CINEMA_LENSES["classic_anamorphic"]
    foc = FOCAL_LENGTHS.get(str(focal_mm)) or FOCAL_LENGTHS["35"]
    ap = APERTURES.get(aperture) or APERTURES["f/2.8"]
    lig = LIGHTING_MOODS.get(lighting_id) or LIGHTING_MOODS["natural_cinema"]

    parts.extend(
        [
            cam["prompt"],
            f"using a {lens['prompt']} at {foc['prompt']}",
            ap["prompt"],
            lig["prompt"],
            "photorealistic cinema frame, ultra-detailed skin and fabric texture, high dynamic range",
        ]
    )
    return ", ".join(p for p in parts if p)


def render_higgsfield_cpu_motion_clip(
    image_path: Path,
    output_mp4: Path,
    motion_id: str = "dolly_in_dramatic",
    width: int = 1280,
    height: int = 720,
    fps: int = 24,
    duration_sec: float = 4.0,
    audio_path: Optional[Path] = None,
    job: Optional[JobRecord] = None,
) -> Dict[str, Any]:
    """Renderiza curvas ópticas especiales en CPU (Crash Zoom, Handheld orgánico, Dolly, Pan) en ~1.5s y 0 GB VRAM."""
    m_spec = HIGGSFIELD_MOTIONS.get(motion_id) or HIGGSFIELD_MOTIONS["dolly_in_dramatic"]
    ff_kind = m_spec.get("ffmpeg_motion", "zoom_in")

    if ff_kind in ("crash_zoom", "handheld"):
        ffmpeg_bin = get_ffmpeg_bin()
        width = max(256, (int(width) // 2) * 2)
        height = max(256, (int(height) // 2) * 2)
        fps = max(12, min(60, int(fps)))
        dur = max(1.0, float(duration_sec))
        if audio_path and audio_path.exists():
            ai = probe_media_info(audio_path)
            if ai["duration_seconds"] > 0.2:
                dur = max(dur, float(ai["duration_seconds"]))
        total_frames = max(fps, int(round(dur * fps)))
        base_pad = (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1"
        )
        if ff_kind == "crash_zoom":
            # Aceleración rápida exponencial en el primer tercio seguida de micro-estabilización
            vf = (
                f"{base_pad},scale={width*2}:{height*2},"
                f"zoompan=z='min(1.42,1.0+0.42*pow(min(1.0,on/{max(1, int(total_frames*0.38))}),2))'"
                f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
                f":d={total_frames}:s={width}x{height}:fps={fps},format=yuv420p"
            )
        else:
            # Cámara al hombro orgánica (oscilación suave senoidal en X/Y con ligero zoom 1.08)
            vf = (
                f"{base_pad},scale={int(width*1.20)}:{int(height*1.20)},"
                f"zoompan=z='1.08':x='(iw-iw/zoom)/2+sin(on/5.5)*14':y='(ih-ih/zoom)/2+cos(on/7.0)*10'"
                f":d={total_frames}:s={width}x{height}:fps={fps},format=yuv420p"
            )

        cmd = [ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-i", str(image_path)]
        if audio_path and audio_path.exists():
            cmd += ["-i", str(audio_path)]
            af_chain = f"[1:a]aresample=48000,aformat=channel_layouts=stereo,apad=whole_dur={dur:.3f},atrim=0:{dur:.3f}[aout]"
        else:
            cmd += ["-f", "lavfi", "-t", f"{dur:.3f}", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000"]
            af_chain = f"[1:a]atrim=0:{dur:.3f}[aout]"

        output_mp4.parent.mkdir(parents=True, exist_ok=True)
        cmd += [
            "-filter_complex",
            f"[0:v]{vf}[vout];{af_chain}",
            "-map",
            "[vout]",
            "-map",
            "[aout]",
            "-t",
            f"{dur:.3f}",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "19",
            "-colorspace",
            "bt709",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-movflags",
            "+faststart",
            "-y",
            str(output_mp4),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if res.returncode != 0:
            raise RuntimeError(f"Error FFmpeg en efecto Higgsfield ({motion_id}): {res.stderr.strip()}")
        return probe_media_info(output_mp4)

    return create_motion_clip_from_image(
        image_path=image_path,
        output_mp4=output_mp4,
        duration_sec=duration_sec,
        width=width,
        height=height,
        fps=fps,
        motion=ff_kind,
        audio_path=audio_path,
        job=job,
    )


# ==================== 3. CASTING VAULT (PERSONAJES PERSISTENTES) ====================

class CharacterVault:
    """Almacén persistente en disco (`state/characters_vault.json`) de personajes reutilizables."""

    def __init__(self) -> None:
        self._lock = threading.RLock()

    def _vault_path(self) -> Path:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        return STATE_DIR / "characters_vault.json"

    def list_characters(self) -> List[Dict[str, Any]]:
        with self._lock:
            fp = self._vault_path()
            if not fp.exists():
                return []
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
                return data if isinstance(data, list) else []
            except Exception:
                return []

    def get_character(self, char_id: str) -> Optional[Dict[str, Any]]:
        cid = (char_id or "").strip()
        if not cid:
            return None
        for ch in self.list_characters():
            if ch.get("id") == cid or ch.get("name", "").lower() == cid.lower():
                return ch
        return None

    def save_character(
        self,
        name: str,
        appearance_prompt: str,
        anchor_image_rel: str = "",
        voice_id: str = "es-ES-AlvaroNeural",
        default_face_side: str = "largest",
        traits: Optional[Dict[str, str]] = None,
        char_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        clean_name = (name or "").strip()
        if not clean_name:
            raise ValueError("El personaje necesita un nombre (ej. 'Marco', 'Lucía').")

        with self._lock:
            items = self.list_characters()
            cid = (char_id or "").strip() or f"char_{uuid.uuid4().hex[:8]}"
            entry = {
                "id": cid,
                "name": clean_name,
                "appearance_prompt": (appearance_prompt or "").strip(),
                "anchor_image_rel": (anchor_image_rel or "").strip(),
                "media_url": f"/media/{anchor_image_rel.lstrip('/')}" if anchor_image_rel else "",
                "voice_id": voice_id or "es-ES-AlvaroNeural",
                "default_face_side": default_face_side or "largest",
                "traits": dict(traits or {}),
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
            replaced = False
            for idx, existing in enumerate(items):
                if existing.get("id") == cid:
                    items[idx] = entry
                    replaced = True
                    break
            if not replaced:
                items.insert(0, entry)
            self._vault_path().write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")
            return entry

    def delete_character(self, char_id: str) -> bool:
        with self._lock:
            items = self.list_characters()
            filtered = [c for c in items if c.get("id") != char_id]
            if len(filtered) == len(items):
                return False
            self._vault_path().write_text(json.dumps(filtered, indent=2, ensure_ascii=False), encoding="utf-8")
            return True


character_vault = CharacterVault()


def build_character_appearance_from_traits(traits: Dict[str, str], extra_notes: str = "") -> str:
    """Convierte selectores visuales estilo AI Influencer Studio en una descripción física precisa."""
    order = ["gender_age", "ethnicity", "hair", "eyes", "outfit", "distinctive", "style"]
    chunks = [str(traits.get(k) or "").strip() for k in order if str(traits.get(k) or "").strip()]
    if extra_notes and extra_notes.strip():
        chunks.append(extra_notes.strip())
    return ", ".join(chunks)


# ==================== 4. SMART CLIPPER & REENCUADRE 9:16 REELS/SHORTS ====================

def _estimate_subject_center_x_ratio(video_path: Path) -> float:
    """Muestrea hasta 7 fotogramas con OpenCV (densidad de bordes/textura facial superior) para centrar 9:16."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0.5
    try:
        total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1))
        sample_indices = np.linspace(0, max(0, total - 1), num=min(7, total), dtype=int)
        centers: List[float] = []
        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            h, w = frame.shape[:2]
            if w <= 16 or h <= 16:
                continue
            # Enfocarse en la franja superior-media (donde suelen estar cabezas y rostros)
            roi = frame[int(h * 0.10) : int(h * 0.72), :]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            energy_x = np.mean(np.hypot(gx, gy), axis=0)
            # Suavizar perfil horizontal con ventana del ancho de un encuadre 9:16
            win_w = max(8, min(w, int(round(h * 9.0 / 16.0))))
            kernel = np.ones(win_w, dtype=np.float32) / float(win_w)
            smooth = np.convolve(energy_x, kernel, mode="same")
            best_x = float(np.argmax(smooth)) / float(w)
            centers.append(float(np.clip(best_x, 0.15, 0.85)))
        return float(np.median(centers)) if centers else 0.5
    finally:
        cap.release()


def smart_clip_and_reframe_video(
    video_path: Path,
    start_sec: float = 0.0,
    end_sec: float = 0.0,
    target_aspect: str = "9:16",
    fade_seconds: float = 0.25,
    job: Optional[JobRecord] = None,
) -> Dict[str, Any]:
    """Recorta un highlight de vídeo y lo reencuadra inteligentemente (9:16 Reels, 1:1 o 16:9) 100 % en local."""
    if not video_path.exists():
        raise FileNotFoundError(f"No existe el vídeo a recortar: {video_path}")

    vi = probe_media_info(video_path)
    src_dur = max(0.3, float(vi["duration_seconds"] or 4.0))
    st = max(0.0, min(src_dur - 0.2, float(start_sec or 0.0)))
    en = float(end_sec) if float(end_sec or 0.0) > st + 0.2 else src_dur
    en = min(src_dur, en)
    clip_dur = max(0.25, en - st)

    cx_ratio = _estimate_subject_center_x_ratio(video_path) if target_aspect in ("9:16", "1:1") else 0.5
    if job:
        job.log(
            f"Smart Clipper: extrayendo {st:.2f}s → {en:.2f}s ({target_aspect}, centro sujeto X={cx_ratio*100:.0f}%)...",
            progress=40.0,
            step="Reencuadrando y recortando clip...",
        )

    if target_aspect == "9:16":
        out_w, out_h = 720, 1280
        crop_vf = (
            f"crop='min(iw,ih*9/16)':'ih':'clip((iw-min(iw,ih*9/16))*{cx_ratio:.4f},0,iw-min(iw,ih*9/16))':0,"
            f"scale={out_w}:{out_h},setsar=1"
        )
    elif target_aspect == "1:1":
        out_w, out_h = 1080, 1080
        crop_vf = (
            f"crop='min(iw,ih)':'min(iw,ih)':'clip((iw-min(iw,ih))*{cx_ratio:.4f},0,iw-min(iw,ih))':'(ih-min(iw,ih))/2',"
            f"scale={out_w}:{out_h},setsar=1"
        )
    else:
        out_w, out_h = 1280, 720
        crop_vf = f"scale={out_w}:{out_h}:force_original_aspect_ratio=decrease,pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2,setsar=1"

    fade_v = ""
    fade_a = ""
    if fade_seconds > 0.05 and clip_dur > fade_seconds * 2.5:
        fo = max(0.0, clip_dur - fade_seconds)
        fade_v = f",fade=t=in:st=0:d={fade_seconds:.2f},fade=t=out:st={fo:.2f}:d={fade_seconds:.2f}"
        fade_a = f",afade=t=in:st=0:d={fade_seconds:.2f},afade=t=out:st={fo:.2f}:d={fade_seconds:.2f}"

    VIDEOS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts_slug = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = f"reel_{target_aspect.replace(':', 'x')}_{ts_slug}.mp4"
    out_path = VIDEOS_OUT_DIR / out_name

    ffmpeg_bin = get_ffmpeg_bin()
    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{st:.3f}",
        "-t",
        f"{clip_dur:.3f}",
        "-i",
        str(video_path),
    ]
    if vi["has_audio"]:
        fc = f"[0:v]{crop_vf},format=yuv420p{fade_v}[vout];[0:a]aresample=48000,aformat=channel_layouts=stereo{fade_a}[aout]"
    else:
        cmd += ["-f", "lavfi", "-t", f"{clip_dur:.3f}", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000"]
        fc = f"[0:v]{crop_vf},format=yuv420p{fade_v}[vout];[1:a]atrim=0:{clip_dur:.3f}[aout]"

    cmd += [
        "-filter_complex",
        fc,
        "-map",
        "[vout]",
        "-map",
        "[aout]",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "19",
        "-colorspace",
        "bt709",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        "-y",
        str(out_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
    if res.returncode != 0:
        raise RuntimeError(f"Error en Smart Clipper FFmpeg: {res.stderr.strip()}")

    out_info = probe_media_info(out_path)
    rel_path = f"outputs/videos/{out_name}"
    meta = {
        "id": out_path.stem,
        "filename": out_name,
        "rel_path": rel_path,
        "media_url": f"/media/{rel_path}",
        "type": "video",
        "aspect": target_aspect,
        "subject_center_x_pct": round(cx_ratio * 100.0, 1),
        "duration_seconds": out_info["duration_seconds"],
        "width": out_w,
        "height": out_h,
        "size_mb": round(out_path.stat().st_size / (1024**2), 2),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    (VIDEOS_OUT_DIR / f"{out_path.stem}.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if job:
        job.log(f"Reel / Clip listo: {out_name} ({meta['duration_seconds']}s).", level="success", progress=100.0)
    return meta


# ==================== 5. AUTO-DIRECTOR DE ESCENA COMPLETA «HIGGSFIELD» ====================

def run_higgsfield_scene_director(
    job: JobRecord,
    scene_prompt: str,
    resolve_rel_fn: Any,
    camera_id: str = "arri_full_frame_8k",
    lens_id: str = "classic_anamorphic",
    focal_mm: str = "35",
    aperture: str = "f/2.8",
    lighting_id: str = "natural_cinema",
    motion_id: str = "dolly_in_dramatic",
    video_engine_mode: str = "auto",  # "auto" | "ai_14b" | "ai_fast" | "ffmpeg_instant"
    aspect_ratio: str = "16:9",
    duration_seconds: float = 4.5,
    base_image_rel: Optional[str] = None,
    character_ids: Optional[List[str]] = None,
    dialogue_lines: Optional[List[Dict[str, Any]]] = None,
    ambience_preset: str = "none",
    apply_lipsync: bool = True,
    seed: int = 42,
) -> Dict[str, Any]:
    """Ejecuta de principio a fin una escena cinematográfica estilo Higgsfield 100 % en local / Colab:
    1) Construye u obtiene el fotograma cinematográfico con óptica real + Casting Vault.
    2) Anima el plano con el movimiento de cámara Higgsfield elegido (`14B FP8`, `1.3B` o `FFmpeg CPU 1.5s`).
    3) Si hay `dialogue_lines`, genera las voces TTS, añade ambiente con Auto-Ducking y aplica LipSync HD si está instalado.
    """
    t0 = time.time()
    chars: List[Dict[str, Any]] = []
    for cid in character_ids or []:
        c_obj = character_vault.get_character(cid)
        if c_obj:
            chars.append(c_obj)

    compiled_img_prompt = compile_cinema_prompt(
        user_prompt=scene_prompt,
        camera_id=camera_id,
        lens_id=lens_id,
        focal_mm=focal_mm,
        aperture=aperture,
        lighting_id=lighting_id,
        characters=chars,
    )

    if aspect_ratio == "9:16":
        img_w, img_h, vid_w, vid_h = 704, 1280, 704, 1280
    elif aspect_ratio == "1:1":
        img_w, img_h, vid_w, vid_h = 1024, 1024, 960, 960
    else:
        img_w, img_h, vid_w, vid_h = 1280, 704, 832, 480

    # Paso 1: Fotograma inicial (reutilizar `base_image_rel`, anclar a personaje del Vault o crear plano cine nuevo)
    keyframe_path: Optional[Path] = resolve_rel_fn(base_image_rel) if base_image_rel else None
    keyframe_rel = base_image_rel or ""
    coherence_score: Optional[float] = None

    anchor_paths: List[Path] = []
    for ch in chars:
        ap = resolve_rel_fn(ch.get("anchor_image_rel")) if ch.get("anchor_image_rel") else None
        if ap and ap.exists():
            anchor_paths.append(ap)

    if keyframe_path is None:
        if video_engine_mode == "ffmpeg_instant" and anchor_paths:
            # En modo instantáneo 0 GB sin modelo de imagen en VRAM, animar directamente el retrato ancla
            keyframe_path = anchor_paths[0]
            keyframe_rel = chars[0]["anchor_image_rel"]
        else:
            job.log(
                f"🎬 [Higgsfield 1/4] Revelando fotograma Cinema Studio ({camera_id} + {lens_id} @ {focal_mm}mm {aperture})...",
                progress=8.0,
                step="Generando fotograma Cinema Studio...",
            )
            img_meta = generate_or_edit_image(
                prompt=compiled_img_prompt,
                model_id="qwen-image-edit-2511-pack",
                base_image_path=anchor_paths[0] if anchor_paths else None,
                reference_paths=anchor_paths[:3] if len(anchor_paths) > 1 else None,
                width=img_w,
                height=img_h,
                steps=4,
                guidance_scale=1.0,
                seed=seed,
                use_lightning_lora=True,
                job=job,
            )
            keyframe_rel = f"outputs/images/{img_meta['filename']}"
            keyframe_path = IMAGES_OUT_DIR / img_meta["filename"]
            if anchor_paths:
                cv_audit = analyze_pair_fast_cv(anchor_paths[0], keyframe_path)
                coherence_score = cv_audit["cv_score"]

    assert keyframe_path is not None and keyframe_path.exists()

    # Paso 2: Sintetizar diálogos si el usuario definió líneas habladas
    turns_for_lipsync: List[Dict[str, Any]] = []
    voice_tracks_for_mix: List[Dict[str, Any]] = []
    cursor_sec = 0.25
    for idx_d, dline in enumerate(dialogue_lines or []):
        txt = str(dline.get("text") or "").strip()
        if not txt:
            continue
        ch_match = character_vault.get_character(str(dline.get("character_id") or "")) if dline.get("character_id") else (
            chars[idx_d] if idx_d < len(chars) else None
        )
        v_id = str(dline.get("voice_id") or (ch_match.get("voice_id") if ch_match else "es-ES-AlvaroNeural"))
        t_face = str(dline.get("target_face") or (ch_match.get("default_face_side") if ch_match else "largest"))
        st_s = float(dline["start_sec"]) if dline.get("start_sec") is not None else cursor_sec

        job.log(
            f"🎙️ [Higgsfield 2/4] Locutando línea #{idx_d + 1} ({v_id}): «{txt[:48]}»...",
            progress=35.0 + idx_d * 5.0,
            step=f"Generando voz #{idx_d + 1}...",
        )
        v_meta = synthesize_voice(text=txt, engine_id="edge-tts-cloud", voice_id=v_id, job=None)
        v_path = resolve_rel_fn(f"outputs/audio/{v_meta['filename']}")
        v_dur = max(0.3, float(v_meta.get("duration_seconds") or 1.5))
        cursor_sec = st_s + v_dur + 0.30

        turns_for_lipsync.append(
            {
                "voice_path": v_path,
                "target_face": t_face,
                "start_sec": round(st_s, 2),
                "volume": 1.0,
            }
        )
        voice_tracks_for_mix.append(
            {
                "path": v_path,
                "start_sec": round(st_s, 2),
                "volume": 1.0,
            }
        )

    effective_dur = max(float(duration_seconds), cursor_sec + 0.20 if turns_for_lipsync else float(duration_seconds))

    # Paso 3: Animar con preset de cámara Higgsfield
    m_spec = HIGGSFIELD_MOTIONS.get(motion_id) or HIGGSFIELD_MOTIONS["dolly_in_dramatic"]
    motion_prompt = f"{compiled_img_prompt}, {m_spec['video_prompt']}"

    eff_mode = (video_engine_mode or "auto").strip().lower()
    if eff_mode == "auto":
        if catalog_manager.is_downloaded("wan21-vace-14b-fp8"):
            eff_mode = "ai_14b"
        elif catalog_manager.is_downloaded("wan21-vace-1.3b"):
            eff_mode = "ai_fast"
        else:
            eff_mode = "ffmpeg_instant"

    job.log(
        f"🌪️ [Higgsfield 3/4] Animando movimiento «{m_spec['label']}» ({eff_mode})...",
        progress=55.0,
        step=f"Animando cámara ({m_spec['label']})...",
    )

    if eff_mode in ("ai_14b", "ai_fast"):
        v_mid = "wan21-vace-14b-fp8" if eff_mode == "ai_14b" else "wan21-vace-1.3b"
        v_lora = "lora-wan21-14b-fusionx-phantom" if eff_mode == "ai_14b" else "lora-wan21-1.3b-bidirect2"
        if not catalog_manager.is_downloaded(v_lora):
            v_lora = ""
        vid_meta = generate_video_clip(
            mode="ai",
            video_submode="i2v",
            model_id=v_mid,
            video_lora_id=v_lora or None,
            prompt=motion_prompt,
            image_path=keyframe_path,
            width=vid_w,
            height=vid_h,
            duration_seconds=min(6.0, effective_dur),
            fps=16,
            steps=6 if v_lora else 24,
            guidance_scale=1.0 if v_lora else 5.0,
            seed=seed,
            job=job,
        )
        raw_vid_path = VIDEOS_OUT_DIR / vid_meta["filename"]
    else:
        VIDEOS_OUT_DIR.mkdir(parents=True, exist_ok=True)
        raw_name = f"hf_cam_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{job.id[:6]}.mp4"
        raw_vid_path = VIDEOS_OUT_DIR / raw_name
        render_higgsfield_cpu_motion_clip(
            image_path=keyframe_path,
            output_mp4=raw_vid_path,
            motion_id=motion_id,
            width=vid_w,
            height=vid_h,
            fps=24,
            duration_sec=effective_dur,
            job=job,
        )

    # Paso 4: Ambiente sonoro opcional + LipSync HD o Mezcla Auto-Ducking
    amb_path: Optional[Path] = None
    if ambience_preset and ambience_preset not in ("none", "off", ""):
        amb_res = audio_studio_engine.generate_music_or_ambience(
            job=job,
            prompt=scene_prompt,
            model_id="procedural-ambience-cpu",
            preset=ambience_preset,
            duration_seconds=effective_dur + 1.0,
            seed=seed,
        )
        amb_path = resolve_rel_fn(amb_res["rel_path"])

    lipsync_used = False
    if turns_for_lipsync and apply_lipsync and catalog_manager.is_downloaded("edtalk-256-lipsync-hd"):
        job.log(
            "👄 [Higgsfield 4/4] Aplicando LipSync HD multi-personaje + Auto-Ducking...",
            progress=80.0,
            step="Sincronizando labios en HD...",
        )
        ls_out = lipsync_engine.render_lipsync(
            job=job,
            media_path=raw_vid_path,
            dialogue_turns=turns_for_lipsync,
            ambient_path=amb_path,
            ambient_volume=0.18,
            extend_video_to_audio=True,
        )
        final_rel = ls_out["rel_path"]
        final_url = ls_out["media_url"]
        final_name = ls_out["filename"]
        final_dur = ls_out["duration_seconds"]
        lipsync_used = True
    elif voice_tracks_for_mix or amb_path is not None:
        out_name = f"higgsfield_scene_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{job.id[:6]}.mp4"
        out_path = VIDEOS_OUT_DIR / out_name
        mix_info = mix_and_master_media(
            output_path=out_path,
            video_path=raw_vid_path,
            voice_tracks=voice_tracks_for_mix,
            ambient_path=amb_path,
            ambient_volume=0.18,
            auto_duck_bg=True,
            extend_video_to_audio=True,
            job=job,
        )
        final_rel = f"outputs/videos/{out_name}"
        final_url = f"/media/{final_rel}"
        final_name = out_name
        final_dur = mix_info["duration_seconds"]
    else:
        final_name = raw_vid_path.name
        final_rel = f"outputs/videos/{final_name}"
        final_url = f"/media/{final_rel}"
        final_dur = probe_media_info(raw_vid_path)["duration_seconds"]

    meta = {
        "id": Path(final_name).stem,
        "filename": final_name,
        "rel_path": final_rel,
        "media_url": final_url,
        "type": "video",
        "mode": "higgsfield_director",
        "keyframe_rel": keyframe_rel,
        "compiled_cinema_prompt": compiled_img_prompt,
        "camera_id": camera_id,
        "lens_id": lens_id,
        "focal_mm": focal_mm,
        "aperture": aperture,
        "lighting_id": lighting_id,
        "motion_id": motion_id,
        "video_engine_used": eff_mode,
        "lipsync_used": lipsync_used,
        "coherence_cv_score": coherence_score,
        "duration_seconds": final_dur,
        "width": vid_w,
        "height": vid_h,
        "elapsed_seconds": round(time.time() - t0, 2),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    (VIDEOS_OUT_DIR / f"{Path(final_name).stem}.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    job.log(
        f"🎬 Escena Higgsfield completada: {final_name} ({final_dur:.2f}s).",
        level="success",
        progress=100.0,
        step="Escena Higgsfield lista",
    )
    return meta


def get_higgsfield_catalog_payload() -> Dict[str, Any]:
    return {
        "cameras": CINEMA_CAMERAS,
        "lenses": CINEMA_LENSES,
        "focal_lengths": FOCAL_LENGTHS,
        "apertures": APERTURES,
        "lighting_moods": LIGHTING_MOODS,
        "motions": HIGGSFIELD_MOTIONS,
        "characters": character_vault.list_characters(),
    }
