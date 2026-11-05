import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageFilter

from studio.config import TEMP_DIR
from studio.jobs import JobRecord


def get_ffmpeg_bin() -> str:
    sys_ff = shutil.which("ffmpeg")
    if sys_ff:
        return sys_ff
    try:
        import imageio_ffmpeg  # type: ignore

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and Path(exe).exists():
            return exe
    except Exception:
        pass
    raise RuntimeError("No se encontró el binario de FFmpeg ni en el sistema ni en imageio-ffmpeg.")


def probe_media_info(file_path: Path) -> Dict[str, Any]:
    """Inspecciona un archivo multimedia con ffmpeg -i y extrae duración, resolución y pistas."""
    if not file_path.exists() or file_path.stat().st_size == 0:
        raise RuntimeError(f"El archivo no existe o está vacío: {file_path}")

    ffmpeg_bin = get_ffmpeg_bin()
    res = subprocess.run(
        [ffmpeg_bin, "-hide_banner", "-i", str(file_path)],
        capture_output=True,
        text=True,
        timeout=20,
    )
    info_text = (res.stderr or "") + "\n" + (res.stdout or "")

    dur_sec = 0.0
    m_dur = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", info_text)
    if m_dur:
        h, m, s = int(m_dur.group(1)), int(m_dur.group(2)), float(m_dur.group(3))
        dur_sec = round(h * 3600 + m * 60 + s, 2)

    width, height = None, None
    m_res = re.search(r"Video:.*?\s(\d{2,5})x(\d{2,5})(?:[\s,\[])", info_text)
    if m_res:
        width, height = int(m_res.group(1)), int(m_res.group(2))

    has_video = "Video:" in info_text
    has_audio = "Audio:" in info_text

    if not has_video and not has_audio:
        raise RuntimeError(f"FFmpeg no reconoce pistas válidas en {file_path.name}")

    return {
        "exists": True,
        "size_bytes": file_path.stat().st_size,
        "duration_seconds": dur_sec,
        "width": width,
        "height": height,
        "has_video": has_video,
        "has_audio": has_audio,
    }


def verify_image_file(file_path: Path) -> Tuple[int, int]:
    """Verifica que una imagen existe en disco, no está vacía y se abre íntegra con PIL."""
    if not file_path.exists() or file_path.stat().st_size == 0:
        raise RuntimeError(f"El archivo de imagen no existe o está vacío: {file_path}")
    with Image.open(file_path) as im:
        im.verify()
    with Image.open(file_path) as im2:
        return im2.size


def process_pil_bt709(image_pil: Image.Image) -> Image.Image:
    """
    Procesamiento FFmpeg exacto (limpieza de metadatos + yuvj444p + espacio de color BT.709)
    tomado del flujo de Qwen-Image-Edit del usuario.
    """
    ffmpeg_bin = get_ffmpeg_bin()
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    fd_in, temp_in = tempfile.mkstemp(suffix=".png", dir=str(TEMP_DIR))
    fd_out, temp_out = tempfile.mkstemp(suffix=".jpg", dir=str(TEMP_DIR))
    os.close(fd_in)
    os.close(fd_out)
    try:
        image_pil.convert("RGB").save(temp_in, "PNG")
        cmd = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            temp_in,
            "-map_metadata",
            "-1",
            "-vf",
            "format=yuvj444p",
            "-colorspace",
            "bt709",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            "-q:v",
            "2",
            "-y",
            temp_out,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if res.returncode != 0:
            raise RuntimeError(f"Error FFmpeg BT.709: {res.stderr.strip()}")
        with Image.open(temp_out) as processed:
            out_img = processed.convert("RGB").copy()
        return out_img
    finally:
        for p in (temp_in, temp_out):
            try:
                if os.path.exists(p):
                    os.unlink(p)
            except OSError:
                pass


def process_image_file_bt709(file_path: Path) -> Image.Image:
    with Image.open(file_path) as im:
        rgb = im.convert("RGB").copy()
    return process_pil_bt709(rgb)


def apply_inpaint_composite(
    original: Image.Image,
    edited: Image.Image,
    mask: Image.Image,
    feather_px: int = 8,
) -> Image.Image:
    """
    Combina el resultado editado sobre la imagen original respetando al 100% los píxeles
    fuera de la máscara blanca (con transición suave opcional en el borde).
    """
    orig_rgb = original.convert("RGB")
    edit_rgb = edited.convert("RGB").resize(orig_rgb.size, Image.Resampling.LANCZOS)
    mask_l = mask.convert("L").resize(orig_rgb.size, Image.Resampling.BILINEAR)
    # Normalizar máscara binaria primero y aplicar suavizado si se pide
    mask_l = mask_l.point(lambda p: 255 if p > 32 else 0)
    if feather_px > 0:
        mask_l = mask_l.filter(ImageFilter.GaussianBlur(radius=float(feather_px)))
    return Image.composite(edit_rgb, orig_rgb, mask_l)


def create_motion_clip_from_image(
    image_path: Path,
    output_mp4: Path,
    duration_sec: float = 4.0,
    width: int = 1280,
    height: int = 720,
    fps: int = 24,
    motion: str = "zoom_in",
    audio_path: Optional[Path] = None,
    job: Optional[JobRecord] = None,
) -> Dict[str, Any]:
    """Convierte una imagen fija en un clip MP4 H.264 BT.709 con movimiento suave opcional y/o audio."""
    ffmpeg_bin = get_ffmpeg_bin()
    verify_image_file(image_path)

    # Asegurar dimensiones pares para yuv420p
    width = max(256, (int(width) // 2) * 2)
    height = max(256, (int(height) // 2) * 2)
    fps = max(12, min(60, int(fps)))

    if audio_path and audio_path.exists():
        a_info = probe_media_info(audio_path)
        if a_info["duration_seconds"] > 0.2:
            duration_sec = max(duration_sec, a_info["duration_seconds"])

    duration_sec = max(0.5, min(120.0, float(duration_sec)))
    total_frames = max(12, int(round(duration_sec * fps)))

    # Construir filtro de vídeo seguro y nítido
    base_pad = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1"
    )
    if motion == "zoom_in":
        vf = (
            f"{base_pad},scale={width*2}:{height*2},"
            f"zoompan=z='min(zoom+0.0012,1.18)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
            f":d={total_frames}:s={width}x{height}:fps={fps},format=yuv420p"
        )
    elif motion == "zoom_out":
        vf = (
            f"{base_pad},scale={width*2}:{height*2},"
            f"zoompan=z='if(eq(on,1),1.18,max(1.0,zoom-0.0012))':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
            f":d={total_frames}:s={width}x{height}:fps={fps},format=yuv420p"
        )
    elif motion == "pan_right":
        vf = (
            f"{base_pad},scale={int(width*1.24)}:{int(height*1.24)},"
            f"zoompan=z='1.12':x='(iw-iw/zoom)*(on/{max(1, total_frames)})':y='(ih-ih/zoom)/2'"
            f":d={total_frames}:s={width}x{height}:fps={fps},format=yuv420p"
        )
    else:
        vf = f"{base_pad},fps={fps},format=yuv420p"

    cmd = [ffmpeg_bin, "-hide_banner", "-loglevel", "error"]
    if motion in ("zoom_in", "zoom_out", "pan_right"):
        cmd += ["-i", str(image_path)]
    else:
        cmd += ["-loop", "1", "-t", f"{duration_sec:.2f}", "-i", str(image_path)]

    if audio_path and audio_path.exists():
        cmd += ["-i", str(audio_path)]
    else:
        cmd += ["-f", "lavfi", "-t", f"{duration_sec:.2f}", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100"]

    cmd += [
        "-t",
        f"{duration_sec:.2f}",
        "-vf",
        vf,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-colorspace",
        "bt709",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-ar",
        "44100",
        "-ac",
        "2",
        "-movflags",
        "+faststart",
        "-shortest",
        "-y",
        str(output_mp4),
    ]

    if job:
        job.log(
            f"Renderizando clip MP4 ({width}x{height} @ {fps}fps, {duration_sec:.1f}s, efecto={motion})...",
            step="Codificando clip con FFmpeg...",
        )

    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if res.returncode != 0:
        raise RuntimeError(f"FFmpeg falló al crear clip: {res.stderr.strip()}")

    return probe_media_info(output_mp4)


def render_montage_mp4(
    items: List[Dict[str, Any]],
    output_mp4: Path,
    width: int = 1280,
    height: int = 720,
    fps: int = 24,
    global_audio_path: Optional[Path] = None,
    global_audio_volume: float = 0.35,
    fade_seconds: float = 0.25,
    job: Optional[JobRecord] = None,
) -> Dict[str, Any]:
    """
    Monta una lista ordenada de planos (imágenes y/o clips de vídeo + audio por plano opcional)
    en un único MP4 final verificado con FFmpeg.
    """
    if not items:
        raise ValueError("Añade al menos una imagen o clip de vídeo al montaje.")

    ffmpeg_bin = get_ffmpeg_bin()
    width = max(256, (int(width) // 2) * 2)
    height = max(256, (int(height) // 2) * 2)
    fps = max(12, min(60, int(fps)))

    work_dir = Path(tempfile.mkdtemp(prefix="montage_", dir=str(TEMP_DIR)))
    segment_paths: List[Path] = []

    try:
        total = len(items)
        for idx, item in enumerate(items, start=1):
            if job:
                job.check_cancelled()
                pct = 5.0 + ((idx - 1) / total) * 70.0
                job.log(
                    f"[Plano {idx}/{total}] Preparando «{Path(item['path']).name}»...",
                    progress=pct,
                    step=f"Normalizando plano {idx}/{total}",
                )

            src_path = Path(item["path"])
            if not src_path.exists():
                raise RuntimeError(f"No se encuentra el archivo del plano {idx}: {src_path}")

            seg_out = work_dir / f"seg_{idx:03d}.mp4"
            shot_audio = Path(item["audio_path"]) if item.get("audio_path") else None
            if shot_audio and not shot_audio.exists():
                shot_audio = None

            ext = src_path.suffix.lower()
            is_image = ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp")

            if is_image:
                dur = float(item.get("duration", 3.5))
                motion = str(item.get("motion", "none"))
                create_motion_clip_from_image(
                    image_path=src_path,
                    output_mp4=seg_out,
                    duration_sec=dur,
                    width=width,
                    height=height,
                    fps=fps,
                    motion=motion,
                    audio_path=shot_audio,
                    job=None,
                )
            else:
                v_info = probe_media_info(src_path)
                clip_dur = float(item.get("duration") or v_info["duration_seconds"] or 4.0)
                if shot_audio:
                    a_info = probe_media_info(shot_audio)
                    if a_info["duration_seconds"] > 0.2 and item.get("fit_to_audio", True):
                        clip_dur = max(clip_dur, a_info["duration_seconds"])

                fade_filter = ""
                if fade_seconds > 0 and clip_dur > fade_seconds * 2.5:
                    fade_out_st = max(0.0, clip_dur - fade_seconds)
                    fade_filter = f",fade=t=in:st=0:d={fade_seconds:.2f},fade=t=out:st={fade_out_st:.2f}:d={fade_seconds:.2f}"

                vf = (
                    f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                    f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,"
                    f"fps={fps},format=yuv420p{fade_filter}"
                )

                cmd = [ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-i", str(src_path)]
                if shot_audio:
                    cmd += ["-i", str(shot_audio), "-map", "0:v:0", "-map", "1:a:0"]
                elif v_info["has_audio"]:
                    cmd += ["-map", "0:v:0", "-map", "0:a:0"]
                else:
                    cmd += [
                        "-f",
                        "lavfi",
                        "-t",
                        f"{clip_dur:.2f}",
                        "-i",
                        "anullsrc=channel_layout=stereo:sample_rate=44100",
                        "-map",
                        "0:v:0",
                        "-map",
                        "1:a:0",
                    ]

                cmd += [
                    "-t",
                    f"{clip_dur:.2f}",
                    "-vf",
                    vf,
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "20",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    "-ar",
                    "44100",
                    "-ac",
                    "2",
                    "-y",
                    str(seg_out),
                ]
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
                if res.returncode != 0:
                    raise RuntimeError(f"Error FFmpeg normalizando vídeo {src_path.name}: {res.stderr.strip()}")

            # Si es imagen y fade está activado, aplicar fade rápido si procede
            probe_media_info(seg_out)
            segment_paths.append(seg_out)

        # Paso 2: Concatenar todos los segmentos normalizados
        if job:
            job.check_cancelled()
            job.log(
                f"Uniendo {len(segment_paths)} plano(s) en línea de tiempo...",
                progress=80.0,
                step="Concatenando planos...",
            )

        concat_txt = work_dir / "concat_list.txt"
        concat_lines = [f"file '{p.as_posix()}'" for p in segment_paths]
        concat_txt.write_text("\n".join(concat_lines), encoding="utf-8")

        joined_mp4 = work_dir / "joined.mp4"
        cmd_concat = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_txt),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "19",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            "-y",
            str(joined_mp4),
        ]
        res_cat = subprocess.run(cmd_concat, capture_output=True, text=True, timeout=300)
        if res_cat.returncode != 0:
            raise RuntimeError(f"Error al concatenar planos: {res_cat.stderr.strip()}")

        output_mp4.parent.mkdir(parents=True, exist_ok=True)

        # Paso 3: Mezclar pista de audio/música global si se proporcionó
        if global_audio_path and global_audio_path.exists():
            if job:
                job.log(
                    f"Mezclando pista de audio global ({global_audio_path.name})...",
                    progress=92.0,
                    step="Mezclando audio final...",
                )
            joined_info = probe_media_info(joined_mp4)
            total_dur = max(0.5, joined_info["duration_seconds"])
            vol = max(0.05, min(2.0, float(global_audio_volume)))
            cmd_mix = [
                ffmpeg_bin,
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(joined_mp4),
                "-stream_loop",
                "-1",
                "-i",
                str(global_audio_path),
                "-t",
                f"{total_dur:.2f}",
                "-filter_complex",
                f"[1:a]volume={vol:.2f}[bg];[0:a][bg]amix=inputs=2:duration=first:dropout_transition=2[aout]",
                "-map",
                "0:v:0",
                "-map",
                "[aout]",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-movflags",
                "+faststart",
                "-y",
                str(output_mp4),
            ]
            res_mix = subprocess.run(cmd_mix, capture_output=True, text=True, timeout=240)
            if res_mix.returncode != 0:
                raise RuntimeError(f"Error mezclando audio global: {res_mix.stderr.strip()}")
        else:
            shutil.move(str(joined_mp4), str(output_mp4))

        final_info = probe_media_info(output_mp4)
        if job:
            job.log(
                f"MP4 final verificado: {output_mp4.name} ({final_info['duration_seconds']}s, "
                f"{final_info['size_bytes'] / (1024**2):.2f} MB).",
                level="success",
                progress=99.0,
                step="Verificando archivo final...",
            )
        return final_info
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
