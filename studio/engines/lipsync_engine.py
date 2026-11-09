"""Motor HD de LipSync y Escenas Habladas Multi-Personaje (Studio IA).

Arquitectura ONNX acelerada en GPU/CPU (`edtalk-256-lipsync-hd`, 0.65 GB en disco, ~1.5 GB VRAM):
- `yunet_2023_mar.onnx`: detección facial multi-rostro + 5 landmarks (`ojo_izq`, `ojo_der`, `nariz`, `comisura_izq`, `comisura_der`)
  con seguimiento temporal suavizado (EMA) y selector espacial (`largest`, `leftmost`, `rightmost`, `center`, `index_0`...).
- `edtalk_256.onnx` (Apache-2.0): animación labial y del tercio inferior facial a 256×256 guiada por espectrograma Mel
  de 80 bandas a 16 kHz (agnóstico al idioma: español, inglés, etc.).
- `xseg_1.onnx`: máscara semántica de oclusión facial para respetar copas, tazas, tenedores, manos o gafas delante de la boca.
- `gfpgan_1.4.onnx`: restauración facial HD a 512×512 mezclada con suavidad (`restore_strength`) para dientes y labios nítidos.

Soporta:
1. Imagen fija + voz → genera micro-movimiento orgánico de cámara/respiración + LipSync completo.
2. Vídeo IA existente (p.ej. `Wan 2.1 VACE 14B` en restaurante) + 1 voz o conversación por turnos (`dialogue_turns`)
   donde el personaje de la izquierda (`leftmost`) habla en su turno y el de la derecha (`rightmost`) responde en el suyo,
   más mezcla automática con música y sonido ambiente con Auto-Ducking.
"""

from __future__ import annotations

import gc
import json
import subprocess
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from studio.catalog import CatalogManager, catalog_manager
from studio.config import VIDEOS_OUT_DIR
from studio.engines.audio_studio_engine import mix_and_master_media
from studio.engines.ffmpeg_utils import (
    create_motion_clip_from_image,
    get_ffmpeg_bin,
    probe_media_info,
)
from studio.jobs import JobRecord
from studio.memory_manager import memory_manager

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


# Plantilla canónica FFHQ 512 (coordenadas normalizadas en [0, 1] para 5 puntos faciales)
FFHQ_512_TEMPLATE = np.array(
    [
        [0.37691676, 0.46864664],  # ojo izquierdo en imagen
        [0.62285697, 0.46912813],  # ojo derecho en imagen
        [0.50123859, 0.61331904],  # punta de la nariz
        [0.39308822, 0.72541100],  # comisura izquierda en imagen
        [0.61150205, 0.72490465],  # comisura derecha en imagen
    ],
    dtype=np.float32,
)


def decode_audio_16k_mono(audio_path: Path) -> np.ndarray:
    """Decodifica cualquier archivo de audio (.mp3, .wav, .m4a, .ogg) a PCM float32 mono a 16000 Hz vía FFmpeg."""
    ffmpeg_bin = get_ffmpeg_bin()
    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(audio_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "f32le",
        "pipe:1",
    ]
    res = subprocess.run(cmd, capture_output=True, timeout=60)
    if res.returncode != 0 or not res.stdout:
        err = res.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Error decodificando audio para LipSync ({audio_path.name}): {err}")
    samples = np.frombuffer(res.stdout, dtype=np.float32).copy()
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak > 1.0:
        samples /= peak
    return samples


def _hz_to_mel(freq_hz: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(freq_hz, dtype=np.float32) / 700.0)


def _mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float32) / 2595.0) - 1.0)


def _build_mel_filterbank_80(sr: int = 16000, n_fft: int = 800, n_mels: int = 80) -> np.ndarray:
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    mels = np.linspace(_hz_to_mel(55.0), _hz_to_mel(7600.0), n_mels + 2, dtype=np.float32)
    bins = np.floor((n_fft + 1) * _mel_to_hz(mels) / float(sr)).astype(int)
    for i in range(n_mels):
        left, center, right = int(bins[i]), int(bins[i + 1]), int(bins[i + 2])
        if center > left:
            fb[i, left:center] = (np.arange(left, center, dtype=np.float32) - left) / float(center - left)
        if right > center:
            fb[i, center:right] = (right - np.arange(center, right, dtype=np.float32)) / float(right - center)
    return fb


_MEL_FB_80 = _build_mel_filterbank_80()


def extract_edtalk_mel_spectrogram(audio_16k: np.ndarray) -> np.ndarray:
    """Extrae el espectrograma Mel [80, T] calibrado exactamente para `edtalk_256.onnx` (80 pasos/s)."""
    if audio_16k.size < 800:
        audio_16k = np.pad(audio_16k, (0, 800 - audio_16k.size), mode="constant")

    pre = np.empty_like(audio_16k, dtype=np.float32)
    pre[0] = audio_16k[0]
    pre[1:] = audio_16k[1:] - 0.97 * audio_16k[:-1]

    nperseg = 800
    hop = 200  # 16000 / 200 = 80 vectores Mel por segundo
    n_idx = np.arange(nperseg, dtype=np.float32)
    win = (0.5 - 0.5 * np.cos(2.0 * np.pi * n_idx / float(nperseg))).astype(np.float32)
    win_sum = float(win.sum())

    pad_right = nperseg // 2 + hop
    padded = np.pad(pre, (nperseg // 2, pad_right), mode="constant")
    frames = np.lib.stride_tricks.sliding_window_view(padded, nperseg)[::hop]
    stft_mag = np.abs(np.fft.rfft(frames * win[None, :], n=nperseg, axis=-1)).T.astype(np.float32) / max(1e-6, win_sum)

    mel = np.dot(_MEL_FB_80, stft_mag)
    mel_db = np.clip(np.log10(np.maximum(1e-5, mel)) * 1.6 + 3.2, -4.0, 4.0).astype(np.float32)
    return mel_db


def slice_mel_window_for_frame(mel_spec: np.ndarray, local_frame_idx: int, fps: float, window_len: int = 16) -> np.ndarray:
    """Extrae la ventana [1, 1, 80, 16] correspondiente al fotograma `local_frame_idx` a `fps`."""
    total_cols = mel_spec.shape[1]
    step = 80.0 / max(1.0, float(fps))
    start = int(round(local_frame_idx * step))
    end = start + window_len
    if end > total_cols:
        start = max(0, total_cols - window_len)
        end = total_cols
    chunk = mel_spec[:, start:end]
    if chunk.shape[1] < window_len:
        chunk = np.pad(chunk, ((0, 0), (0, window_len - chunk.shape[1])), mode="edge")
    return chunk[None, None, :, :].astype(np.float32)


def _create_static_anchors(stride: int, h: int, w: int) -> np.ndarray:
    grid_x, grid_y = np.mgrid[:w, :h]
    anchors = np.stack((grid_y, grid_x), axis=-1).astype(np.float32)
    return (anchors * float(stride)).reshape(-1, 2)


def _greedy_nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float = 0.4) -> List[int]:
    if len(boxes) == 0:
        return []
    order = np.argsort(-scores)
    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        area_i = max(1.0, (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1]))
        area_r = np.maximum(1.0, (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1]))
        iou = inter / (area_i + area_r - inter + 1e-6)
        order = rest[iou <= iou_thresh]
    return keep


def detect_faces_yunet_onnx(
    yunet_sess: Any,
    frame_bgr: np.ndarray,
    det_size: int = 640,
    score_thresh: float = 0.45,
) -> List[Dict[str, Any]]:
    """Detecta todos los rostros en `frame_bgr` devolviendo bbox, score y `landmarks_5` [5, 2]."""
    orig_h, orig_w = frame_bgr.shape[:2]
    scale = min(float(det_size) / max(1, orig_w), float(det_size) / max(1, orig_h), 1.0)
    resized_w = max(16, int(round(orig_w * scale)))
    resized_h = max(16, int(round(orig_h * scale)))
    temp = cv2.resize(frame_bgr, (resized_w, resized_h), interpolation=cv2.INTER_AREA) if scale < 1.0 else frame_bgr

    ratio_w = float(orig_w) / float(temp.shape[1])
    ratio_h = float(orig_h) / float(temp.shape[0])

    canvas = np.zeros((det_size, det_size, 3), dtype=np.float32)
    canvas[: temp.shape[0], : temp.shape[1], :] = temp.astype(np.float32)
    inp = np.expand_dims(canvas.transpose(2, 0, 1), axis=0)

    in_name = yunet_sess.get_inputs()[0].name
    outputs = yunet_sess.run(None, {in_name: inp})

    strides = [8, 16, 32]
    fmc = 3
    all_boxes: List[np.ndarray] = []
    all_scores: List[float] = []
    all_kps: List[np.ndarray] = []

    for idx, stride in enumerate(strides):
        scores_raw = (outputs[idx] * outputs[idx + fmc]).reshape(-1)
        keep = np.where(scores_raw >= score_thresh)[0]
        if not np.any(keep):
            continue
        sh = det_size // stride
        sw = det_size // stride
        anchors = _create_static_anchors(stride, sh, sw)
        reg = outputs[idx + fmc * 2].squeeze(0)
        centers = reg[:, :2] * stride + anchors
        sizes = np.exp(np.clip(reg[:, 2:4], -10.0, 10.0)) * stride
        kps_raw = outputs[idx + fmc * 3].squeeze(0)

        boxes_s = np.stack(
            [
                (centers[:, 0] - sizes[:, 0] * 0.5) * ratio_w,
                (centers[:, 1] - sizes[:, 1] * 0.5) * ratio_h,
                (centers[:, 0] + sizes[:, 0] * 0.5) * ratio_w,
                (centers[:, 1] + sizes[:, 1] * 0.5) * ratio_h,
            ],
            axis=-1,
        )
        kps_s = np.concatenate(
            [
                (kps_raw[:, [0, 1]] * stride + anchors),
                (kps_raw[:, [2, 3]] * stride + anchors),
                (kps_raw[:, [4, 5]] * stride + anchors),
                (kps_raw[:, [6, 7]] * stride + anchors),
                (kps_raw[:, [8, 9]] * stride + anchors),
            ],
            axis=-1,
        ).reshape(-1, 5, 2)

        for ki in keep:
            all_boxes.append(boxes_s[ki])
            all_scores.append(float(scores_raw[ki]))
            all_kps.append(kps_s[ki] * np.array([ratio_w, ratio_h], dtype=np.float32))

    if not all_boxes:
        return []

    boxes_arr = np.asarray(all_boxes, dtype=np.float32)
    scores_arr = np.asarray(all_scores, dtype=np.float32)
    keep_nms = _greedy_nms(boxes_arr, scores_arr, iou_thresh=0.40)

    faces: List[Dict[str, Any]] = []
    for ki in keep_nms:
        b = boxes_arr[ki]
        w_box = max(1.0, float(b[2] - b[0]))
        h_box = max(1.0, float(b[3] - b[1]))
        faces.append(
            {
                "bbox": b,
                "score": float(scores_arr[ki]),
                "landmarks_5": all_kps[ki].astype(np.float32),
                "area": w_box * h_box,
                "center_x": float(b[0] + b[2]) * 0.5,
                "center_y": float(b[1] + b[3]) * 0.5,
            }
        )
    return faces


def select_target_face(
    faces: List[Dict[str, Any]],
    selector: str = "largest",
    frame_width: int = 832,
    prev_center: Optional[Tuple[float, float]] = None,
) -> Optional[Dict[str, Any]]:
    """Selecciona el personaje objetivo (`largest`, `leftmost`, `rightmost`, `center`, `index_0`...) con tracking."""
    if not faces:
        return None
    if prev_center is not None:
        px, py = prev_center
        # Mantener identidad espacial si hay un rostro cercano al fotograma previo
        ranked = sorted(faces, key=lambda f: (f["center_x"] - px) ** 2 + (f["center_y"] - py) ** 2)
        nearest = ranked[0]
        max_jump = max(90.0, frame_width * 0.22)
        if ((nearest["center_x"] - px) ** 2 + (nearest["center_y"] - py) ** 2) ** 0.5 <= max_jump:
            return nearest

    sel = (selector or "largest").strip().lower()
    by_x = sorted(faces, key=lambda f: f["center_x"])
    if sel in ("left", "leftmost", "izquierda"):
        return by_x[0]
    if sel in ("right", "rightmost", "derecha"):
        return by_x[-1]
    if sel in ("center", "middle", "centro"):
        mid_x = frame_width * 0.5
        return min(faces, key=lambda f: abs(f["center_x"] - mid_x))
    if sel.startswith("index_") or sel.isdigit():
        idx_str = sel.replace("index_", "")
        idx = max(0, min(len(by_x) - 1, int(idx_str) if idx_str.isdigit() else 0))
        return by_x[idx]
    return max(faces, key=lambda f: f["area"])


def warp_face_ffhq_512(frame_bgr: np.ndarray, landmarks_5: np.ndarray, crop_size: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    norm_template = FFHQ_512_TEMPLATE * float(crop_size)
    affine_mat, _ = cv2.estimateAffinePartial2D(
        landmarks_5.astype(np.float32),
        norm_template,
        method=cv2.RANSAC,
        ransacReprojThreshold=float(crop_size) * 0.10,
    )
    if affine_mat is None:
        affine_mat, _ = cv2.estimateAffinePartial2D(landmarks_5.astype(np.float32), norm_template, method=cv2.LMEDS)
    crop = cv2.warpAffine(
        frame_bgr,
        affine_mat,
        (crop_size, crop_size),
        flags=cv2.INTER_AREA,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return crop, affine_mat


def create_feathered_mouth_face_mask(crop_size: int = 512, blur_ratio: float = 0.28) -> np.ndarray:
    """Máscara suave centrada en el rostro/tercio inferior para integrar LipSync sin bordes visibles."""
    mask = np.ones((crop_size, crop_size), dtype=np.float32)
    # Proteger frente superior/pelo (top 24%) y bordes laterales (8%) para preservar 100% los ojos/pelo originales
    top_pad = int(crop_size * 0.24)
    bot_pad = int(crop_size * 0.05)
    side_pad = int(crop_size * 0.08)
    mask[:top_pad, :] = 0.0
    mask[-bot_pad:, :] = 0.0
    mask[:, :side_pad] = 0.0
    mask[:, -side_pad:] = 0.0
    sigma = max(4.0, crop_size * 0.5 * blur_ratio * 0.25)
    mask = cv2.GaussianBlur(mask, (0, 0), sigma).clip(0.0, 1.0)
    return mask


def compute_xseg_occlusion_mask(xseg_sess: Optional[Any], crop_512_bgr: np.ndarray) -> np.ndarray:
    if xseg_sess is None:
        return np.ones((512, 512), dtype=np.float32)
    inp = cv2.resize(crop_512_bgr, (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    inp = inp[None, :, :, :]  # [1, 256, 256, 3]
    in_name = xseg_sess.get_inputs()[0].name
    raw = xseg_sess.run(None, {in_name: inp})[0]
    mask = np.squeeze(raw).clip(0.0, 1.0).astype(np.float32)
    mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_LINEAR)
    mask = ((cv2.GaussianBlur(mask, (0, 0), 5.0).clip(0.5, 1.0) - 0.5) * 2.0).astype(np.float32)
    return mask


def run_edtalk_and_gfpgan_on_crop(
    edtalk_sess: Any,
    gfpgan_sess: Optional[Any],
    crop_512_bgr: np.ndarray,
    mel_window: np.ndarray,
    lip_weight: float = 0.85,
    restore_strength: float = 0.50,
) -> np.ndarray:
    # 1. EDTalk 256x256
    crop_256 = cv2.resize(crop_512_bgr, (256, 256), interpolation=cv2.INTER_AREA)
    rgb_in = (crop_256[:, :, ::-1].astype(np.float32) / 255.0).transpose(2, 0, 1)[None, :, :, :]
    w_arr = np.array([float(np.clip(lip_weight, 0.2, 1.2))], dtype=np.float32)
    out_ed = edtalk_sess.run(
        None,
        {
            "source": mel_window,
            "target": rgb_in,
            "weight": w_arr,
        },
    )[0]
    ed_bgr_256 = np.clip(out_ed[0].transpose(1, 2, 0)[:, :, ::-1] * 255.0, 0.0, 255.0).astype(np.uint8)
    synced_512 = cv2.resize(ed_bgr_256, (512, 512), interpolation=cv2.INTER_CUBIC)

    # 2. Restauración HD opcional con GFPGAN 1.4 para labios/dientes nítidos a 512x512
    if gfpgan_sess is not None and restore_strength > 0.02:
        gfp_in = ((synced_512[:, :, ::-1].astype(np.float32) / 255.0) - 0.5) / 0.5
        gfp_in = gfp_in.transpose(2, 0, 1)[None, :, :, :]
        in_name = gfpgan_sess.get_inputs()[0].name
        gfp_out = gfpgan_sess.run(None, {in_name: gfp_in})[0][0]
        gfp_bgr = np.clip(((gfp_out.transpose(1, 2, 0)[:, :, ::-1] + 1.0) * 0.5) * 255.0, 0.0, 255.0).astype(np.uint8)
        alpha = float(np.clip(restore_strength, 0.0, 1.0))
        synced_512 = cv2.addWeighted(gfp_bgr, alpha, synced_512, 1.0 - alpha, 0.0)

    return synced_512


def paste_back_crop(
    frame_bgr: np.ndarray,
    synced_crop_512: np.ndarray,
    blend_mask_512: np.ndarray,
    affine_mat: np.ndarray,
) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    ch, cw = synced_crop_512.shape[:2]
    inv_mat = cv2.invertAffineTransform(affine_mat)
    corners = np.array([[0, 0], [cw, 0], [cw, ch], [0, ch]], dtype=np.float32).reshape(-1, 1, 2)
    pts = cv2.transform(corners, inv_mat).reshape(-1, 2)
    x1, y1 = np.clip(np.floor(pts.min(axis=0)).astype(int), 0, [w, h])
    x2, y2 = np.clip(np.ceil(pts.max(axis=0)).astype(int), 0, [w, h])
    pw, ph = int(x2 - x1), int(y2 - y1)
    if pw <= 2 or ph <= 2:
        return frame_bgr

    paste_mat = inv_mat.copy()
    paste_mat[0, 2] -= float(x1)
    paste_mat[1, 2] -= float(y1)

    inv_mask = cv2.warpAffine(blend_mask_512, paste_mat, (pw, ph), flags=cv2.INTER_LINEAR).clip(0.0, 1.0)[:, :, None]
    inv_crop = cv2.warpAffine(synced_crop_512, paste_mat, (pw, ph), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    out = frame_bgr.copy()
    roi = out[y1:y2, x1:x2].astype(np.float32)
    blended = roi * (1.0 - inv_mask) + inv_crop.astype(np.float32) * inv_mask
    out[y1:y2, x1:x2] = np.clip(blended, 0.0, 255.0).astype(np.uint8)
    return out


class LipSyncEngine:
    """Motor gestionado de sincronización labial HD (`edtalk-256-lipsync-hd`) con soporte multi-personaje."""

    def __init__(self, catalog: CatalogManager) -> None:
        self.catalog = catalog
        self._lock = threading.Lock()
        self.loaded_model_id: Optional[str] = None
        self._sessions: Dict[str, Any] = {}
        self._provider_used: str = "CPUExecutionProvider"

    def unload(self) -> None:
        with self._lock:
            self._sessions.clear()
            self.loaded_model_id = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _ensure_loaded(self, model_id: str, job: Optional[JobRecord] = None) -> Dict[str, Any]:
        definition = self.catalog.get_definition(model_id)
        if definition and definition.get("validation_state") == "candidate":
            raise ValueError("MODEL_RECIPE_UNVALIDATED: candidato LipSync sin receta revisada.")
        with self._lock:
            if self.loaded_model_id == model_id and self._sessions.get("edtalk") is not None:
                return self._sessions

            if not self.catalog.is_downloaded(model_id):
                defn = self.catalog.get_definition(model_id)
                mname = defn["name"] if defn else model_id
                raise RuntimeError(
                    f"El paquete LipSync «{mname}» ({model_id}) no está descargado aún. "
                    f"Descárgalo primero con studio_download_model('{model_id}') (solo 0.65 GB, tarda ~4 segundos en Colab)."
                )

            import onnxruntime as ort

            avail = ort.get_available_providers()
            providers: List[Any] = []
            if "CUDAExecutionProvider" in avail:
                providers.append("CUDAExecutionProvider")
            providers.append("CPUExecutionProvider")
            self._provider_used = providers[0]

            onnx_dir = self.catalog.model_dir(model_id) / "onnx"
            if job:
                job.log(
                    f"Cargando motores ONNX de LipSync HD ({self._provider_used})...",
                    progress=12.0,
                    step="Cargando EDTalk + YuNet + XSeg + GFPGAN...",
                )

            sess_opts = ort.SessionOptions()
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            self._sessions = {
                "yunet": ort.InferenceSession(str(onnx_dir / "yunet_2023_mar.onnx"), sess_options=sess_opts, providers=providers),
                "edtalk": ort.InferenceSession(str(onnx_dir / "edtalk_256.onnx"), sess_options=sess_opts, providers=providers),
            }
            xseg_fp = onnx_dir / "xseg_1.onnx"
            if xseg_fp.exists():
                self._sessions["xseg"] = ort.InferenceSession(str(xseg_fp), sess_options=sess_opts, providers=providers)
            gfp_fp = onnx_dir / "gfpgan_1.4.onnx"
            if gfp_fp.exists():
                self._sessions["gfpgan"] = ort.InferenceSession(str(gfp_fp), sess_options=sess_opts, providers=providers)

            self.loaded_model_id = model_id
            defn = self.catalog.get_definition(model_id)
            memory_manager.register_loaded("lipsync", model_id, defn["name"] if defn else model_id, self)
            return self._sessions

    def render_lipsync(
        self,
        job: JobRecord,
        media_path: Path,
        voice_audio_path: Optional[Path] = None,
        target_face: str = "largest",
        dialogue_turns: Optional[List[Dict[str, Any]]] = None,
        model_id: str = "edtalk-256-lipsync-hd",
        lip_weight: float = 0.85,
        enhance_face: bool = True,
        restore_strength: float = 0.50,
        music_path: Optional[Path] = None,
        music_volume: float = 0.22,
        ambient_path: Optional[Path] = None,
        ambient_volume: float = 0.18,
        extend_video_to_audio: bool = True,
    ) -> Dict[str, Any]:
        """Anima los labios de 1 o varios personajes sobre una foto o un clip MP4 y masteriza el sonido final."""
        if not media_path.exists():
            raise FileNotFoundError(f"No existe la imagen o vídeo origen: {media_path}")

        turns: List[Dict[str, Any]] = []
        if dialogue_turns:
            for idx_t, t_item in enumerate(dialogue_turns):
                vp = Path(t_item["voice_path"])
                if not vp.exists():
                    raise FileNotFoundError(f"No existe el audio del turno #{idx_t + 1}: {vp}")
                turns.append(
                    {
                        "voice_path": vp,
                        "target_face": str(t_item.get("target_face") or target_face or "largest"),
                        "start_sec": max(0.0, float(t_item.get("start_sec") or 0.0)),
                        "volume": max(0.1, min(2.0, float(t_item.get("volume", 1.0)))),
                    }
                )
        elif voice_audio_path is not None:
            if not voice_audio_path.exists():
                raise FileNotFoundError(f"No existe el audio de voz: {voice_audio_path}")
            turns.append(
                {
                    "voice_path": voice_audio_path,
                    "target_face": str(target_face or "largest"),
                    "start_sec": 0.0,
                    "volume": 1.0,
                }
            )
        else:
            raise ValueError("Debes indicar `voice_audio_path` o al menos un turno en `dialogue_turns`.")

        sessions = self._ensure_loaded(model_id=model_id, job=job)
        yunet_sess = sessions["yunet"]
        edtalk_sess = sessions["edtalk"]
        xseg_sess = sessions.get("xseg")
        gfpgan_sess = sessions.get("gfpgan") if enhance_face else None

        # Precalcular espectrogramas Mel de cada intervención hablada
        max_dialogue_end = 0.0
        for turn in turns:
            wav_16k = decode_audio_16k_mono(turn["voice_path"])
            dur_s = max(0.15, float(len(wav_16k)) / 16000.0)
            turn["duration_sec"] = dur_s
            turn["end_sec"] = turn["start_sec"] + dur_s
            turn["mel"] = extract_edtalk_mel_spectrogram(wav_16k)
            turn["prev_center"] = None
            turn["prev_kps"] = None
            max_dialogue_end = max(max_dialogue_end, turn["end_sec"] + 0.15)

        work_dir = Path(tempfile.mkdtemp(prefix="studio_lipsync_"))
        ffmpeg_bin = get_ffmpeg_bin()
        VIDEOS_OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_name = f"lipsync_{job.id[:8]}.mp4"
        final_mp4 = VIDEOS_OUT_DIR / out_name

        try:
            # 1. Preparar vídeo base normalizado (si la entrada es imagen, crear movimiento suave; si es vídeo corto, extender sin congelar)
            prep_mp4 = work_dir / "prepared_base.mp4"
            is_img = media_path.suffix.lower() in IMAGE_EXTS
            if is_img:
                from PIL import Image

                with Image.open(media_path) as im:
                    im_w, im_h = im.size
                # Mantener proporción par limpia hasta 960px máximo
                scale_f = min(1.0, 960.0 / float(max(im_w, im_h, 1)))
                vid_w = max(256, int(round(im_w * scale_f / 2.0)) * 2)
                vid_h = max(256, int(round(im_h * scale_f / 2.0)) * 2)
                fps = 24
                total_dur = max(1.5, max_dialogue_end)
                job.log(
                    f"Preparando plano animado base desde imagen ({vid_w}×{vid_h} @ {fps}fps, {total_dur:.2f}s)...",
                    progress=18.0,
                    step="Animando base desde retrato...",
                )
                create_motion_clip_from_image(
                    image_path=media_path,
                    output_mp4=prep_mp4,
                    width=vid_w,
                    height=vid_h,
                    fps=fps,
                    duration_sec=total_dur,
                    motion="zoom_in",
                )
            else:
                v_info = probe_media_info(media_path)
                orig_dur = max(0.3, float(v_info["duration_seconds"] or 4.0))
                total_dur = max(orig_dur, max_dialogue_end) if extend_video_to_audio else orig_dur
                if total_dur > orig_dur + 0.08:
                    job.log(
                        f"Extendiendo fluidez del vídeo base ({orig_dur:.2f}s → {total_dur:.2f}s) sin fotogramas congelados...",
                        progress=18.0,
                        step="Adaptando duración del vídeo al diálogo...",
                    )
                    use_pp = total_dur > orig_dur * 1.40
                    cmd_ext = [ffmpeg_bin, "-hide_banner", "-loglevel", "error"]
                    if use_pp:
                        cycles = max(1, int((total_dur / max(0.3, orig_dur * 2.0)) + 1.5))
                        if cycles > 1:
                            cmd_ext += ["-stream_loop", str(cycles)]
                    cmd_ext += ["-i", str(media_path)]
                    if use_pp:
                        vf_ext = (
                            f"[0:v]split=2[vf][vr];[vr]reverse[vrev];"
                            f"[vf][vrev]concat=n=2:v=1:a=0,trim=0:{total_dur:.3f},setpts=PTS-STARTPTS,format=yuv420p[vout]"
                        )
                    else:
                        ratio = min(1.40, total_dur / orig_dur)
                        vf_ext = f"[0:v]setpts={ratio:.5f}*PTS,trim=0:{total_dur:.3f},setpts=PTS-STARTPTS,format=yuv420p[vout]"
                    cmd_ext += [
                        "-filter_complex",
                        vf_ext,
                        "-map",
                        "[vout]",
                        "-an",
                        "-c:v",
                        "libx264",
                        "-preset",
                        "veryfast",
                        "-crf",
                        "18",
                        "-y",
                        str(prep_mp4),
                    ]
                    r_ext = subprocess.run(cmd_ext, capture_output=True, text=True, timeout=240)
                    if r_ext.returncode != 0:
                        raise RuntimeError(f"Error preparando vídeo base para LipSync: {r_ext.stderr.strip()}")
                else:
                    prep_mp4 = media_path

            # 2. Procesar fotograma a fotograma aplicando LipSync únicamente a los personajes que hablan en cada instante
            cap = cv2.VideoCapture(str(prep_mp4))
            if not cap.isOpened():
                raise RuntimeError(f"No se pudo abrir el vídeo para LipSync: {prep_mp4}")

            fps_val = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
            if fps_val <= 1.0 or fps_val > 120.0:
                fps_val = 24.0
            frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 832)
            frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
            # Asegurar dimensiones pares para libx264 yuv420p
            even_w = max(2, (frame_w // 2) * 2)
            even_h = max(2, (frame_h // 2) * 2)
            est_frames = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or round(total_dur * fps_val)))

            silent_synced_mp4 = work_dir / "synced_visual.mp4"
            enc_proc = subprocess.Popen(
                [
                    ffmpeg_bin,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "bgr24",
                    "-s",
                    f"{even_w}x{even_h}",
                    "-r",
                    f"{fps_val:.4f}",
                    "-i",
                    "pipe:0",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "18",
                    "-pix_fmt",
                    "yuv420p",
                    "-colorspace",
                    "bt709",
                    "-y",
                    str(silent_synced_mp4),
                ],
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            base_face_mask = create_feathered_mouth_face_mask(crop_size=512)
            frames_synced = 0
            faces_detected_frames = 0
            frame_idx = 0

            try:
                while True:
                    job.check_cancelled()
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        break
                    if frame.shape[1] != even_w or frame.shape[0] != even_h:
                        frame = cv2.resize(frame, (even_w, even_h), interpolation=cv2.INTER_AREA)

                    t_sec = frame_idx / fps_val
                    active_turns = [
                        tr for tr in turns if (tr["start_sec"] - 0.04) <= t_sec <= (tr["end_sec"] + 0.04)
                    ]

                    if active_turns:
                        faces = detect_faces_yunet_onnx(yunet_sess, frame)
                        if faces:
                            faces_detected_frames += 1
                        for tr in active_turns:
                            chosen = select_target_face(
                                faces,
                                selector=tr["target_face"],
                                frame_width=even_w,
                                prev_center=tr["prev_center"],
                            )
                            kps_5: Optional[np.ndarray] = None
                            if chosen is not None:
                                raw_kps = chosen["landmarks_5"]
                                if tr["prev_kps"] is not None:
                                    kps_5 = (0.68 * raw_kps + 0.32 * tr["prev_kps"]).astype(np.float32)
                                else:
                                    kps_5 = raw_kps
                                tr["prev_kps"] = kps_5
                                tr["prev_center"] = (chosen["center_x"], chosen["center_y"])
                            elif tr["prev_kps"] is not None:
                                kps_5 = tr["prev_kps"]

                            if kps_5 is None:
                                continue

                            local_t = max(0.0, t_sec - tr["start_sec"])
                            local_fidx = int(round(local_t * fps_val))
                            mel_win = slice_mel_window_for_frame(tr["mel"], local_fidx, fps_val)

                            crop_512, aff_mat = warp_face_ffhq_512(frame, kps_5, crop_size=512)
                            occ_mask = compute_xseg_occlusion_mask(xseg_sess, crop_512)
                            synced_512 = run_edtalk_and_gfpgan_on_crop(
                                edtalk_sess=edtalk_sess,
                                gfpgan_sess=gfpgan_sess,
                                crop_512_bgr=crop_512,
                                mel_window=mel_win,
                                lip_weight=lip_weight,
                                restore_strength=restore_strength,
                            )

                            # Fundido suave de entrada/salida de la máscara labial al empezar y terminar de hablar
                            ramp_s = 0.14
                            in_gain = min(1.0, max(0.0, (t_sec - tr["start_sec"] + 0.04) / ramp_s))
                            out_gain = min(1.0, max(0.0, (tr["end_sec"] + 0.04 - t_sec) / ramp_s))
                            env_gain = float(min(in_gain, out_gain))

                            final_mask = (base_face_mask * occ_mask * env_gain).astype(np.float32)
                            frame = paste_back_crop(frame, synced_512, final_mask, aff_mat)
                            frames_synced += 1

                    assert enc_proc.stdin is not None
                    enc_proc.stdin.write(frame.tobytes())
                    frame_idx += 1

                    if frame_idx == 1 or frame_idx % 12 == 0:
                        pct = 24.0 + min(56.0, (frame_idx / float(max(1, est_frames))) * 56.0)
                        job.heartbeat(
                            progress=pct,
                            step=f"LipSync HD fotograma {frame_idx}/{est_frames} ({self._provider_used.replace('ExecutionProvider', '')})",
                        )
            finally:
                cap.release()
                if enc_proc.stdin:
                    enc_proc.stdin.close()
                enc_stderr = enc_proc.stderr.read().decode("utf-8", errors="replace") if enc_proc.stderr else ""
                rc = enc_proc.wait(timeout=120)
                if rc != 0:
                    raise RuntimeError(f"Error codificando pista visual de LipSync: {enc_stderr.strip()}")

            if faces_detected_frames == 0 and frames_synced == 0:
                job.log(
                    "⚠️ Aviso: YuNet no detectó un rostro frontal claro en los fotogramas activos; el vídeo conserva la imagen original con el audio sincronizado.",
                    level="warning",
                )

            # 3. Mezclar todas las voces de los turnos + música + sonido ambiente con Auto-Ducking
            voice_mix_specs = [
                {
                    "path": tr["voice_path"],
                    "start_sec": tr["start_sec"],
                    "volume": tr["volume"],
                }
                for tr in turns
            ]
            mix_meta = mix_and_master_media(
                output_path=final_mp4,
                video_path=silent_synced_mp4,
                voice_tracks=voice_mix_specs,
                music_path=music_path,
                music_volume=music_volume,
                ambient_path=ambient_path,
                ambient_volume=ambient_volume,
                keep_original_audio=(not is_img),
                original_audio_volume=0.25,
                auto_duck_bg=True,
                extend_video_to_audio=False,
                job=job,
            )

            rel_path = f"outputs/videos/{out_name}"
            defn = self.catalog.get_definition(model_id)
            meta_doc = {
                "id": final_mp4.stem,
                "filename": out_name,
                "rel_path": rel_path,
                "media_url": f"/media/{rel_path}",
                "type": "video",
                "model_id": model_id,
                "model_name": defn["name"] if defn else model_id,
                "provider": self._provider_used,
                "duration_seconds": mix_meta["duration_seconds"],
                "width": even_w,
                "height": even_h,
                "fps": round(fps_val, 2),
                "turns_count": len(turns),
                "frames_total": frame_idx,
                "frames_lipsynced": frames_synced,
                "faces_detected_frames": faces_detected_frames,
                "auto_ducking_applied": mix_meta["auto_ducking_applied"],
                "size_mb": mix_meta["size_mb"],
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }
            (VIDEOS_OUT_DIR / f"{final_mp4.stem}.json").write_text(
                json.dumps(meta_doc, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            job.log(
                f"LipSync HD completado: {out_name} ({mix_meta['duration_seconds']:.2f}s · {frames_synced} sincronizaciones labiales).",
                level="success",
                progress=100.0,
                step="LipSync HD completado",
            )
            return meta_doc
        finally:
            import shutil

            shutil.rmtree(work_dir, ignore_errors=True)


lipsync_engine = LipSyncEngine(catalog_manager)

