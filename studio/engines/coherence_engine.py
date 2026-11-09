import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

from studio.catalog import catalog_manager
from studio.compat_patches import ensure_patchable_runtime
from studio.config import DATA_ROOT
from studio.hardware import colab_console_log, query_gpu_info
from studio.memory_safety_hooks import guarded_model_load, generation_memory_criteria
from studio.memory_policy import MemoryPressureError
from studio.jobs import JobRecord
from studio.memory_manager import (
    memory_manager,
    run_with_live_memory_progress,
    serialized_gpu_operation,
    checkpoint_current_operation,
)


def _pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    rgb = np.asarray(pil_img.convert("RGB"), dtype=np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _resize_max_side_bgr(bgr: np.ndarray, max_side: int = 640) -> np.ndarray:
    h, w = bgr.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return bgr
    scale = max_side / float(longest)
    nw = max(16, int(round(w * scale)))
    nh = max(16, int(round(h * scale)))
    return cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)


def _resize_for_qwen_vl(pil_img: Image.Image, max_side: int = 756) -> Image.Image:
    """
    Ajusta el lado mayor a <= 756 px y redondea ancho y alto a múltiplos exactos de 28
    (factor de fusión espacial 14*2 de Qwen2.5-VL), evitando dependencias externas como qwen-vl-utils
    y manteniendo el consumo de VRAM en ~2.8 GB incluso comparando varias imágenes.
    """
    rgb = pil_img.convert("RGB")
    w, h = rgb.size
    scale = min(1.0, float(max_side) / float(max(w, h, 1)))
    nw = max(56, int(round((w * scale) / 28.0)) * 28)
    nh = max(56, int(round((h * scale) / 28.0)) * 28)
    if (nw, nh) == (w, h):
        return rgb
    return rgb.resize((nw, nh), Image.Resampling.LANCZOS)


def analyze_pair_fast_cv(anchor_path: Path, candidate_path: Path) -> Dict[str, Any]:
    """
    Etapa 1 (0 GB VRAM · ~0.12 s CPU): Análisis fotométrico y estructural objetivo con OpenCV.
    Detecta al instante:
      - Incompatibilidad de proporción/aspecto (crítico antes de interpolar Frame a Frame).
      - Saltos de temperatura de color / paleta cromática (histogramas 2D HSV + CIE LAB).
      - Saltos bruscos de exposición / iluminación / contraste (canal L*).
      - Continuidad estructural de fondo/encuadre (keypoints ORB + correlación espacial).
      - Desenfoque o degradación de nitidez (varianza Laplaciana).
    """
    t0 = time.time()
    pil_a = Image.open(anchor_path).convert("RGB")
    pil_c = Image.open(candidate_path).convert("RGB")

    wa, ha = pil_a.size
    wc, hc = pil_c.size
    ar_a = wa / max(float(ha), 1.0)
    ar_c = wc / max(float(hc), 1.0)
    ar_diff_pct = abs(ar_a - ar_c) / max(ar_a, 1e-6) * 100.0
    aspect_compatible = ar_diff_pct <= 4.5

    bgr_a = _resize_max_side_bgr(_pil_to_bgr(pil_a), 640)
    bgr_c = _resize_max_side_bgr(_pil_to_bgr(pil_c), 640)

    # 1. Paleta cromática (HSV Hue-Sat + CIE LAB a*-b*)
    hsv_a = cv2.cvtColor(bgr_a, cv2.COLOR_BGR2HSV)
    hsv_c = cv2.cvtColor(bgr_c, cv2.COLOR_BGR2HSV)
    h_hsv_a = cv2.calcHist([hsv_a], [0, 1], None, [30, 32], [0, 180, 0, 256])
    h_hsv_c = cv2.calcHist([hsv_c], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(h_hsv_a, h_hsv_a)
    cv2.normalize(h_hsv_c, h_hsv_c)
    bhatt_hsv = float(cv2.compareHist(h_hsv_a, h_hsv_c, cv2.HISTCMP_BHATTACHARYYA))

    lab_a = cv2.cvtColor(bgr_a, cv2.COLOR_BGR2LAB)
    lab_c = cv2.cvtColor(bgr_c, cv2.COLOR_BGR2LAB)
    h_lab_a = cv2.calcHist([lab_a], [1, 2], None, [32, 32], [0, 256, 0, 256])
    h_lab_c = cv2.calcHist([lab_c], [1, 2], None, [32, 32], [0, 256, 0, 256])
    cv2.normalize(h_lab_a, h_lab_a)
    cv2.normalize(h_lab_c, h_lab_c)
    bhatt_lab = float(cv2.compareHist(h_lab_a, h_lab_c, cv2.HISTCMP_BHATTACHARYYA))

    color_sim = max(0.0, min(100.0, (1.0 - (0.5 * bhatt_hsv + 0.5 * bhatt_lab)) * 100.0))

    # 2. Exposición e iluminación (canal L*)
    l_a = lab_a[:, :, 0].astype(np.float32)
    l_c = lab_c[:, :, 0].astype(np.float32)
    mean_diff = abs(float(l_a.mean()) - float(l_c.mean()))
    std_diff = abs(float(l_a.std()) - float(l_c.std()))
    p15_diff = abs(float(np.percentile(l_a, 15)) - float(np.percentile(l_c, 15)))
    p85_diff = abs(float(np.percentile(l_a, 85)) - float(np.percentile(l_c, 85)))
    lum_penalty = (mean_diff * 0.9 + std_diff * 0.7 + 0.25 * (p15_diff + p85_diff)) / 1.6
    lighting_score = max(0.0, min(100.0, 100.0 - lum_penalty))

    # 3. Anclaje estructural (rejilla espacial LAB + descriptores ORB)
    grid_a = cv2.resize(lab_a, (24, 24), interpolation=cv2.INTER_AREA).astype(np.float32)
    grid_c = cv2.resize(lab_c, (24, 24), interpolation=cv2.INTER_AREA).astype(np.float32)
    mad_grid = float(np.mean(np.abs(grid_a - grid_c)))
    spatial_sim = max(0.0, min(100.0, 100.0 - mad_grid * 1.45))

    orb = cv2.ORB_create(nfeatures=600)
    kp_a, des_a = orb.detectAndCompute(cv2.cvtColor(bgr_a, cv2.COLOR_BGR2GRAY), None)
    kp_c, des_c = orb.detectAndCompute(cv2.cvtColor(bgr_c, cv2.COLOR_BGR2GRAY), None)
    orb_ratio_pct = 0.0
    if des_a is not None and des_c is not None and len(des_a) >= 8 and len(des_c) >= 8:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        raw_matches = bf.knnMatch(des_a, des_c, k=2)
        good = 0
        for pair in raw_matches:
            if len(pair) == 2 and pair[0].distance < 0.78 * pair[1].distance:
                good += 1
        denom = max(1, min(len(kp_a), len(kp_c)))
        orb_ratio_pct = min(100.0, (good / float(denom)) * 220.0)

    structural_score = round(0.60 * spatial_sim + 0.40 * orb_ratio_pct, 1)

    # 4. Nitidez / foco (varianza Laplaciana normalizada)
    gray_a_512 = _resize_max_side_bgr(bgr_a, 512)
    gray_c_512 = _resize_max_side_bgr(bgr_c, 512)
    lap_a = float(cv2.Laplacian(cv2.cvtColor(gray_a_512, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
    lap_c = float(cv2.Laplacian(cv2.cvtColor(gray_c_512, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
    # Una imagen nítida típica tiene lap >= 80; < 25 suele estar muy borrosa
    sharp_c_score = max(0.0, min(100.0, (lap_c / max(80.0, min(lap_a, 220.0))) * 92.0))
    if lap_c >= 90.0:
        sharp_c_score = max(sharp_c_score, 88.0)

    cv_overall = (
        0.42 * color_sim
        + 0.28 * lighting_score
        + 0.18 * structural_score
        + 0.12 * sharp_c_score
        - (0.0 if aspect_compatible else 22.0)
    )
    cv_overall = round(max(0.0, min(100.0, cv_overall)), 1)

    warnings: List[str] = []
    if not aspect_compatible:
        warnings.append(
            f"Proporción distinta entre ancla ({wa}×{ha}) y candidato ({wc}×{hc}): diferencia {ar_diff_pct:.1f}% (deformará Frame a Frame)."
        )
    if color_sim < 58.0:
        warnings.append(
            f"Deriva fuerte de paleta de color / temperatura ({color_sim:.1f}/100): la ropa o la luz ambiental cambiaron notablemente."
        )
    if lighting_score < 60.0:
        warnings.append(
            f"Salto brusco de iluminación/exposición ({lighting_score:.1f}/100): brillo medio pasó de {l_a.mean():.0f} a {l_c.mean():.0f}."
        )
    if sharp_c_score < 45.0:
        warnings.append(f"El fotograma candidato presenta baja nitidez o desenfoque (Laplaciano={lap_c:.1f}).")

    return {
        "anchor_resolution": f"{wa}x{ha}",
        "candidate_resolution": f"{wc}x{hc}",
        "aspect_ratio_compatible": aspect_compatible,
        "color_palette_score": round(color_sim, 1),
        "lighting_exposure_score": round(lighting_score, 1),
        "structural_anchor_score": structural_score,
        "spatial_displacement_pct": round(min(100.0, mad_grid * 3.8), 1),
        "sharpness_score": round(sharp_c_score, 1),
        "cv_overall_score": cv_overall,
        "cv_warnings": warnings,
        "elapsed_ms": round((time.time() - t0) * 1000, 1),
        "limitation_note": (
            "OpenCV mide color, luz, proporción, nitidez y desplazamiento espacial en 0.1s, pero NO distingue rostros semánticos; "
            "para verificar identidad de personajes usa mode='hybrid_auto' o mode='vlm_qwen' (Qwen2.5-VL-3B)."
        ),
    }


_VLM_UNPATCHED_KEY = "loaded_before_compat_patches"


def _check_four_bit_modules_patchable(cls_vlm: Any) -> None:
    """Fail early with a diagnosis instead of `AssertionError` mid-inference.

    bitsandbytes rebuilds a 4-bit module's parameter from the module when the
    quantization state was lost. If Studio's constructor guard is not active on
    that class, the rebuild raises a bare `AssertionError` the first time the
    model runs. Nothing is downloaded or re-quantized here.
    """
    if cls_vlm is None:
        return
    try:
        import bitsandbytes as bnb  # type: ignore
    except Exception:
        return  # configured elsewhere; the loader reports the real error
    if getattr(bnb.nn, "Linear4bit", None) is None:
        return
    if getattr(bnb.nn.Params4bit.__new__, "_studio_safe_wrapped", False):
        return
    raise RuntimeError(
        "RUNTIME_PATCHES_STALE: las clases 4-bit de bitsandbytes se crearon antes de los parches "
        "de compatibilidad de Studio IA. Reinicia el proceso del servidor (no recargues solo el "
        "módulo de memoria) y vuelve a auditar; los 7 GB del inspector ya están en disco, no hay "
        "que descargarlos otra vez."
    )

@serialized_gpu_operation(slots=("vision",), managed=True)
def load_coherence_vlm_into_memory(
    model_id: str = "qwen25-vl-3b-coherence",
    auto_download: bool = True,
    job: Optional[JobRecord] = None,
) -> Dict[str, Any]:
    """
    Carga Qwen2.5-VL-3B-Instruct en 4-bit NF4 (con torre visual ViT en BF16 intacto, ~2.8 GB VRAM)
    en el slot ligero `vision`. En una GPU L4 (22.5 GB) puede convivir junto a Qwen-Image-Edit-2511
    sin descargar el modelo de imagen de la VRAM.
    """
    defn = catalog_manager.get_definition(model_id)
    if not defn:
        raise ValueError(f"Modelo de visión no encontrado en el catálogo: {model_id}")

    if defn.get("validation_state") == "candidate":
        raise ValueError("MODEL_RECIPE_UNVALIDATED: candidato de visión sin receta revisada.")
    cur = memory_manager.get_slot("vision")
    if cur and cur.get("model_id") == model_id and not cur.get("handle", {}).get(_VLM_UNPATCHED_KEY):
        vram_now = query_gpu_info()["vram_used_gb"]
        if job:
            job.log(
                f"Inspector «{defn['name']}» ya está activo en VRAM ({vram_now} GB totales en uso).",
                level="info",
                progress=35.0,
                step="Inspector Qwen2.5-VL activo en memoria",
            )
        return {"model_id": model_id, "vram_used_gb": vram_now}
    if cur and cur.get("handle", {}).get(_VLM_UNPATCHED_KEY):
        # Built before the compatibility patches were active: its 4-bit modules are
        # half-patched. Reloading is the supported recovery (no new weights needed).
        if job:
            job.log(
                "El inspector se cargó antes de los parches de compatibilidad; se recarga desde los archivos locales.",
                level="warning",
                progress=8.0,
            )
        memory_manager.unload_slot("vision")
        cur = None

    # Primes the guard on the 4-bit classes BEFORE transformers/bitsandbytes build them.
    runtime_report = ensure_patchable_runtime(modules=("transformers", "bitsandbytes", "accelerate", "peft", "diffusers"))
    if runtime_report.get("notice") and job:
        job.log(runtime_report["notice"], level="warning")

    import torch  # type: ignore

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Qwen2.5-VL requiere GPU NVIDIA activa en Colab (o usa mode='fast_cv' para análisis instantáneo OpenCV en CPU)."
        )

    if not auto_download and not catalog_manager.is_downloaded(model_id):
        raise RuntimeError(f"«{defn['name']}» no está descargado; no se modifica la memoria ni el disco.")
    if cur:
        memory_manager.unload_slot("vision")
    del cur

    memory_manager.prepare_for_load(defn, job=job)

    dl_job = job or JobRecord("download", f"Auto-descarga {model_id}")
    if not catalog_manager.is_downloaded(model_id):
        if not auto_download:
            raise RuntimeError(f"«{defn['name']}» no está descargado aún.")
        if job:
            job.log(
                f"Descargando inspector multimodal «{defn['name']}» de Hugging Face (~6.4 GB, solo la primera vez)...",
                progress=5.0,
                step="Descargando Qwen2.5-VL-3B...",
            )
        catalog_manager.download_model(model_id, dl_job, progress_range=(5.0, 55.0))
        load_start_pct = 55.0
    else:
        load_start_pct = 12.0

    base_artifact = next(a for a in defn.get("artifacts", []) if a.get("role") == "base")
    base_dir = catalog_manager.model_dir(model_id) / base_artifact["subdir"]

    memory_manager.prepare_for_load(defn, job=job)

    def _do_load() -> Dict[str, Any]:
        ensure_patchable_runtime(modules=("transformers", "bitsandbytes", "accelerate", "peft"))
        import transformers  # type: ignore
        from transformers import AutoProcessor, BitsAndBytesConfig  # type: ignore

        cls_vlm = getattr(transformers, "Qwen2_5_VLForConditionalGeneration", None) or getattr(
            transformers, "AutoModelForImageTextToText", None
        )
        _check_four_bit_modules_patchable(cls_vlm)
        if cls_vlm is None:
            raise RuntimeError(
                f"Tu versión de transformers ({transformers.__version__}) no incluye Qwen2_5_VLForConditionalGeneration. "
                "Ejecuta pip install -U 'transformers>=4.49.0'."
            )

        processor = guarded_model_load(AutoProcessor.from_pretrained, str(base_dir), local_files_only=True, trust_remote_code=False)
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=["visual"],
        )
        # A quantization failure is not permission to allocate another full model
        # in BF16. hybrid_auto may return an explicitly limited CV result instead.
        model = guarded_model_load(cls_vlm.from_pretrained,
            str(base_dir),
            quantization_config=bnb_cfg,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            low_cpu_mem_usage=True,
            local_files_only=True,
            trust_remote_code=False,
        )
        model.eval()
        return {"model": model, "processor": processor}

    handle = run_with_live_memory_progress(
        job=job,
        start_pct=load_start_pct,
        end_pct=68.0,
        step_label="Cargando Qwen2.5-VL-3B (4-bit NF4) en GPU",
        fn=_do_load,
        expected_seconds=18.0,
    )
    # A rebuild notice means this load may still be half-patched; do not hand the
    # cached 4-bit instance to a later job as if it were healthy.
    handle[_VLM_UNPATCHED_KEY] = bool(runtime_report.get("notice")) and not runtime_report.get("reimported")
    memory_manager.register_loaded(
        slot="vision",
        model_id=model_id,
        name=defn["name"],
        handle=handle,
        adapters=[],
    )
    vram_after = query_gpu_info()["vram_used_gb"]
    if job:
        job.log(
            f"Inspector Qwen2.5-VL-3B listo en GPU (VRAM total en uso: {vram_after} GB).",
            level="success",
            progress=70.0,
            step="Qwen2.5-VL-3B listo",
        )
    return {"model_id": model_id, "vram_used_gb": vram_after}


def _extract_first_json_dict(raw_text: str) -> Dict[str, Any]:
    cleaned = raw_text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    if fence:
        cleaned = fence.group(1)
    else:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start : end + 1]
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


@serialized_gpu_operation(slots=("vision",), managed=True)
def analyze_pair_qwen_vl(
    anchor_path: Path,
    candidate_path: Path,
    scene_goal: str = "",
    expect_spatial_progression: bool = False,
) -> Dict[str, Any]:
    """
    Etapa 2 (Semántica Multimodal con Qwen2.5-VL-3B-Instruct):
    Compara la Imagen 1 (Ancla Maestra) frente a la Imagen 2 (Fotograma Candidato)
    evaluando identidad facial, peinado, ropa, número de personas, objetos del escenario y viabilidad Frame a Frame.
    """
    import torch  # type: ignore

    slot = memory_manager.get_slot("vision")
    if not slot:
        raise RuntimeError("Qwen2.5-VL no está cargado en el slot `vision`.")

    model = slot["handle"]["model"]
    processor = slot["handle"]["processor"]

    img_anchor = _resize_for_qwen_vl(Image.open(anchor_path), max_side=756)
    img_cand = _resize_for_qwen_vl(Image.open(candidate_path), max_side=756)

    goal_line = f"Contexto de la escena pedida por el usuario: «{scene_goal.strip()}».\n" if scene_goal.strip() else ""
    env_criterion = (
        "4. `environment_props_score` (0-100): tratándose de una ESCENA EN AVANCE (ej. caminando por la calle), ¿se mantiene la MISMA calle/entorno, estilo arquitectónico, hora del día e iluminación aunque el fondo se haya desplazado por perspectiva al caminar? (NO penalices el paralaje natural del fondo al avanzar).\n"
        if expect_spatial_progression
        else "4. `environment_props_score` (0-100): ¿se mantiene el mismo lugar, mobiliario (ej. mesa, restaurante, pared), objetos e iluminación?\n"
    )
    advance_criterion = (
        "5. `flf2v_pose_transition_score` (0-100): ¿los personajes REALMENTE han avanzado caminando/moviéndose respecto a la Imagen 1 (pierna contraria adelantada, nuevo gesto de conversación y perspectiva de calle desplazada) sin quedarse congelados en la misma postura exacta? Si están casi idénticos/congelados en el sitio, pon < 50.\n"
        if expect_spatial_progression
        else "5. `flf2v_pose_transition_score` (0-100): ¿el cambio de postura entre Imagen 1 e Imagen 2 es conectable de forma natural en un clip de 3 a 5 segundos?\n"
    )
    system_prompt = (
        "Eres un supervisor experto de continuidad cinematográfica (Script Supervisor) para producción de vídeo con IA. "
        "Sé estricto y objetivo evaluando si dos fotogramas mantienen EXACTAMENTE los mismos personajes y coherencia espacial."
    )
    user_instruction = (
        f"{goal_line}"
        "Compara la IMAGEN 1 (FOTOGRAMA ANCLA DE REFERENCIA) con la IMAGEN 2 (NUEVO FOTOGRAMA CANDIDATO).\n"
        "Evalúa con rigor de 0 a 100:\n"
        "1. `same_people_count` (boolean): ¿hay exactamente el mismo número de personas protagonistas?\n"
        "2. `character_identity_score` (0-100): ¿son exactamente las mismas personas con los mismos rasgos faciales, edad, gafas/barba y etnia? (Si parece otro actor/persona, pon < 55).\n"
        "3. `clothing_hair_score` (0-100): ¿mantienen exactamente el mismo peinado, color de pelo y las mismas prendas/colores de ropa?\n"
        f"{env_criterion}"
        f"{advance_criterion}"
        "6. `anchor_characters_summary`: descripción breve y precisa de los personajes de la Imagen 1.\n"
        "7. `drifts_detected`: lista de fallos concretos de consistencia (o de falta de avance si están congelados) observados en la Imagen 2 respecto a la Imagen 1 (vacía `[]` si todo es correcto).\n"
        "8. `repair_edit_prompt`: si hay fallos, redacta la instrucción exacta para `Qwen-Image-Edit-2511` manteniendo la identidad de la Imagen 1.\n\n"
        "Responde ÚNICAMENTE con un objeto JSON válido con esas 8 claves exactas, sin texto adicional."
    )

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "image"},
                {"type": "text", "text": user_instruction},
            ],
        },
    ]

    t0 = time.time()
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text_prompt],
        images=[img_anchor, img_cand],
        padding=True,
        return_tensors="pt",
    )
    target_dev = next(model.parameters()).device
    inputs = {k: (v.to(target_dev) if hasattr(v, "to") else v) for k, v in inputs.items()}

    checkpoint_current_operation(force=True)
    _check_four_bit_modules_patchable(type(model))
    with torch.inference_mode():
        try:
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=420,
                do_sample=False,
                stopping_criteria=generation_memory_criteria(),
            )
        except AssertionError as exc:
            # bitsandbytes' state-recovery assert, not a measurement of image quality.
            raise RuntimeError(
                "QWEN_VL_4BIT_STATE_LOST: el inspector 4-bit perdió su estado de cuantización al "
                "ejecutarse. Reinicia el proceso del servidor (los pesos ya están en disco) y "
                "reaudita; mientras tanto la auditoría es solo fotométrica y no verifica identidad."
            ) from exc

    in_len = inputs["input_ids"].shape[1]
    trimmed = [out_ids[in_len:] for out_ids in generated_ids]
    raw_output = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    parsed = _extract_first_json_dict(raw_output)

    def _clamp_score(val: Any, fallback: float = 70.0) -> float:
        try:
            return round(max(0.0, min(100.0, float(val))), 1)
        except Exception:
            return fallback

    same_cnt = bool(parsed.get("same_people_count", True))
    char_sc = _clamp_score(parsed.get("character_identity_score"), 70.0)
    cloth_sc = _clamp_score(parsed.get("clothing_hair_score"), 70.0)
    env_sc = _clamp_score(parsed.get("environment_props_score"), 75.0)
    pose_sc = _clamp_score(parsed.get("flf2v_pose_transition_score"), 78.0)

    vlm_overall = round(
        0.42 * char_sc + 0.28 * cloth_sc + 0.18 * env_sc + 0.12 * pose_sc - (0.0 if same_cnt else 25.0),
        1,
    )
    vlm_overall = max(0.0, min(100.0, vlm_overall))

    drifts = parsed.get("drifts_detected")
    if not isinstance(drifts, list):
        drifts = [str(drifts)] if drifts else []
    drifts = [str(x).strip() for x in drifts if str(x).strip()]

    return {
        "same_people_count": same_cnt,
        "character_identity_score": char_sc,
        "clothing_hair_score": cloth_sc,
        "environment_props_score": env_sc,
        "flf2v_pose_transition_score": pose_sc,
        "vlm_overall_score": vlm_overall,
        "anchor_characters_summary": str(parsed.get("anchor_characters_summary") or "").strip(),
        "drifts_detected": drifts,
        "repair_edit_prompt": str(parsed.get("repair_edit_prompt") or "").strip(),
        "raw_vlm_response": raw_output.strip()[:800],
        "elapsed_seconds": round(time.time() - t0, 2),
    }


def _to_rel_str(path: Path) -> str:
    try:
        return path.resolve().relative_to(DATA_ROOT.resolve()).as_posix()
    except Exception:
        return path.name


@serialized_gpu_operation(slots=lambda args: () if args["mode"] == "fast_cv" else ("vision",), managed=True)
def run_coherence_analysis(
    anchor_path: Path,
    candidate_paths: List[Path],
    scene_goal: str = "",
    mode: str = "hybrid_auto",
    strictness_threshold: float = 78.0,
    expect_spatial_progression: bool = False,
    job: Optional[JobRecord] = None,
) -> Dict[str, Any]:
    """
    Ejecuta el análisis completo de coherencia (1 fotograma Ancla vs 1..N fotogramas candidatos).
    Si `expect_spatial_progression=True` (ej. personajes caminando por la calle), verifica ADEMÁS
    que cada fotograma consecutivo tenga desplazamiento espacial real y no sea una copia estática congelada.
    """
    mode = (mode or "hybrid_auto").strip().lower()
    if mode not in ("hybrid_auto", "vlm_qwen", "fast_cv"):
        raise ValueError("`mode` debe ser 'hybrid_auto', 'vlm_qwen' o 'fast_cv'.")

    if not candidate_paths:
        raise ValueError("Debes proporcionar al menos 1 fotograma candidato en `candidate_paths`.")

    threshold = max(40.0, min(95.0, float(strictness_threshold)))
    anchor_rel = _to_rel_str(anchor_path)

    gpu_info = query_gpu_info()
    use_vlm = mode == "vlm_qwen" or (mode == "hybrid_auto" and bool(gpu_info.get("has_gpu")))

    if job:
        job.log(
            f"Iniciando auditoría de coherencia ({mode} · umbral={threshold}/100 · avance_espacial={expect_spatial_progression}) sobre {len(candidate_paths)} fotograma(s)...",
            progress=3.0,
            step="Etapa 1: Métricas físicas OpenCV...",
        )

    # Etapa 1: OpenCV instantáneo en todos los pares (contra ancla + desplazamiento paso a paso contra el fotograma previo)
    cv_results: List[Dict[str, Any]] = []
    prev_step_path = anchor_path
    for idx, cpath in enumerate(candidate_paths, 1):
        checkpoint_current_operation(job=job, force=True)
        if job:
            job.check_cancelled()
        cv_r = analyze_pair_fast_cv(anchor_path, cpath)
        if prev_step_path == anchor_path:
            step_disp = cv_r["spatial_displacement_pct"]
        else:
            step_cv = analyze_pair_fast_cv(prev_step_path, cpath)
            step_disp = step_cv["spatial_displacement_pct"]
        cv_r["step_displacement_from_prev_pct"] = step_disp
        prev_step_path = cpath
        cv_results.append(cv_r)
        if job:
            job.log(
                f"OpenCV Fotograma #{idx} (`{cpath.name}`): color={cv_r['color_palette_score']} · luz={cv_r['lighting_exposure_score']} · avance={step_disp}% → {cv_r['cv_overall_score']}/100",
                progress=5.0 + (idx / len(candidate_paths)) * (15.0 if use_vlm else 90.0),
                step=f"OpenCV #{idx}/{len(candidate_paths)} completado",
            )

    # Etapa 2: Qwen2.5-VL-3B-Instruct 4-bit (si procede)
    vlm_loaded_ok = False
    vlm_fallback_reason: Optional[str] = None
    if use_vlm:
        try:
            load_coherence_vlm_into_memory(model_id="qwen25-vl-3b-coherence", auto_download=True, job=job)
            vlm_loaded_ok = True
        except MemoryPressureError:
            raise
        except Exception as exc:
            if mode == "vlm_qwen":
                raise
            vlm_fallback_reason = str(exc)
            runtime_blocker = any(
                token in str(exc)
                for token in ("RUNTIME_PATCHES_STALE", "QWEN_VL_4BIT_STATE_LOST", "MEMORY_", "FP8_RUNTIME_UNSUPPORTED")
            )
            if job:
                if runtime_blocker:
                    job.log(
                        "⛔ El inspector quedó bloqueado por un problema de compatibilidad/runtime, no por la escena: "
                        f"{str(exc)[:220]}. El resultado será SOLO fotométrico y NO autoriza vídeo. "
                        "No se reintenta ni se descarga nada: reinicia el proceso del servidor y reaudita.",
                        level="warning",
                    )
                else:
                    job.log(
                        f"Aviso: No se pudo activar Qwen2.5-VL ({exc}); entregando resultado fotométrico OpenCV.",
                        level="warning",
                    )

    comparisons: List[Dict[str, Any]] = []
    inconsistent_rels: List[str] = []

    for idx, (cpath, cv_r) in enumerate(zip(candidate_paths, cv_results), 1):
        crel = _to_rel_str(cpath)
        prev_rel = anchor_rel if idx == 1 else _to_rel_str(candidate_paths[idx - 2])
        vlm_r: Optional[Dict[str, Any]] = None
        if vlm_loaded_ok:
            if job:
                pct = 70.0 + ((idx - 1) / len(candidate_paths)) * 26.0
                job.log(
                    f"Qwen2.5-VL inspeccionando rostros, ropa y avance espacial en Fotograma #{idx} (`{cpath.name}`)...",
                    progress=pct,
                    step=f"Inspección Qwen2.5-VL #{idx}/{len(candidate_paths)}...",
                )
            vlm_r = analyze_pair_qwen_vl(
                anchor_path=anchor_path,
                candidate_path=cpath,
                scene_goal=scene_goal,
                expect_spatial_progression=expect_spatial_progression,
            )

        step_disp_pct = float(cv_r.get("step_displacement_from_prev_pct", cv_r.get("spatial_displacement_pct", 25.0)))
        too_static = bool(
            expect_spatial_progression
            and (step_disp_pct < 8.0 or (vlm_r is not None and vlm_r.get("flf2v_pose_transition_score", 80.0) < 52.0))
        )

        if vlm_r is not None:
            combined = round(0.80 * vlm_r["vlm_overall_score"] + 0.20 * cv_r["cv_overall_score"], 1)
            hard_fail = (
                (not vlm_r["same_people_count"])
                or (vlm_r["character_identity_score"] < threshold)
                or (vlm_r["clothing_hair_score"] < (threshold - 5.0))
                or (not cv_r["aspect_ratio_compatible"])
            )
            passed = (combined >= threshold) and (not hard_fail) and (not too_static)
        else:
            # En escenas en avance el fondo cambia por paralaje, por lo que relajamos el umbral estructural puro
            eff_cv_thresh = (threshold - 16.0) if expect_spatial_progression else (threshold - 10.0)
            combined = cv_r["cv_overall_score"]
            passed = (combined >= eff_cv_thresh) and cv_r["aspect_ratio_compatible"] and (not too_static)

        if passed:
            verdict = "PASS"
        elif too_static:
            verdict = "TOO_STATIC_NEEDS_ADVANCE"
        else:
            verdict = "NEEDS_REPAIR"

        if not passed:
            inconsistent_rels.append(crel)

        repair_prompt = ""
        if not passed:
            if too_static:
                goal_txt = scene_goal.strip() or "walking forward down the same street while conversing"
                repair_prompt = (
                    "Next Scene: The exact same people from the reference image with identical faces, hairstyles and exact same outfits "
                    f"have walked 4 meters further forward ({goal_txt}). Camera tracks backward ahead of them; "
                    "ALTERNATE LEG STRIDE with the opposite foot stepping forward, new natural hand gesture while speaking, "
                    "and background storefronts, trees and streetlamps shifted further back in perspective along the same street under identical lighting."
                )
            elif vlm_r and vlm_r.get("repair_edit_prompt"):
                repair_prompt = vlm_r["repair_edit_prompt"]
            else:
                goal_txt = scene_goal.strip() or "la nueva pose deseada para este plano"
                repair_prompt = (
                    "Mantén EXACTAMENTE las mismas personas, rostros, peinados, ropa, iluminación y escenario de la imagen original; "
                    f"modifica únicamente la acción/postura de los personajes para mostrar: {goal_txt}"
                )

        repair_call = None
        if not passed:
            ref_list = [anchor_rel] if prev_rel == anchor_rel else [anchor_rel, prev_rel]
            repair_args: Dict[str, Any] = {
                "model_id": "qwen-image-edit-2511-pack",
                "base_image_rel": prev_rel if expect_spatial_progression else anchor_rel,
                "reference_rels": ref_list,
                "use_official_defaults": True,
                "prompt": repair_prompt,
            }
            if expect_spatial_progression:
                repair_args["custom_lora_ids"] = [
                    "qwen-lora-next-scene",
                    "qwen-lora-skin-realism",
                    "qwen-lora-anylight-env",
                ]
                repair_args["lora_scales"] = {
                    "qwen-lora-next-scene": 0.85,
                    "qwen-lora-skin-realism": 0.75,
                    "qwen-lora-anylight-env": 0.65,
                }
            repair_call = {
                "tool": "studio_generate_or_edit_image",
                "arguments": repair_args,
            }

        comp_entry: Dict[str, Any] = {
            "index": idx,
            "candidate_rel": crel,
            "verdict": verdict,
            "overall_coherence_score": combined,
            "assessment_scope": "semantic_and_photometric" if vlm_loaded_ok else "photometric_only_not_identity",
            "effective_threshold": threshold if vlm_loaded_ok else eff_cv_thresh,
            "fast_cv": cv_r,
            "semantic_vlm": vlm_r,
            "recommended_repair_tool_call": repair_call,
        }
        comparisons.append(comp_entry)

        icon_v = "✅" if passed else "⚠️"
        detail_msg = (
            f"identidad={vlm_r['character_identity_score']} · ropa/pelo={vlm_r['clothing_hair_score']} · entorno={vlm_r['environment_props_score']}"
            if vlm_r
            else f"color={cv_r['color_palette_score']} · luz={cv_r['lighting_exposure_score']}"
        )
        colab_console_log(
            f"[Coherencia #{idx}] `{cpath.name}` vs `{anchor_path.name}` → {combined}/100 ({verdict}) [{detail_msg}]",
            icon=icon_v,
        )

    comparisons_ok = len(inconsistent_rels) == 0
    audit_blocked = bool(vlm_fallback_reason) and any(
        token in str(vlm_fallback_reason)
        for token in ("RUNTIME_PATCHES_STALE", "QWEN_VL_4BIT_STATE_LOST")
    )
    cv_threshold = (threshold - 16.0) if expect_spatial_progression else (threshold - 10.0)
    photometric_ok = all(
        r["cv_overall_score"] >= cv_threshold and r["aspect_ratio_compatible"]
        and (not expect_spatial_progression or r["step_displacement_from_prev_pct"] >= 8.0)
        for r in cv_results
    )
    # Explicit fast_cv remains a photometric filter for callers that requested it.
    # A requested semantic audit that fell back to CV cannot clear the GPU
    # workflow's identity gate, even if all relaxed CV thresholds pass.
    all_ok = comparisons_ok and (vlm_loaded_ok or mode == "fast_cv")
    avg_score = round(sum(c["overall_coherence_score"] for c in comparisons) / len(comparisons), 1)

    return {
        "ok": True,
        "engine_used": "hybrid_opencv_plus_qwen25_vl_3b" if vlm_loaded_ok else "opencv_fast_cv_only",
        "vlm_semantic_used": vlm_loaded_ok,
        "vlm_fallback_reason": vlm_fallback_reason,
        "anchor_image_rel": anchor_rel,
        "strictness_threshold": threshold,
        "effective_comparison_threshold": threshold if vlm_loaded_ok else ((threshold - 16.0) if expect_spatial_progression else (threshold - 10.0)),
        "assessment_scope": "semantic_and_photometric" if vlm_loaded_ok else "photometric_only_not_identity",
        "audit_blocked_by_runtime": audit_blocked,
        "recommended_action": (
            "Restart the server process, then rerun mode='vlm_qwen' for these frames; "
            "weights are already on disk and nothing needs re-downloading."
            if audit_blocked
            else ""
        ),
        "photometric_consistent": photometric_ok,
        "identity_verified": vlm_loaded_ok and all_ok,
        "ready_for_video": vlm_loaded_ok and all_ok,
        "all_consistent": all_ok,
        "average_coherence_score": avg_score,
        "inconsistent_frames": inconsistent_rels,
        "comparisons": comparisons,
        "agent_next_step": (
            ("Solo se ha evaluado color/luz/estructura, NO identidad de personajes ni avance 3D real. "
             "No iniciar vídeo automáticamente ni dar por aprobada la auditoría semántica. "
             "Revisar vlm_fallback_reason y repetir mode='vlm_qwen' o validar visualmente de forma explícita.")
            if not vlm_loaded_ok else
            "Auditoría semántica y fotométrica superada según los umbrales del inspector; no es una garantía de identidad perfecta. Puedes continuar con `studio_generate_video` tras revisar el resultado."
            if all_ok
            else (
                "Uno o más fotogramas perdieron consistencia respecto al ancla (`NEEDS_REPAIR`). "
                "Ejecuta `recommended_repair_tool_call` de cada fotograma fallido (usando `qwen-image-edit-2511-pack` con `base_image_rel` apuntando al ancla) "
                "y vuelve a verificar con `studio_analyze_coherence` antes de generar el vídeo."
            )
        ),
    }
