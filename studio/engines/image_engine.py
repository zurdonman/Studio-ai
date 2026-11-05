import json
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from studio.catalog import catalog_manager
from studio.config import IMAGES_OUT_DIR
from studio.engines.ffmpeg_utils import (
    apply_inpaint_composite,
    process_image_file_bt709,
    process_pil_bt709,
    verify_image_file,
)
from studio.hardware import query_gpu_info
from studio.jobs import JobRecord
from studio.memory_manager import (
    flush_ram_and_vram,
    memory_manager,
    run_with_live_memory_progress,
)


def _safe_load_lora(pipe: Any, lora_dir: Path, filename: Optional[str], adapter_name: str) -> None:
    if filename:
        file_path = lora_dir / filename
        try:
            pipe.load_lora_weights(str(lora_dir), weight_name=filename, adapter_name=adapter_name)
            return
        except Exception:
            pipe.load_lora_weights(str(file_path), adapter_name=adapter_name)
            return
    pipe.load_lora_weights(str(lora_dir), adapter_name=adapter_name)


def load_image_model_into_memory(
    model_id: str = "qwen-image-edit-2511-pack",
    extra_lora_ids: Optional[List[str]] = None,
    auto_download: bool = False,
    job: Optional[JobRecord] = None,
) -> Dict[str, Any]:
    """Carga Qwen-Image-Edit 2511 (4-bit) + LoRA Lightning 4 pasos (+ LoRAs opcionales) con progreso continuo 0-100%."""
    defn = catalog_manager.get_definition(model_id)
    if not defn:
        raise ValueError(f"Modelo de imagen no encontrado: {model_id}")

    try:
        import torch  # type: ignore
        from diffusers import QwenImageEditPlusPipeline  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            f"Faltan librerías de GPU/inferencia ({exc}). "
            "En Google Colab se instalan automáticamente con requirements-colab.txt."
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No se detectó GPU NVIDIA activa en este entorno (estás en modo CPU). "
            "Activa una GPU en Google Colab (T4, L4 o A100) para cargar Qwen 4-bit."
        )

    was_downloaded = catalog_manager.is_downloaded(model_id)
    if not was_downloaded:
        if auto_download and job is not None:
            job.log(
                "Fase 1/2: El pack aún no está en disco. Descargando de Hugging Face (0% → 70%)...",
                progress=1.0,
                step="Fase 1/2: Iniciando descarga HF...",
            )
            catalog_manager.download_model(model_id, job, progress_range=(1.0, 70.0))
            load_p_start = 70.0
        else:
            raise RuntimeError(
                f"«{defn['name']}» aún no está descargado en disco. "
                "Pulsa «Preparar Qwen + Lightning» o «Descargar + Cargar»."
            )
    else:
        load_p_start = 4.0
        if job:
            job.log(
                "Archivos ya verificados en disco. Iniciando carga directa en RAM y VRAM...",
                progress=load_p_start,
                step="Verificando memoria GPU...",
            )

    def _map_load_pct(frac: float) -> float:
        return round(load_p_start + frac * (100.0 - load_p_start), 1)

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Si hay un modelo pesado de vídeo ocupando VRAM, liberarlo automáticamente primero
    vid_slot = memory_manager.get_slot("video")
    if vid_slot:
        if job:
            job.log(
                f"Liberando modelo de vídeo «{vid_slot['name']}» de VRAM antes de cargar imagen...",
                progress=_map_load_pct(0.06),
                step="Liberando VRAM previa...",
            )
        memory_manager.unload_slot("video")

    flush_ram_and_vram()

    m_dir = catalog_manager.model_dir(model_id)
    base_dir = m_dir / "base"
    if not base_dir.exists():
        base_dir = m_dir / defn["artifacts"][0]["subdir"]

    p_pre_start = _map_load_pct(0.10)
    p_pre_end = _map_load_pct(0.68)
    if job:
        job.log(
            f"Cargando pesos 4-bit de QwenImageEditPlusPipeline desde disco a RAM/VRAM...",
            progress=p_pre_start,
            step="Cargando pesos 4-bit en memoria...",
        )

    pipe = run_with_live_memory_progress(
        job=job,
        start_pct=p_pre_start,
        end_pct=p_pre_end,
        step_label="Cargando tensores Qwen 4-bit",
        fn=lambda: QwenImageEditPlusPipeline.from_pretrained(
            str(base_dir),
            torch_dtype=torch.bfloat16,
        ),
        expected_seconds=50.0,
    )

    if job:
        job.log(
            "Activando CPU Offload inteligente para optimizar VRAM...",
            progress=_map_load_pct(0.72),
            step="Activando CPU Offload...",
        )
    pipe.enable_model_cpu_offload()

    loaded_adapters: Dict[str, str] = {}

    # Cargar LoRA Lightning 4 pasos incluido en el pack
    lightning_dir = m_dir / "lora_lightning"
    lightning_name = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
    lightning_file = lightning_dir / lightning_name
    if lightning_file.exists():
        p_l_start = _map_load_pct(0.76)
        p_l_end = _map_load_pct(0.90)
        if job:
            job.log(
                "Integrando LoRA Lightning (aceleración 4 pasos)...",
                progress=p_l_start,
                step="Cargando LoRA Lightning...",
            )
        run_with_live_memory_progress(
            job=job,
            start_pct=p_l_start,
            end_pct=p_l_end,
            step_label="Fusionando LoRA Lightning",
            fn=lambda: _safe_load_lora(pipe, lightning_dir, lightning_name, "lightning"),
            expected_seconds=12.0,
        )
        loaded_adapters["lightning"] = model_id

    # Cargar LoRAs adicionales descargados (ej. qwen-lora-nsfw o LoRAs custom de HF)
    candidate_loras = set(extra_lora_ids or [])
    for item in catalog_manager.get_all_definitions():
        if item["category"] == "lora" and catalog_manager.is_downloaded(item["id"]):
            candidate_loras.add(item["id"])

    for lid in sorted(candidate_loras):
        ldef = catalog_manager.get_definition(lid)
        if not ldef or not catalog_manager.is_downloaded(lid):
            continue
        art = ldef["artifacts"][0]
        l_sub = catalog_manager.model_dir(lid) / art["subdir"]
        l_fn = art.get("filename")
        adapter_name = ldef.get("adapter_name") or lid
        try:
            if job:
                job.log(
                    f"Cargando LoRA opcional «{ldef['name']}»...",
                    progress=_map_load_pct(0.93),
                    step=f"Cargando {adapter_name}...",
                )
            _safe_load_lora(pipe, l_sub, l_fn, adapter_name)
            loaded_adapters[adapter_name] = lid
        except Exception as e:
            if job:
                job.log(f"Aviso al cargar LoRA opcional {lid}: {e}", level="warning")

    if loaded_adapters:
        default_active = ["lightning"] if "lightning" in loaded_adapters else list(loaded_adapters.keys())[:1]
        pipe.set_adapters(default_active, adapter_weights=[1.0] * len(default_active))

    gpu_now = query_gpu_info()
    vram_gb = gpu_now["vram_used_gb"]
    memory_manager.register_loaded(
        slot="image",
        model_id=model_id,
        name=defn["name"],
        handle={"pipe": pipe, "adapters": loaded_adapters},
        adapters=list(loaded_adapters.values()),
    )

    if job:
        job.log(
            f"Qwen + LoRAs listos ({', '.join(loaded_adapters.keys()) or 'base'}). VRAM en uso: {vram_gb} GB.",
            level="success",
            progress=100.0,
            step="Modelo listo en RAM/VRAM (100%)",
        )
    return {
        "model_id": model_id,
        "adapters_available": list(loaded_adapters.keys()),
        "vram_allocated_gb": vram_gb,
    }


def generate_or_edit_image(
    prompt: str,
    base_image_path: Optional[Path] = None,
    reference_paths: Optional[List[Path]] = None,
    mask_image_path: Optional[Path] = None,
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1024,
    steps: int = 4,
    guidance_scale: float = 4.0,
    seed: int = 0,
    use_lightning_lora: bool = True,
    use_nsfw_lora: bool = False,
    nsfw_lora_weight: float = 1.0,
    custom_lora_ids: Optional[List[str]] = None,
    use_ffmpeg_bt709: bool = True,
    mask_feather_px: int = 8,
    job: Optional[JobRecord] = None,
) -> Dict[str, Any]:
    """Ejecuta edición simple, edición multi-referencia, inpainting por máscara o creación base."""
    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("Escribe un prompt describiendo qué quieres generar o editar.")

    slot = memory_manager.get_slot("image")
    if not slot:
        raise RuntimeError(
            "No hay ningún modelo de imagen cargado en RAM/VRAM. "
            "Pulsa «Preparar Qwen + Lightning» arriba para cargarlo antes de generar."
        )

    import torch  # type: ignore

    pipe = slot["handle"]["pipe"]
    available_adapters: Dict[str, str] = slot["handle"].get("adapters", {})

    t0 = time.time()
    flush_ram_and_vram()

    width = max(512, min(1536, (int(width) // 64) * 64))
    height = max(512, min(1536, (int(height) // 64) * 64))
    steps = max(1, min(50, int(steps)))
    actual_seed = int(seed) if int(seed) > 0 else random.randint(1, 2_147_483_647)

    # Configurar adaptadores LoRA solicitados para esta ejecución
    active_names: List[str] = []
    active_weights: List[float] = []
    if use_lightning_lora and "lightning" in available_adapters:
        active_names.append("lightning")
        active_weights.append(1.0)
    if use_nsfw_lora:
        if "mcnl-nsfw-v1" in available_adapters:
            active_names.append("mcnl-nsfw-v1")
            active_weights.append(float(nsfw_lora_weight))
        elif job:
            job.log("Aviso: LoRA NSFW marcado pero no está descargado/cargado; continuando sin él.", level="warning")

    for cid in custom_lora_ids or []:
        cdef = catalog_manager.get_definition(cid)
        aname = (cdef.get("adapter_name") if cdef else None) or cid
        if aname in available_adapters and aname not in active_names:
            active_names.append(aname)
            active_weights.append(1.0)

    if hasattr(pipe, "set_adapters") and available_adapters:
        if active_names:
            pipe.set_adapters(active_names, adapter_weights=active_weights)
        elif hasattr(pipe, "disable_lora"):
            pipe.disable_lora()

    if job:
        job.log(
            f"LoRAs activos: {', '.join(active_names) if active_names else 'ninguno'} · Semilla: {actual_seed}",
            progress=10.0,
            step="Preparando imágenes de entrada...",
        )

    # Preparar imagen principal (o lienzo limpio si es generación desde cero)
    original_for_inpaint: Optional[Image.Image] = None
    if base_image_path and base_image_path.exists():
        base_pil = (
            process_image_file_bt709(base_image_path)
            if use_ffmpeg_bt709
            else Image.open(base_image_path).convert("RGB")
        )
        original_for_inpaint = base_pil.copy()
    else:
        base_pil = Image.new("RGB", (width, height), color=(128, 128, 128))

    if base_pil.size[0] > 1024 or base_pil.size[1] > 1024:
        base_pil.thumbnail((1024, 1024), Image.Resampling.LANCZOS)

    # Preparar referencias multi-imagen si las hay
    refs_pil: List[Image.Image] = []
    for rpath in reference_paths or []:
        if rpath and rpath.exists():
            r_img = process_image_file_bt709(rpath) if use_ffmpeg_bt709 else Image.open(rpath).convert("RGB")
            if r_img.size[0] > 1024 or r_img.size[1] > 1024:
                r_img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            refs_pil.append(r_img)

    # Determinar modo de operación para trazabilidad
    has_mask = bool(mask_image_path and mask_image_path.exists() and original_for_inpaint is not None)
    if has_mask:
        mode_name = "inpainting_mascara"
    elif refs_pil:
        mode_name = f"multi_referencia ({len(refs_pil)} ref)"
    elif base_image_path and base_image_path.exists():
        mode_name = "edicion_simple"
    else:
        mode_name = "lienzo_nuevo"

    pipe_images: Any = [base_pil] + refs_pil if refs_pil else base_pil
    generator = torch.Generator(device="cuda").manual_seed(actual_seed)

    step_start = {"ts": time.time()}

    def _on_step_end(_pipe: Any, step_index: int, _timestep: Any, callback_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        if job:
            job.check_cancelled()
            now = time.time()
            dt = max(0.01, now - step_start["ts"])
            step_start["ts"] = now
            cur = step_index + 1
            pct = 15.0 + (cur / steps) * 70.0
            job.log(
                f"Paso de difusión {cur}/{steps} completado ({dt:.2f} s/paso)",
                progress=pct,
                step=f"Inferencia paso {cur}/{steps}",
                speed_text=f"{dt:.2f} s/paso",
            )
        return callback_kwargs

    if job:
        job.log(
            f"Ejecutando {mode_name} ({width}x{height}, {steps} pasos, guía {guidance_scale})...",
            progress=15.0,
            step=f"Paso 0/{steps}...",
        )

    call_kwargs: Dict[str, Any] = {
        "image": pipe_images,
        "prompt": prompt,
        "negative_prompt": negative_prompt.strip() or None,
        "guidance_scale": float(guidance_scale),
        "num_inference_steps": steps,
        "generator": generator,
        "output_type": "pil",
        "height": height,
        "width": width,
        "callback_on_step_end": _on_step_end,
    }

    try:
        out = pipe(**call_kwargs)
    except TypeError:
        # Compatibilidad si una versión concreta de pipeline no acepta callback_on_step_end
        call_kwargs.pop("callback_on_step_end", None)
        out = pipe(**call_kwargs)

    result_pil: Image.Image = out.images[0].convert("RGB")

    # Si había máscara de inpainting pintada, componer sobre la imagen original preservando el exterior
    if has_mask and mask_image_path and original_for_inpaint is not None:
        if job:
            job.log(
                f"Aplicando máscara de inpainting con borde suavizado ({mask_feather_px}px)...",
                progress=88.0,
                step="Fusionando máscara de inpainting...",
            )
        with Image.open(mask_image_path) as m_im:
            mask_pil = m_im.convert("L").copy()
        result_pil = apply_inpaint_composite(
            original=original_for_inpaint,
            edited=result_pil,
            mask=mask_pil,
            feather_px=int(mask_feather_px),
        )

    if use_ffmpeg_bt709:
        if job:
            job.log(
                "Normalizando salida con FFmpeg (sin metadatos + BT.709 JPG calidad 95)...",
                progress=93.0,
                step="Postprocesado FFmpeg BT.709...",
            )
        result_pil = process_pil_bt709(result_pil)

    IMAGES_OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts_slug = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = f"img_{ts_slug}_{actual_seed % 10000:04d}.jpg"
    out_path = IMAGES_OUT_DIR / out_name
    result_pil.save(out_path, "JPEG", quality=95)

    w_out, h_out = verify_image_file(out_path)
    elapsed = round(time.time() - t0, 2)

    meta = {
        "id": out_path.stem,
        "filename": out_name,
        "media_url": f"/media/outputs/images/{out_name}",
        "type": "image",
        "model_id": slot["model_id"],
        "model_name": slot["name"],
        "active_loras": active_names,
        "mode": mode_name,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": actual_seed,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "width": w_out,
        "height": h_out,
        "ffmpeg_bt709": use_ffmpeg_bt709,
        "elapsed_seconds": elapsed,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    meta_path = IMAGES_OUT_DIR / f"{out_path.stem}.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    flush_ram_and_vram()
    return meta


def process_image_cpu_ffmpeg_only(
    base_image_path: Path,
    width: Optional[int] = None,
    height: Optional[int] = None,
    job: Optional[JobRecord] = None,
) -> Dict[str, Any]:
    """Limpia metadatos, normaliza a BT.709 y reescala opcionalmente usando solo FFmpeg (0 VRAM, funciona en CPU)."""
    t0 = time.time()
    if job:
        job.log("Procesando imagen con FFmpeg BT.709 en CPU...", progress=30.0, step="Ejecutando FFmpeg BT.709...")

    img = process_image_file_bt709(base_image_path)
    if width and height:
        img = img.resize((int(width), int(height)), Image.Resampling.LANCZOS)

    IMAGES_OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts_slug = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = f"bt709_{ts_slug}.jpg"
    out_path = IMAGES_OUT_DIR / out_name
    img.save(out_path, "JPEG", quality=95)
    w_out, h_out = verify_image_file(out_path)

    meta = {
        "id": out_path.stem,
        "filename": out_name,
        "media_url": f"/media/outputs/images/{out_name}",
        "type": "image",
        "model_id": "ffmpeg-bt709-cpu",
        "model_name": "FFmpeg BT.709 (Normalización CPU)",
        "active_loras": [],
        "mode": "normalizacion_ffmpeg",
        "prompt": f"Normalización BT.709 de {base_image_path.name}",
        "negative_prompt": "",
        "seed": 0,
        "steps": 1,
        "guidance_scale": 0.0,
        "width": w_out,
        "height": h_out,
        "ffmpeg_bt709": True,
        "elapsed_seconds": round(time.time() - t0, 2),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    (IMAGES_OUT_DIR / f"{out_path.stem}.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return meta
