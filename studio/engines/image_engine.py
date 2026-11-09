import inspect
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
        pipe.load_lora_weights(str(lora_dir), weight_name=filename, adapter_name=adapter_name)
        return
    pipe.load_lora_weights(str(lora_dir), adapter_name=adapter_name)


def _instantiate_image_pipeline(engine: str, base_dir: Path) -> Any:
    import torch  # type: ignore
    import diffusers  # type: ignore

    bdir = str(base_dir)
    if engine == "qwen_image_edit":
        from diffusers import QwenImageEditPlusPipeline  # type: ignore

        return QwenImageEditPlusPipeline.from_pretrained(bdir, torch_dtype=torch.bfloat16)

    if engine == "qwen_image_t2i":
        from diffusers import QwenImagePipeline  # type: ignore

        return QwenImagePipeline.from_pretrained(bdir, torch_dtype=torch.bfloat16)

    if engine == "sdxl_turbo":
        from diffusers import AutoPipelineForText2Image  # type: ignore

        try:
            return AutoPipelineForText2Image.from_pretrained(bdir, torch_dtype=torch.float16, variant="fp16")
        except Exception:
            return AutoPipelineForText2Image.from_pretrained(bdir, torch_dtype=torch.float16)

    return diffusers.DiffusionPipeline.from_pretrained(bdir, torch_dtype=torch.bfloat16)


def _is_image_lora_compatible(ldef: Dict[str, Any], model_id: str, engine: Optional[str] = None) -> bool:
    if ldef.get("category") != "lora" or ldef.get("lora_target", "image") != "image":
        return False
    compat = ldef.get("compatible_models")
    if compat:
        return model_id in compat
    if engine is None:
        mdef = catalog_manager.get_definition(model_id) or {}
        engine = str(mdef.get("engine") or "")
    parent = str(ldef.get("parent_model") or "")
    l_eng = str(ldef.get("engine") or "")
    if engine == "sdxl_turbo":
        return "sdxl" in parent or "sdxl" in l_eng
    if engine in ("qwen_image_edit", "qwen_image_t2i"):
        return "sdxl" not in l_eng and ("qwen" in parent or "qwen" in l_eng or not parent)
    return True


def load_image_model_into_memory(
    model_id: str = "qwen-image-edit-2511-pack",
    extra_lora_ids: Optional[List[str]] = None,
    auto_download: bool = False,
    job: Optional[JobRecord] = None,
) -> Dict[str, Any]:
    """Carga un modelo de imagen del catálogo (+ LoRAs opcionales compatibles) con progreso continuo 0-100%."""
    defn = catalog_manager.get_definition(model_id)
    if not defn:
        raise ValueError(f"Modelo de imagen no encontrado: {model_id}")
    target_engine = defn.get("engine", "qwen_image_edit")

    # Comprobar qué LoRAs de imagen compatibles con este modelo están hoy descargados en disco
    downloaded_loras = {
        item["id"]
        for item in catalog_manager.get_all_definitions()
        if _is_image_lora_compatible(item, model_id, target_engine)
        and catalog_manager.is_downloaded(item["id"])
    }
    if extra_lora_ids:
        for lid in extra_lora_ids:
            ld = catalog_manager.get_definition(lid)
            if ld and _is_image_lora_compatible(ld, model_id, target_engine) and catalog_manager.is_downloaded(lid):
                downloaded_loras.add(lid)

    cur_img = memory_manager.get_slot("image")
    if cur_img and cur_img.get("model_id") == model_id:
        already_loaded_ids = set(cur_img.get("adapters", []))
        if downloaded_loras.issubset(already_loaded_ids):
            vram_now = query_gpu_info()["vram_used_gb"]
            avail = list(cur_img["handle"].get("adapters", {}).keys())
            if job:
                job.log(
                    f"«{defn['name']}» ya está cargado y listo en RAM/VRAM ({vram_now} GB).",
                    level="success",
                    progress=100.0,
                    step="Modelo ya activo en memoria",
                )
            return {
                "model_id": model_id,
                "adapters_available": avail,
                "vram_allocated_gb": vram_now,
            }

    try:
        import torch  # type: ignore
        import diffusers  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            f"Faltan librerías de GPU/inferencia ({exc}). "
            "En Google Colab se instalan automáticamente con requirements-colab.txt."
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No se detectó GPU NVIDIA activa en este entorno (estás en modo CPU). "
            "Activa una GPU en Google Colab (T4, L4 o A100) para cargar el modelo de imagen."
        )

    was_downloaded = catalog_manager.is_downloaded(model_id)
    if not was_downloaded:
        if auto_download and job is not None:
            job.log(
                f"Fase 1/2: Descargando «{defn['name']}» de Hugging Face (0% → 70%)...",
                progress=1.0,
                step="Fase 1/2: Iniciando descarga HF...",
            )
            catalog_manager.download_model(model_id, job, progress_range=(1.0, 70.0))
            load_p_start = 70.0
        else:
            raise RuntimeError(
                f"«{defn['name']}» aún no está descargado en disco. "
                "Pulsa «Preparar Modelo» o «Descargar + Cargar»."
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

    # Liberar cualquier modelo previo (vídeo o imagen anterior) antes de alojar el nuevo
    for prev_slot in ("video", "image"):
        ps = memory_manager.get_slot(prev_slot)
        if ps:
            if job:
                job.log(
                    f"Liberando «{ps['name']}» de RAM/VRAM antes de cargar «{defn['name']}»...",
                    progress=_map_load_pct(0.06),
                    step=f"Liberando VRAM previa ({prev_slot})...",
                )
            memory_manager.unload_slot(prev_slot)

    flush_ram_and_vram()

    m_dir = catalog_manager.model_dir(model_id)
    base_dir = m_dir / "base"
    if not base_dir.exists():
        base_dir = m_dir / defn["artifacts"][0]["subdir"]
    engine = defn.get("engine", "qwen_image_edit")

    p_pre_start = _map_load_pct(0.10)
    p_pre_end = _map_load_pct(0.68)
    if job:
        job.log(
            f"Cargando pesos de «{defn['name']}» desde disco a RAM/VRAM...",
            progress=p_pre_start,
            step="Cargando pesos en memoria...",
        )

    pipe: Any = None
    try:
        pipe = run_with_live_memory_progress(
            job=job,
            start_pct=p_pre_start,
            end_pct=p_pre_end,
            step_label=f"Cargando tensores {defn['name']}",
            fn=lambda: _instantiate_image_pipeline(engine, base_dir),
            expected_seconds=45.0,
        )

        if job:
            job.log(
                "Activando CPU Offload inteligente para optimizar VRAM...",
                progress=_map_load_pct(0.72),
                step="Activando CPU Offload...",
            )
        if hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")

        loaded_adapters: Dict[str, str] = {}

        # Cargar LoRA Lightning 4 pasos incluido en packs Qwen (Edit 2511 o T2I 2512)
        lightning_dir = m_dir / "lora_lightning"
        lightning_candidates = [
            "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
            "Qwen-Image-Lightning-4steps-V2.0-bf16.safetensors",
        ]
        lightning_name = next((fn for fn in lightning_candidates if (lightning_dir / fn).exists()), None)
        if lightning_name:
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

        # Cargar LoRAs adicionales compatibles con la arquitectura de imagen activa (Qwen Edit, Qwen T2I o SDXL Turbo)
        if engine in ("qwen_image_edit", "qwen_image_t2i", "sdxl_turbo"):
            for lid in sorted(downloaded_loras):
                ldef = catalog_manager.get_definition(lid)
                if not ldef or not catalog_manager.is_downloaded(lid):
                    continue
                if not _is_image_lora_compatible(ldef, model_id, engine):
                    continue
                art = ldef["artifacts"][0]
                l_sub = catalog_manager.model_dir(lid) / art["subdir"]
                l_fn = art.get("filename")
                adapter_name = ldef.get("adapter_name") or lid
                try:
                    if job:
                        job.log(
                            f"Cargando LoRA compatible «{ldef['name']}»...",
                            progress=_map_load_pct(0.93),
                            step=f"Cargando {adapter_name}...",
                        )
                    _safe_load_lora(pipe, l_sub, l_fn, adapter_name)
                    loaded_adapters[adapter_name] = lid
                except Exception as e:
                    if job:
                        job.log(f"Aviso al cargar LoRA opcional {lid}: {e}", level="warning")

        if loaded_adapters and hasattr(pipe, "set_adapters"):
            default_active = ["lightning"] if "lightning" in loaded_adapters else list(loaded_adapters.keys())[:1]
            pipe.set_adapters(default_active, adapter_weights=[1.0] * len(default_active))

        gpu_now = query_gpu_info()
        vram_gb = gpu_now["vram_used_gb"]
        memory_manager.register_loaded(
            slot="image",
            model_id=model_id,
            name=defn["name"],
            handle={"pipe": pipe, "adapters": loaded_adapters, "engine": engine},
            adapters=list(loaded_adapters.values()),
        )
    except Exception:
        if pipe is not None:
            del pipe
        flush_ram_and_vram()
        raise

    if job:
        job.log(
            f"«{defn['name']}» listo ({', '.join(loaded_adapters.keys()) or 'base'}). VRAM en uso: {vram_gb} GB.",
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
    model_id: Optional[str] = None,
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
    lora_scales: Optional[Dict[str, float]] = None,
    auto_download_loras: bool = True,
    use_ffmpeg_bt709: bool = True,
    mask_feather_px: int = 8,
    job: Optional[JobRecord] = None,
) -> Dict[str, Any]:
    """Ejecuta edición simple, edición multi-referencia, inpainting por máscara o creación base con apilado Multi-LoRA."""
    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("Escribe un prompt describiendo qué quieres generar o editar.")

    slot = memory_manager.get_slot("image")
    if model_id and (not slot or slot.get("model_id") != model_id):
        if catalog_manager.is_downloaded(model_id):
            load_image_model_into_memory(model_id=model_id, extra_lora_ids=custom_lora_ids, auto_download=False, job=job)
            slot = memory_manager.get_slot("image")
    if not slot:
        raise RuntimeError(
            "No hay ningún modelo de imagen cargado en RAM/VRAM. "
            "Pulsa «Preparar Modelo» arriba para cargarlo antes de generar."
        )

    import torch  # type: ignore

    pipe = slot["handle"]["pipe"]
    engine = slot["handle"].get("engine", "qwen_image_edit")
    loaded_model_id = str(slot.get("model_id") or "qwen-image-edit-2511-pack")
    available_adapters: Dict[str, str] = slot["handle"].get("adapters", {})
    scales_map: Dict[str, float] = {str(k): float(v) for k, v in (lora_scales or {}).items()}

    t0 = time.time()
    flush_ram_and_vram()

    width = max(512, min(1536, (int(width) // 64) * 64))
    height = max(512, min(1536, (int(height) // 64) * 64))
    steps = max(1, min(50, int(steps)))
    actual_seed = int(seed) if int(seed) > 0 else random.randint(1, 2_147_483_647)

    # Configurar adaptadores LoRA solicitados (y cargar en caliente si se acaban de descargar)
    active_names: List[str] = []
    active_weights: List[float] = []
    injected_triggers: List[str] = []

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
        cid = str(cid or "").strip()
        if not cid:
            continue
        cdef = catalog_manager.get_definition(cid)
        if not cdef:
            continue
        if not _is_image_lora_compatible(cdef, loaded_model_id, engine):
            if job:
                job.log(
                    f"Aviso: el LoRA «{cdef['name']}» no es compatible con «{loaded_model_id}»; se omite.",
                    level="warning",
                )
            continue
        if not catalog_manager.is_downloaded(cid):
            if auto_download_loras and job is not None:
                job.log(
                    f"Descargando LoRA creativo «{cdef['name']}» ({cdef.get('estimated_gb', 0.3)} GB)...",
                    progress=6.0,
                    step=f"Descargando {cdef['name'][:32]}...",
                )
                catalog_manager.download_model(cid, job, progress_range=(5.0, 9.0))
            else:
                if job:
                    job.log(f"Aviso: LoRA «{cid}» no descargado aún; descárgalo primero.", level="warning")
                continue

        aname = cdef.get("adapter_name") or cid
        if aname not in available_adapters:
            try:
                art = cdef["artifacts"][0]
                l_sub = catalog_manager.model_dir(cid) / art["subdir"]
                l_fn = art.get("filename")
                if hasattr(pipe, "remove_all_hooks"):
                    try:
                        pipe.remove_all_hooks()
                    except Exception:
                        pass
                _safe_load_lora(pipe, l_sub, l_fn, aname)
                if hasattr(pipe, "enable_model_cpu_offload"):
                    pipe.enable_model_cpu_offload()
                available_adapters[aname] = cid
                if cid not in slot.get("adapters", []):
                    slot.setdefault("adapters", []).append(cid)
            except Exception as exc:
                if job:
                    job.log(f"Aviso al acoplar LoRA {cid} en caliente: {exc}", level="warning")
                continue

        if aname in available_adapters and aname not in active_names:
            w_val = float(scales_map.get(cid, cdef.get("default_lora_scale", 0.85)))
            active_names.append(aname)
            active_weights.append(max(0.05, min(2.0, w_val)))
            trig = str(cdef.get("trigger_phrase") or "").strip()
            if trig and trig.lower() not in prompt.lower():
                injected_triggers.append(trig)

    if injected_triggers:
        prompt = f"{', '.join(injected_triggers)}, {prompt}"

    if hasattr(pipe, "set_adapters") and available_adapters:
        if active_names:
            pipe.set_adapters(active_names, adapter_weights=active_weights)
        elif hasattr(pipe, "disable_lora"):
            pipe.disable_lora()

    if job:
        job.log(
            f"Modelo: {slot['name']} · LoRAs: {', '.join(active_names) if active_names else 'ninguno'} · Semilla: {actual_seed}",
            progress=10.0,
            step="Preparando imágenes de entrada...",
        )

    original_for_inpaint: Optional[Image.Image] = None
    has_base = bool(base_image_path and base_image_path.exists())
    base_resolved = base_image_path.resolve() if has_base and base_image_path else None

    if has_base and base_image_path:
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

    refs_pil: List[Image.Image] = []
    seen_resolved = {base_resolved} if base_resolved else set()
    for rpath in reference_paths or []:
        if rpath and rpath.exists():
            rp_res = rpath.resolve()
            if rp_res in seen_resolved:
                continue
            seen_resolved.add(rp_res)
            r_img = process_image_file_bt709(rpath) if use_ffmpeg_bt709 else Image.open(rpath).convert("RGB")
            if r_img.size[0] > 1024 or r_img.size[1] > 1024:
                r_img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            refs_pil.append(r_img)

    has_mask = bool(mask_image_path and mask_image_path.exists() and original_for_inpaint is not None)
    if has_mask:
        mode_name = "inpainting_mascara"
    elif has_base and refs_pil:
        mode_name = f"multi_referencia ({1 + len(refs_pil)} imgs)"
    elif refs_pil:
        mode_name = f"referencia_ancla ({len(refs_pil)} ref)"
    elif has_base:
        mode_name = "edicion_anclada"
    else:
        mode_name = "lienzo_nuevo"

    if has_base:
        pipe_images: Any = [base_pil] + refs_pil if refs_pil else base_pil
    elif refs_pil:
        pipe_images = refs_pil if len(refs_pil) > 1 else refs_pil[0]
    else:
        pipe_images = base_pil
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

    call_sig = inspect.signature(pipe.__call__).parameters
    if (has_base or refs_pil) and "image" not in call_sig:
        raise ValueError(
            f"El modelo actualmente cargado («{slot['name']}») es solo Texto→Imagen puro y NO admite imágenes de referencia ni anclas. "
            "Carga o selecciona «qwen-image-edit-2511-pack» para mantener la identidad de personajes y editar fotogramas."
        )
    effective_true_cfg = 1.0 if ("lightning" in active_names and not negative_prompt.strip()) else float(guidance_scale)
    effective_neg = (negative_prompt.strip() or " ") if effective_true_cfg > 1.0 else None

    call_kwargs: Dict[str, Any] = {
        "prompt": prompt,
        "num_inference_steps": steps,
        "generator": generator,
        "output_type": "pil",
        "height": height,
        "width": width,
    }
    if "image" in call_sig:
        call_kwargs["image"] = pipe_images
    if "true_cfg_scale" in call_sig:
        call_kwargs["true_cfg_scale"] = effective_true_cfg
        call_kwargs["negative_prompt"] = effective_neg
    else:
        if "guidance_scale" in call_sig:
            call_kwargs["guidance_scale"] = 0.0 if engine == "sdxl_turbo" else float(guidance_scale)
        if "negative_prompt" in call_sig and negative_prompt.strip():
            call_kwargs["negative_prompt"] = negative_prompt.strip()
    if "callback_on_step_end" in call_sig:
        call_kwargs["callback_on_step_end"] = _on_step_end

    try:
        out = pipe(**call_kwargs)
    except TypeError:
        call_kwargs.pop("callback_on_step_end", None)
        out = pipe(**call_kwargs)

    result_pil: Image.Image = out.images[0].convert("RGB")

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
