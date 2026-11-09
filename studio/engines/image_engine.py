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
from studio.compat_patches import apply_runtime_compatibility_patches
from studio.config import IMAGES_OUT_DIR
from studio.engines.ffmpeg_utils import (
    apply_inpaint_composite,
    process_image_file_bt709,
    process_pil_bt709,
    verify_image_file,
)
from studio.hardware import query_gpu_info
from studio.memory_safety_hooks import guarded_model_load
from studio.jobs import JobRecord
from studio.memory_policy import MemoryPressureError
from studio.model_recipes import REVIEWED_IMAGE_ENGINES
from studio.memory_manager import (
    flush_ram_and_vram,
    memory_manager,
    run_with_live_memory_progress,
    serialized_gpu_operation,
    checkpoint_current_operation,
)


def _safe_load_lora(pipe: Any, lora_dir: Path, filename: Optional[str], adapter_name: str) -> None:
    if filename:
        guarded_model_load(pipe.load_lora_weights, str(lora_dir), weight_name=filename, adapter_name=adapter_name)
        return
    guarded_model_load(pipe.load_lora_weights, str(lora_dir), adapter_name=adapter_name)


def _instantiate_image_pipeline(engine: str, base_dir: Path, definition: Optional[Dict[str, Any]] = None) -> Any:
    apply_runtime_compatibility_patches()
    import torch  # type: ignore
    import diffusers  # type: ignore

    bdir = str(base_dir)
    if engine == "qwen_image_edit":
        from diffusers import QwenImageEditPlusPipeline  # type: ignore

        return guarded_model_load(QwenImageEditPlusPipeline.from_pretrained, bdir, torch_dtype=torch.bfloat16)

    if engine == "qwen_image_t2i":
        from diffusers import QwenImagePipeline  # type: ignore

        return guarded_model_load(QwenImagePipeline.from_pretrained, bdir, torch_dtype=torch.bfloat16)

    if engine == "sdxl_turbo":
        from diffusers import AutoPipelineForText2Image  # type: ignore

        return guarded_model_load(AutoPipelineForText2Image.from_pretrained, bdir, torch_dtype=torch.float16,
                                  variant="fp16", low_cpu_mem_usage=True, local_files_only=True)

    if engine in REVIEWED_IMAGE_ENGINES:
        contract = REVIEWED_IMAGE_ENGINES[engine]
        class_name = contract["classes"][0]
        manifest_path = base_dir / "model_index.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("_class_name") != class_name:
                raise ValueError("La clase local no coincide con la receta revisada; detenerse y revisar el manifiesto.")
            for component, spec in manifest.items():
                if not component.startswith("_") and isinstance(spec, list) and len(spec) == 2 and spec[0] not in (None, "diffusers", "transformers"):
                    raise ValueError("Componente de código externo no autorizado en la receta local.")
        pipeline_cls = getattr(diffusers, class_name, None)
        if pipeline_cls is None:
            raise RuntimeError(f"Falta {class_name}. Requiere diffusers>={contract['diffusers']}; actualizar dependencias y reiniciar, no descargar alternativas.")
        if contract.get("native_bf16"):
            props = torch.cuda.get_device_properties(0)
            if getattr(props, "major", 0) < 8:
                raise RuntimeError("Esta receta necesita BF16 nativo (GPU Ampere/Ada/Hopper). No hay conversión automática a FP16/4-bit para T4.")
        kwargs = {"torch_dtype": getattr(torch, contract["dtype"]), "local_files_only": True, "use_safetensors": True}
        if engine == "sdxl_base":
            kwargs["variant"] = "fp16"
        pipe = guarded_model_load(pipeline_cls.from_pretrained, bdir, **kwargs)
        component_dtypes = ((definition or {}).get("official_spec") or {}).get("component_dtypes", {})
        for component, dtype in component_dtypes.items():
            module = getattr(pipe, component, None)
            if module is not None:
                module.to(getattr(torch, dtype))
        # SANA explicitly requires a BF16/FP32 text encoder and a BF16/FP32 VAE;
        # keep the decode in FP32 rather than downcasting every component to FP16.
        if engine in ("sana_image", "sana_sprint") and getattr(pipe, "vae", None) is not None:
            pipe.vae.to(torch.float32)
        return pipe

    raise ValueError(f"MODEL_ENGINE_UNSUPPORTED: no existe cargador revisado de imagen para {engine!r}. No se ejecutará código remoto.")


def _is_image_lora_compatible(ldef: Dict[str, Any], model_id: str, engine: Optional[str] = None) -> bool:
    if ldef.get("validation_state") == "candidate":
        return False
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
    return False


@serialized_gpu_operation(slots=("image",), managed=True)
def load_image_model_into_memory(
    model_id: str = "qwen-image-edit-2511-pack",
    extra_lora_ids: Optional[List[str]] = None,
    auto_download: bool = False,
    job: Optional[JobRecord] = None,
    force_reload: bool = False,
) -> Dict[str, Any]:
    """Carga un modelo de imagen del catálogo (+ LoRAs opcionales compatibles) con progreso continuo 0-100%."""
    return _load_image_model_into_memory_locked(
        model_id=model_id,
        extra_lora_ids=extra_lora_ids,
        auto_download=auto_download,
        force_reload=force_reload,
        job=job,
    )


def _load_image_model_into_memory_locked(
    model_id: str = "qwen-image-edit-2511-pack",
    extra_lora_ids: Optional[List[str]] = None,
    auto_download: bool = False,
    job: Optional[JobRecord] = None,
    force_reload: bool = False,
) -> Dict[str, Any]:
    defn = catalog_manager.get_definition(model_id)
    if not defn:
        raise ValueError(f"Modelo de imagen no encontrado: {model_id}")
    if defn.get("category") != "image" or defn.get("validation_state") == "candidate":
        raise ValueError("MODEL_RECIPE_UNVALIDATED: se necesita una receta revisada de imagen; no se descargan ni cargan pesos.")
    target_engine = defn.get("engine", "qwen_image_edit")
    dl_job = job or JobRecord("download", f"Auto-descarga {model_id}")

    requested_loras = list(dict.fromkeys(str(lid).strip() for lid in (extra_lora_ids or []) if str(lid).strip()))
    downloaded_loras = {
        item["id"]
        for item in catalog_manager.get_all_definitions()
        if _is_image_lora_compatible(item, model_id, target_engine)
        and catalog_manager.is_downloaded(item["id"])
        and (extra_lora_ids is None or item["id"] in requested_loras)
    }

    cur_img = memory_manager.get_slot("image")
    if not force_reload and cur_img and cur_img.get("model_id") == model_id:
        already_loaded_ids = set(cur_img.get("adapters", []))
        if (downloaded_loras | set(requested_loras)).issubset(already_loaded_ids):
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

    # Do not retain the old pipeline while allocating its replacement.
    del cur_img
    apply_runtime_compatibility_patches()
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

    if target_engine in REVIEWED_IMAGE_ENGINES and REVIEWED_IMAGE_ENGINES[target_engine].get("native_bf16"):
        if getattr(torch.cuda.get_device_properties(0), "major", 0) < 8:
            raise RuntimeError("La receta seleccionada exige BF16 nativo; no descargues pesos para una T4/V100 con esta receta.")

    if not auto_download and not catalog_manager.is_downloaded(model_id):
        raise RuntimeError(f"«{defn['name']}» no está descargado; no se modifica la memoria ni el disco.")
    adapter_defs = []
    for lid in sorted(downloaded_loras | set(requested_loras)):
        ld = catalog_manager.get_definition(lid)
        if not ld or not _is_image_lora_compatible(ld, model_id, target_engine):
            raise ValueError(f"LoRA de imagen desconocido o incompatible con {model_id}: {lid}")
        adapter_defs.append(ld)
    # Liberar cualquier modelo previo (vídeo o imagen anterior) antes de alojar el nuevo
    for prev_slot in ("video", "image"):
        ps = memory_manager.get_slot(prev_slot)
        if ps:
            if job:
                job.log(
                    f"Liberando «{ps['name']}» de RAM/VRAM antes de cargar «{defn['name']}»...",
                    progress=2.0,
                    step=f"Liberando VRAM previa ({prev_slot})...",
                )
            del ps
            memory_manager.unload_slot(prev_slot)

    flush_ram_and_vram()

    memory_manager.prepare_for_load(defn, adapters=adapter_defs, job=job)

    was_downloaded = catalog_manager.is_downloaded(model_id)
    if not was_downloaded:
        if auto_download:
            if job:
                job.log(
                    f"Fase 1/2: Descargando «{defn['name']}» de Hugging Face (0% → 70%)...",
                    progress=5.0,
                    step="Fase 1/2: Iniciando descarga HF...",
                )
            catalog_manager.download_model(model_id, dl_job, progress_range=(5.0, 70.0))
            load_p_start = 70.0
        else:
            raise RuntimeError(
                f"«{defn['name']}» aún no está descargado en disco. "
                "Pulsa «Preparar Modelo» o «Descargar + Cargar»."
            )
    else:
        load_p_start = 6.0
        if job:
            job.log(
                "Archivos ya verificados en disco. Iniciando carga directa en RAM y VRAM...",
                progress=load_p_start,
                step="Verificando memoria GPU...",
            )

    # Mount requested adapters on real tensors, never after CPU offload.
    for lid in requested_loras:
        ld = catalog_manager.get_definition(lid)
        if not ld or not _is_image_lora_compatible(ld, model_id, target_engine):
            raise ValueError(f"LoRA de imagen desconocido o incompatible con {model_id}: {lid}")
        if not catalog_manager.is_downloaded(lid):
            if not auto_download:
                raise RuntimeError(f"LoRA «{lid}» no descargado. Activa auto_download o descárgalo primero.")
            if job:
                job.log(f"Descargando LoRA «{ld['name']}» antes de activar CPU Offload...", step="Preparando adaptadores...")
            catalog_manager.download_model(lid, dl_job, progress_range=(load_p_start, load_p_start))
        downloaded_loras.add(lid)

    def _map_load_pct(frac: float) -> float:
        return round(load_p_start + frac * (100.0 - load_p_start), 1)

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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

    memory_manager.prepare_for_load(defn, adapters=adapter_defs, job=job)
    pipe: Any = None
    try:
        pipe = run_with_live_memory_progress(
            job=job,
            start_pct=p_pre_start,
            end_pct=p_pre_end,
            step_label=f"Cargando tensores {defn['name']}",
            fn=lambda: _instantiate_image_pipeline(engine, base_dir, defn),
            expected_seconds=45.0,
        )

        loaded_adapters: Dict[str, str] = {}

        # IMPORTANTE: Acoplar TODOS los adaptadores LoRA ANTES de `enable_model_cpu_offload()`
        # para evitar errores `Cannot copy out of meta tensor; no data!` sobre capas 4-bit cuantizadas.
        lightning_dir = m_dir / "lora_lightning"
        lightning_candidates = [
            "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
            "Qwen-Image-Lightning-4steps-V2.0-bf16.safetensors",
        ]
        lightning_name = next((fn for fn in lightning_candidates if (lightning_dir / fn).exists()), None)
        if lightning_name:
            p_l_start = _map_load_pct(0.70)
            p_l_end = _map_load_pct(0.84)
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
                            progress=_map_load_pct(0.89),
                            step=f"Cargando {adapter_name}...",
                        )
                    _safe_load_lora(pipe, l_sub, l_fn, adapter_name)
                    loaded_adapters[adapter_name] = lid
                except MemoryPressureError:
                    raise
                except Exception as e:
                    if lid in requested_loras:
                        raise RuntimeError(f"No se pudo cargar el LoRA solicitado {lid}: {e}") from e
                    if job:
                        job.log(f"Aviso al cargar LoRA opcional {lid}: {e}", level="warning")

        if loaded_adapters and hasattr(pipe, "set_adapters"):
            default_active = ["lightning"] if "lightning" in loaded_adapters else list(loaded_adapters.keys())[:1]
            pipe.set_adapters(default_active, adapter_weights=[1.0] * len(default_active))

        if job:
            job.log(
                "Activando CPU Offload inteligente para optimizar VRAM...",
                progress=_map_load_pct(0.95),
                step="Activando CPU Offload...",
            )
        if hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")

        if engine in REVIEWED_IMAGE_ENGINES and getattr(pipe, "vae", None) is not None:
            for method in ("enable_tiling", "enable_slicing"):
                if hasattr(pipe.vae, method):
                    getattr(pipe.vae, method)()

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


@serialized_gpu_operation(slots=("image",), managed=True)
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

    apply_runtime_compatibility_patches()
    slot = memory_manager.get_slot("image")
    target_mid = str(model_id or (slot.get("model_id") if slot else None) or "qwen-image-edit-2511-pack").strip()
    target_def = catalog_manager.get_definition(target_mid) or {}
    if not target_def or target_def.get("category") != "image" or target_def.get("validation_state") == "candidate":
        raise ValueError("MODEL_RECIPE_UNVALIDATED: candidato sin receta revisada; conserva los archivos y consulta el inspector.")
    if (base_image_path or reference_paths) and "i2i" not in target_def.get("capabilities", []):
        raise ValueError("El modelo elegido solo crea imágenes desde texto. No descargues ni cargues esta receta para una edición/referencia.")
    requested_loras = []
    for lid in dict.fromkeys(str(x).strip() for x in (custom_lora_ids or []) if str(x).strip()):
        ld = catalog_manager.get_definition(lid)
        if not ld or not _is_image_lora_compatible(ld, target_mid, target_def.get("engine")):
            raise ValueError(f"LoRA de imagen desconocido o incompatible con {target_mid}: {lid}")
        if auto_download_loras or catalog_manager.is_downloaded(lid):
            requested_loras.append(lid)
        elif job:
            job.log(f"Aviso: LoRA «{lid}» no descargado; auto_download_loras=False, se omite.", level="warning")

    # Preserve the separate optional-adapter toggle on automatic model loads,
    # without downloading that adapter unless explicitly requested in custom_lora_ids.
    optional_id = "qwen-lora-nsfw"
    optional_def = catalog_manager.get_definition(optional_id) or {}
    if (
        use_nsfw_lora
        and optional_id not in requested_loras
        and catalog_manager.is_downloaded(optional_id)
        and _is_image_lora_compatible(optional_def, target_mid, target_def.get("engine"))
    ):
        requested_loras.append(optional_id)

    same_model = bool(slot and slot.get("model_id") == target_mid)
    resident_ids = list((slot["handle"].get("adapters") or {}).values()) if same_model else []
    missing_loras = set(requested_loras) - set(resident_ids)
    if not same_model or missing_loras:
        # A new LoRA on a 4-bit offloaded model may touch meta tensors. Reload
        # once with the complete requested set BEFORE offload instead of trying
        # an unsafe hot injection. Preloaded frames only call set_adapters below.
        preload_ids = list(dict.fromkeys([lid for lid in resident_ids if lid != target_mid] + requested_loras))
        if job:
            job.log(
                f"Preparando automáticamente «{target_mid}» con sus adaptadores antes de CPU Offload...",
                progress=2.0,
                step="Preparando motor de imagen...",
            )
        slot = None
        load_image_model_into_memory(
            model_id=target_mid,
            extra_lora_ids=preload_ids,
            auto_download=True,
            force_reload=bool(same_model and missing_loras),
            job=job,
        )
        slot = memory_manager.get_slot("image")
    if not slot or slot.get("model_id") != target_mid:
        raise RuntimeError(f"No fue posible activar el modelo de imagen «{target_mid}» en memoria.")

    import torch  # type: ignore

    pipe = slot["handle"]["pipe"]
    engine = slot["handle"].get("engine", "qwen_image_edit")
    available_adapters: Dict[str, str] = slot["handle"].get("adapters", {})
    scales_map: Dict[str, float] = {str(k): float(v) for k, v in (lora_scales or {}).items()}

    t0 = time.time()
    flush_ram_and_vram()

    width = max(512, min(1536, (int(width) // 64) * 64))
    height = max(512, min(1536, (int(height) // 64) * 64))
    steps = max(1, min(50, int(steps)))
    actual_seed = int(seed) if int(seed) > 0 else random.randint(1, 2_147_483_647)

    # Switch preloaded adapters without loading weights or altering offload hooks.
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

    for cid in requested_loras:
        cdef = catalog_manager.get_definition(cid) or {}
        aname = cdef.get("adapter_name") or cid
        if aname not in available_adapters:
            raise RuntimeError(f"El LoRA solicitado {cid} no quedó cargado antes de CPU Offload.")

        if aname in available_adapters and aname not in active_names:
            w_val = float(scales_map.get(cid, cdef.get("default_lora_scale", 0.85)))
            if w_val <= 0.001:
                # Escala 0.0: adaptador precargado en VRAM/RAM para conmutación instantánea sin activarlo en este frame
                continue
            active_names.append(aname)
            active_weights.append(max(0.05, min(2.0, w_val)))
            trig = str(cdef.get("trigger_phrase") or "").strip()
            if trig and trig.lower() not in prompt.lower():
                injected_triggers.append(trig)

    if injected_triggers:
        prompt = f"{', '.join(injected_triggers)}, {prompt}"

    if hasattr(pipe, "set_adapters") and available_adapters:
        if active_names:
            if hasattr(pipe, "enable_lora"):
                pipe.enable_lora()
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
        checkpoint_current_operation(job=job, force=True)
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
    if "image" in call_sig and (has_base or refs_pil or engine == "qwen_image_edit"):
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

    checkpoint_current_operation(job=job, force=True)
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
