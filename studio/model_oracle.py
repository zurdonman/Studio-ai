"""Deterministic recommendation policy over reviewed, hardware-assessed recipes.

The oracle never installs packages, downloads weights, executes remote code or
promotes discoveries. Preferences are workflow policy, not quality/speed benchmarks.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from studio.model_compatibility import assess_definition, assess_model_stack, assessment_context
from studio.model_recipes import ENGINE_CONTRACTS, image_defaults

TASKS = ("video_flf2v", "video_i2v", "video_t2v", "consistent_keyframes", "coherence_check",
         "lipsync_talking_scene", "music_and_ambience", "image_create", "image_edit", "tts", "montage")
PRIORITIES = ("max_quality_14b", "balanced_fast", "ultra_fast_draft", "zero_vram_cpu")


def _eligible(definition: Dict[str, Any], task: str) -> bool:
    if definition.get("validation_state") == "candidate" or definition.get("category") == "lora":
        return False
    if task.startswith("video_"):
        return definition.get("category") == "video" and task[6:] in definition.get("video_modes", []) and definition.get("engine") != "ffmpeg_motion"
    if task in ("image_create", "image_edit", "consistent_keyframes"):
        return (definition.get("category") == "image"
                and (task == "image_create" or "i2i" in definition.get("capabilities", []))
                and (task != "consistent_keyframes" or definition["id"] == "qwen-image-edit-2511-pack"))
    return definition.get("category") == {"tts": "tts", "coherence_check": "vision", "music_and_ambience": "audio", "lipsync_talking_scene": "lipsync"}.get(task)


def _preference(task: str, priority: str) -> List[str]:
    if task.startswith("video_"):
        if priority == "max_quality_14b":
            return ["wan21-vace-14b-fp8", "wan21-flf2v-14b-fp8", "ltx-video-13b-distilled", "wan22-ti2v-5b-turbo", "wan21-vace-1.3b"]
        if priority == "balanced_fast":
            return ["ltx-video-13b-distilled", "wan22-ti2v-5b-turbo", "wan21-vace-1.3b", "sana-video-2b-480p"]
        return ["wan21-vace-1.3b", "sana-video-2b-480p", "wan21-t2v-1.3b", "ltx-video-2b"]
    if task == "image_create":
        if priority in ("ultra_fast_draft", "zero_vram_cpu"):
            return ["sdxl-turbo-fp16", "sana-sprint-600m", "flux2-klein-4b"]
        return ["qwen-image-2512-pack", "flux2-klein-4b", "z-image-turbo", "sdxl-base-1.0", "pixart-sigma-xl-1024"]
    return {"tts": ["edge-tts-cloud", "mms-tts-spa", "kokoro-82m-tts", "bark-small-tts"],
            "coherence_check": ["qwen25-vl-3b-coherence", "opencv-fast-coherence"],
            "music_and_ambience": ["procedural-ambience-cpu", "musicgen-small-bgm", "audioldm2-sfx"],
            "image_edit": ["qwen-image-edit-2511-pack", "flux2-klein-4b"],
            "consistent_keyframes": ["qwen-image-edit-2511-pack"]}.get(task, [])


# Explicit, reviewed acceleration recipes. This list is planning metadata, not
# proof of successful loading, speed, or visual quality on the user's GPU.
FAST_VIDEO_RECIPES = {
    "wan21-flf2v-14b-fp8": "lora-wan21-14b-fusionx-i2v",
    "wan21-vace-14b-fp8": "lora-wan21-14b-fusionx-phantom",
    "wan22-ti2v-5b": "lora-wan22-ti2v-5b-fastwan",
    "wan21-vace-1.3b": "lora-wan21-1.3b-bidirect2",
    "ltx-video-13b-distilled": None,
    "ltx-2.3-distilled-av": None,
    "ltx-2.5-distilled-av": None,
    "ltx-2.3-distilled-fp8-av": None,
    "ltx-2.5-distilled-fp8-av": None,
    "wan22-ti2v-5b-turbo": None,
}

def fast_video_recipe(definition: Dict[str, Any], submode: str, catalog: Any) -> Optional[Dict[str, Any]]:
    mid = definition.get("id")
    if mid not in FAST_VIDEO_RECIPES or submode not in definition.get("video_modes", []):
        return None
    adapter_id = FAST_VIDEO_RECIPES[mid]
    if adapter_id is None:
        return {"kind": "distilled", "adapter": None}
    adapter = catalog.get_definition(adapter_id)
    if (not adapter or adapter.get("validation_state") == "candidate" or adapter.get("lora_group") != "speed"
            or mid not in adapter.get("compatible_models", []) or submode in adapter.get("incompatible_submodes", [])
            or (submode == "flf2v" and adapter.get("causal_attention"))):
        return None
    return {"kind": "speed_lora", "adapter": adapter}


def recommend_pipeline(task: str, priority: str, *, catalog: Any,
                       snapshot: Optional[Dict[str, Any]] = None,
                       context: Optional[Dict[str, Any]] = None, require_fast: bool = False) -> Dict[str, Any]:
    if task not in TASKS or priority not in PRIORITIES:
        raise ValueError("Tarea o prioridad no reconocida: consulta el esquema de studio_recommend_pipeline.")
    context = context or assessment_context(snapshot=snapshot)
    hardware = context["hardware"]
    definitions = catalog.get_all_definitions()
    candidates = [d for d in definitions if _eligible(d, task)]
    if task == "montage":
        candidates = [d for d in definitions if d["id"] == "ffmpeg-motion-cpu"]
    if priority == "zero_vram_cpu":
        candidates = [d for d in candidates if ENGINE_CONTRACTS.get(d.get("engine"), {}).get("cpu") or
                      (ENGINE_CONTRACTS.get(d.get("engine"), {}).get("cpu_capable") and not context["target"].get("has_gpu"))]
        if task in ("video_i2v", "video_flf2v"):
            candidates = [d for d in definitions if d["id"] == "ffmpeg-motion-cpu"]
    preferences = _preference(task, priority)
    fast_recipes: Dict[str, Dict[str, Any]] = {}
    if require_fast and task.startswith("video_"):
        submode = task[6:]
        for definition in candidates:
            recipe = fast_video_recipe(definition, submode, catalog)
            if recipe:
                fast_recipes[definition["id"]] = recipe
        candidates = [d for d in candidates if d["id"] in fast_recipes]
        if priority == "max_quality_14b":
            # Do not silently downgrade a quality-first job to the 1.3B draft.
            candidates = [d for d in candidates if d["id"] != "wan21-vace-1.3b"]
        scored = [(d, assess_model_stack(d, [fast_recipes[d["id"]]["adapter"]], context, catalog=catalog)
                   if fast_recipes[d["id"]]["adapter"] else assess_definition(d, context, catalog=catalog))
                  for d in candidates]
        if priority == "max_quality_14b":
            # Prefer larger reviewed candidates, not a pre-downloaded 1.3B draft.
            preferences = ["wan21-vace-14b-fp8", "wan21-flf2v-14b-fp8",
                           "ltx-video-13b-distilled", "wan22-ti2v-5b", "wan22-ti2v-5b-turbo"]
    else:
        scored = [(d, assess_definition(d, context, catalog=catalog)) for d in candidates]

    def rank(row: Any) -> Any:
        definition, assessment = row
        # Builtins are always "downloaded", but an image histogram is not a
        # semantic identity audit. Do not trade away the requested quality just
        # because a reduced-scope CPU tool has no weights to prepare.
        reduced_scope = 0
        if priority not in ("zero_vram_cpu", "ultra_fast_draft"):
            if task == "coherence_check" and definition.get("engine") == "opencv_fast":
                reduced_scope = 1
            if task == "music_and_ambience" and priority == "max_quality_14b" and definition.get("engine") == "procedural_dsp":
                reduced_scope = 1
        quality_order = preferences.index(definition["id"]) if definition["id"] in preferences else 100
        if require_fast and priority == "max_quality_14b" and task.startswith("video_"):
            return (not assessment["compatible"], quality_order, not assessment["runnable_now"], definition["id"])
        return (not assessment["compatible"], reduced_scope, not assessment["runnable_now"], not catalog.is_downloaded(definition["id"]),
                preferences.index(definition["id"]) if definition["id"] in preferences else 100,
                definition.get("min_vram_gb", 0), definition["id"])

    scored.sort(key=rank)
    feasible = [row for row in scored if row[1]["compatible"]]
    base = {"task": task, "priority": priority, "detected_gpu": hardware.get("gpu", {}).get("gpu_name", "CPU"),
            "vram_total_gb": hardware.get("gpu", {}).get("vram_total_gb"),
            "free_disk_gb": hardware.get("disk", {}).get("free_gb"), "hardware": hardware,
            "selection_policy": "Solo recetas revisadas, modo exacto, capacidad estimada VRAM/RAM/disco y precisión. Reutilizar pesos descargados entre recetas de alcance equivalente; no confundir OpenCV/DSP con auditoría semántica/música IA. Preferencias de flujo, no benchmark.",
            "alternatives": [{"model_id": d["id"], "assessment": a} for d, a in scored[:8]],
            "automatic_weight_downloads": False, "remote_code_allowed": False, "gpu_validation": "No ejecutada por este oráculo",
            "manual_uri": "studio://manual/model-oracle"}
    if not feasible:
        return {**base, "status": "no_compatible_recipe", "executable_now": False, "recommended_tool": None,
                "recommended_arguments": {}, "why_this_choice": "Ninguna receta revisada satisface la tarea, prioridad y recursos conocidos (incluida aceleración si se solicitó). No se cambia silenciosamente de modelo ni de tarea.",
                "next_actions": ["Leer reasons/resources/runtime de alternatives. Revisar GPU/RAM/disco reales y no sumar la VRAM de varias GPU.",
                                 "Buscar candidatos con studio_search_models e inspeccionar con studio_inspect_model; descubrirlos no habilita su ejecución.",
                                 "Para imagen/vídeo generativo sin GPU no hay equivalencia automática con filtros o movimiento FFmpeg."],
                "discovery_suggestion": {"name": "studio_search_models", "arguments": {"query": "", "category": "video" if task.startswith("video_") else "image" if "image" in task or task == "consistent_keyframes" else "all", "only_compatible": False}}}
    definition, assessment = feasible[0]
    mid = definition["id"]
    args: Dict[str, Any] = {"model_id": mid}
    required_inputs: List[str] = []
    workflow, caveats = [], []
    if task.startswith("video_"):
        tool = "studio_generate_video"
        submode = task[6:]
        args.update(mode="ffmpeg_motion" if mid == "ffmpeg-motion-cpu" else "ai", video_submode=submode, use_official_defaults=True)
        if require_fast and mid != "ffmpeg-motion-cpu":
            fast_recipe = fast_recipes[mid]
            args["require_fast"] = True
            if fast_recipe["adapter"]:
                args["video_lora_id"] = fast_recipe["adapter"]["id"]
            base["acceleration"] = {"kind": fast_recipe["kind"], "video_lora_id": args.get("video_lora_id"),
                                    "gpu_tested": False}
            caveats.append("Receta rápida estimada, no medida: confirmar carga real del adaptador y pasos; no garantiza velocidad ni calidad.")
        required_inputs = ["prompt"] + (["image_rel"] if submode in ("i2v", "flf2v") else []) + (["end_image_rel"] if submode == "flf2v" else [])
        if mid == "ffmpeg-motion-cpu":
            args.update(motion_style="zoom_in", duration_sec=4.5)
            caveats.append("Movimiento/interpolación CPU de imágenes existentes; NO síntesis generativa ni equivalente a T2V IA.")
    elif task in ("image_create", "image_edit", "consistent_keyframes"):
        tool = "studio_generate_or_edit_image"
        args.update(use_official_defaults=True, use_lightning_lora=definition.get("engine") in ("qwen_image_edit", "qwen_image_t2i"), aspect_ratio="1:1")
        required_inputs = ["prompt"] + (["base_image_rel"] if task == "image_edit" else [])
        base["resolved_image_defaults"] = image_defaults(definition, args["use_lightning_lora"])
        if task == "consistent_keyframes":
            workflow = ["studio_get_workflow_preset antes de producir la secuencia.",
                        "Todos los keyframes derivados del ancla con referencias, auditoría y reparación; no cargar vídeo aún.",
                        "Después todos los clips; después diálogo medido, LipSync y montaje. Precargar LoRAs futuros con escala 0 antes del offload."]
            caveats.append("Referencias y auditoría reducen deriva; no garantizan identidad perfecta. Una auditoría fallida requiere reparación o decisión del usuario.")
    elif task == "coherence_check":
        tool = "studio_analyze_coherence"
        args = {"mode": "vlm_qwen" if mid == "qwen25-vl-3b-coherence" else "fast_cv", "strictness_threshold": 78.0}
        required_inputs = ["anchor_image_rel", "candidate_image_rels"]
        if mid == "opencv-fast-coherence":
            caveats.append("Alcance reducido: se recomienda OpenCV por límites de recursos/prioridad; no sustituye una auditoría semántica de identidad.")
        caveats.append("OpenCV mide semejanza fotométrica, no prueba identidad semántica; VLM y reparación tampoco son infalibles.")
    elif task == "tts":
        tool = "studio_generate_speech"
        args = {"engine_id": mid, "voice_id": "es-ES-AlvaroNeural", "rate_pct": 0}
        required_inputs = ["text"]
        caveats.append("Los voice_id de Edge no seleccionan voces de modelos locales; consultar el contrato del motor. Edge necesita internet; MMS es local y tiene sus propios términos.")
    elif task == "lipsync_talking_scene":
        tool = "studio_lipsync_character"
        args.update(target_face="largest", lip_weight=0.85, enhance_face=True, extend_video_to_audio=True)
        required_inputs = ["media_rel"]
        base["required_one_of"] = [["voice_audio_rel", "dialogue_turns"]]
        caveats.append("Asociar leftmost/rightmost a posiciones verificadas y medir cada audio; 2 personajes requieren turnos explícitos.")
    elif task == "music_and_ambience":
        tool = "studio_generate_music_or_sfx"
        args.update(preset="restaurant_warm_murmur", duration_seconds=12.0)
        required_inputs = ["preset"] if mid == "procedural-ambience-cpu" else ["prompt"]
        if mid != "procedural-ambience-cpu":
            args["preset"] = "auto"
        caveats.append("DSP genera ambiente sintético, no una canción cantada. MusicGen instrumental/AudioLDM2 tienen funciones y licencias distintas.")
    else:
        tool = "studio_render_montage"
        args = {"width": 1280, "height": 720, "fps": 24, "fade_seconds": 0.2, "global_audio_volume": 0.25}
        required_inputs = ["items"]
    args["wait"] = False
    next_actions = []
    if not assessment["runtime"]["ready"]:
        next_actions.append("Instalar/comprobar las dependencias indicadas en assessment.runtime y reiniciar el proceso Python. No sustituir el modelo ni descargar alternativas por un error de librerías.")
    if assessment["resources"]["vram"].get("needs_unload") or assessment["resources"]["ram"].get("needs_release"):
        next_actions.append("Liberar modelos de RAM/VRAM (studio_free_memory_or_disk action=unload_vram), conservar pesos y volver a medir. No lanzar dos inferencias simultáneas.")
    if not catalog.is_downloaded(mid):
        next_actions.append("Antes de ejecutar, confirmar presupuesto de descarga, licencia/gate y espacio real. La operación de generación puede descargar SOLO la receta elegida; este oráculo no descarga nada.")
    free_disk = hardware.get("disk", {}).get("free_gb")
    low_output_space = free_disk is not None and float(free_disk) < 1.0
    if low_output_space:
        next_actions.append("Menos de 1 GiB libre para salidas/temporales: exportar o liberar espacio con autorización antes de generar.")
    next_actions.append("Completar required_inputs con rutas/valores reales, usar wait=false y esperar el mismo job_id con studio_wait_for_job(wait_seconds=80) hasta completed. Running/queued no son fallo.")
    return {**base, "status": "ready" if assessment["runnable_now"] and not low_output_space else "needs_preparation",
            "executable_now": assessment["runnable_now"] and not low_output_space, "model_id": mid,
            "readiness_level": "static_prerequisites_only_not_inference_validation",
            "assessment": assessment, "recommended_tool": tool, "recommended_arguments": args,
            "required_inputs": required_inputs, "workflow_steps": workflow, "caveats": caveats,
            "why_this_choice": f"{mid}: tarea admitida por la receta, encaje estimado de recursos; " + ("reutiliza pesos locales." if catalog.is_downloaded(mid) else "requiere preparar sus pesos, con autorización y comprobación de acceso."),
            "next_actions": next_actions, "timing_guarantee": None,
            "no_cascade_rule": "Ante error conservar pesos y logs, esperar el job o diagnosticar. No descargar variantes 14B/13B/1.3B en cascada sin autorización."}
