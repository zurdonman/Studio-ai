"""
Presets Oficiales de Flujo Operativo para Agentes MCP en Studio IA.

Cada preset actúa como un «Director Técnico Compilado»:
  1. Enseña al agente las reglas anti-error específicas de la escena (ej. evitar el «movimiento estático»
     cuando dos personas van andando por la calle hablando).
  2. Selecciona automáticamente los modelos base y el apilamiento Multi-LoRA óptimo según la GPU real
     detectada en Google Colab (L4 22.5 GB, A100 40/80 GB o T4 15 GB).
  3. Devuelve una hoja de ruta (`execution_plan`) paso a paso con llamadas MCP listas para ejecutar,
     incluyendo ingeniería de prompts en 3 vectores para vídeo Frame a Frame (`flf2v`), auditoría
     anti-estática (`expect_spatial_progression=true`), voces TTS, LipSync HD por turnos y ambiente con Auto-Ducking.
"""

import math
import re
from typing import Any, Dict, List, Optional, Tuple

from studio.hardware import query_gpu_info


_DURATION_PATTERN = (
    r"\b(?:\d+(?:[.,]\d+)?|un|uno|medio|one|a|half)\s*"
    r"(?:minutos?|minutes?|mins?|segundos?|seconds?|secs?|seg|s)\b"
)
_META_INSTRUCTION_PATTERNS = [
    r"\b(?:crea(?:r|me)?|genera(?:r|me)?|escribe|create|generate|write)\s+"
    r"(?:(?:los|el|unos|the|some)\s+)?di[aá]log(?:os?|ues?)\b(?:\s+(?:de|para)\s+(?:los\s+)?personajes)?",
    r"\b(?:los\s+personajes\s+)?que\s+tengan?\s+(?:todos?\s+)?coherencia\b",
    rf"\b(?:(?:que\s+sea|haz(?:me)?|quiero)\s+)?(?:(?:un|el|a)\s+)?(?:clip|v[ií]deo)\s+(?:de\s+)?{_DURATION_PATTERN}",
    rf"\b(?:duraci[oó]n\s*(?:de\s*)?|duration\s*:?\s*)?{_DURATION_PATTERN}",
    r"\b(?:en\s+)?total\s+coherencia\b",
    r"\b(?:utiliza(?:ndo|r)?|usa(?:ndo|r)?|use)\s+(?:las\s+)?loras?\s*(?:r[aá]pid[oa]s?|fast)?\b",
    r"\bsincroniza(?:ndo|r)?\s+(?:los\s+)?labios?\b",
    r"\bcon\s+lip[ -]?sync\b",
]


def _sanitize_visual_scene_description(raw: str) -> str:
    """Keep physical scene details; never feed workflow directives to a visual model."""
    cleaned = (raw or "").strip()
    for pattern in _META_INSTRUCTION_PATTERNS:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\s+([,.;])", r"\1", cleaned)
    cleaned = re.sub(r"([,;])\s*[,;]+", r"\1", cleaned)
    return cleaned.strip(" ,.;:-")


def _infer_clip_plan_from_text(scene_raw: str, requested_num_clips: Optional[int]) -> Tuple[int, float]:
    """Infer 5s batches (60s = 12 clips), without replacing an explicit non-default count."""
    match = re.search(_DURATION_PATTERN, (scene_raw or "").lower())
    target_seconds = 0.0
    if match:
        duration = match.group(0)
        number = re.match(r"\d+(?:[.,]\d+)?|un|uno|medio|one|a|half", duration)
        if number:
            word = number.group(0)
            values = {"un": 1.0, "uno": 1.0, "one": 1.0, "a": 1.0, "medio": 0.5, "half": 0.5}
            amount = values[word] if word in values else float(word.replace(",", "."))
            target_seconds = amount * (60.0 if "min" in duration else 1.0)

    if target_seconds > 0 and (requested_num_clips is None or int(requested_num_clips) == 2):
        return max(1, min(12, math.ceil(target_seconds / 5.0))), 5.0
    n = int(requested_num_clips) if requested_num_clips is not None else 2
    return max(1, min(12, n)), 5.0


_DEFAULT_STREET_DIALOGUE_POOL = [
    "¿Has visto la luz de esta avenida? Parece una película.",
    "Sí, hoy apetece caminar sin prisa y mirar alrededor.",
    "Me encantan los nuevos escaparates y las terrazas del barrio.",
    "Antes pasábamos corriendo. Hoy podemos disfrutarlo con calma.",
    "¿Paramos en aquella cafetería de la esquina a tomar algo?",
    "¡Buena idea! Dicen que preparan un café estupendo.",
    "Lleguemos al final del bulevar y nos ponemos al día.",
    "Perfecto, tengo mil cosas que contarte de esta semana.",
    "Cuéntame, que aún queda un buen tramo por delante.",
    "No te vas a creer lo bien que salió todo.",
    "¡Qué alegría! Me encanta verte con tanta energía.",
    "Gracias. Caminar y charlar contigo siempre me arregla el día.",
]

AGENT_PRESET_SUMMARY_CATALOG: Dict[str, Dict[str, Any]] = {
    "coherencia_personajes_entorno_avance": {
        "id": "coherencia_personajes_entorno_avance",
        "aliases": [
            "walking_talking_coherence",
            "andando_por_la_calle",
            "avance_coherente",
            "character_environment_coherence",
            "walking_dialogue",
            "caminando_hablando",
        ],
        "title": "🚶‍♂️🗣️ Coherencia de Personajes y Entorno en Avance Real (Caminando y Hablando · Anti-Estático)",
        "badge": "★ Preset Estrella · Avance Real Frame-a-Frame + Multi-LoRA + LipSync",
        "when_to_use": (
            "Úsalo SIEMPRE que los personajes deban desplazarse/avanzar por un escenario manteniendo la misma cara, "
            "ropa y calle/entorno (ej. dos personas andando por la calle hablando, paseando por un parque, "
            "recorriendo un pasillo o caminando hacia cámara)."
        ),
        "solves_problem": (
            "Elimina el fallo clásico donde el agente edita solo la boca/manos dejando las piernas y el fondo "
            "congelados (produciendo un vídeo estático donde parecen flotar o moverse en el sitio). Fuerza "
            "desplazamiento espacial real metro a metro con encadenamiento de fotogramas, LoRA `Next Scene`, "
            "auditoría `expect_spatial_progression=true` y prompt de vídeo en 3 vectores (Steadicam + Zancada + Paralaje)."
        ),
    },
    "coherencia_escena_fija_dialogo": {
        "id": "coherencia_escena_fija_dialogo",
        "aliases": [
            "stationary_dialogue_coherence",
            "dialogo_fijo",
            "restaurante_conversacion",
            "mesa_hablando",
        ],
        "title": "🍽️🗣️ Coherencia de Personajes y Entorno Fijo con Diálogo (Mesa / Salón / Café + LipSync Turnos)",
        "badge": "Escenario Anclado + Multi-Personaje LipSync HD",
        "when_to_use": (
            "Cuando 1 o 2 personajes conversan en un mismo lugar sin desplazarse metros (ej. sentados en una terraza, "
            "restaurante, sofá, estudio u oficina) con gestos naturales, brindis o reacciones."
        ),
        "solves_problem": (
            "Mantiene intactos rostros, peinados, ropa, objetos de la mesa e iluminación entre todos los planos, "
            "sincroniza los labios por turnos (`leftmost` / `rightmost`) con `EDTalk + GFPGAN` y añade ambiente con Auto-Ducking."
        ),
    },
    "cambio_angulo_3d_mismo_personaje": {
        "id": "cambio_angulo_3d_mismo_personaje",
        "aliases": [
            "character_3d_multi_angle",
            "plano_contraplano",
            "multi_angulos",
            "camera_angles_3d",
        ],
        "title": "🔄📷 Cobertura Multicámara 3D Coherente (Plano / Contraplano / Perfil del Mismo Actor y Entorno)",
        "badge": "LoRA Multiple-Angles 3D + AnyLight + LTX/Wan",
        "when_to_use": (
            "Cuando necesitas grabar la misma escena desde distintos ángulos de cámara (frontal, tres cuartos, perfil, "
            "plano sobre el hombro) sin que cambie el actor ni el entorno."
        ),
        "solves_problem": (
            "Usa `qwen-lora-multi-angles-3d` + `qwen-lora-anylight-env` sobre el fotograma ancla para rotar la cámara "
            "virtual respetando identidad y luz antes de animar cada corte."
        ),
    },
    "corto_1_minuto_multiescena": {
        "id": "corto_1_minuto_multiescena",
        "aliases": [
            "short_film_1min_coherent",
            "corto_1_minuto",
            "historia_larga",
            "montaje_60s",
        ],
        "title": "🎞️🎬 Producción Coherente de 30–60 Segundos (Casting Vault + Cadena de 6–12 Planos + BSO)",
        "badge": "Arquitectura Narrativa 1 Minuto · 85–90/100 Calidad",
        "when_to_use": (
            "Para montar clips largos de 20 a 60 segundos combinando varios planos coherentes (establecimiento, "
            "avance caminando, diálogo en primer plano y plano final) con música y mezcla maestra."
        ),
        "solves_problem": (
            "Registra fichas en Casting Vault, ancla todos los keyframes al retrato maestro evitando deriva acumulada, "
            "encadena clips de 5 s y masteriza voz + música + ambiente con Auto-Ducking en `studio_render_montage`."
        ),
    },
    "edicion_fotografica_relight_cine": {
        "id": "edicion_fotografica_relight_cine",
        "aliases": [
            "photo_relight_and_realism",
            "editar_foto_pro",
            "relight_entorno",
        ],
        "title": "🌅✨ Edición Fotográfica Pro, Armonización de Luz de Entorno y Piel Realista",
        "badge": "Studio DeLight + AnyLight + Skin Realism + Unblur",
        "when_to_use": (
            "Para editar fotos subidas por el usuario, cambiar la hora del día o el fondo integrando sombras reales "
            "sobre la piel y ropa, y opcionalmente dar vida al retrato en vídeo."
        ),
        "solves_problem": (
            "Evita el efecto 'recorte pegado' combinando limpieza de sombras (`Studio-DeLight`), integración de luz "
            "ambiental (`AnyLight`) y textura real de poros (`Skin Realism`)."
        ),
    },
    "animacion_estilizada_coherente": {
        "id": "animacion_estilizada_coherente",
        "aliases": [
            "coherent_animation_style",
            "anime_coherente",
            "animacion_clasica",
        ],
        "title": "🎨🎞️ Animación Estilizada Coherente (Anime Moderno / Clásico 1928 / Ilustración a Real)",
        "badge": "LoRAs Anime v2 · Steamboat Classic · Anything2Real",
        "when_to_use": (
            "Para producir secuencias animadas estilizadas manteniendo diseño de personaje uniforme entre planos."
        ),
        "solves_problem": (
            "Aplica adaptadores dedicados de estilo tanto en los fotogramas ancla como en la interpolación de vídeo."
        ),
    },
}


def resolve_preset_id(raw_preset: str, scene_hint: str = "") -> str:
    clean = (raw_preset or "").strip().lower()
    if clean in AGENT_PRESET_SUMMARY_CATALOG:
        return clean
    for pid, meta in AGENT_PRESET_SUMMARY_CATALOG.items():
        if clean in [a.lower() for a in meta.get("aliases", [])]:
            return pid

    combined = f"{clean} {(scene_hint or '').lower()}"
    if any(k in combined for k in ("andando", "caminan", "calle", "pasean", "avanz", "walk", "street", "corridor", "pasillo")):
        return "coherencia_personajes_entorno_avance"
    if any(k in combined for k in ("angulo", "ángulo", "contraplano", "3d", "angle", "orbit", "perfil")):
        return "cambio_angulo_3d_mismo_personaje"
    if any(k in combined for k in ("minuto", "60s", "60 seg", "corto", "historia", "1 min", "secuencia larga")):
        return "corto_1_minuto_multiescena"
    if any(k in combined for k in ("relight", "iluminac", "editar foto", "piel", "upscale", "restaurar")):
        return "edicion_fotografica_relight_cine"
    if any(k in combined for k in ("anime", "dibujo", "cartoon", "steamboat", "clasic", "ilustrac")):
        return "animacion_estilizada_coherente"
    if any(k in combined for k in ("mesa", "restaur", "cafe", "café", "sentad", "entrevista", "fijo")):
        return "coherencia_escena_fija_dialogo"
    return "coherencia_personajes_entorno_avance"


def list_agent_workflow_presets() -> List[Dict[str, Any]]:
    return list(AGENT_PRESET_SUMMARY_CATALOG.values())


def _select_video_stack_for_gpu(vram_gb: float, has_gpu: bool) -> Dict[str, Any]:
    if not has_gpu or vram_gb < 10.0:
        return {
            "tier": "cpu_fallback",
            "model_id": "ffmpeg-motion-cpu",
            "mode": "ffmpeg_motion",
            "video_lora_id": "",
            "extra_video_lora_ids": [],
            "extra_video_lora_scales": {},
            "note": "Sin GPU suficiente: usando motor óptico CPU FFmpeg.",
        }
    if vram_gb >= 19.5:
        # L4 (22.5 GB) o A100 (40/80 GB): Wan 2.1 VACE 14B FP8 + FusionX 6p + UniAnimate + MoviiGen Cinema
        return {
            "tier": "high_14b_fp8",
            "model_id": "wan21-vace-14b-fp8",
            "mode": "ai",
            "video_lora_id": "lora-wan21-14b-fusionx-phantom",
            "extra_video_lora_ids": [
                "lora-wan21-14b-unianimate-char",
                "lora-wan21-14b-moviigen-cinema",
            ],
            "extra_video_lora_scales": {
                "lora-wan21-14b-unianimate-char": 0.65,
                "lora-wan21-14b-moviigen-cinema": 0.60,
            },
            "note": "GPU >= 20 GB (L4/A100): Wan 2.1 VACE 14B FP8 + FusionX 6p + UniAnimate (identidad/anatomía) + MoviiGen (paralaje cine).",
        }
    # T4 (15 GB): Wan 2.1 VACE 1.3B + Bidirect2 (6-8 pasos)
    return {
        "tier": "fast_1_3b",
        "model_id": "wan21-vace-1.3b",
        "mode": "ai",
        "video_lora_id": "lora-wan21-1.3b-bidirect2",
        "extra_video_lora_ids": [],
        "extra_video_lora_scales": {},
        "note": "GPU ~15 GB (T4): Wan 2.1 VACE 1.3B + LoRA Bidireccional v2 (8 pasos seguros en Frame a Frame).",
    }


def compile_agent_workflow_preset(
    preset_id: str = "coherencia_personajes_entorno_avance",
    scene_description: str = "",
    characters_description: str = "",
    dialogue_lines: Optional[List[str]] = None,
    num_clips: Optional[int] = None,
    aspect_ratio: str = "16:9",
    gpu_override: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Compila un plan operativo completo y ejecutable para que el agente MCP siga exactamente
    las mejores prácticas de coherencia de personajes, entorno, avance espacial y vídeo.
    """
    resolved_id = resolve_preset_id(preset_id, scene_hint=scene_description)
    meta = AGENT_PRESET_SUMMARY_CATALOG[resolved_id]

    gpu = gpu_override or query_gpu_info()
    vram_gb = float(gpu.get("vram_total_gb") or gpu.get("total_vram_gb") or 0.0)
    has_gpu = bool(gpu.get("has_gpu", vram_gb > 0))
    vid_stack = _select_video_stack_for_gpu(vram_gb=vram_gb, has_gpu=has_gpu)

    n_clips, clip_dur_sec = _infer_clip_plan_from_text(scene_description, num_clips)
    ar = aspect_ratio if aspect_ratio in ("16:9", "9:16", "1:1") else "16:9"
    orient = "portrait" if ar == "9:16" else ("square" if ar == "1:1" else "landscape")

    clean_scene = _sanitize_visual_scene_description(scene_description)
    scene_txt = (
        clean_scene
        or "dos amigos (un hombre joven con chaqueta vaquera oscura y una mujer joven con gabardina beige) andando por una calle peatonal arbolada al atardecer mientras conversan"
    )
    chars_txt = (
        _sanitize_visual_scene_description(characters_description)
        or "Personaje izquierda: hombre de 30 años, pelo castaño corto, barba incipiente, chaqueta vaquera azul oscuro sobre camiseta blanca y pantalón chino gris. Personaje derecha: mujer de 29 años, melena castaña ondulada por los hombros, gabardina beige abierta sobre jersey negro y vaqueros."
    )
    d_lines = [str(x).strip() for x in (dialogue_lines or []) if str(x).strip()]
    num_turns = max(2, n_clips, len(d_lines))
    while len(d_lines) < num_turns:
        d_lines.append(_DEFAULT_STREET_DIALOGUE_POOL[len(d_lines) % len(_DEFAULT_STREET_DIALOGUE_POOL)])

    # ==================== PRESET 1: CAMINANDO Y HABLANDO CON AVANCE REAL ====================
    if resolved_id == "coherencia_personajes_entorno_avance":
        num_keyframes = n_clips + 1
        steps: List[Dict[str, Any]] = []

        # Paso 1: Fotograma 1 (Ancla Maestra en Punto A de la calle)
        anchor_prompt = (
            f"Cinematic Steadicam front-tracking medium-wide shot (knee-up framing showing walking legs and full upper bodies) "
            f"of {scene_txt}. {chars_txt} "
            "POSITION A ON THE STREET: Both characters are actively walking forward toward the camera along a detailed sidewalk "
            "(person on the left has LEFT leg stepping forward and gestures naturally while speaking; person on the right has RIGHT leg stepping forward, smiling and looking slightly toward them). "
            "Clear architectural street depth behind them: a warm café awning and vintage streetlamp right behind them on the left, trees and shopfronts receding in perspective, "
            "natural golden-hour cinema lighting, 35mm ARRI Alexa lens, natural skin texture."
        )
        steps.append(
            {
                "step_number": 1,
                "phase": "keyframe_1_master_anchor",
                "purpose": (
                    "Generar el FOTOGRAMA 1 (ANCLA MAESTRA en el Punto A de la calle). "
                    "NO llames antes a `studio_prepare_model`: esta llamada auto-descarga y auto-carga `qwen-image-edit-2511-pack` "
                    "+ `Lightning` + los 4 LoRAs del flujo de una sola vez (dejando `next-scene` y `anylight` precargados a escala 0.0 "
                    "para conmutar los Fotogramas 2..N sin volver a cargar los pesos). "
                    "REGLA CRÍTICA: Describe en el prompt SOLO la escena visual física (sin frases como 'crea los diálogos' o 'clip de un minuto')."
                ),
                "tool": "studio_generate_or_edit_image",
                "arguments": {
                    "model_id": "qwen-image-edit-2511-pack",
                    "aspect_ratio": ar,
                    "use_official_defaults": True,
                    "custom_lora_ids": [
                        "qwen-lora-skin-realism",
                        "qwen-lora-flatlog-cinema",
                        "qwen-lora-next-scene",
                        "qwen-lora-anylight-env",
                    ],
                    "lora_scales": {
                        "qwen-lora-skin-realism": 0.80,
                        "qwen-lora-flatlog-cinema": 0.65,
                        "qwen-lora-next-scene": 0.0,
                        "qwen-lora-anylight-env": 0.0,
                    },
                    "prompt": anchor_prompt,
                },
                "save_output_as": "FRAME_1_REL",
            }
        )

        # Pasos 2..K: Fotogramas progresivos (Punto B, Punto C...) avanzando 4 metros cada uno
        parallax_variants = [
            "the café awning and streetlamp from Position A have now receded noticeably further back in the background distance behind them, while a new bookstore window and leafy tree appear along the sidewalk beside them",
            "the bookstore window is now behind their shoulders in mid-ground perspective and a stone crosswalk with warm evening reflections enters the foreground pavement",
            "previous shopfronts recede into the far background depth as new warm street details pass by on both sides",
        ]

        for kf_idx in range(2, num_keyframes + 1):
            prev_var = f"FRAME_{kf_idx - 1}_REL"
            refs_vars = ["<FRAME_1_REL>"] if kf_idx == 2 else ["<FRAME_1_REL>", f"<{prev_var}>"]
            parallax_txt = parallax_variants[min(kf_idx - 2, len(parallax_variants) - 1)]
            pos_txt = (
                f"4 meters further forward than the previous frame (POSITION {chr(64 + kf_idx)}, "
                f"{(kf_idx - 1) * 4} meters beyond the master anchor at POSITION A)"
            )
            left_leg, right_leg = ("RIGHT", "LEFT") if kf_idx % 2 == 0 else ("LEFT", "RIGHT")
            stride_txt = (
                f"person on the left steps forward with their {left_leg} leg while person on the right steps "
                f"forward with their {right_leg} leg; a fresh natural conversational gesture, continuous walking gait"
            )

            prog_prompt = (
                f"Next Scene: The exact same two people from the reference image ({chars_txt}) with 100% identical faces, "
                f"hairstyles and exact same outfits have walked {pos_txt}. "
                f"Camera continues tracking backward ahead of them at the same knee-up framing. "
                f"TRUE WALKING PROGRESSION (NOT STATIC): {stride_txt}. "
                f"ENVIRONMENT PERSPECTIVE SHIFT: {parallax_txt}. "
                "Exact same street architecture, golden-hour lighting direction and cinematic color grade."
            )
            steps.append(
                {
                    "step_number": len(steps) + 1,
                    "phase": f"keyframe_{kf_idx}_spatial_advance",
                    "purpose": (
                        f"Generar FOTOGRAMA {kf_idx} avanzando ~4 metros reales por la calle respecto a `{prev_var}` "
                        f"manteniendo bloqueada la identidad del ancla `FRAME_1_REL`. "
                        "PROHIBIDO pedir 'mantén exactamente el mismo fondo sin mover nada': debes exigir zancada alterna "
                        "y que las tiendas/farolas del fondo retrocedan en perspectiva (`Next Scene:` + LoRA `qwen-lora-next-scene`)."
                    ),
                    "tool": "studio_generate_or_edit_image",
                    "arguments": {
                        "model_id": "qwen-image-edit-2511-pack",
                        "base_image_rel": f"<{prev_var}>",
                        "reference_rels": refs_vars,
                        "aspect_ratio": ar,
                        "use_official_defaults": True,
                        "custom_lora_ids": [
                            "qwen-lora-next-scene",
                            "qwen-lora-skin-realism",
                            "qwen-lora-anylight-env",
                        ],
                        "lora_scales": {
                            "qwen-lora-next-scene": 0.85,
                            "qwen-lora-skin-realism": 0.75,
                            "qwen-lora-anylight-env": 0.65,
                        },
                        "prompt": prog_prompt,
                    },
                    "save_output_as": f"FRAME_{kf_idx}_REL",
                }
            )

        # Paso de Auditoría Anti-Estática + Identidad
        cand_placeholders = [f"<FRAME_{i}_REL>" for i in range(2, num_keyframes + 1)]
        steps.append(
            {
                "step_number": len(steps) + 1,
                "phase": "coherence_and_anti_static_audit",
                "purpose": (
                    "OBLIGATORIO antes de renderizar vídeo: audita que (1) sigan siendo exactamente las mismas personas "
                    "con la misma ropa y luz, Y (2) `expect_spatial_progression=true` confirme que REALMENTE avanzaron "
                    "por la calle (`PASS`) y no quedaron congelados (`TOO_STATIC_NEEDS_ADVANCE`). Si algún fotograma no sale `PASS`, "
                    "ejecuta el `recommended_repair_tool_call` devuelto."
                ),
                "tool": "studio_analyze_coherence",
                "arguments": {
                    "anchor_image_rel": "<FRAME_1_REL>",
                    "candidate_image_rels": cand_placeholders,
                    "scene_goal": scene_txt,
                    "expect_spatial_progression": True,
                    "mode": "hybrid_auto",
                },
                "save_output_as": "COHERENCE_AUDIT_RESULT",
                "required_result": {"all_consistent": True, "vlm_semantic_used": has_gpu},
                "on_failure": "Reparar fotogramas fallidos y reauditar TODO el lote; no cargar vídeo todavía.",
            }
        )

        # Pasos de Vídeo Frame a Frame (`flf2v`) encadenados: Frame 1->2, Frame 2->3...
        for clip_idx in range(1, n_clips + 1):
            start_var = f"<FRAME_{clip_idx}_REL>"
            end_var = f"<FRAME_{clip_idx + 1}_REL>"
            video_3vector_prompt = (
                "Smooth backward-moving Steadicam tracking shot leading two friends walking forward together down a lively city sidewalk, "
                "both characters walk forward continuously at a natural steady pace with realistic alternating leg strides, subtle hip and shoulder motion, "
                "natural blinking, warm eye contact and expressive conversational hand gestures as they talk while walking, "
                "sidewalk pavement moves beneath them while storefronts, streetlamps and trees glide past smoothly in the background with natural 3D depth parallax, "
                "consistent facial identity, stable anatomy, cinematic 35mm golden hour lighting."
            )
            steps.append(
                {
                    "step_number": len(steps) + 1,
                    "phase": f"video_clip_{clip_idx}_flf2v_progressive",
                    "purpose": (
                        f"Animar el Tramo {clip_idx} (`{start_var}` → `{end_var}`) en modo `flf2v` con prompt en 3 vectores "
                        "(1: Steadicam retrocediendo delante de ellos + 2: locomoción continua de piernas/hombros y gestos de habla + "
                        "3: paralaje 3D de escaparates y acera deslizándose atrás). Como el Tramo 1 termina exactamente en `FRAME_2_REL` "
                        "y el Tramo 2 arranca en `FRAME_2_REL`, la unión entre clips será continua."
                    ),
                    "tool": "studio_generate_video",
                    "arguments": {
                        "video_submode": "flf2v",
                        "mode": vid_stack["mode"],
                        "model_id": vid_stack["model_id"],
                        "video_lora_id": vid_stack["video_lora_id"],
                        "extra_video_lora_ids": vid_stack["extra_video_lora_ids"],
                        "extra_video_lora_scales": vid_stack["extra_video_lora_scales"],
                        "orientation": orient,
                        "duration_sec": clip_dur_sec,
                        "use_official_defaults": True,
                        "image_rel": start_var,
                        "end_image_rel": end_var,
                        "prompt": video_3vector_prompt,
                    },
                    "save_output_as": f"RAW_VIDEO_CLIP_{clip_idx}_REL",
                }
            )

        # Voces + Ambiente de Calle + LipSync HD por turnos + Montaje
        voice_calls: List[Dict[str, Any]] = []
        for t_idx in range(1, num_turns + 1):
            is_left = (t_idx % 2 == 1)
            voice_calls.append(
                {
                    "tool": "studio_generate_speech",
                    "arguments": {
                        "voice_id": "es-ES-AlvaroNeural" if is_left else "es-ES-ElviraNeural",
                        "text": d_lines[t_idx - 1],
                    },
                    "save_output_as": f"VOICE_{t_idx}_REL",
                }
            )
        voice_calls.append(
            {
                "tool": "studio_generate_music_or_sfx",
                "arguments": {
                    "model_id": "procedural-ambience-cpu",
                    "preset": "city_street_walk",
                    "duration_seconds": float(n_clips * clip_dur_sec + 2.0),
                },
                "save_output_as": "STREET_AMBIENCE_REL",
            }
        )

        steps.append(
            {
                "step_number": len(steps) + 1,
                "phase": "voices_and_street_ambience",
                "purpose": (
                    f"Generar las {num_turns} réplicas de conversación alternas (`es-ES-AlvaroNeural` para personaje izquierdo, "
                    f"`es-ES-ElviraNeural` para personaje derecho) más la pista de ambiente exterior con pasos suaves (`city_street_walk`, 0 GB VRAM)."
                ),
                "parallel_or_sequential_calls": voice_calls,
            }
        )

        # Para secuencias largas (>= 4 clips, ej. 30-60s), sincronizar labios clip por clip (~5s cada uno)
        # evita timeouts MCP y mantiene cadencia exacta; luego unir con fade_seconds=0.0 + ambiente urbano.
        if n_clips >= 4:
            post_calls: List[Dict[str, Any]] = [
                {
                    "tool": "studio_prepare_model",
                    "arguments": {"model_id": "edtalk-256-lipsync-hd"},
                }
            ]
            for c_idx in range(1, n_clips + 1):
                clip_turns = [i for i in range(1, num_turns + 1) if min(n_clips, (i - 1) * n_clips // num_turns + 1) == c_idx]
                first_turn = clip_turns[0]
                post_calls.append(
                    {
                        "tool": "studio_lipsync_character",
                        "arguments": {
                            "media_rel": f"<RAW_VIDEO_CLIP_{c_idx}_REL>",
                            "dialogue_turns": [
                                {
                                    "voice_audio_rel": f"<VOICE_{t_idx}_REL>",
                                    "target_face": "leftmost" if t_idx % 2 == 1 else "rightmost",
                                    "start_sec": round((t_idx - first_turn) * clip_dur_sec / len(clip_turns), 2),
                                }
                                for t_idx in clip_turns
                            ],
                        },
                        "save_output_as": f"LIPSYNC_CLIP_{c_idx}_REL",
                    }
                )
            post_calls.append(
                {
                    "tool": "studio_render_montage",
                    "arguments": {
                        "title": "paseo_calle_conversacion_master",
                        "items": [{"media_rel": f"<LIPSYNC_CLIP_{i}_REL>"} for i in range(1, n_clips + 1)],
                        "fade_seconds": 0.0,
                        "global_audio_rel": "<STREET_AMBIENCE_REL>",
                        "global_audio_volume": 0.18,
                    },
                    "save_output_as": "FINAL_MASTER_VIDEO_REL",
                }
            )
        else:
            post_calls = [
                {
                    "tool": "studio_prepare_model",
                    "arguments": {"model_id": "edtalk-256-lipsync-hd"},
                },
                *(
                    [
                        {
                            "tool": "studio_render_montage",
                            "arguments": {
                                "title": "paseo_continuo_base",
                                "items": [
                                    {"media_rel": f"<RAW_VIDEO_CLIP_{i}_REL>"} for i in range(1, n_clips + 1)
                                ],
                                "fade_seconds": 0.0,
                            },
                            "save_output_as": "JOINED_WALK_VIDEO_REL",
                        }
                    ]
                    if n_clips >= 2
                    else []
                ),
                {
                    "tool": "studio_lipsync_character",
                    "arguments": {
                        "media_rel": "<JOINED_WALK_VIDEO_REL>" if n_clips >= 2 else "<RAW_VIDEO_CLIP_1_REL>",
                        "dialogue_turns": [
                            {
                                "voice_audio_rel": f"<VOICE_{i}_REL>",
                                "target_face": "leftmost" if (i % 2 == 1) else "rightmost",
                                "start_sec": round((i - 1) * (n_clips * clip_dur_sec / num_turns) + 0.15, 2),
                            }
                            for i in range(1, num_turns + 1)
                        ],
                        "ambient_rel": "<STREET_AMBIENCE_REL>",
                        "ambient_volume": 0.18,
                    },
                    "save_output_as": "FINAL_MASTER_VIDEO_REL",
                },
            ]

        steps.append(
            {
                "step_number": len(steps) + 1,
                "phase": "lipsync_and_final_master",
                "purpose": (
                    "Sincronizar labios por turnos (`leftmost` / `rightmost` con `edtalk-256-lipsync-hd`) y unir los tramos "
                    "con `fade_seconds=0.0` (porque cada `FRAME_{i+1}` enlaza exactamente el final del Clip `i` con el inicio del Clip `i+1`) "
                    "más sonido urbano `STREET_AMBIENCE_REL` con Auto-Ducking."
                ),
                "recommended_calls": post_calls,
            }
        )

        keyframe_outputs = [f"FRAME_{i}_REL" for i in range(1, num_keyframes + 1)]
        for step in steps:
            phase = step["phase"]
            if phase.startswith("keyframe_"):
                step["batch"] = "1_all_keyframes"
            elif phase == "coherence_and_anti_static_audit":
                step["batch"] = "2_audit_and_repair"
                step["requires_outputs"] = keyframe_outputs
            elif phase.startswith("video_clip_"):
                step["batch"] = "3_all_video_clips"
                step["requires_outputs"] = keyframe_outputs + ["COHERENCE_AUDIT_RESULT"]
                step["requires_audit"] = {"all_consistent": True, "vlm_semantic_used": has_gpu}
            else:
                step["batch"] = "4_sound_and_finish"

        return {
            "ok": True,
            "preset_id": resolved_id,
            "title": meta["title"],
            "creative_brief": {
                "visual_scene": scene_txt,
                "dialogue_request": scene_description.strip(),
                "dialogue_lines_source": "provided" if dialogue_lines else "example_draft",
                "dialogue_instruction": "Respetar los diálogos proporcionados. Si son ejemplos de plantilla, reescribirlos según el tema/tono del usuario antes del TTS, manteniendo voces, turnos y duración del plan.",
            },
            "planned_duration_summary": {
                "num_clips": n_clips,
                "num_keyframes": num_keyframes,
                "clip_duration_sec": clip_dur_sec,
                "estimated_total_seconds": round(n_clips * clip_dur_sec, 1),
            },
            "detected_gpu": {
                "gpu_name": gpu.get("gpu_name", "CPU"),
                "vram_total_gb": vram_gb,
                "selected_video_stack": vid_stack,
            },
            "execution_policy": {
                "order": ["1_all_keyframes", "2_audit_and_repair", "3_all_video_clips", "4_sound_and_finish"],
                "gpu_calls_parallel": False,
                "wait_for_each_job_to_complete": True,
                "running_job_action": "studio_wait_for_job con el MISMO job_id; un timeout HTTP no es un fallo del motor.",
                "on_model_load_error": "STOP: informar del error, conservar el modelo elegido y los archivos ya descargados. No probar otros 14B/13B/1.3B automáticamente.",
                "allow_automatic_fallback_downloads": False,
                "dialogue_timing": "Medir duración real de cada voz; acortar o dividir las réplicas que no quepan en su tramo antes de LipSync.",
            },
            "golden_rules_mandatory": [
                "1. EJECUCIÓN POR LOTES EN VRAM (NO CONMUTES MODELOS IDA Y VUELTA): Genera PRIMERO TODOS los fotogramas (`FRAME_1`..`FRAME_{N+1}`) seguidos mientras Qwen 2511 está en VRAM; audita con `studio_analyze_coherence`; y DESPUÉS genera TODOS los clips de vídeo (`CLIP_1`..`CLIP_N`) seguidos mientras el modelo de vídeo está en VRAM. NO llames a `studio_prepare_model` por separado: `studio_generate_or_edit_image` y `studio_generate_video` auto-cargan el modelo y sus LoRAs automáticamente.",
                "2. PROHIBIDO EL ENCUADRE CERRADO SIN SUELO/FONDO: En el Fotograma 1 usa plano americano (rodillas arriba) o cuerpo entero con elementos claros de calle (escaparates, árboles, farolas, acera) y un prompt visual limpio (sin instrucciones meta como 'crea los diálogos' o 'clip de un minuto').",
                "3. PROHIBIDO CONGELAR EL FONDO ENTRE FOTOGRAMAS: En los Fotogramas 2..N usa SIEMPRE `custom_lora_ids=['qwen-lora-next-scene', 'qwen-lora-skin-realism', 'qwen-lora-anylight-env']`, empieza el prompt con `'Next Scene:'`, alterna la pierna adelantada de cada personaje y pide explícitamente que los escaparates/farolas anteriores retrocedan en perspectiva 4 metros más atrás.",
                "4. CADENA DE REFERENCIAS SIN DERIVA: Para el Fotograma `k` usa `base_image_rel=FRAME_{k-1}` (continuidad espacial progresiva) y `reference_rels=[FRAME_1, FRAME_{k-1}]` (bloquea los rostros y ropa originales del Ancla `FRAME_1`).",
                "5. AUDITORÍA ANTI-ESTÁTICA OBLIGATORIA: Llama a `studio_analyze_coherence` con `expect_spatial_progression=true` antes de animar vídeo. Si sale `TOO_STATIC_NEEDS_ADVANCE` o `NEEDS_REPAIR`, ejecuta `recommended_repair_tool_call`.",
                "6. PROMPT DE VÍDEO EN 3 VECTORES (`flf2v`) Y REGLA ANTI-PÁNICO DE DISCO: Incluye SIEMPRE (a) cámara Steadicam retrocediendo, (b) zancada continua + gestos al hablar, y (c) paralaje 3D de acera/tiendas. Si un trabajo falla, NUNCA descargues otros 3 modelos pesados en paralelo llenando el disco.",
            ],
            "execution_plan": steps,
        }

    # ==================== PRESET 2: ESCENA FIJA CON DIÁLOGO (RESTAURANTE / MESA / INTERIOR) ====================
    if resolved_id == "coherencia_escena_fija_dialogo":
        return {
            "ok": True,
            "preset_id": resolved_id,
            "title": meta["title"],
            "detected_gpu": {"gpu_name": gpu.get("gpu_name", "CPU"), "vram_total_gb": vram_gb, "selected_video_stack": vid_stack},
            "golden_rules_mandatory": [
                "1. Genera solo el Fotograma 1 desde texto (`qwen-image-edit-2511-pack` + `qwen-lora-skin-realism`).",
                "2. Deriva Fotogramas 2..N con `base_image_rel=FRAME_1`, `reference_rels=[FRAME_1]` pidiendo mantener mesa/pared/luz idénticas y cambiando solo postura/mirada/gesto.",
                "3. Audita con `studio_analyze_coherence(expect_spatial_progression=false)`.",
                "4. Anima con `studio_generate_video(video_submode='flf2v')` + `UniAnimate 14B` y aplica `studio_lipsync_character` con `dialogue_turns` + ambiente `restaurant_warm_murmur`.",
            ],
            "execution_plan": [
                {
                    "step_number": 1,
                    "tool": "studio_generate_or_edit_image",
                    "arguments": {
                        "model_id": "qwen-image-edit-2511-pack",
                        "aspect_ratio": ar,
                        "custom_lora_ids": ["qwen-lora-skin-realism", "qwen-lora-flatlog-cinema"],
                        "prompt": f"Medium two-shot cinematic photograph: {scene_txt}. {chars_txt}",
                    },
                    "save_output_as": "FRAME_1_REL",
                },
                {
                    "step_number": 2,
                    "tool": "studio_generate_or_edit_image",
                    "arguments": {
                        "model_id": "qwen-image-edit-2511-pack",
                        "base_image_rel": "<FRAME_1_REL>",
                        "reference_rels": ["<FRAME_1_REL>"],
                        "aspect_ratio": ar,
                        "custom_lora_ids": ["qwen-lora-skin-realism"],
                        "prompt": "Keep the exact same two people, faces, hairstyles, outfits, table, background and warm lighting; change only their pose so the person on the right leans slightly forward smiling and gesturing while replying.",
                    },
                    "save_output_as": "FRAME_2_REL",
                },
                {
                    "step_number": 3,
                    "tool": "studio_analyze_coherence",
                    "arguments": {
                        "anchor_image_rel": "<FRAME_1_REL>",
                        "candidate_image_rels": ["<FRAME_2_REL>"],
                        "expect_spatial_progression": False,
                        "mode": "hybrid_auto",
                    },
                },
                {
                    "step_number": 4,
                    "tool": "studio_generate_video",
                    "arguments": {
                        "video_submode": "flf2v",
                        "mode": vid_stack["mode"],
                        "model_id": vid_stack["model_id"],
                        "video_lora_id": vid_stack["video_lora_id"],
                        "extra_video_lora_ids": vid_stack["extra_video_lora_ids"],
                        "image_rel": "<FRAME_1_REL>",
                        "end_image_rel": "<FRAME_2_REL>",
                        "duration_sec": 5.0,
                        "use_official_defaults": True,
                        "prompt": "Subtle cinematic dolly-in on two people conversing naturally, realistic head turns, blinking, expressive hand gestures, warm ambient lighting, stable faces and hands.",
                    },
                },
            ],
        }

    # ==================== PRESET 3: COBERTURA MULTICÁMARA 3D (PLANO / CONTRAPLANO) ====================
    if resolved_id == "cambio_angulo_3d_mismo_personaje":
        return {
            "ok": True,
            "preset_id": resolved_id,
            "title": meta["title"],
            "detected_gpu": {"gpu_name": gpu.get("gpu_name", "CPU"), "vram_total_gb": vram_gb, "selected_video_stack": vid_stack},
            "golden_rules_mandatory": [
                "1. Crea el plano maestro frontal (`FRAME_1_REL`).",
                "2. Para cada nuevo ángulo (perfil 45°, escorzo sobre el hombro, plano lateral), usa `qwen-image-edit-2511-pack` con `base_image_rel=FRAME_1_REL`, `reference_rels=[FRAME_1_REL]` y `custom_lora_ids=['qwen-lora-multi-angles-3d', 'qwen-lora-anylight-env', 'qwen-lora-skin-realism']`.",
                "3. Anima cada ángulo por separado en modo `i2v` (5 s cada plano) y únelos por corte limpio (`fade_seconds=0.0`) en `studio_render_montage`.",
            ],
            "execution_plan": [
                {
                    "step_number": 1,
                    "tool": "studio_generate_or_edit_image",
                    "arguments": {
                        "model_id": "qwen-image-edit-2511-pack",
                        "aspect_ratio": ar,
                        "custom_lora_ids": ["qwen-lora-skin-realism", "qwen-lora-flatlog-cinema"],
                        "prompt": f"Medium cinematic establishing shot: {scene_txt}. {chars_txt}",
                    },
                    "save_output_as": "ANGLE_1_MASTER_REL",
                },
                {
                    "step_number": 2,
                    "tool": "studio_generate_or_edit_image",
                    "arguments": {
                        "model_id": "qwen-image-edit-2511-pack",
                        "base_image_rel": "<ANGLE_1_MASTER_REL>",
                        "reference_rels": ["<ANGLE_1_MASTER_REL>"],
                        "aspect_ratio": ar,
                        "custom_lora_ids": [
                            "qwen-lora-multi-angles-3d",
                            "qwen-lora-anylight-env",
                            "qwen-lora-skin-realism",
                        ],
                        "lora_scales": {
                            "qwen-lora-multi-angles-3d": 0.90,
                            "qwen-lora-anylight-env": 0.70,
                            "qwen-lora-skin-realism": 0.75,
                        },
                        "prompt": "Rotate the camera 45 degrees to a cinematic three-quarter close-up angle of the exact same character in the same environment and lighting, preserving identical face, hair and outfit.",
                    },
                    "save_output_as": "ANGLE_2_SIDE_REL",
                },
                {
                    "step_number": 3,
                    "tool": "studio_generate_video",
                    "arguments": {
                        "video_submode": "i2v",
                        "mode": vid_stack["mode"],
                        "model_id": vid_stack["model_id"],
                        "video_lora_id": vid_stack["video_lora_id"],
                        "extra_video_lora_ids": vid_stack["extra_video_lora_ids"],
                        "image_rel": "<ANGLE_2_SIDE_REL>",
                        "duration_sec": 5.0,
                        "use_official_defaults": True,
                        "prompt": "Subtle cinematic handheld camera breath, natural eye movement, speaking expression and realistic shallow depth of field.",
                    },
                },
            ],
        }

    # ==================== PRESET 4: CORTO COHERENTE DE 1 MINUTO (6-12 PLANOS) ====================
    if resolved_id == "corto_1_minuto_multiescena":
        return {
            "ok": True,
            "preset_id": resolved_id,
            "title": meta["title"],
            "detected_gpu": {"gpu_name": gpu.get("gpu_name", "CPU"), "vram_total_gb": vram_gb, "selected_video_stack": vid_stack},
            "golden_rules_mandatory": [
                "1. FASE 1 (SOLO IMAGEN + VAULT): Genera el Retrato Ancla de cada protagonista y guárdalos con `studio_higgsfield_save_character`.",
                "2. FASE 2 (TODOS LOS KEYFRAMES ANTES DE CARGAR VÍDEO): Genera los 8–12 keyframes de una sola tacada mientras Qwen 2511 está en VRAM (así evitas descargar y recargar modelos pesados 12 veces). Usa `qwen-lora-next-scene` para tramos de avance y `qwen-lora-multi-angles-3d` para cortes de ángulo.",
                "3. FASE 3 (AUDITORÍA EN LOTE): Pasa todos los keyframes por `studio_analyze_coherence` de una vez y repara solo el que falle.",
                "4. FASE 4 (LOTE DE VÍDEO EN VRAM): Con `Wan 14B` cargado una sola vez en VRAM, renderiza los 10–12 clips de 5 s seguidos (~1.6 min/clip en L4).",
                "5. FASE 5 (SONIDO + LIPSYNC + MONTAJE MAESTRO): Aplica `studio_lipsync_character` únicamente en los planos con diálogo visible, genera BSO con `musicgen-small-bgm` + ambiente, y ensambla el minuto entero en `studio_render_montage` (Auto-Ducking activo).",
            ],
            "execution_plan": [
                {"phase": "1_register_cast", "tool": "studio_higgsfield_save_character", "note": "Guardar protagonistas con su foto ancla y voz."},
                {"phase": "2_batch_all_keyframes", "tool": "studio_generate_or_edit_image", "note": "Generar Keyframes 1..12 manteniendo Qwen-Image-Edit 2511 en VRAM con Skin Realism + Next Scene / Multi-Angles."},
                {"phase": "3_batch_coherence_check", "tool": "studio_analyze_coherence", "note": "Verificar identidad antes de pasar al motor de vídeo."},
                {"phase": "4_batch_video_clips", "tool": "studio_generate_video", "arguments": {"model_id": vid_stack["model_id"], "video_lora_id": vid_stack["video_lora_id"], "extra_video_lora_ids": vid_stack["extra_video_lora_ids"], "duration_sec": 5.0}},
                {"phase": "5_lipsync_and_bgm_montage", "tool": "studio_render_montage", "note": "Unir los 12 clips con `global_audio_rel` (MusicGen) y Auto-Ducking sobre los diálogos."},
            ],
        }

    # ==================== PRESET 5: EDICIÓN FOTOGRÁFICA PRO + RELIGHT ====================
    if resolved_id == "edicion_fotografica_relight_cine":
        return {
            "ok": True,
            "preset_id": resolved_id,
            "title": meta["title"],
            "detected_gpu": {"gpu_name": gpu.get("gpu_name", "CPU"), "vram_total_gb": vram_gb, "selected_video_stack": vid_stack},
            "golden_rules_mandatory": [
                "1. Pasa la foto original en `base_image_rel` y `reference_rels` con `model_id='qwen-image-edit-2511-pack'`.",
                "2. Apila `qwen-lora-relight-scene` (0.80) + `qwen-lora-anylight-env` (0.75) + `qwen-lora-skin-realism` (0.80) para integrar la nueva iluminación sobre el rostro y ropa sin aspecto artificial.",
            ],
            "execution_plan": [
                {
                    "step_number": 1,
                    "tool": "studio_generate_or_edit_image",
                    "arguments": {
                        "model_id": "qwen-image-edit-2511-pack",
                        "base_image_rel": "<INPUT_PHOTO_REL>",
                        "reference_rels": ["<INPUT_PHOTO_REL>"],
                        "custom_lora_ids": [
                            "qwen-lora-relight-scene",
                            "qwen-lora-anylight-env",
                            "qwen-lora-skin-realism",
                        ],
                        "lora_scales": {
                            "qwen-lora-relight-scene": 0.80,
                            "qwen-lora-anylight-env": 0.75,
                            "qwen-lora-skin-realism": 0.80,
                        },
                        "prompt": f"Relight Scene: AnyLight: Preserve the exact same person, facial features and outfit while harmonizing lighting and environment to: {scene_txt}",
                    },
                }
            ],
        }

    # ==================== PRESET 6: ANIMACIÓN ESTILIZADA COHERENTE ====================
    return {
        "ok": True,
        "preset_id": resolved_id,
        "title": meta["title"],
        "detected_gpu": {"gpu_name": gpu.get("gpu_name", "CPU"), "vram_total_gb": vram_gb, "selected_video_stack": vid_stack},
        "golden_rules_mandatory": [
            "1. Para Anime moderno usa `qwen-lora-anime-v2` + `qwen-lora-next-scene` en los fotogramas y `flf2v` en vídeo.",
            "2. Para Animación clásica años 30 usa `wan21-vace-1.3b` con `video_lora_id='lora-wan21-1.3b-bidirect2'` + `extra_video_lora_ids=['lora-wan21-1.3b-classic-anim']`.",
        ],
        "execution_plan": [
            {
                "step_number": 1,
                "tool": "studio_generate_or_edit_image",
                "arguments": {
                    "model_id": "qwen-image-edit-2511-pack",
                    "aspect_ratio": ar,
                    "custom_lora_ids": ["qwen-lora-anime-v2"],
                    "prompt": f"Anime Style: {scene_txt}. {chars_txt}",
                },
                "save_output_as": "ANIME_FRAME_1_REL",
            },
            {
                "step_number": 2,
                "tool": "studio_generate_or_edit_image",
                "arguments": {
                    "model_id": "qwen-image-edit-2511-pack",
                    "base_image_rel": "<ANIME_FRAME_1_REL>",
                    "reference_rels": ["<ANIME_FRAME_1_REL>"],
                    "aspect_ratio": ar,
                    "custom_lora_ids": ["qwen-lora-anime-v2", "qwen-lora-next-scene"],
                    "prompt": "Next Scene: Anime Style: Same exact characters and outfits advancing dynamically to the next action beat in the same setting.",
                },
                "save_output_as": "ANIME_FRAME_2_REL",
            },
        ],
    }
