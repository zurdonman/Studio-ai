import fnmatch
import io
import json
import os
import shutil
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests
from huggingface_hub import HfApi, hf_hub_download
from tqdm.std import tqdm as std_tqdm

from studio.config import MODELS_DIR, STATE_DIR
from studio.hardware import colab_console_log, query_gpu_info
from studio.jobs import JobRecord
from studio.model_recipes import EXTRA_IMAGE_MODELS, ENGINE_CONTRACTS, image_recipe_files_complete

CUSTOM_MODELS_FILE = STATE_DIR / "custom_models.json"
DISK_SAFETY_MARGIN_GB = 1.0
_DOWNLOAD_LOCKS_GUARD = threading.Lock()
_MODEL_DOWNLOAD_LOCKS: Dict[str, threading.RLock] = {}


def _get_model_download_lock(model_id: str) -> threading.RLock:
    with _DOWNLOAD_LOCKS_GUARD:
        lk = _MODEL_DOWNLOAD_LOCKS.get(model_id)
        if lk is None:
            lk = threading.RLock()
            _MODEL_DOWNLOAD_LOCKS[model_id] = lk
        return lk

# Perfiles oficiales de GPU disponibles en Google Colab para filtrado y estimación realista
COLAB_GPU_PROFILES: List[Dict[str, Any]] = [
    {
        "id": "auto",
        "label": "🎯 Mi GPU detectada (Auto)",
        "short": "Mi GPU (Auto)",
        "vram_gb": None,
        "speed_factor": 1.0,
        "rate_uh": None,
    },
    {
        "id": "T4",
        "label": "NVIDIA T4 · 15 GB VRAM (~1.17 u/h)",
        "short": "T4 (15 GB)",
        "vram_gb": 15.0,
        "speed_factor": 1.65,
        "rate_uh": 1.17,
    },
    {
        "id": "L4",
        "label": "NVIDIA L4 · 22.5 GB VRAM (1.92 u/h)",
        "short": "L4 (22.5 GB)",
        "vram_gb": 22.5,
        "speed_factor": 1.0,
        "rate_uh": 1.92,
    },
    {
        "id": "A100_40",
        "label": "NVIDIA A100 · 40 GB VRAM (~6.93 u/h)",
        "short": "A100 (40 GB)",
        "vram_gb": 40.0,
        "speed_factor": 0.52,
        "rate_uh": 6.93,
    },
    {
        "id": "A100_80",
        "label": "NVIDIA A100 · 80 GB VRAM (~11.77 u/h)",
        "short": "A100 (80 GB)",
        "vram_gb": 80.0,
        "speed_factor": 0.45,
        "rate_uh": 11.77,
    },
    {
        "id": "H100",
        "label": "NVIDIA H100 · 80 GB VRAM (~13.10 u/h)",
        "short": "H100 (80 GB)",
        "vram_gb": 80.0,
        "speed_factor": 0.28,
        "rate_uh": 13.10,
    },
    {
        "id": "CPU",
        "label": "Solo CPU · Sin GPU (0 GB VRAM)",
        "short": "Solo CPU",
        "vram_gb": 0.0,
        "speed_factor": 12.0,
        "rate_uh": 0.08,
    },
]

_DIFFUSERS_VIDEO_ALLOW = [
    "model_index.json",
    "config.json",
    "scheduler/*",
    "text_encoder/*",
    "text_encoder_2/*",
    "tokenizer/*",
    "tokenizer_2/*",
    "transformer/*",
    "transformer_2/*",
    "vae/*",
    "image_encoder/*",
    "image_processor/*",
    "audio_vae/*",
    "vocoder/*",
    "connectors/*",
    "latent_upsampler/*",
    "processor/*",
]

DEFAULT_CATALOG: List[Dict[str, Any]] = [
    # ==================== IMAGEN ====================
    {
        "id": "qwen-image-edit-2511-pack",
        "name": "Qwen-Image-Edit 2511 (4-bit) + LoRA Lightning (4 pasos)",
        "category": "image",
        "is_default": True,
        "badge": "Predeterminado ★ · Edición + Multi-Ref + Inpaint",
        "capabilities": ["i2i", "multi_ref", "inpaint", "t2i"],
        "description": (
            "Pack principal verificado: descarga y carga juntos Qwen-Image-Edit-2511 en 4-bit (bitsandbytes) "
            "y su LoRA Lightning de 4 pasos. Edita fotos, combina hasta 3 referencias y pinta máscaras."
        ),
        "estimated_gb": 16.6,
        "min_vram_gb": 14.0,
        "recommended_vram_gb": 19.0,
        "l4_seconds_hint": "18–26 s (4 pasos)",
        "engine": "qwen_image_edit",
        "artifacts": [
            {
                "role": "base",
                "repo_id": "ovedrive/Qwen-Image-Edit-2511-4bit",
                "subdir": "base",
                "filename": None,
                "verify_file": "model_index.json",
            },
            {
                "role": "lora_lightning",
                "repo_id": "lightx2v/Qwen-Image-Edit-2511-Lightning",
                "subdir": "lora_lightning",
                "filename": "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
                "verify_file": "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
            },
        ],
    },
    {
        "id": "qwen-image-2512-pack",
        "name": "Qwen-Image 2512 Texto→Imagen (4-bit) + LoRA Lightning (4 pasos)",
        "category": "image",
        "is_default": False,
        "badge": "Creación Pura Texto→Imagen · 4 pasos",
        "capabilities": ["t2i"],
        "description": (
            "Modelo generativo Qwen-Image 2512 cuantizado en 4-bit junto a LoRA Lightning V2 (4 pasos) "
            "para crear escenas nuevas desde cero con tipografía y realismo altos."
        ),
        "estimated_gb": 16.6,
        "min_vram_gb": 14.0,
        "recommended_vram_gb": 19.0,
        "l4_seconds_hint": "16–24 s (4 pasos)",
        "engine": "qwen_image_t2i",
        "artifacts": [
            {
                "role": "base",
                "repo_id": "ovedrive/Qwen-Image-2512-4bit",
                "subdir": "base",
                "filename": None,
                "verify_file": "model_index.json",
            },
            {
                "role": "lora_lightning",
                "repo_id": "lightx2v/Qwen-Image-Lightning",
                "subdir": "lora_lightning",
                "filename": "Qwen-Image-Lightning-4steps-V2.0-bf16.safetensors",
                "verify_file": "Qwen-Image-Lightning-4steps-V2.0-bf16.safetensors",
            },
        ],
    },
    {
        "id": "sdxl-turbo-fp16",
        "name": "SDXL Turbo FP16 (1–4 pasos · Ultraligero para T4 y L4)",
        "category": "image",
        "is_default": False,
        "badge": "Ultraligero · 1–4 pasos",
        "capabilities": ["t2i"],
        "description": (
            "Generador y bocetador instantáneo de Stability AI en media precisión (FP16 limpio sin archivos ONNX). "
            "Ideal cuando usas una GPU T4 gratuita o buscas bocetos en 2 segundos."
        ),
        "estimated_gb": 6.5,
        "min_vram_gb": 5.5,
        "recommended_vram_gb": 8.0,
        "l4_seconds_hint": "2–4 s",
        "engine": "sdxl_turbo",
        "artifacts": [
            {
                "role": "base",
                "repo_id": "stabilityai/sdxl-turbo",
                "subdir": "base",
                "filename": None,
                "verify_file": "model_index.json",
                "allow_patterns": [
                    "model_index.json",
                    "scheduler/*",
                    "tokenizer/*",
                    "tokenizer_2/*",
                    "text_encoder/config.json",
                    "text_encoder/model.fp16.safetensors",
                    "text_encoder_2/config.json",
                    "text_encoder_2/model.fp16.safetensors",
                    "unet/config.json",
                    "unet/diffusion_pytorch_model.fp16.safetensors",
                    "vae/config.json",
                    "vae/diffusion_pytorch_model.fp16.safetensors",
                ],
            }
        ],
    },
    # ==================== LORAS ====================
    {
        "id": "qwen-lora-nsfw",
        "name": "LoRA Extra: Qwen-Image-Edit Plus (Sin censura / Realismo)",
        "category": "lora",
        "lora_target": "image",
        "lora_group": "quality",
        "compatible_models": ["qwen-image-edit-2511-pack"],
        "default_lora_scale": 1.0,
        "is_default": False,
        "parent_model": "qwen-image-edit-2511-pack",
        "adapter_name": "mcnl-nsfw-v1",
        "badge": "Complemento LoRA · Opcional",
        "description": (
            "LoRA opcional compatible con Qwen-Image-Edit-2511. Se descarga aparte y puedes activarlo "
            "o desactivarlo con un interruptor y control de fuerza al editar."
        ),
        "estimated_gb": 0.35,
        "min_vram_gb": 0.2,
        "recommended_vram_gb": 0.5,
        "l4_seconds_hint": "+0.5 s",
        "engine": "qwen_lora",
        "artifacts": [
            {
                "role": "lora",
                "repo_id": "ScottzillaSystems/qwen-image-edit-plus-nsfw-lora",
                "subdir": "lora_nsfw",
                "filename": "qwen-image-edit-plus-nsfw-lora.safetensors",
                "verify_file": "qwen-image-edit-plus-nsfw-lora.safetensors",
            }
        ],
    },
    {
        "id": "qwen-lora-lightning-8steps",
        "name": "LoRA Extra: Qwen-Image-Edit Detalle Fino (8 pasos)",
        "category": "lora",
        "lora_target": "image",
        "lora_group": "speed",
        "compatible_models": ["qwen-image-edit-2511-pack"],
        "default_lora_scale": 1.0,
        "is_default": False,
        "parent_model": "qwen-image-edit-2511-pack",
        "adapter_name": "lightning-8steps",
        "badge": "Complemento LoRA · 8 pasos",
        "description": (
            "Alternativa oficial de 8 pasos de LightX2V cuando prefieres un punto extra de textura fina "
            "sobre el modo rápido de 4 pasos."
        ),
        "estimated_gb": 0.80,
        "min_vram_gb": 0.3,
        "recommended_vram_gb": 0.8,
        "l4_seconds_hint": "35–42 s (8 pasos)",
        "engine": "qwen_lora",
        "artifacts": [
            {
                "role": "lora",
                "repo_id": "lightx2v/Qwen-Image-Lightning",
                "subdir": "lora_8steps",
                "filename": "Qwen-Image-Edit-2509-Lightning-8steps-V1.0-bf16.safetensors",
                "verify_file": "Qwen-Image-Edit-2509-Lightning-8steps-V1.0-bf16.safetensors",
            }
        ],
    },
    # ==================== LoRAs DE IMAGEN: COHERENCIA DE PERSONAJES, ENTORNO Y CALIDAD ====================
    {
        "id": "qwen-lora-multi-angles-3d",
        "name": "👤🎥 LoRA Multi-Ángulo 3D (96 Poses de Cámara · Coherencia Total de Personaje y Entorno)",
        "category": "lora",
        "lora_target": "image",
        "lora_group": "character",
        "compatible_models": ["qwen-image-edit-2511-pack"],
        "default_lora_scale": 0.90,
        "trigger_phrase": "Front-right quarter view, consistent character and environment",
        "is_default": False,
        "parent_model": "qwen-image-edit-2511-pack",
        "adapter_name": "fal_multi_angles_3d",
        "badge": "★ Coherencia 3D · 96 Ángulos · 295 MB",
        "description": (
            "Entrenado con +3.000 escenas 3D Gaussian Splatting (fal.ai). Permite rotar la cámara (perfil, contrapicado, primer plano, plano general) "
            "alrededor del MISMO personaje y entorno sin perder identidad facial ni vestimenta. Ideal para generar fotogramas Inicio → Fin coherentes."
        ),
        "estimated_gb": 0.30,
        "min_vram_gb": 0.2,
        "recommended_vram_gb": 0.5,
        "l4_seconds_hint": "+0.4 s",
        "engine": "qwen_lora",
        "artifacts": [
            {
                "role": "lora",
                "repo_id": "fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA",
                "subdir": "lora_multi_angles",
                "filename": "qwen-image-edit-2511-multiple-angles-lora.safetensors",
                "verify_file": "qwen-image-edit-2511-multiple-angles-lora.safetensors",
            }
        ],
    },
    {
        "id": "qwen-lora-next-scene",
        "name": "🎬🏛️ LoRA Next-Scene Continuidad de Guion (Siguiente Plano Coherente de Escena y Actor)",
        "category": "lora",
        "lora_target": "image",
        "lora_group": "environment",
        "compatible_models": ["qwen-image-edit-2511-pack"],
        "default_lora_scale": 0.85,
        "trigger_phrase": "Next scene:",
        "is_default": False,
        "parent_model": "qwen-image-edit-2511-pack",
        "adapter_name": "next_scene_continuity",
        "badge": "★ Continuidad Storyboard · 295 MB",
        "description": (
            "Especialista en continuidad cinematográfica: a partir de un fotograma base genera el siguiente instante narrativo "
            "conservando exactamente el mismo escenario, iluminación, objetos de la habitación y rostro/ropa del personaje."
        ),
        "estimated_gb": 0.30,
        "min_vram_gb": 0.2,
        "recommended_vram_gb": 0.5,
        "l4_seconds_hint": "+0.4 s",
        "engine": "qwen_lora",
        "artifacts": [
            {
                "role": "lora",
                "repo_id": "lovis93/next-scene-qwen-image-lora-2509",
                "subdir": "lora_next_scene",
                "filename": "next-scene_lora-v2-3000.safetensors",
                "verify_file": "next-scene_lora-v2-3000.safetensors",
            }
        ],
    },
    {
        "id": "qwen-lora-anylight-env",
        "name": "🏛️💡 LoRA AnyLight Coherencia de Luz entre Planos (Igualar Iluminación de Entorno)",
        "category": "lora",
        "lora_target": "image",
        "lora_group": "environment",
        "compatible_models": ["qwen-image-edit-2511-pack"],
        "default_lora_scale": 0.85,
        "trigger_phrase": "Apply consistent ambient lighting across the scene",
        "is_default": False,
        "parent_model": "qwen-image-edit-2511-pack",
        "adapter_name": "qie_anylight",
        "badge": "Coherencia Lumínica · Multi-Ref · 560 MB",
        "description": (
            "Transfiere la atmósfera lumínica y temperatura de color de una imagen de referencia (Imagen 2) sobre tu personaje o plano (Imagen 1) "
            "para que todas las tomas de una secuencia compartan exactamente la misma luz de entorno."
        ),
        "estimated_gb": 0.56,
        "min_vram_gb": 0.2,
        "recommended_vram_gb": 0.6,
        "l4_seconds_hint": "+0.5 s",
        "engine": "qwen_lora",
        "artifacts": [
            {
                "role": "lora",
                "repo_id": "lilylilith/QIE-2511-MP-AnyLight",
                "subdir": "lora_anylight",
                "filename": "QIE-2511-AnyLight_.safetensors",
                "verify_file": "QIE-2511-AnyLight_.safetensors",
            }
        ],
    },
    {
        "id": "qwen-lora-relight-scene",
        "name": "🌅 LoRA Re-Iluminación de Entorno (Cambiar Hora del Día / Luces sin Alterar Personaje)",
        "category": "lora",
        "lora_target": "image",
        "lora_group": "environment",
        "compatible_models": ["qwen-image-edit-2511-pack"],
        "default_lora_scale": 0.85,
        "is_default": False,
        "parent_model": "qwen-image-edit-2511-pack",
        "adapter_name": "qwen_relight",
        "badge": "Re-Light Escenario · 290 MB",
        "description": (
            "Cambia la dirección e intensidad de la luz del escenario (velas cálidas, neón nocturno, hora dorada, contraluz) "
            "manteniendo intactos los rasgos faciales, pose y proporciones del sujeto."
        ),
        "estimated_gb": 0.29,
        "min_vram_gb": 0.2,
        "recommended_vram_gb": 0.5,
        "l4_seconds_hint": "+0.4 s",
        "engine": "qwen_lora",
        "artifacts": [
            {
                "role": "lora",
                "repo_id": "dx8152/Qwen-Image-Edit-2509-Relight",
                "subdir": "lora_relight",
                "filename": "Qwen-Edit-Relight.safetensors",
                "verify_file": "Qwen-Edit-Relight.safetensors",
            }
        ],
    },
    {
        "id": "qwen-lora-studio-delight",
        "name": "👤✨ LoRA Studio DeLight (Limpiar Sombras para Ficha Ancla de Personaje)",
        "category": "lora",
        "lora_target": "image",
        "lora_group": "character",
        "compatible_models": ["qwen-image-edit-2511-pack"],
        "default_lora_scale": 0.90,
        "trigger_phrase": "Neutral uniform lighting. Preserve identity and composition.",
        "is_default": False,
        "parent_model": "qwen-image-edit-2511-pack",
        "adapter_name": "studio_delight",
        "badge": "Casting Limpio · Identidad Pura · 236 MB",
        "description": (
            "Elimina sombras duras, contraluces y tintes de color de cualquier foto para obtener un retrato limpio y neutro "
            "del personaje, óptimo para guardar como foto ancla en el Casting Vault."
        ),
        "estimated_gb": 0.24,
        "min_vram_gb": 0.2,
        "recommended_vram_gb": 0.5,
        "l4_seconds_hint": "+0.4 s",
        "engine": "qwen_lora",
        "artifacts": [
            {
                "role": "lora",
                "repo_id": "prithivMLmods/QIE-2511-Studio-DeLight",
                "subdir": "lora_delight",
                "filename": "QIE-2511-Studio-DeLight-5000.safetensors",
                "verify_file": "QIE-2511-Studio-DeLight-5000.safetensors",
            }
        ],
    },
    {
        "id": "qwen-lora-skin-realism",
        "name": "💎 LoRA Textura de Piel Humana Natural (Poros Reales · Anti-Plástico IA)",
        "category": "lora",
        "lora_target": "image",
        "lora_group": "quality",
        "compatible_models": ["qwen-image-edit-2511-pack"],
        "default_lora_scale": 0.75,
        "trigger_phrase": "make the subjects skin details more prominent and natural",
        "is_default": False,
        "parent_model": "qwen-image-edit-2511-pack",
        "adapter_name": "qwen_skin_real",
        "badge": "★ Calidad Retrato · Piel Orgánica · 295 MB",
        "description": (
            "Añade micro-textura real de piel humana, poros naturales y acabado fotográfico sin redibujar ni cambiar "
            "la identidad de la persona. Excelente antes de aplicar LipSync HD."
        ),
        "estimated_gb": 0.30,
        "min_vram_gb": 0.2,
        "recommended_vram_gb": 0.5,
        "l4_seconds_hint": "+0.4 s",
        "engine": "qwen_lora",
        "artifacts": [
            {
                "role": "lora",
                "repo_id": "tlennon-ie/qwen-edit-skin",
                "subdir": "lora_skin",
                "filename": "qwen-edit-skin.safetensors",
                "verify_file": "qwen-edit-skin.safetensors",
            }
        ],
    },
    {
        "id": "qwen-lora-ultra-portrait",
        "name": "💎 LoRA Retrato Ultra-Realista de Cine (Nitidez Ojos, Cabello y Tejidos)",
        "category": "lora",
        "lora_target": "image",
        "lora_group": "quality",
        "compatible_models": ["qwen-image-edit-2511-pack"],
        "default_lora_scale": 0.80,
        "trigger_phrase": "Ultra-realistic portrait.",
        "is_default": False,
        "parent_model": "qwen-image-edit-2511-pack",
        "adapter_name": "qwen_ultra_portrait",
        "badge": "Calidad Óptica Retrato · 236 MB",
        "description": (
            "Refuerza el enfoque en iris, pestañas, hebras de pelo y textura de ropa en primeros planos y planos medios "
            "cinematográficos con Qwen-Image-Edit 2511."
        ),
        "estimated_gb": 0.24,
        "min_vram_gb": 0.2,
        "recommended_vram_gb": 0.5,
        "l4_seconds_hint": "+0.4 s",
        "engine": "qwen_lora",
        "artifacts": [
            {
                "role": "lora",
                "repo_id": "prithivMLmods/Qwen-Image-Edit-2511-Ultra-Realistic-Portrait",
                "subdir": "lora_ultra_portrait",
                "filename": "URP_20.safetensors",
                "verify_file": "URP_20.safetensors",
            }
        ],
    },
    {
        "id": "qwen-lora-upscale-unblur",
        "name": "🔍 LoRA Restaurador HD Unblur + Upscale (Recuperar Fotos Borrosas o Fotogramas)",
        "category": "lora",
        "lora_target": "image",
        "lora_group": "quality",
        "compatible_models": ["qwen-image-edit-2511-pack"],
        "default_lora_scale": 0.85,
        "trigger_phrase": "Unblur and upscale.",
        "is_default": False,
        "parent_model": "qwen-image-edit-2511-pack",
        "adapter_name": "qwen_unblur_upscale",
        "badge": "Restauración Fotográfica · 295 MB",
        "description": (
            "Elimina desenfoque de movimiento, artefactos de compresión y falta de definición en fotografías antiguas "
            "o en el último fotograma extraído de un vídeo para seguir encadenando clips nítidos."
        ),
        "estimated_gb": 0.30,
        "min_vram_gb": 0.2,
        "recommended_vram_gb": 0.5,
        "l4_seconds_hint": "+0.4 s",
        "engine": "qwen_lora",
        "artifacts": [
            {
                "role": "lora",
                "repo_id": "prithivMLmods/Qwen-Image-Edit-2511-Unblur-Upscale",
                "subdir": "lora_unblur",
                "filename": "Qwen-Image-Edit-Unblur-Upscale_15.safetensors",
                "verify_file": "Qwen-Image-Edit-Unblur-Upscale_15.safetensors",
            }
        ],
    },
    {
        "id": "qwen-lora-flatlog-cinema",
        "name": "🎞️ LoRA Perfil ARRI Flat-Log Cine (Alto Rango Dinámico para Etalonaje y Vídeo)",
        "category": "lora",
        "lora_target": "image",
        "lora_group": "quality",
        "compatible_models": ["qwen-image-edit-2511-pack"],
        "default_lora_scale": 0.75,
        "trigger_phrase": "Transform into a cinematic flat log.",
        "is_default": False,
        "parent_model": "qwen-image-edit-2511-pack",
        "adapter_name": "qwen_flatlog",
        "badge": "HDR Cine Log · 236 MB",
        "description": (
            "Suaviza contrastes quemados y recupera información en sombras profundas con curva de cámara de cine profesional, "
            "evitando parpadeos de exposición al animar después con Wan 14B o LTX."
        ),
        "estimated_gb": 0.24,
        "min_vram_gb": 0.2,
        "recommended_vram_gb": 0.5,
        "l4_seconds_hint": "+0.4 s",
        "engine": "qwen_lora",
        "artifacts": [
            {
                "role": "lora",
                "repo_id": "prithivMLmods/QIE-2511-Cinematic-FlatLog-Control",
                "subdir": "lora_flatlog",
                "filename": "QIE-2511-Cinematic-FlatLog-Control-3200.safetensors",
                "verify_file": "QIE-2511-Cinematic-FlatLog-Control-3200.safetensors",
            }
        ],
    },
    {
        "id": "qwen-lora-anything2real",
        "name": "🎨➡️📷 LoRA Anything2Real (Convertir Boceto, Anime o 3D en Persona y Entorno Reales)",
        "category": "lora",
        "lora_target": "image",
        "lora_group": "style",
        "compatible_models": ["qwen-image-edit-2511-pack"],
        "default_lora_scale": 0.90,
        "trigger_phrase": "Change the picture to realistic photograph.",
        "is_default": False,
        "parent_model": "qwen-image-edit-2511-pack",
        "adapter_name": "qwen_anything2real",
        "badge": "★ +114k Descargas · 3D/Anime→Real · 580 MB",
        "description": (
            "Convierte cualquier boceto, ilustración, personaje anime o bloqueo 3D sencillo en una fotografía hiperrealista "
            "clavando exactamente la postura, encuadre y distribución del entorno."
        ),
        "estimated_gb": 0.58,
        "min_vram_gb": 0.3,
        "recommended_vram_gb": 0.6,
        "l4_seconds_hint": "+0.5 s",
        "engine": "qwen_lora",
        "artifacts": [
            {
                "role": "lora",
                "repo_id": "lrzjason/Anything2Real",
                "subdir": "lora_anything2real",
                "filename": "anything2real_2601.safetensors",
                "verify_file": "anything2real_2601.safetensors",
            }
        ],
    },
    {
        "id": "qwen-lora-anime-v2",
        "name": "🎨 LoRA Animación Japonesa Coherente V2 (Personaje + Fondo Anime Consistente)",
        "category": "lora",
        "lora_target": "image",
        "lora_group": "style",
        "compatible_models": ["qwen-image-edit-2511-pack"],
        "default_lora_scale": 0.90,
        "trigger_phrase": "Transform into anime while preserving composition and details.",
        "is_default": False,
        "parent_model": "qwen-image-edit-2511-pack",
        "adapter_name": "qwen_anime_v2",
        "badge": "Animación Coherente · 236 MB",
        "description": (
            "Transforma fotografías o personajes reales en fotogramas de animación japonesa limpia conservando vestuario, "
            "expresión y arquitectura del escenario para crear series animadas coherentes."
        ),
        "estimated_gb": 0.24,
        "min_vram_gb": 0.2,
        "recommended_vram_gb": 0.5,
        "l4_seconds_hint": "+0.4 s",
        "engine": "qwen_lora",
        "artifacts": [
            {
                "role": "lora",
                "repo_id": "prithivMLmods/Qwen-Image-Edit-2511-Anime",
                "subdir": "lora_anime_v2",
                "filename": "Qwen-Image-Edit-2511-Anime-2000.safetensors",
                "verify_file": "Qwen-Image-Edit-2511-Anime-2000.safetensors",
            }
        ],
    },
    {
        "id": "qwen-t2i-lora-snapshot-real",
        "name": "📱 LoRA Fotografía Espontánea Realista (Para Qwen-Image 2512 T2I y Edit 2511)",
        "category": "lora",
        "lora_target": "image",
        "lora_group": "quality",
        "compatible_models": ["qwen-image-2512-pack", "qwen-image-edit-2511-pack"],
        "default_lora_scale": 0.70,
        "trigger_phrase": "amateur photo, natural human skin texture",
        "is_default": False,
        "parent_model": "qwen-image-2512-pack",
        "adapter_name": "qwen_snapshot_real",
        "badge": "Qwen 2512 + 2511 · Realismo Natural · 295 MB",
        "description": (
            "Elimina el acabado artificial/plástico al crear escenas desde texto con Qwen-Image 2512 o editar con 2511, "
            "aportando óptica natural de cámara cotidiana, piel auténtica e iluminación creíble."
        ),
        "estimated_gb": 0.29,
        "min_vram_gb": 0.2,
        "recommended_vram_gb": 0.5,
        "l4_seconds_hint": "+0.4 s",
        "engine": "qwen_lora",
        "artifacts": [
            {
                "role": "lora",
                "repo_id": "ribilu74/qwensnapshot",
                "subdir": "lora_snapshot",
                "filename": "Qwen-Smartphone.safetensors",
                "verify_file": "Qwen-Smartphone.safetensors",
            }
        ],
    },
    {
        "id": "sdxl-lora-high-detail",
        "name": "💎 LoRA Micro-Detalle HD v5 para SDXL Turbo (Texturas Nítidas en T4/L4)",
        "category": "lora",
        "lora_target": "image",
        "lora_group": "quality",
        "compatible_models": ["sdxl-turbo-fp16"],
        "default_lora_scale": 0.75,
        "is_default": False,
        "parent_model": "sdxl-turbo-fp16",
        "adapter_name": "sdxl_high_detail",
        "badge": "SDXL Turbo · Calidad HD · 228 MB",
        "description": (
            "Potenciador de detalle fino para SDXL Turbo FP16: enriquece rostros, materiales y fondos arquitectónicos "
            "manteniendo la generación ultrarrápida en GPUs T4 y L4."
        ),
        "estimated_gb": 0.23,
        "min_vram_gb": 0.2,
        "recommended_vram_gb": 0.4,
        "l4_seconds_hint": "+0.2 s",
        "engine": "sdxl_lora",
        "artifacts": [
            {
                "role": "lora",
                "repo_id": "JCTN/JCTN_LORAxl",
                "subdir": "lora_sdxl_detail",
                "filename": "SDXLHighDetail_v5.safetensors",
                "verify_file": "SDXLHighDetail_v5.safetensors",
            }
        ],
    },
    {
        "id": "sdxl-lora-cinematic-light",
        "name": "🏛️🎬 LoRA Iluminación Cinematográfica para SDXL Turbo (Atmósfera y Entorno · 44 MB)",
        "category": "lora",
        "lora_target": "image",
        "lora_group": "environment",
        "compatible_models": ["sdxl-turbo-fp16"],
        "default_lora_scale": 1.2,
        "trigger_phrase": "cinematic lighting",
        "is_default": False,
        "parent_model": "sdxl-turbo-fp16",
        "adapter_name": "sdxl_cinematic_light",
        "badge": "SDXL Turbo · Luz de Cine · 44 MB",
        "description": (
            "Slider ultraligero (44 MB) de iluminación dramática y coherencia ambiental volumétrica para bocetar storyboards "
            "cinematográficos coherentes con SDXL Turbo."
        ),
        "estimated_gb": 0.05,
        "min_vram_gb": 0.1,
        "recommended_vram_gb": 0.3,
        "l4_seconds_hint": "+0.1 s",
        "engine": "sdxl_lora",
        "artifacts": [
            {
                "role": "lora",
                "repo_id": "ntc-ai/SDXL-LoRA-slider.cinematic-lighting",
                "subdir": "lora_sdxl_light",
                "filename": "cinematic lighting.safetensors",
                "verify_file": "cinematic lighting.safetensors",
            }
        ],
    },
    # ==================== LoRAs ACELERADORES DE VÍDEO ====================
    {
        "id": "lora-wan22-ti2v-5b-turbo",
        "name": "⚡ LoRA Turbo 4–6 Pasos para Wan 2.2 TI2V-5B (Self-Forcing · Solo Imagen/Texto→Vídeo)",
        "category": "lora",
        "lora_target": "video",
        "lora_group": "speed",
        "compatible_models": ["wan22-ti2v-5b"],
        "incompatible_submodes": ["flf2v"],
        "causal_attention": True,
        "recommended_steps": 6,
        "min_safe_steps": 4,
        "max_steps": 8,
        "recommended_cfg": 1.0,
        "default_lora_scale": 1.0,
        "is_default": False,
        "parent_model": "wan22-ti2v-5b",
        "adapter_name": "wan22_5b_turbo",
        "badge": "⚡ Vídeo ×12 más rápido · I2V/T2V · 332 MB",
        "description": (
            "LoRA causal Self-Forcing para Wan 2.2 TI2V-5B (332 MB) en Imagen a Vídeo y Texto a Vídeo. "
            "Baja de 45 pasos (~18 min) a solo 4–6 pasos con CFG 1.0 (~1.2 min en L4). No se usa en Frame a Frame porque la atención causal ignora el fotograma final."
        ),
        "estimated_gb": 0.33,
        "min_vram_gb": 0.4,
        "recommended_vram_gb": 1.0,
        "l4_seconds_hint": "~1.2 min (6 pasos @ CFG 1.0)",
        "engine": "video_lora",
        "artifacts": [
            {
                "role": "lora",
                "repo_id": "Kijai/WanVideo_comfy",
                "subdir": "lora_turbo",
                "filename": "LoRAs/Wan22-Turbo/Wan22_TI2V_5B_Turbo_lora_rank_64_fp16.safetensors",
                "verify_file": "LoRAs/Wan22-Turbo/Wan22_TI2V_5B_Turbo_lora_rank_64_fp16.safetensors",
            }
        ],
    },
    {
        "id": "lora-wan22-ti2v-5b-fastwan",
        "name": "🚀 LoRA FastWan 6 Pasos para Wan 2.2 TI2V-5B (Hao AI Lab DMD · Fidelidad Foto)",
        "category": "lora",
        "lora_target": "video",
        "lora_group": "speed",
        "compatible_models": ["wan22-ti2v-5b"],
        "incompatible_submodes": ["flf2v"],
        "causal_attention": False,
        "recommended_steps": 6,
        "min_safe_steps": 4,
        "max_steps": 10,
        "recommended_cfg": 1.0,
        "default_lora_scale": 0.75,
        "is_default": False,
        "parent_model": "wan22-ti2v-5b",
        "adapter_name": "wan22_5b_fastwan",
        "badge": "🚀 FastWan DMD · 661 MB",
        "description": (
            "LoRA FastWan DMD oficial extraído de FastVideo (Hao AI Lab, 661 MB) para Wan 2.2 TI2V-5B. "
            "Genera Imagen a Vídeo o Texto a Vídeo en 6 pasos con CFG 1.0 y escala 0.75 manteniendo máxima fidelidad a la imagen inicial."
        ),
        "estimated_gb": 0.66,
        "min_vram_gb": 0.6,
        "recommended_vram_gb": 1.2,
        "l4_seconds_hint": "~1.2 min (6 pasos @ CFG 1.0)",
        "engine": "video_lora",
        "artifacts": [
            {
                "role": "lora",
                "repo_id": "Kijai/WanVideo_comfy",
                "subdir": "lora_fastwan",
                "filename": "FastWan/Wan2_2_5B_FastWanFullAttn_lora_rank_128_bf16.safetensors",
                "verify_file": "FastWan/Wan2_2_5B_FastWanFullAttn_lora_rank_128_bf16.safetensors",
            }
        ],
    },
    {
        "id": "lora-wan21-14b-fusionx-phantom",
        "name": "👻 LoRA FusionX Phantom 6 Pasos para Wan 2.1 VACE 14B (★ Oficial Producción Rápida Frame a Frame)",
        "category": "lora",
        "lora_target": "video",
        "lora_group": "speed",
        "compatible_models": ["wan21-vace-14b-fp8"],
        "incompatible_submodes": [],
        "causal_attention": False,
        "recommended_steps": 6,
        "flf2v_recommended_steps": 6,
        "min_safe_steps": 6,
        "max_steps": 10,
        "recommended_cfg": 1.0,
        "flf2v_recommended_cfg": 1.0,
        "default_lora_scale": 1.0,
        "flf2v_recommended_scale": 0.90,
        "recommended_flow_shift": 2.0,
        "is_default": False,
        "parent_model": "wan21-vace-14b-fp8",
        "adapter_name": "wan21_14b_fusionx_phantom",
        "badge": "★ Oficial HF VACE-14B Fast · 6 Pasos · 317 MB",
        "description": (
            "LoRA bidireccional FusionX Phantom (317 MB) usado en el Space oficial de Hugging Face «Fast 6-step Wan 2.1 VACE 14B» "
            "para Frame a Frame (Inicio → Fin), Imagen a Vídeo y Texto a Vídeo en solo 6 pasos con CFG 1.0 y flow_shift 2.0 (~1.6 min en L4)."
        ),
        "estimated_gb": 0.31,
        "min_vram_gb": 0.4,
        "recommended_vram_gb": 0.9,
        "l4_seconds_hint": "~1.5–2.0 min en 14B (6 pasos @ CFG 1.0)",
        "engine": "video_lora",
        "artifacts": [
            {
                "role": "lora",
                "repo_id": "vrgamedevgirl84/Wan14BT2VFusioniX",
                "subdir": "lora_fusionx_phantom",
                "filename": "FusionX_LoRa/Phantom_Wan_14B_FusionX_LoRA.safetensors",
                "verify_file": "FusionX_LoRa/Phantom_Wan_14B_FusionX_LoRA.safetensors",
            }
        ],
    },
    {
        "id": "lora-wan21-14b-fusionx-i2v",
        "name": "🚀 LoRA FusionX I2V/FLF2V 6–8 Pasos para Wan 2.1 14B (FLF2V 720P / Fun-InP 14B / I2V 14B)",
        "category": "lora",
        "lora_target": "video",
        "lora_group": "speed",
        "compatible_models": [
            "wan21-flf2v-14b-fp8",
            "wan21-fun-inp-14b-fp8",
            "wan21-i2v-14b-480p",
            "wan21-flf2v-14b-720p",
        ],
        "incompatible_submodes": [],
        "causal_attention": False,
        "recommended_steps": 6,
        "flf2v_recommended_steps": 8,
        "min_safe_steps": 6,
        "max_steps": 10,
        "recommended_cfg": 1.0,
        "flf2v_recommended_cfg": 1.0,
        "default_lora_scale": 0.90,
        "flf2v_recommended_scale": 0.85,
        "recommended_flow_shift": 3.0,
        "is_default": False,
        "parent_model": "wan21-flf2v-14b-fp8",
        "adapter_name": "wan21_14b_fusionx_i2v",
        "badge": "🚀 Producción 14B · 6–8 Pasos · 371 MB",
        "description": (
            "Acelerador bidireccional FusionX I2V/FLF2V (371 MB) para los modelos Wan 2.1 14B de 36 canales (FLF2V 720P FP8, Fun-InP 14B FP8 e I2V 14B). "
            "Permite interpolar Frame a Frame en 8 pasos (escala 0.85, CFG 1.0) o animar 1 imagen en 6 pasos manteniendo fluidez cinematográfica."
        ),
        "estimated_gb": 0.37,
        "min_vram_gb": 0.4,
        "recommended_vram_gb": 1.0,
        "l4_seconds_hint": "~1.8–2.3 min en 14B (6–8 pasos)",
        "engine": "video_lora",
        "artifacts": [
            {
                "role": "lora",
                "repo_id": "vrgamedevgirl84/Wan14BT2VFusioniX",
                "subdir": "lora_fusionx_i2v",
                "filename": "FusionX_LoRa/Wan2.1_I2V_14B_FusionX_LoRA.safetensors",
                "verify_file": "FusionX_LoRa/Wan2.1_I2V_14B_FusionX_LoRA.safetensors",
            }
        ],
    },
    {
        "id": "lora-wan21-14b-lightx2v-i2v",
        "name": "⚡ LoRA LightX2V Bidireccional 6 Pasos para Wan 2.1 14B (FLF2V 720P / Fun-InP 14B / I2V 14B)",
        "category": "lora",
        "lora_target": "video",
        "lora_group": "speed",
        "compatible_models": [
            "wan21-flf2v-14b-fp8",
            "wan21-fun-inp-14b-fp8",
            "wan21-i2v-14b-480p",
            "wan21-flf2v-14b-720p",
        ],
        "incompatible_submodes": [],
        "causal_attention": False,
        "recommended_steps": 6,
        "flf2v_recommended_steps": 6,
        "min_safe_steps": 4,
        "max_steps": 8,
        "recommended_cfg": 1.0,
        "flf2v_recommended_cfg": 1.0,
        "default_lora_scale": 0.85,
        "flf2v_recommended_scale": 0.75,
        "recommended_flow_shift": 5.0,
        "is_default": False,
        "parent_model": "wan21-flf2v-14b-fp8",
        "adapter_name": "wan21_14b_lightx2v_i2v",
        "badge": "⚡ LightX2V 14B · 4–6 Pasos · 373 MB",
        "description": (
            "Destilación bidireccional oficial LightX2V (CFG+Step Distill Rank 32, 373 MB) para Wan 2.1 FLF2V 14B, Fun-InP 14B e I2V 14B. "
            "En Frame a Frame ajusta escala 0.75 y 6 pasos @ CFG 1.0 para lograr transiciones rápidas sin rigidez sobre la foto inicial."
        ),
        "estimated_gb": 0.37,
        "min_vram_gb": 0.4,
        "recommended_vram_gb": 1.0,
        "l4_seconds_hint": "~1.6–2.0 min en 14B (6 pasos)",
        "engine": "video_lora",
        "artifacts": [
            {
                "role": "lora",
                "repo_id": "Kijai/WanVideo_comfy",
                "subdir": "lora_lightx2v_i2v_14b",
                "filename": "Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank32_bf16.safetensors",
                "verify_file": "Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank32_bf16.safetensors",
            }
        ],
    },
    {
        "id": "lora-wan21-14b-lightx2v-v2",
        "name": "⚡ LoRA LightX2V v2 Bidireccional 6 Pasos para Wan 2.1 VACE 14B (Frame a Frame / I2V / T2V)",
        "category": "lora",
        "lora_target": "video",
        "lora_group": "speed",
        "compatible_models": ["wan21-vace-14b-fp8"],
        "incompatible_submodes": [],
        "causal_attention": False,
        "recommended_steps": 6,
        "flf2v_recommended_steps": 6,
        "min_safe_steps": 4,
        "max_steps": 8,
        "recommended_cfg": 1.0,
        "flf2v_recommended_cfg": 1.0,
        "default_lora_scale": 0.80,
        "flf2v_recommended_scale": 0.70,
        "recommended_flow_shift": 3.0,
        "is_default": False,
        "parent_model": "wan21-vace-14b-fp8",
        "adapter_name": "wan21_14b_lightx2v_v2",
        "badge": "⚡ LightX2V v2 VACE-14B · 319 MB",
        "description": (
            "Segunda generación LightX2V v2 bidireccional (Rank 32 BF16, 319 MB) compatible con Wan 2.1 VACE 14B Quantized FP8. "
            "Reduce la generación de 28 pasos a solo 6 pasos @ CFG 1.0 (escala 0.70 en Frame a Frame para conservar toda la plasticidad VACE)."
        ),
        "estimated_gb": 0.32,
        "min_vram_gb": 0.4,
        "recommended_vram_gb": 0.9,
        "l4_seconds_hint": "~1.5–1.9 min en 14B (6 pasos)",
        "engine": "video_lora",
        "artifacts": [
            {
                "role": "lora",
                "repo_id": "Kijai/WanVideo_comfy",
                "subdir": "lora_lightx2v_v2_14b",
                "filename": "Lightx2v/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank32_bf16.safetensors",
                "verify_file": "Lightx2v/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank32_bf16.safetensors",
            }
        ],
    },
    {
        "id": "lora-wan21-1.3b-bidirect2",
        "name": "⚡ LoRA Bidireccional v2 (6–8 Pasos) para Modelos 1.3B (VACE / Fun-InP / SkyReels / T2V)",
        "category": "lora",
        "lora_target": "video",
        "lora_group": "speed",
        "compatible_models": [
            "wan21-vace-1.3b",
            "wan21-fun-1.3b-inp",
            "skyreels-v2-i2v-1.3b",
            "wan21-t2v-1.3b",
        ],
        "incompatible_submodes": [],
        "causal_attention": False,
        "recommended_steps": 6,
        "flf2v_recommended_steps": 8,
        "min_safe_steps": 6,
        "max_steps": 10,
        "recommended_cfg": 1.0,
        "flf2v_recommended_cfg": 1.5,
        "default_lora_scale": 0.75,
        "flf2v_recommended_scale": 0.65,
        "is_default": False,
        "parent_model": "wan21-vace-1.3b",
        "adapter_name": "wan21_1_3b_bidirect2",
        "badge": "⚡ Producción 1.3B · ~22 s en L4 · 91 MB",
        "description": (
            "LoRA acelerador ultraligero (91 MB) entrenado con atención BIDIRECCIONAL v2 (no causal) para toda la familia Wan 2.1 1.3B "
            "(VACE 1.3B, Fun-InP 1.3B, SkyReels V2 1.3B y T2V 1.3B). Permite producir clips Frame a Frame en 8 pasos (escala 0.65, CFG 1.5) en ~22–30 s en L4."
        ),
        "estimated_gb": 0.09,
        "min_vram_gb": 0.2,
        "recommended_vram_gb": 0.6,
        "l4_seconds_hint": "~20–30 s (6–8 pasos)",
        "engine": "video_lora",
        "artifacts": [
            {
                "role": "lora",
                "repo_id": "Kijai/WanVideo_comfy",
                "subdir": "lora_1_3b_bidirect2",
                "filename": "Wan21_CausVid_bidirect2_T2V_1_3B_lora_rank32.safetensors",
                "verify_file": "Wan21_CausVid_bidirect2_T2V_1_3B_lora_rank32.safetensors",
            }
        ],
    },
    {
        "id": "lora-wan21-1.3b-rcm-turbo",
        "name": "🚀 LoRA rCM NVLabs (6–8 Pasos) para Modelos 1.3B (VACE / Fun-InP / SkyReels / T2V)",
        "category": "lora",
        "lora_target": "video",
        "lora_group": "speed",
        "compatible_models": [
            "wan21-vace-1.3b",
            "wan21-fun-1.3b-inp",
            "skyreels-v2-i2v-1.3b",
            "wan21-t2v-1.3b",
        ],
        "incompatible_submodes": [],
        "causal_attention": False,
        "recommended_steps": 6,
        "flf2v_recommended_steps": 8,
        "min_safe_steps": 4,
        "max_steps": 10,
        "recommended_cfg": 1.0,
        "flf2v_recommended_cfg": 1.5,
        "default_lora_scale": 0.85,
        "flf2v_recommended_scale": 0.70,
        "is_default": False,
        "parent_model": "wan21-vace-1.3b",
        "adapter_name": "wan21_1_3b_rcm",
        "badge": "🚀 Consistencia rCM 1.3B · 187 MB",
        "description": (
            "Destilación de consistencia bidireccional rCM (NVLabs, 187 MB) compatible con VACE 1.3B, Fun-InP 1.3B, SkyReels V2 1.3B y T2V 1.3B. "
            "Genera Texto/Imagen a Vídeo en 6 pasos (CFG 1.0) y Frame a Frame rápido en 8 pasos (escala 0.70, CFG 1.5) en ~25–35 s en L4."
        ),
        "estimated_gb": 0.19,
        "min_vram_gb": 0.3,
        "recommended_vram_gb": 0.8,
        "l4_seconds_hint": "~25–35 s (6–8 pasos)",
        "engine": "video_lora",
        "artifacts": [
            {
                "role": "lora",
                "repo_id": "Kijai/WanVideo_comfy",
                "subdir": "lora_rcm",
                "filename": "LoRAs/rCM/Wan_2_1_T2V_1_3B_480p_rCM_lora_average_rank_64_bf16.safetensors",
                "verify_file": "LoRAs/rCM/Wan_2_1_T2V_1_3B_480p_rCM_lora_average_rank_64_bf16.safetensors",
            }
        ],
    },
    # ==================== LoRAs DE VÍDEO: COHERENCIA DE PERSONAJES, ENTORNO, CALIDAD Y CÁMARA ====================
    {
        "id": "lora-wan21-14b-unianimate-char",
        "name": "👤🎬 LoRA UniAnimate 14B — Coherencia de Identidad y Anatomía en Movimiento (Wan 2.1 14B)",
        "category": "lora",
        "lora_target": "video",
        "lora_group": "character",
        "stackable_creative": True,
        "compatible_models": [
            "wan21-vace-14b-fp8",
            "wan21-flf2v-14b-fp8",
            "wan21-fun-inp-14b-fp8",
            "wan21-i2v-14b-480p",
            "wan21-flf2v-14b-720p",
        ],
        "default_lora_scale": 0.65,
        "is_default": False,
        "parent_model": "wan21-vace-14b-fp8",
        "adapter_name": "wan21_unianimate_char",
        "badge": "★ Coherencia Personaje 14B · Apilable",
        "description": (
            "Adaptador UniAnimate entrenado para preservar los rasgos faciales, proporciones corporales y ropa del personaje "
            "durante giros, caminatas y gestos complejos en toda la familia Wan 2.1 14B. Apilable junto a FusionX / LightX2V."
        ),
        "estimated_gb": 1.23,
        "min_vram_gb": 0.5,
        "recommended_vram_gb": 1.2,
        "l4_seconds_hint": "Apilable (+0 s extra)",
        "engine": "video_lora",
        "artifacts": [
            {
                "role": "lora",
                "repo_id": "Kijai/WanVideo_comfy",
                "subdir": "lora_unianimate_14b",
                "filename": "UniAnimate-Wan2.1-14B-Lora-12000-fp16.safetensors",
                "verify_file": "UniAnimate-Wan2.1-14B-Lora-12000-fp16.safetensors",
            }
        ],
    },
    {
        "id": "lora-wan21-14b-moviigen-cinema",
        "name": "🏛️🎥 LoRA MoviiGen Cinema 14B — Coherencia Espacial de Entorno y Luz de Cine (Wan 2.1 14B)",
        "category": "lora",
        "lora_target": "video",
        "lora_group": "environment",
        "stackable_creative": True,
        "compatible_models": [
            "wan21-vace-14b-fp8",
            "wan21-flf2v-14b-fp8",
            "wan21-fun-inp-14b-fp8",
            "wan21-i2v-14b-480p",
            "wan21-flf2v-14b-720p",
        ],
        "default_lora_scale": 0.60,
        "is_default": False,
        "parent_model": "wan21-vace-14b-fp8",
        "adapter_name": "wan21_moviigen_cinema",
        "badge": "★ Coherencia Entorno Cine · 317 MB",
        "description": (
            "Entrenado para composición cinematográfica 1080p/720p estable: evita que paredes, mesas o fondos se deformen "
            "cuando la cámara hace paneos o travellings en modelos Wan 2.1 14B. Apilable encima de tu LoRA rápido de 6 pasos."
        ),
        "estimated_gb": 0.32,
        "min_vram_gb": 0.3,
        "recommended_vram_gb": 0.8,
        "l4_seconds_hint": "Apilable (+0 s extra)",
        "engine": "video_lora",
        "artifacts": [
            {
                "role": "lora",
                "repo_id": "Kijai/WanVideo_comfy",
                "subdir": "lora_moviigen_14b",
                "filename": "Wan21_T2V_14B_MoviiGen_lora_rank32_fp16.safetensors",
                "verify_file": "Wan21_T2V_14B_MoviiGen_lora_rank32_fp16.safetensors",
            }
        ],
    },
    {
        "id": "lora-wan21-14b-detailz",
        "name": "💎 LoRA Detailz Wan 14B — Nitidez de Rostros, Ropa y Escenario (Anti-Suavizado)",
        "category": "lora",
        "lora_target": "video",
        "lora_group": "quality",
        "stackable_creative": True,
        "compatible_models": [
            "wan21-vace-14b-fp8",
            "wan21-flf2v-14b-fp8",
            "wan21-fun-inp-14b-fp8",
            "wan21-i2v-14b-480p",
            "wan21-flf2v-14b-720p",
        ],
        "default_lora_scale": 0.55,
        "is_default": False,
        "parent_model": "wan21-vace-14b-fp8",
        "adapter_name": "wan21_detailz",
        "badge": "★ Potenciador Nitidez 14B · 307 MB",
        "description": (
            "Recupera micro-detalle facial, poros, texturas de telas y nitidez del escenario que suelen perderse al usar "
            "destilaciones rápidas de 6 pasos en Wan 2.1 14B."
        ),
        "estimated_gb": 0.31,
        "min_vram_gb": 0.3,
        "recommended_vram_gb": 0.8,
        "l4_seconds_hint": "Apilable (+0 s extra)",
        "engine": "video_lora",
        "artifacts": [
            {
                "role": "lora",
                "repo_id": "Thelocallab/WAN-2.1-loras",
                "subdir": "lora_detailz_wan",
                "filename": "detailz-wan.safetensors",
                "verify_file": "detailz-wan.safetensors",
            }
        ],
    },
    {
        "id": "lora-wan21-14b-snapshot-real",
        "name": "📱 LoRA Realismo Cámara Orgánica 14B (Movimiento y Piel de Vídeo Real · Anti-CGI)",
        "category": "lora",
        "lora_target": "video",
        "lora_group": "quality",
        "stackable_creative": True,
        "compatible_models": [
            "wan21-vace-14b-fp8",
            "wan21-flf2v-14b-fp8",
            "wan21-fun-inp-14b-fp8",
            "wan21-i2v-14b-480p",
            "wan21-flf2v-14b-720p",
        ],
        "default_lora_scale": 0.60,
        "trigger_phrase": "natural handheld camera realism, authentic skin texture",
        "is_default": False,
        "parent_model": "wan21-vace-14b-fp8",
        "adapter_name": "wan21_snapshot_real",
        "badge": "Realismo Documental 14B · 307 MB",
        "description": (
            "Elimina el brillo sintético CGI en vídeos Wan 2.1 14B logrando aspecto de grabación real, tonos de piel naturales "
            "y micro-gestos creíbles para escenas habladas."
        ),
        "estimated_gb": 0.31,
        "min_vram_gb": 0.3,
        "recommended_vram_gb": 0.8,
        "l4_seconds_hint": "Apilable (+0 s extra)",
        "engine": "video_lora",
        "artifacts": [
            {
                "role": "lora",
                "repo_id": "Thelocallab/WAN-2.1-loras",
                "subdir": "lora_wan_snapshot",
                "filename": "WAN2.1_SmartphoneSnapshotPhotoReality_v1_by-AI_Characters.safetensors",
                "verify_file": "WAN2.1_SmartphoneSnapshotPhotoReality_v1_by-AI_Characters.safetensors",
            }
        ],
    },
    {
        "id": "lora-wan21-1.3b-classic-anim",
        "name": "🎨 LoRA Animación Clásica Coherente 1.3B (Estilo Dibujo Clásico para VACE / T2V / SkyReels 1.3B)",
        "category": "lora",
        "lora_target": "video",
        "lora_group": "style",
        "stackable_creative": True,
        "compatible_models": [
            "wan21-vace-1.3b",
            "wan21-fun-1.3b-inp",
            "skyreels-v2-i2v-1.3b",
            "wan21-t2v-1.3b",
        ],
        "default_lora_scale": 0.80,
        "trigger_phrase": "steamboat willie style, golden era animation",
        "is_default": False,
        "parent_model": "wan21-vace-1.3b",
        "adapter_name": "wan21_1_3b_classic_anim",
        "badge": "Estilo Animación 1.3B · 45 MB",
        "description": (
            "LoRA ligero oficial de ejemplo Diffusers (CC0, 45 MB) para estilizar clips de la familia 1.3B con estética "
            "coherente de animación clásica entintada. Apilable con el LoRA Bidireccional 1.3B."
        ),
        "estimated_gb": 0.05,
        "min_vram_gb": 0.1,
        "recommended_vram_gb": 0.3,
        "l4_seconds_hint": "Apilable (+0 s extra)",
        "engine": "video_lora",
        "artifacts": [
            {
                "role": "lora",
                "repo_id": "benjamin-paine/steamboat-willie-1.3b",
                "subdir": "lora_classic_anim",
                "filename": "steamboat-willie.bf16.safetensors",
                "verify_file": "steamboat-willie.bf16.safetensors",
            }
        ],
    },
    {
        "id": "lora-wan22-a14b-svi-continuity",
        "name": "👤🏛️ LoRA Stable-Video-Infinity v2 PRO (Continuidad Larga de Personaje y Escena · Wan 2.2 A14B)",
        "category": "lora",
        "lora_target": "video",
        "lora_group": "character",
        "stackable_creative": True,
        "compatible_models": ["wan22-i2v-a14b"],
        "default_lora_scale": 0.85,
        "is_default": False,
        "parent_model": "wan22-i2v-a14b",
        "adapter_name": "wan22_svi_pro",
        "badge": "★ Continuidad Infinita Wan 2.2 · 1.23 GB",
        "description": (
            "Evita la deriva de color, la mutación de rostros y la degradación del escenario al encadenar múltiples clips "
            "consecutivos (extend video) sobre Wan 2.2 I2V-A14B."
        ),
        "estimated_gb": 1.23,
        "min_vram_gb": 0.6,
        "recommended_vram_gb": 1.5,
        "l4_seconds_hint": "Apilable en A100/H100",
        "engine": "video_lora",
        "artifacts": [
            {
                "role": "lora",
                "repo_id": "Kijai/WanVideo_comfy",
                "subdir": "lora_svi_pro",
                "filename": "LoRAs/Stable-Video-Infinity/v2.0/SVI_v2_PRO_Wan2.2-I2V-A14B_HIGH_lora_rank_128_fp16.safetensors",
                "verify_file": "LoRAs/Stable-Video-Infinity/v2.0/SVI_v2_PRO_Wan2.2-I2V-A14B_HIGH_lora_rank_128_fp16.safetensors",
            }
        ],
    },
    {
        "id": "lora-wan22-a14b-lightx2v-4step",
        "name": "⚡ LoRA LightX2V 4–6 Pasos para Wan 2.2 I2V-A14B (Aceleración 10× en A100/H100)",
        "category": "lora",
        "lora_target": "video",
        "lora_group": "speed",
        "compatible_models": ["wan22-i2v-a14b"],
        "recommended_steps": 6,
        "min_safe_steps": 4,
        "max_steps": 8,
        "recommended_cfg": 1.0,
        "default_lora_scale": 0.90,
        "is_default": False,
        "parent_model": "wan22-i2v-a14b",
        "adapter_name": "wan22_a14b_lightx2v",
        "badge": "⚡ Turbo 4–6p Wan 2.2 A14B · 631 MB",
        "description": (
            "Destilación oficial LightX2V Rank 64 BF16 para acelerar Wan 2.2 I2V-A14B de 40 pasos a solo 4–6 pasos @ CFG 1.0."
        ),
        "estimated_gb": 0.63,
        "min_vram_gb": 0.5,
        "recommended_vram_gb": 1.2,
        "l4_seconds_hint": "~55 s en A100 (6 pasos)",
        "engine": "video_lora",
        "artifacts": [
            {
                "role": "lora",
                "repo_id": "Kijai/WanVideo_comfy",
                "subdir": "lora_wan22_a14b_lx2v",
                "filename": "LoRAs/Wan22_Lightx2v/Wan_2_2_I2V_A14B_HIGH_lightx2v_4step_lora_v1030_rank_64_bf16.safetensors",
                "verify_file": "LoRAs/Wan22_Lightx2v/Wan_2_2_I2V_A14B_HIGH_lightx2v_4step_lora_v1030_rank_64_bf16.safetensors",
            }
        ],
    },
    {
        "id": "lora-ltx13b-detailer-ic",
        "name": "💎 LoRA Oficial Lightricks IC-Detailer para LTX-Video 13B (Nitidez Facial y Entorno)",
        "category": "lora",
        "lora_target": "video",
        "lora_group": "quality",
        "stackable_creative": True,
        "compatible_models": ["ltx-video-13b-distilled"],
        "default_lora_scale": 0.75,
        "is_default": False,
        "parent_model": "ltx-video-13b-distilled",
        "adapter_name": "ltx13b_ic_detailer",
        "badge": "★ Oficial Lightricks 13B · Nitidez HD",
        "description": (
            "Adaptador oficial en formato Diffusers de Lightricks para LTX-Video 0.9.8 13B Distilled: refuerza la definición "
            "de rostros, manos y detalles arquitectónicos manteniendo sus 8 pasos ultrarrápidos (~45 s en L4)."
        ),
        "estimated_gb": 1.36,
        "min_vram_gb": 0.6,
        "recommended_vram_gb": 1.4,
        "l4_seconds_hint": "Mantiene 8 pasos (~48 s)",
        "engine": "video_lora",
        "artifacts": [
            {
                "role": "lora",
                "repo_id": "Lightricks/LTX-Video-ICLoRA-detailer-13b-0.9.8",
                "subdir": "lora_ltx13b_detailer",
                "filename": "ltxv-098-ic-lora-detailer-diffusers.safetensors",
                "verify_file": "ltxv-098-ic-lora-detailer-diffusers.safetensors",
            }
        ],
    },
    {
        "id": "lora-ltx2-detailer-ic",
        "name": "💎 LoRA Oficial Lightricks IC-Detailer para LTX-2 19B Audio+Vídeo",
        "category": "lora",
        "lora_target": "video",
        "lora_group": "quality",
        "stackable_creative": True,
        "compatible_models": ["ltx-2-19b-av"],
        "default_lora_scale": 0.80,
        "is_default": False,
        "parent_model": "ltx-2-19b-av",
        "adapter_name": "ltx2_ic_detailer",
        "badge": "Oficial Lightricks LTX-2 · Detalle Cine",
        "description": (
            "Refinador oficial de alta definición para LTX-2 19B que estabiliza texturas finas de piel, cabello y entorno."
        ),
        "estimated_gb": 2.62,
        "min_vram_gb": 0.8,
        "recommended_vram_gb": 1.8,
        "l4_seconds_hint": "Apilable en LTX-2",
        "engine": "video_lora",
        "artifacts": [
            {
                "role": "lora",
                "repo_id": "Lightricks/LTX-2-19b-IC-LoRA-Detailer",
                "subdir": "lora_ltx2_detailer",
                "filename": "ltx-2-19b-ic-lora-detailer.safetensors",
                "verify_file": "ltx-2-19b-ic-lora-detailer.safetensors",
            }
        ],
    },
    {
        "id": "lora-ltx2-cam-static",
        "name": "🏛️🎥 LoRA Oficial Cámara Estática en Trípode para LTX-2 19B (Entorno Fijo para Diálogos)",
        "category": "lora",
        "lora_target": "video",
        "lora_group": "camera",
        "stackable_creative": True,
        "compatible_models": ["ltx-2-19b-av"],
        "default_lora_scale": 0.90,
        "is_default": False,
        "parent_model": "ltx-2-19b-av",
        "adapter_name": "ltx2_cam_static",
        "badge": "Oficial LTX-2 · Trípode Fijo · 327 MB",
        "description": (
            "Bloquea la cámara en trípode fijo eliminando derivas del fondo mientras los personajes conversan y gesticulan "
            "(ideal para escenas de diálogo y LipSync HD)."
        ),
        "estimated_gb": 0.33,
        "min_vram_gb": 0.3,
        "recommended_vram_gb": 0.8,
        "l4_seconds_hint": "Apilable (+0 s extra)",
        "engine": "video_lora",
        "artifacts": [
            {
                "role": "lora",
                "repo_id": "Lightricks/LTX-2-19b-LoRA-Camera-Control-Static",
                "subdir": "lora_ltx2_static",
                "filename": "ltx-2-19b-lora-camera-control-static.safetensors",
                "verify_file": "ltx-2-19b-lora-camera-control-static.safetensors",
            }
        ],
    },
    {
        "id": "lora-ltx2-cam-dolly-in",
        "name": "🎯 LoRA Oficial Cámara Dolly-In Cinematográfico para LTX-2 19B",
        "category": "lora",
        "lora_target": "video",
        "lora_group": "camera",
        "stackable_creative": True,
        "compatible_models": ["ltx-2-19b-av"],
        "default_lora_scale": 0.90,
        "is_default": False,
        "parent_model": "ltx-2-19b-av",
        "adapter_name": "ltx2_cam_dolly_in",
        "badge": "Oficial LTX-2 · Dolly-In · 327 MB",
        "description": (
            "Control físico de cámara Dolly-In entrenado por Lightricks para acercamiento rectilíneo con paralaje 3D coherente."
        ),
        "estimated_gb": 0.33,
        "min_vram_gb": 0.3,
        "recommended_vram_gb": 0.8,
        "l4_seconds_hint": "Apilable (+0 s extra)",
        "engine": "video_lora",
        "artifacts": [
            {
                "role": "lora",
                "repo_id": "Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In",
                "subdir": "lora_ltx2_dolly_in",
                "filename": "ltx-2-19b-lora-camera-control-dolly-in.safetensors",
                "verify_file": "ltx-2-19b-lora-camera-control-dolly-in.safetensors",
            }
        ],
    },
    # ==================== VÍDEO ====================
    {
        "id": "wan22-ti2v-5b",
        "name": "Wan 2.2 TI2V-5B HD 720P (Imagen a Vídeo y Texto a Vídeo · 24 fps)",
        "category": "video",
        "is_default": True,
        "badge": "HD 720P ★ · Imagen/Texto + LoRA Turbo",
        "video_modes": ["i2v", "t2v"],
        "default_fps": 24,
        "default_steps": 45,
        "default_cfg": 5.0,
        "default_duration_sec": 5.0,
        "official_spec": {
            "landscape": [1280, 704],
            "portrait": [704, 1280],
            "square": [960, 960],
            "max_area": 704 * 1280,
            "min_area": 640 * 960,
            "mod_value": 32,
            "frame_rule": "4k+1",
            "default_frames": 121,
            "default_duration_sec": 5.0,
            "default_fps": 24,
            "default_steps": 45,
            "fast_safe_steps": 28,
            "min_safe_steps": 22,
            "default_cfg": 5.0,
            "flow_shift": 5.0,
            "prompt_lang": "es_en",
            "official_note": "720P nativo (1280×704 / 704×1280) · Especialista 1 Foto→Vídeo y Texto→Vídeo · 6 pasos recomendados con LoRA Turbo; rendimiento real sin validar",
        },
        "description": (
            "Modelo HD 720P@24fps de Alibaba Wan 2.2 entrenado para animar 1 Imagen Inicial o generar desde Texto (no entrenado para 2 poses Frame a Frame). "
            "Con LoRA Turbo o FastWan se recomiendan 6 pasos; tiempo y calidad deben medirse. No admite Frame a Frame en esta receta."
        ),
        "estimated_gb": 31.9,
        "min_vram_gb": 13.5,
        "recommended_vram_gb": 19.5,
        "l4_seconds_hint": "Sin benchmark real; medir tiempo con LoRA y sin LoRA",
        "engine": "wan_ti2v",
        "artifacts": [
            {
                "role": "base",
                "repo_id": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
                "subdir": "base",
                "filename": None,
                "verify_file": "model_index.json",
                "allow_patterns": _DIFFUSERS_VIDEO_ALLOW,
            }
        ],
    },
    {
        "id": "wan22-ti2v-5b-turbo",
        "name": "Wan 2.2 TI2V-5B TURBO 4 Pasos (Imagen a Vídeo y Texto a Vídeo · 22.8 GB)",
        "category": "video",
        "is_default": False,
        "badge": "⚡ 4 Pasos de Fábrica · Solo I2V / T2V",
        "video_modes": ["i2v", "t2v"],
        "default_fps": 24,
        "default_steps": 4,
        "default_cfg": 1.0,
        "default_duration_sec": 5.0,
        "official_spec": {
            "landscape": [1280, 704],
            "portrait": [704, 1280],
            "square": [960, 960],
            "max_area": 704 * 1280,
            "min_area": 640 * 960,
            "mod_value": 32,
            "frame_rule": "4k+1",
            "default_frames": 121,
            "default_duration_sec": 5.0,
            "default_fps": 24,
            "default_steps": 4,
            "fast_safe_steps": 4,
            "min_safe_steps": 4,
            "default_cfg": 1.0,
            "flow_shift": 5.0,
            "prompt_lang": "es_en",
            "official_note": "Versión Self-Forcing 4 pasos @ CFG 1.0 · Solo 1 Foto→Vídeo o Texto→Vídeo · ~1 min en L4 (no apto para Frame a Frame por atención causal)",
        },
        "description": (
            "Edición destilada Self-Forcing a 4 pasos (bfloat16, 22.8 GB) de Wan 2.2 TI2V-5B para animar 1 imagen o texto en ~1 min en L4. "
            "Al usar atención causal hacia delante, no es compatible con Frame a Frame (para Inicio → Fin usa Wan 2.1 VACE 1.3B)."
        ),
        "estimated_gb": 22.8,
        "min_vram_gb": 12.5,
        "recommended_vram_gb": 18.0,
        "l4_seconds_hint": "~1 min / clip 720p (4 pasos)",
        "engine": "wan_ti2v",
        "artifacts": [
            {
                "role": "base",
                "repo_id": "yetter-ai/Wan2.2-TI2V-5B-Turbo-Diffusers",
                "subdir": "base",
                "filename": None,
                "verify_file": "model_index.json",
                "allow_patterns": _DIFFUSERS_VIDEO_ALLOW,
            }
        ],
    },
    {
        "id": "wan21-vace-14b-fp8",
        "name": "Wan 2.1 VACE 14B Quantized FP8 (★ 14B Pro Frame a Frame + Imagen + Texto · Cabe en L4)",
        "category": "video",
        "is_default": False,
        "badge": "★ #1 14B Cuantizado FP8 · Frame a Frame + LoRA 6p",
        "video_modes": ["flf2v", "i2v", "t2v"],
        "default_fps": 16,
        "default_steps": 28,
        "default_cfg": 5.0,
        "default_duration_sec": 5.0,
        "official_spec": {
            "landscape": [832, 480],
            "portrait": [480, 832],
            "square": [640, 640],
            "max_area": 480 * 832,
            "mod_value": 16,
            "frame_rule": "4k+1",
            "default_frames": 81,
            "default_duration_sec": 5.0,
            "default_fps": 16,
            "default_steps": 28,
            "fast_safe_steps": 22,
            "min_safe_steps": 18,
            "default_cfg": 5.0,
            "flow_shift": 3.0,
            "prompt_lang": "es_en",
            "official_note": "★ 14B VACE Cuantizado FP8 (29.1 GB vs 75 GB) · Con «👻 LoRA FusionX Phantom» genera en 6 pasos (~1.6 min en L4) · Admite 480P y 720P",
        },
        "description": (
            "El gigante oficial de 14.000 millones de parámetros (10× mayor que 1.3B) de Alibaba Wan-AI VACE cuantizado en FP8 (e4m3fn) con Layerwise Casting "
            "para caber cómodamente en GPU L4 (22.5 GB VRAM) ocupando solo 29.1 GB en disco. Combínalo con «👻 LoRA FusionX Phantom 6 Pasos» para producción ultrarrápida Frame a Frame."
        ),
        "estimated_gb": 29.1,
        "min_vram_gb": 15.8,
        "recommended_vram_gb": 20.5,
        "l4_seconds_hint": "~1.6 min con LoRA 6p · ~6.5 min base (28p)",
        "engine": "wan_vace_fp8",
        "artifacts": [
            {
                "role": "base",
                "repo_id": "Wan-AI/Wan2.1-VACE-1.3B-diffusers",
                "subdir": "base",
                "filename": None,
                "verify_file": "model_index.json",
                "allow_patterns": _DIFFUSERS_VIDEO_ALLOW,
                "ignore_patterns": ["transformer/*"],
            },
            {
                "role": "transformer_cfg",
                "repo_id": "Wan-AI/Wan2.1-VACE-14B-diffusers",
                "subdir": "base",
                "filename": "transformer/config.json",
                "verify_file": "transformer/config.json",
            },
            {
                "role": "transformer_fp8",
                "repo_id": "Kamikaze-88/Wan2.1-VACE-14B-fp8",
                "subdir": "fp8",
                "filename": "wan2.1_vace_14B_fp8_e4m3fn.safetensors",
                "verify_file": "wan2.1_vace_14B_fp8_e4m3fn.safetensors",
            },
        ],
    },
    {
        "id": "wan21-flf2v-14b-fp8",
        "name": "Wan 2.1 FLF2V 14B 720P Quantized FP8 (★ 14B Dedicado Frame a Frame HD · Cabe en L4)",
        "category": "video",
        "is_default": False,
        "badge": "★ 14B FLF2V Nativo FP8 · Doble CLIP 514 + LoRA 6p",
        "video_modes": ["flf2v", "i2v"],
        "default_fps": 16,
        "default_steps": 30,
        "default_cfg": 5.0,
        "default_duration_sec": 5.0,
        "official_spec": {
            "landscape": [960, 544],
            "portrait": [544, 960],
            "square": [720, 720],
            "max_area": 544 * 960,
            "mod_value": 16,
            "frame_rule": "4k+1",
            "default_frames": 81,
            "default_duration_sec": 5.0,
            "default_fps": 16,
            "default_steps": 30,
            "fast_safe_steps": 24,
            "min_safe_steps": 20,
            "default_cfg": 5.0,
            "flow_shift": 5.0,
            "prompt_lang": "es_en",
            "official_note": "★ 14B FLF2V Puro Cuantizado Comfy-Org FP8 (30.7 GB vs 88 GB) · Doble CLIP 514 tokens · Con LoRA FusionX/LightX2V 14B tarda ~1.8 min en L4",
        },
        "description": (
            "Versión cuantizada FP8 oficial de Comfy-Org del especialista puro Wan 2.1 FLF2V 14B 720P (36 canales + embedding posicional dual de 514 tokens "
            "para Fotograma Inicial y Final). Baja de 88 GB a 30.7 GB en disco, entra en L4 (22.5 GB VRAM) y admite LoRAs rápidos FusionX y LightX2V de 6–8 pasos."
        ),
        "estimated_gb": 30.7,
        "min_vram_gb": 16.2,
        "recommended_vram_gb": 21.0,
        "l4_seconds_hint": "~1.8 min con LoRA 6–8p · ~7.5 min base",
        "engine": "wan_i2v_14b_fp8",
        "artifacts": [
            {
                "role": "base",
                "repo_id": "Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers",
                "subdir": "base",
                "filename": None,
                "verify_file": "model_index.json",
                "allow_patterns": _DIFFUSERS_VIDEO_ALLOW,
                "ignore_patterns": ["transformer/*.safetensors", "transformer/*.index.json", "text_encoder/*"],
            },
            {
                "role": "text_encoder_bf16",
                "repo_id": "Wan-AI/Wan2.1-VACE-1.3B-diffusers",
                "subdir": "base",
                "filename": None,
                "verify_file": "text_encoder/config.json",
                "allow_patterns": ["text_encoder/*"],
            },
            {
                "role": "transformer_fp8",
                "repo_id": "Comfy-Org/Wan-2.1",
                "subdir": "fp8",
                "filename": "split_files/diffusion_models/wan2.1_flf2v_720p_14B_fp8_e4m3fn.safetensors",
                "verify_file": "split_files/diffusion_models/wan2.1_flf2v_720p_14B_fp8_e4m3fn.safetensors",
            },
        ],
    },
    {
        "id": "wan21-fun-inp-14b-fp8",
        "name": "Wan 2.1 Fun-InP 14B Quantized FP8 (Alibaba PAI 14B Interpolación Inicio → Fin · Cabe en L4)",
        "category": "video",
        "is_default": False,
        "badge": "14B Fun-InP FP8 · 36 Canales + LoRA 6p",
        "video_modes": ["flf2v", "i2v"],
        "default_fps": 16,
        "default_steps": 30,
        "default_cfg": 5.0,
        "default_duration_sec": 5.0,
        "official_spec": {
            "landscape": [832, 480],
            "portrait": [480, 832],
            "square": [640, 640],
            "max_area": 480 * 832,
            "mod_value": 16,
            "frame_rule": "4k+1",
            "default_frames": 81,
            "default_duration_sec": 5.0,
            "default_fps": 16,
            "default_steps": 30,
            "fast_safe_steps": 24,
            "min_safe_steps": 20,
            "default_cfg": 5.0,
            "flow_shift": 3.0,
            "prompt_lang": "es_en",
            "official_note": "Alibaba PAI Fun-InP 14B en FP8 (30.7 GB) · Máscara explícita Inicio/Fin 36 canales · Con LoRA FusionX/LightX2V 14B genera en 6–8 pasos en L4",
        },
        "description": (
            "Hermano mayor de 14B parámetros de Alibaba PAI Fun-InP cuantizado en FP8 (30.7 GB en disco) con transformador de 36 canales y codificador CLIP dual. "
            "Compatible con GPU L4 (22.5 GB VRAM) y con los aceleradores bidireccionales FusionX I2V 14B y LightX2V I2V 14B (6–8 pasos)."
        ),
        "estimated_gb": 30.7,
        "min_vram_gb": 16.0,
        "recommended_vram_gb": 20.8,
        "l4_seconds_hint": "~1.6 min con LoRA 6p · ~6.8 min base",
        "engine": "wan_i2v_14b_fp8",
        "artifacts": [
            {
                "role": "base",
                "repo_id": "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
                "subdir": "base",
                "filename": None,
                "verify_file": "model_index.json",
                "allow_patterns": _DIFFUSERS_VIDEO_ALLOW,
                "ignore_patterns": ["transformer/*.safetensors", "transformer/*.index.json", "text_encoder/*"],
            },
            {
                "role": "text_encoder_bf16",
                "repo_id": "Wan-AI/Wan2.1-VACE-1.3B-diffusers",
                "subdir": "base",
                "filename": None,
                "verify_file": "text_encoder/config.json",
                "allow_patterns": ["text_encoder/*"],
            },
            {
                "role": "transformer_fp8",
                "repo_id": "Kijai/WanVideo_comfy",
                "subdir": "fp8",
                "filename": "Fun/Wan2.1-Fun-InP-14B_fp8_e4m3fn.safetensors",
                "verify_file": "Fun/Wan2.1-Fun-InP-14B_fp8_e4m3fn.safetensors",
            },
        ],
    },
    {
        "id": "ltx-video-13b-distilled",
        "name": "LTX-Video 0.9.8 13B Distilled FP8-Cast (⚡ 13B Destilado a 8 Pasos · Frame a Frame + I2V + T2V)",
        "category": "video",
        "is_default": False,
        "badge": "⚡ 13B 8 Pasos de Fábrica · ~45s en L4 (3 en 1)",
        "video_modes": ["flf2v", "i2v", "t2v"],
        "default_fps": 24,
        "default_steps": 8,
        "default_cfg": 1.0,
        "default_duration_sec": 4.0,
        "official_spec": {
            "landscape": [896, 512],
            "portrait": [512, 896],
            "square": [704, 704],
            "max_area": 512 * 896,
            "mod_value": 32,
            "frame_rule": "8k+1",
            "default_frames": 97,
            "default_duration_sec": 4.0,
            "default_fps": 24,
            "default_steps": 8,
            "fast_safe_steps": 8,
            "min_safe_steps": 6,
            "default_cfg": 1.0,
            "distilled_timesteps": [1000, 993, 987, 981, 975, 909, 725, 0.03],
            "decode_timestep": 0.05,
            "decode_noise_scale": 0.025,
            "image_cond_noise_scale": 0.0,
            "prompt_lang": "en",
            "official_note": "⚡ 13B Destilado Oficial Lightricks (8 pasos exactos @ CFG 1.0 sin necesitar LoRA) · FP8 Layerwise Casting (~14.5 GB VRAM) · Mejor prompt en Inglés",
        },
        "description": (
            "Modelo grande de 13.000 millones de parámetros de Lightricks ya destilado de fábrica en pasos y guía (8 pasos exactos @ CFG 1.0 sin necesitar LoRA externo). "
            "Se ejecuta con FP8 Layerwise Casting en L4 (~40–55 s por clip de 97 fotogramas @ 24 fps) y soporta Frame a Frame, Imagen a Vídeo y Texto a Vídeo."
        ),
        "estimated_gb": 45.8,
        "min_vram_gb": 14.5,
        "recommended_vram_gb": 19.5,
        "l4_seconds_hint": "~35–55 s / clip (8 pasos nativos)",
        "engine": "ltx_video_distilled",
        "artifacts": [
            {
                "role": "base",
                "repo_id": "Lightricks/LTX-Video-0.9.8-13B-distilled",
                "subdir": "base",
                "filename": None,
                "verify_file": "model_index.json",
                "allow_patterns": _DIFFUSERS_VIDEO_ALLOW,
                "ignore_patterns": [
                    "vae/transformer/*",
                    "vae/text_encoder/*",
                    "vae/tokenizer/*",
                    "vae/scheduler/*",
                    "vae/model_index.json",
                    "media/*",
                ],
            }
        ],
    },
    {
        "id": "wan21-vace-1.3b",
        "name": "Wan 2.1 VACE 1.3B Oficial (Ligero T4/L4 · Frame a Frame + LoRA opcional)",
        "category": "video",
        "is_default": False,
        "badge": "Ligero 1.3B · Ideal T4/L4 + LoRA Bidireccional",
        "video_modes": ["flf2v", "i2v", "t2v"],
        "default_fps": 16,
        "default_steps": 30,
        "default_cfg": 5.0,
        "default_duration_sec": 5.0,
        "official_spec": {
            "landscape": [832, 480],
            "portrait": [480, 832],
            "square": [640, 640],
            "max_area": 480 * 832,
            "mod_value": 16,
            "frame_rule": "4k+1",
            "default_frames": 81,
            "default_duration_sec": 5.0,
            "default_fps": 16,
            "default_steps": 30,
            "fast_safe_steps": 24,
            "min_safe_steps": 20,
            "default_cfg": 5.0,
            "flow_shift": 3.0,
            "prompt_lang": "es_en",
            "official_note": "Especialista VACE para Frame a Frame · 30 pasos @ CFG 5.0 sin LoRA · 81 frames @ 16 fps; tiempo real no validado",
        },
        "description": (
            "Modelo oficial de Alibaba Wan-AI con 15 capas bidireccionales VACE entrenadas específicamente para interpolar poses complejas de Inicio → Fin "
            "(ej. guardia → patada voladora); no garantiza calidad o continuidad. Pesa 17.7 GB; tiempo real no validado (30 pasos, CFG 5.0, sin LoRA)."
        ),
        "estimated_gb": 17.7,
        "min_vram_gb": 8.0,
        "recommended_vram_gb": 13.5,
        "l4_seconds_hint": "No validado en L4; medir cada fase (30 pasos sin LoRA)",
        "engine": "wan_vace",
        "artifacts": [
            {
                "role": "base",
                "repo_id": "Wan-AI/Wan2.1-VACE-1.3B-diffusers",
                "subdir": "base",
                "filename": None,
                "verify_file": "model_index.json",
                "allow_patterns": _DIFFUSERS_VIDEO_ALLOW,
            }
        ],
    },
    {
        "id": "wan21-fun-1.3b-inp",
        "name": "Wan 2.1 Fun 1.3B InP (Alibaba PAI · Interpolación Dedicada Inicio → Fin)",
        "category": "video",
        "is_default": False,
        "badge": "Especialista InP 36 Canales · Doble CLIP",
        "video_modes": ["flf2v", "i2v"],
        "default_fps": 16,
        "default_steps": 35,
        "default_cfg": 5.0,
        "default_duration_sec": 5.0,
        "official_spec": {
            "landscape": [832, 480],
            "portrait": [480, 832],
            "square": [640, 640],
            "max_area": 480 * 832,
            "mod_value": 16,
            "frame_rule": "4k+1",
            "default_frames": 81,
            "default_duration_sec": 5.0,
            "default_fps": 16,
            "default_steps": 35,
            "fast_safe_steps": 26,
            "min_safe_steps": 22,
            "default_cfg": 5.0,
            "flow_shift": 3.0,
            "prompt_lang": "es_en",
            "official_note": "Alibaba PAI Fun-InP (36 canales + doble CLIP Inicio/Fin) · 35 pasos @ CFG 5.0 (~1.4 min en L4) · Transiciones continuas pose a pose",
        },
        "description": (
            "Modelo dedicado de interpolación (InP) de Alibaba PAI con transformador de 36 canales (máscara inicio/fin explícita) y codificador CLIP dual "
            "para ambos fotogramas. Ideal como segunda opción especializada en transiciones Frame a Frame fluidas en T4 y L4."
        ),
        "estimated_gb": 29.5,
        "min_vram_gb": 8.5,
        "recommended_vram_gb": 14.0,
        "l4_seconds_hint": "~1.2–1.6 min / clip (35 pasos)",
        "engine": "wan_i2v_14b",
        "artifacts": [
            {
                "role": "base",
                "repo_id": "weizhou03/Wan2.1-Fun-1.3B-InP-Diffusers",
                "subdir": "base",
                "filename": None,
                "verify_file": "model_index.json",
                "allow_patterns": _DIFFUSERS_VIDEO_ALLOW,
            }
        ],
    },
    {
        "id": "skyreels-v2-i2v-1.3b",
        "name": "SkyReels V2 1.3B 540P (Frame a Frame + Imagen a Vídeo · 24 fps)",
        "category": "video",
        "is_default": False,
        "badge": "Cine 540p · Frame a Frame + I2V",
        "video_modes": ["flf2v", "i2v"],
        "default_fps": 24,
        "default_steps": 45,
        "default_cfg": 5.0,
        "default_duration_sec": 4.0,
        "official_spec": {
            "landscape": [960, 544],
            "portrait": [544, 960],
            "square": [720, 720],
            "max_area": 544 * 960,
            "mod_value": 16,
            "frame_rule": "4k+1",
            "default_frames": 97,
            "default_duration_sec": 4.0,
            "default_fps": 24,
            "default_steps": 45,
            "default_cfg": 5.0,
            "flow_shift": 5.0,
            "prompt_lang": "es_en",
            "official_note": "540P nativo (960×544 / 544×960) · 4 s (97 frames @ 24 fps) · 45–50 pasos · CFG 5.0 · Entiende Español e Inglés",
        },
        "description": (
            "Especialista cinematográfico 540p@24fps con condicionamiento nativo de primer y último fotograma (Frame a Frame) "
            "e Imagen a Vídeo. Resolución oficial 960×544 / 544×960, 97 fotogramas (4 s) y 45–50 pasos."
        ),
        "estimated_gb": 29.8,
        "min_vram_gb": 9.5,
        "recommended_vram_gb": 15.0,
        "l4_seconds_hint": "60–110 s / clip",
        "engine": "skyreels_v2",
        "artifacts": [
            {
                "role": "base",
                "repo_id": "Skywork/SkyReels-V2-I2V-1.3B-540P-Diffusers",
                "subdir": "base",
                "filename": None,
                "verify_file": "model_index.json",
                "allow_patterns": _DIFFUSERS_VIDEO_ALLOW,
            }
        ],
    },
    {
        "id": "ltx-video-2b",
        "name": "LTX-Video 0.9.5 2B (Texto, Imagen y Frame a Frame · Ultrarrápido)",
        "category": "video",
        "is_default": False,
        "badge": "Ultrarrápido · 3 en 1 (Prompt Inglés)",
        "video_modes": ["t2v", "i2v", "flf2v"],
        "default_fps": 24,
        "default_steps": 40,
        "default_cfg": 3.0,
        "default_duration_sec": 4.0,
        "official_spec": {
            "landscape": [768, 512],
            "portrait": [512, 768],
            "square": [640, 640],
            "max_area": 512 * 768,
            "mod_value": 32,
            "frame_rule": "8k+1",
            "default_frames": 97,
            "default_duration_sec": 4.0,
            "default_fps": 24,
            "default_steps": 40,
            "default_cfg": 3.0,
            "prompt_lang": "en",
            "official_note": "Resolución óptima 768×512 / 512×768 · 4–6 s (97–161 frames @ 24 fps) · 40 pasos · CFG 3.0 · Mejor prompt en Inglés",
        },
        "description": (
            "Versión 0.9.5 de Lightricks con soporte nativo de Texto a Vídeo, Imagen a Vídeo y Frame a Frame. "
            "Usa guía baja oficial (CFG 3.0), 40 pasos y resolución 768×512 / 512×768; rinde mejor con prompts detallados en inglés."
        ),
        "estimated_gb": 23.6,
        "min_vram_gb": 9.5,
        "recommended_vram_gb": 15.0,
        "l4_seconds_hint": "22–45 s / clip",
        "engine": "ltx_video",
        "artifacts": [
            {
                "role": "base",
                "repo_id": "Lightricks/LTX-Video-0.9.5",
                "subdir": "base",
                "filename": None,
                "verify_file": "model_index.json",
                "allow_patterns": _DIFFUSERS_VIDEO_ALLOW,
            }
        ],
    },
    {
        "id": "sana-video-2b-480p",
        "name": "SANA-Video 2B 480p (NVIDIA Lab · Muy Ligero I2V + T2V)",
        "category": "video",
        "is_default": False,
        "badge": "13 GB Disco · Ideal T4 y L4",
        "video_modes": ["i2v", "t2v"],
        "default_fps": 16,
        "default_steps": 50,
        "default_cfg": 6.0,
        "default_duration_sec": 5.0,
        "official_spec": {
            "landscape": [832, 480],
            "portrait": [480, 832],
            "square": [640, 640],
            "max_area": 480 * 832,
            "mod_value": 32,
            "frame_rule": "4k+1",
            "default_frames": 81,
            "default_duration_sec": 5.0,
            "default_fps": 16,
            "default_steps": 50,
            "default_cfg": 6.0,
            "motion_score": 30,
            "prompt_lang": "en",
            "official_note": "480P nativo (832×480 / 480×832) · 5 s (81 frames @ 16 fps) · 50 pasos · CFG 6.0 · Añade motion score automático",
        },
        "description": (
            "Arquitectura lineal ultracompacta de NVIDIA Lab (13 GB en disco). "
            "Ajustado con sus parámetros oficiales: 832×480 / 480×832, 81 fotogramas @ 16 fps, 50 pasos, CFG 6.0 y motion score 30."
        ),
        "estimated_gb": 13.0,
        "min_vram_gb": 7.0,
        "recommended_vram_gb": 12.0,
        "l4_seconds_hint": "30–50 s / clip",
        "engine": "sana_video",
        "artifacts": [
            {
                "role": "base",
                "repo_id": "Efficient-Large-Model/SANA-Video_2B_480p_diffusers",
                "subdir": "base",
                "filename": None,
                "verify_file": "model_index.json",
                "allow_patterns": _DIFFUSERS_VIDEO_ALLOW,
            }
        ],
    },
    {
        "id": "wan21-t2v-1.3b",
        "name": "Wan 2.1 T2V 1.3B (Texto a Vídeo Puro · Ligero Español/Inglés)",
        "category": "video",
        "is_default": False,
        "badge": "Texto→Vídeo · T4 y L4",
        "video_modes": ["t2v"],
        "default_fps": 16,
        "default_steps": 40,
        "default_cfg": 5.0,
        "default_duration_sec": 5.0,
        "official_spec": {
            "landscape": [832, 480],
            "portrait": [480, 832],
            "square": [640, 640],
            "max_area": 480 * 832,
            "mod_value": 16,
            "frame_rule": "4k+1",
            "default_frames": 81,
            "default_duration_sec": 5.0,
            "default_fps": 16,
            "default_steps": 40,
            "default_cfg": 5.0,
            "flow_shift": 3.0,
            "prompt_lang": "es_en",
            "official_note": "480P nativo (832×480 / 480×832) · 5 s (81 frames @ 16 fps) · 40–50 pasos · CFG 5.0 · Entiende Español e Inglés",
        },
        "description": (
            "Generador clásico y muy estable de escenas de vídeo desde texto (832×480 / 480×832 @ 16 fps, 81 fotogramas). "
            "Funciona de forma excelente tanto en T4 como en L4."
        ),
        "estimated_gb": 27.0,
        "min_vram_gb": 8.0,
        "recommended_vram_gb": 13.0,
        "l4_seconds_hint": "40–65 s / clip",
        "engine": "wan_t2v",
        "artifacts": [
            {
                "role": "base",
                "repo_id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                "subdir": "base",
                "filename": None,
                "verify_file": "model_index.json",
                "allow_patterns": _DIFFUSERS_VIDEO_ALLOW,
            }
        ],
    },
    {
        "id": "cogvideox-5b-i2v",
        "name": "CogVideoX 5B I2V (Imagen a Vídeo Expresivo · 720×480 Fijo)",
        "category": "video",
        "is_default": False,
        "badge": "5B BF16 · 720×480 @ 8 fps",
        "video_modes": ["i2v"],
        "default_fps": 8,
        "default_steps": 50,
        "default_cfg": 6.0,
        "default_duration_sec": 6.0,
        "official_spec": {
            "landscape": [720, 480],
            "portrait": [720, 480],
            "square": [720, 480],
            "fixed_resolution": [720, 480],
            "max_area": 720 * 480,
            "mod_value": 16,
            "frame_rule": "cogvideo49",
            "default_frames": 49,
            "default_duration_sec": 6.0,
            "default_fps": 8,
            "default_steps": 50,
            "default_cfg": 6.0,
            "use_dynamic_cfg": True,
            "prompt_lang": "en",
            "official_note": "Resolución fija obligatoria 720×480 · 6 s (49 frames @ 8 fps) · 50 pasos · CFG 6.0 dinámico · Prompt en Inglés",
        },
        "description": (
            "Modelo de 5B parámetros en BF16. Su arquitectura exige estrictamente 720×480 píxeles, 49 fotogramas (6 s a 8 fps) "
            "y 50 pasos con dynamic CFG (Studio IA los fija automáticamente)."
        ),
        "estimated_gb": 20.6,
        "min_vram_gb": 13.0,
        "recommended_vram_gb": 24.0,
        "l4_seconds_hint": "2–3.5 min / clip",
        "engine": "cogvideox_i2v",
        "artifacts": [
            {
                "role": "base",
                "repo_id": "zai-org/CogVideoX-5b-I2V",
                "subdir": "base",
                "filename": None,
                "verify_file": "model_index.json",
                "allow_patterns": _DIFFUSERS_VIDEO_ALLOW,
            }
        ],
    },
    {
        "id": "ltx-2-19b-av",
        "name": "LTX-2 19B Audio+Vídeo (T2V, I2V, Frame a Frame y Sonido)",
        "category": "video",
        "is_default": False,
        "badge": "Vídeo + Audio + Frame a Frame · A100",
        "video_modes": ["t2v", "i2v", "flf2v", "av"],
        "default_fps": 24,
        "default_steps": 40,
        "default_cfg": 4.0,
        "default_duration_sec": 5.0,
        "official_spec": {
            "landscape": [768, 512],
            "portrait": [512, 768],
            "square": [640, 640],
            "max_area": 512 * 768,
            "mod_value": 32,
            "frame_rule": "8k+1",
            "default_frames": 121,
            "default_duration_sec": 5.0,
            "default_fps": 24,
            "default_steps": 40,
            "default_cfg": 4.0,
            "prompt_lang": "es_en",
            "official_note": "768×512 / 512×768 · 5 s (121 frames @ 24 fps) · 40 pasos · CFG 4.0 · Genera pista de audio sincronizada",
        },
        "description": (
            "Modelo multimodal de 19B de Lightricks que genera vídeo (desde texto, 1 imagen o Frame a Frame) "
            "y pista de audio sincronizada a la vez. Óptimo en A100/H100."
        ),
        "estimated_gb": 58.0,
        "min_vram_gb": 21.5,
        "recommended_vram_gb": 38.0,
        "l4_seconds_hint": "No validado en L4; la receta requiere mucha RAM",
        "engine": "ltx2_av",
        "artifacts": [
            {
                "role": "base",
                "repo_id": "Lightricks/LTX-2",
                "subdir": "base",
                "filename": None,
                "verify_file": "model_index.json",
                "allow_patterns": _DIFFUSERS_VIDEO_ALLOW,
            }
        ],
    },
    {
        "id": "ltx-2.3-distilled-av",
        "name": "LTX-2.3 Destilado · Audio+Vídeo (experimental)",
        "category": "video",
        "is_default": False,
        "badge": "Audio + vídeo · experimental · GPU no validada",
        "video_modes": ["t2v", "i2v", "flf2v"],
        "default_fps": 24,
        "default_steps": 8,
        "default_cfg": 1.0,
        "default_duration_sec": 5.0,
        "official_spec": {
            "landscape": [768, 512], "portrait": [512, 768], "square": [640, 640],
            "max_area": 512 * 768, "mod_value": 32, "frame_rule": "8k+1",
            "default_frames": 121, "default_duration_sec": 5.0, "default_fps": 24,
            "default_steps": 8, "default_cfg": 1.0, "prompt_lang": "en",
            "official_note": "Una etapa destilada: 8 sigmas, CFG 1; sin refinador x2. Sin pruebas en GPU.",
        },
        "description": "Vídeo y audio conjuntos; versión destilada BF16 de una etapa. Memoria estimada conservadora, sin probar en L4.",
        "validation_state": "reviewed_recipe", "gpu_tested": False,
        "estimated_gb": 75.0, "min_ram_gb": 80.0, "min_vram_gb": 24.0,
        "recommended_vram_gb": 40.0, "resource_confidence": "engineering_estimate",
        "l4_seconds_hint": "No validado; la receta BF16 puede superar la RAM de Colab",
        "engine": "ltx2_distilled_av", "ltx_distilled": True,
        "artifacts": [{"role": "base", "repo_id": "diffusers/LTX-2.3-Distilled-Diffusers", "subdir": "base", "filename": None,
                       "verify_file": "model_index.json",
                       "allow_patterns": _DIFFUSERS_VIDEO_ALLOW + ["duration_head/*"]}],
    },
    {
        "id": "ltx-2.5-distilled-av",
        "name": "LTX-2.5 Destilado · Audio+Vídeo (experimental)",
        "category": "video",
        "is_default": False,
        "badge": "Audio + vídeo · experimental · GPU no validada",
        "video_modes": ["t2v", "i2v", "flf2v"],
        "default_fps": 24,
        "default_steps": 8,
        "default_cfg": 1.0,
        "default_duration_sec": 5.0,
        "official_spec": {
            "landscape": [768, 512], "portrait": [512, 768], "square": [640, 640],
            "max_area": 512 * 768, "mod_value": 32, "frame_rule": "8k+1",
            "default_frames": 121, "default_duration_sec": 5.0, "default_fps": 24,
            "default_steps": 8, "default_cfg": 1.0, "prompt_lang": "en",
            "official_note": "Una etapa destilada: 8 sigmas, CFG 1; sin refinador x2. Sin pruebas en GPU.",
        },
        "description": "Vídeo y audio conjuntos; versión destilada BF16 de una etapa. Requiere aceptar el acceso en Hugging Face; no es la variante GGUF/FP8.",
        "validation_state": "reviewed_recipe", "gpu_tested": False,
        "estimated_gb": 75.0, "min_ram_gb": 80.0, "min_vram_gb": 24.0,
        "recommended_vram_gb": 40.0, "resource_confidence": "engineering_estimate",
        "l4_seconds_hint": "No validado; la receta BF16 puede superar la RAM de Colab",
        "engine": "ltx2_distilled_av", "ltx_distilled": True,
        "artifacts": [{"role": "base", "repo_id": "Lightricks/LTX-2.5-Diffusers", "subdir": "base", "filename": None,
                       "verify_file": "model_index.json",
                       "allow_patterns": _DIFFUSERS_VIDEO_ALLOW + ["duration_head/*"]}],
    },
    {
        "id": "wan21-i2v-14b-480p",
        "name": "Wan 2.1 I2V 14B 480P (Imagen a Vídeo y Frame a Frame · A100)",
        "category": "video",
        "is_default": False,
        "badge": "14B · I2V + Frame a Frame",
        "video_modes": ["i2v", "flf2v"],
        "default_fps": 16,
        "default_steps": 40,
        "default_cfg": 5.0,
        "default_duration_sec": 5.0,
        "official_spec": {
            "landscape": [832, 480],
            "portrait": [480, 832],
            "square": [640, 640],
            "max_area": 480 * 832,
            "mod_value": 16,
            "frame_rule": "4k+1",
            "default_frames": 81,
            "default_duration_sec": 5.0,
            "default_fps": 16,
            "default_steps": 40,
            "default_cfg": 5.0,
            "flow_shift": 3.0,
            "prompt_lang": "es_en",
            "official_note": "480P (832×480 / 480×832) · 5 s (81 frames @ 16 fps) · 40 pasos · CFG 5.0 · Entiende Español e Inglés",
        },
        "description": (
            "Modelo grande de 14B parámetros para animación detallada de 1 imagen o transición Frame a Frame. "
            "Diseñado para A100 (40/80 GB); en L4 usa offload secuencial."
        ),
        "estimated_gb": 68.0,
        "min_vram_gb": 21.5,
        "recommended_vram_gb": 38.0,
        "l4_seconds_hint": "6–10 min (Offload pesado)",
        "engine": "wan_i2v_14b",
        "artifacts": [
            {
                "role": "base",
                "repo_id": "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
                "subdir": "base",
                "filename": None,
                "verify_file": "model_index.json",
                "allow_patterns": _DIFFUSERS_VIDEO_ALLOW,
            }
        ],
    },
    {
        "id": "wan21-flf2v-14b-720p",
        "name": "Wan 2.1 FLF2V 14B 720P (Especialista Dedicado Frame a Frame · A100)",
        "category": "video",
        "is_default": False,
        "badge": "Especialista Frame a Frame 720p · A100",
        "video_modes": ["flf2v", "i2v"],
        "default_fps": 16,
        "default_steps": 40,
        "default_cfg": 5.0,
        "default_duration_sec": 5.0,
        "official_spec": {
            "landscape": [1280, 720],
            "portrait": [720, 1280],
            "square": [960, 960],
            "max_area": 720 * 1280,
            "mod_value": 16,
            "frame_rule": "4k+1",
            "default_frames": 81,
            "default_duration_sec": 5.0,
            "default_fps": 16,
            "default_steps": 40,
            "default_cfg": 5.0,
            "flow_shift": 5.0,
            "prompt_lang": "es_en",
            "official_note": "720P obligatorio (1280×720 / 720×1280) · 5 s (81 frames @ 16 fps) · 40 pasos · CFG 5.0 · Entiende Español e Inglés",
        },
        "description": (
            "Entrenado específicamente por Alibaba Wan-AI para interpolar la escena completa entre un Fotograma Inicial "
            "y un Fotograma Final en alta definición 720p. Recomendado en A100/H100."
        ),
        "estimated_gb": 88.0,
        "min_vram_gb": 24.0,
        "recommended_vram_gb": 42.0,
        "l4_seconds_hint": "Requiere A100 (40/80 GB)",
        "engine": "wan_i2v_14b",
        "artifacts": [
            {
                "role": "base",
                "repo_id": "Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers",
                "subdir": "base",
                "filename": None,
                "verify_file": "model_index.json",
                "allow_patterns": _DIFFUSERS_VIDEO_ALLOW,
            }
        ],
    },
    {
        "id": "wan22-i2v-a14b",
        "name": "Wan 2.2 I2V-A14B MoE (27B Dos Expertos · Exclusivo A100/H100)",
        "category": "video",
        "is_default": False,
        "badge": "Gama Alta · Solo A100 / H100",
        "video_modes": ["i2v", "flf2v"],
        "default_fps": 16,
        "default_steps": 40,
        "default_cfg": 3.5,
        "default_duration_sec": 5.0,
        "official_spec": {
            "landscape": [1280, 720],
            "portrait": [720, 1280],
            "square": [960, 960],
            "max_area": 720 * 1280,
            "mod_value": 16,
            "frame_rule": "4k+1",
            "default_frames": 81,
            "default_duration_sec": 5.0,
            "default_fps": 16,
            "default_steps": 40,
            "default_cfg": 3.5,
            "flow_shift": 5.0,
            "prompt_lang": "es_en",
            "official_note": "720P (1280×720 / 720×1280) · 5 s (81 frames @ 16 fps) · 40 pasos · CFG 3.5 (MoE 2 expertos)",
        },
        "description": (
            "Buque insignia Mixture-of-Experts (27B totales, dos expertos de ruido). "
            "Máxima calidad abierta actual; pensado para GPUs A100 (40/80 GB) o H100."
        ),
        "estimated_gb": 117.7,
        "min_vram_gb": 36.0,
        "recommended_vram_gb": 65.0,
        "l4_seconds_hint": "No cabe en L4",
        "engine": "wan_i2v_14b",
        "artifacts": [
            {
                "role": "base",
                "repo_id": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
                "subdir": "base",
                "filename": None,
                "verify_file": "model_index.json",
                "allow_patterns": _DIFFUSERS_VIDEO_ALLOW,
            }
        ],
    },
    {
        "id": "ffmpeg-motion-cpu",
        "name": "Animación Instantánea FFmpeg (Zoom / Paneo / Transición 2 Fotos · 0 GB)",
        "category": "video",
        "is_default": False,
        "builtin_no_download": True,
        "badge": "0 GB · Instantáneo CPU/GPU",
        "video_modes": ["i2v", "flf2v"],
        "default_fps": 24,
        "default_steps": 1,
        "description": (
            "Convierte cualquier imagen en un clip MP4 fluido con movimiento Ken Burns (zoom o paneo) "
            "en 2 segundos sin descargar modelos ni consumir VRAM."
        ),
        "estimated_gb": 0.0,
        "min_vram_gb": 0.0,
        "recommended_vram_gb": 0.0,
        "l4_seconds_hint": "2 s",
        "engine": "ffmpeg_motion",
        "artifacts": [],
    },
    # ==================== VOZ / TTS ====================
    {
        "id": "edge-tts-cloud",
        "name": "Edge-TTS Neural (Álvaro, Elvira y +10 voces · Instantáneo)",
        "category": "tts",
        "is_default": True,
        "builtin_no_download": True,
        "badge": "0 GB · Listo sin descargar",
        "description": (
            "Síntesis de voz neuronal en español (España, México, Argentina) e inglés sin ocupar disco ni VRAM."
        ),
        "estimated_gb": 0.0,
        "min_vram_gb": 0.0,
        "recommended_vram_gb": 0.0,
        "l4_seconds_hint": "1–2 s",
        "engine": "edge_tts",
        "artifacts": [],
    },
    {
        "id": "mms-tts-spa",
        "name": "Meta MMS-TTS Español (Voz Local Offline · 0.14 GB)",
        "category": "tts",
        "is_default": False,
        "badge": "Local HF · Español Offline",
        "description": (
            "Modelo VITS oficial de Meta AI para síntesis de voz en español 100% local en GPU o CPU."
        ),
        "estimated_gb": 0.14,
        "min_vram_gb": 1.0,
        "recommended_vram_gb": 1.0,
        "l4_seconds_hint": "1–2 s",
        "engine": "mms_tts",
        "artifacts": [
            {
                "role": "base",
                "repo_id": "facebook/mms-tts-spa",
                "subdir": "base",
                "filename": None,
                "verify_file": "config.json",
                "ignore_patterns": ["*.bin"],
            }
        ],
    },
    {
        "id": "kokoro-82m-tts",
        "name": "Kokoro-82M Multilingüe (Voz Local Ligera HF · 0.35 GB)",
        "category": "tts",
        "is_default": False,
        "badge": "Local HF · 0.35 GB",
        "description": (
            "Modelo abierto ultraligero de Hugging Face (Apache 2.0) para generar voz localmente en GPU o CPU."
        ),
        "estimated_gb": 0.35,
        "min_vram_gb": 0.5,
        "recommended_vram_gb": 2.0,
        "l4_seconds_hint": "2–3 s",
        "engine": "kokoro_tts",
        "artifacts": [
            {
                "role": "base",
                "repo_id": "hexgrad/Kokoro-82M",
                "subdir": "base",
                "filename": None,
                "verify_file": "kokoro-v1_0.pth",
                "allow_patterns": ["config.json", "kokoro-v1_0.pth", "voices/ef_dora.pt"],
                "ignore_patterns": ["samples/*", "eval/*"],
            }
        ],
    },
    {
        "id": "bark-small-tts",
        "name": "Suno Bark Small Expresivo (Voz Local Multilingüe · 1.7 GB)",
        "category": "tts",
        "is_default": False,
        "badge": "Local HF · Expresivo",
        "description": (
            "Modelo generativo de voz expresiva de Suno (MIT) compatible nativamente con Transformers en español e inglés."
        ),
        "estimated_gb": 1.7,
        "min_vram_gb": 2.0,
        "recommended_vram_gb": 4.5,
        "l4_seconds_hint": "5–10 s",
        "engine": "bark_tts",
        "artifacts": [
            {
                "role": "base",
                "repo_id": "suno/bark-small",
                "subdir": "base",
                "filename": None,
                "verify_file": "config.json",
            }
        ],
    },
    # ==================== INSPECTOR DE COHERENCIA VISUAL (MCP / MULTI-FOTOGRAMA) ====================
    {
        "id": "qwen25-vl-3b-coherence",
        "name": "Qwen 2.5 VL 3B Instruct 4-bit (★ Inspector Multimodal de Coherencia y Personajes · ~2.8 GB VRAM)",
        "category": "vision",
        "is_default": True,
        "badge": "★ Inspector Coherencia MCP · 4-bit (~2.8 GB VRAM)",
        "description": (
            "Modelo multimodal de visión y lenguaje de Alibaba Qwen (3B parámetros en 4-bit NF4 con torre visual en BF16). "
            "Analiza varios fotogramas a la vez para auditar si las personas, rostros, ropa, iluminación y escenario se mantienen idénticos, "
            "y genera automáticamente el prompt de reparación para tu agente MCP."
        ),
        "estimated_gb": 6.4,
        "min_vram_gb": 3.2,
        "recommended_vram_gb": 6.0,
        "l4_seconds_hint": "3–5 s / auditoría",
        "engine": "qwen25_vl_coherence",
        "artifacts": [
            {
                "role": "base",
                "repo_id": "Qwen/Qwen2.5-VL-3B-Instruct",
                "subdir": "base",
                "filename": None,
                "verify_file": "config.json",
                "allow_patterns": [
                    "*.json",
                    "*.safetensors",
                    "*.txt",
                    "*.tiktoken",
                    "*.model",
                    "*.jinja",
                ],
            }
        ],
    },
    {
        "id": "opencv-fast-coherence",
        "name": "OpenCV + Métricas Fotométricas Rápidas (Color, Luz, Estructura y Nitidez · 0 GB)",
        "category": "vision",
        "is_default": False,
        "builtin_no_download": True,
        "badge": "0 GB · 0.12s Instantáneo CPU",
        "description": (
            "Filtro físico ultrarrápido en CPU con OpenCV (histogramas HSV/LAB, luminancia, keypoints ORB, SSIM y nitidez Laplaciana). "
            "Detecta al instante saltos bruscos de iluminación, temperatura de color, proporción o desenfoque sin gastar VRAM."
        ),
        "estimated_gb": 0.0,
        "min_vram_gb": 0.0,
        "recommended_vram_gb": 0.0,
        "l4_seconds_hint": "0.12 s",
        "engine": "opencv_fast",
        "artifacts": [],
    },
    # ==================== LIPSYNC HD Y MÚSICA / AMBIENTE SONORO ====================
    {
        "id": "edtalk-256-lipsync-hd",
        "name": "EDTalk-256 HD + GFPGAN 1.4 + XSeg Oclusión (★ LipSync Habla Realista · 0.65 GB · Multi-Persona)",
        "category": "lipsync",
        "is_default": True,
        "badge": "★ LipSync 256→512 HD · Cualquier Idioma · 0.65 GB",
        "description": (
            "Motor completo de sincronización labial fonética en alta definición (Apache-2.0 EDTalk 256×256 + restauración facial "
            "GFPGAN 1.4 a 512×512 + máscara de oclusión XSeg para copas/cubiertos/manos + detector YuNet multi-rostro). "
            "Anima la boca de cualquier personaje en una foto o en un vídeo de Wan 14B eligiendo izquierda, derecha o rostro principal."
        ),
        "estimated_gb": 0.65,
        "min_vram_gb": 1.5,
        "recommended_vram_gb": 3.0,
        "l4_seconds_hint": "4–9 s / clip",
        "engine": "edtalk_onnx_hd",
        "artifacts": [
            {
                "role": "yunet",
                "repo_id": "facefusion/models-3.4.0",
                "direct_url": "https://github.com/facefusion/facefusion-assets/releases/download/models-3.4.0/yunet_2023_mar.onnx",
                "subdir": "onnx",
                "filename": "yunet_2023_mar.onnx",
                "verify_file": "yunet_2023_mar.onnx",
                "expected_bytes": 232475,
            },
            {
                "role": "edtalk",
                "repo_id": "facefusion/models-3.3.0",
                "direct_url": "https://github.com/facefusion/facefusion-assets/releases/download/models-3.3.0/edtalk_256.onnx",
                "subdir": "onnx",
                "filename": "edtalk_256.onnx",
                "verify_file": "edtalk_256.onnx",
                "expected_bytes": 270681365,
            },
            {
                "role": "xseg",
                "repo_id": "facefusion/models-3.1.0",
                "direct_url": "https://github.com/facefusion/facefusion-assets/releases/download/models-3.1.0/xseg_1.onnx",
                "subdir": "onnx",
                "filename": "xseg_1.onnx",
                "verify_file": "xseg_1.onnx",
                "expected_bytes": 70324286,
            },
            {
                "role": "gfpgan",
                "repo_id": "facefusion/models-3.0.0",
                "direct_url": "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/gfpgan_1.4.onnx",
                "subdir": "onnx",
                "filename": "gfpgan_1.4.onnx",
                "verify_file": "gfpgan_1.4.onnx",
                "expected_bytes": 340299087,
            },
        ],
    },
    {
        "id": "musicgen-small-bgm",
        "name": "Meta MusicGen Small (★ Música Instrumental Original por Prompt · 2.0 GB)",
        "category": "audio",
        "is_default": True,
        "badge": "★ Música IA · Jazz / Cine / Lo-Fi · 2.0 GB",
        "description": (
            "Generador oficial de música instrumental de Meta AI (MusicGen 300M integrado en Transformers). "
            "Crea bandas sonoras instrumentales (jazz, piano, ambiente cinematográfico). Revisar licencia del modelo y derechos aplicables a cada uso; no se garantiza libertad comercial."
        ),
        "estimated_gb": 2.0,
        "min_vram_gb": 2.3,
        "recommended_vram_gb": 4.0,
        "l4_seconds_hint": "4–8 s / pista",
        "engine": "musicgen",
        "artifacts": [
            {
                "role": "base",
                "repo_id": "facebook/musicgen-small",
                "subdir": "base",
                "filename": None,
                "verify_file": "config.json",
                "ignore_patterns": ["*.bin", "flax_model*", "tf_model*"],
            }
        ],
    },
    {
        "id": "audioldm2-sfx",
        "name": "AudioLDM 2 Efectos y Sonido Ambiental (★ Ambiente Restaurante / Lluvia / Foley · 3.8 GB)",
        "category": "audio",
        "is_default": False,
        "badge": "SFX + Ambiente Realista · Diffusers",
        "description": (
            "Generador de paisajes sonoros y efectos Foley realistas desde texto (murmullo cálido de restaurante, platos y copas, "
            "lluvia en cristal, pasos, tráfico lejano, viento)."
        ),
        "estimated_gb": 3.8,
        "min_vram_gb": 3.5,
        "recommended_vram_gb": 6.0,
        "l4_seconds_hint": "6–10 s / efecto",
        "engine": "audioldm2",
        "artifacts": [
            {
                "role": "base",
                "repo_id": "cvssp/audioldm2",
                "subdir": "base",
                "filename": None,
                "verify_file": "model_index.json",
                "ignore_patterns": ["*.bin"],
            }
        ],
    },
    {
        "id": "procedural-ambience-cpu",
        "name": "Sintetizador Acústico de Ambiente en Estudio (Restaurante / Lluvia / Sala / Pad Cálido · 0 GB)",
        "category": "audio",
        "is_default": False,
        "builtin_no_download": True,
        "badge": "0 GB · 0.2s Instantáneo CPU",
        "description": (
            "Generador DSP estéreo integrado (0 GB VRAM, 0.2 s) con presets calibrados de ambiente acústico cálido de restaurante/cafetería, "
            "lluvia suave, aire exterior, pad armónico cinematográfico y room-tone limpio."
        ),
        "estimated_gb": 0.0,
        "min_vram_gb": 0.0,
        "recommended_vram_gb": 0.0,
        "l4_seconds_hint": "0.2 s",
        "engine": "procedural_dsp",
        "artifacts": [],
    },
]


DEFAULT_CATALOG.extend(EXTRA_IMAGE_MODELS)
for _definition in DEFAULT_CATALOG:
    if _definition.get("category") == "lora":
        _definition["vram_budget_scope"] = "incremental_adapter_requires_base"


def _dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for root, _, files in os.walk(path):
        for fn in files:
            fp = Path(root) / fn
            try:
                if not fp.is_symlink():
                    total += fp.stat().st_size
            except OSError:
                pass
    return total


def resolve_gpu_profile(
    gpu_profile_id: Any = "auto",
    gpu_info: Optional[Dict[str, Any]] = None,
    detected_gpu: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Resuelve el perfil de GPU elegido ('auto', 'T4', 'L4', 'A100_40', 'A100_80', 'H100', 'CPU')."""
    if isinstance(gpu_profile_id, dict) and "vram_gb" in gpu_profile_id:
        return gpu_profile_id

    ginfo = gpu_info or detected_gpu or query_gpu_info()
    clean_id = str(gpu_profile_id or "auto").strip()
    if clean_id != "auto":
        for prof in COLAB_GPU_PROFILES:
            if prof["id"].lower() == clean_id.lower():
                vram_val = float(prof["vram_gb"] or 0.0)
                return {
                    "id": prof["id"],
                    "profile_id": prof["id"],
                    "tier": prof["id"].split("_", 1)[0],
                    "short": prof["short"],
                    "gpu_name": prof["short"],
                    "vram_gb": vram_val,
                    "has_gpu": vram_val > 0.0,
                    "is_simulated": True,
                }

    # Modo 'auto': usar la GPU real detectada en la sesión actual
    has_gpu = bool(ginfo.get("has_gpu"))
    vram = float(ginfo.get("vram_total_gb") or 0.0)
    tier = str(ginfo.get("gpu_tier") or ("CPU" if not has_gpu else "GPU"))
    gname = f"{tier} ({vram:.1f} GB)" if has_gpu else "Solo CPU (0 GB)"
    return {
        "id": tier if has_gpu else "CPU",
        "profile_id": "auto",
        "short": gname,
        "gpu_name": gname,
        "tier": tier,
        "vram_gb": vram,
        "has_gpu": has_gpu,
        "is_simulated": False,
    }


def evaluate_gpu_tier(
    defn: Dict[str, Any],
    gpu_profile_id: Any = "auto",
    gpu_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Evalúa cómo rendirá un modelo en la GPU filtrada y devuelve un sticker claro:
    - excellent: '🟢 Excelente'
    - normal:    '🟡 Normal'
    - slow:      '🟠 Lento'
    - incompatible: '🔴 No cabe'
    """
    ginfo = gpu_info or query_gpu_info()
    target = resolve_gpu_profile(gpu_profile_id, ginfo)
    vram = float(target["vram_gb"])
    gname = target["gpu_name"]
    min_v = float(defn.get("min_vram_gb") or 0.0)
    rec_v = float(defn.get("recommended_vram_gb") or min_v)
    l4_hint = defn.get("l4_seconds_hint") or ""

    if defn.get("validation_state") == "candidate":
        return {"status": "candidate", "tier": "candidate", "tier_rank": 4, "compatible": False,
                "sticker": "🧪 Candidato sin receta", "short_sticker": "Sin validar",
                "label": "🧪 Candidato guardado: no se ha validado el motor ni los recursos", "gpu_target": gname,
                "detail": "El tamaño declarado no demuestra compatibilidad ni habilita descargas."}

    if ENGINE_CONTRACTS.get(str(defn.get("engine")), {}).get("native_bf16") and target.get("tier") in ("T4", "V100"):
        return {"status": "dtype_unsupported", "tier": "incompatible", "tier_rank": 3, "compatible": False,
                "sticker": "🔴 Requiere BF16 nativo", "short_sticker": "BF16 no nativo", "gpu_target": gname,
                "label": "Esta receta BF16 no está autorizada para T4/V100", "detail": "Sin conversión de precisión ni cuantización automática."}

    # Modelos integrados o que corren en CPU sin exigir VRAM
    if defn.get("builtin_no_download") or min_v <= 0.0:
        return {
            "status": "ok",
            "tier": "excellent",
            "tier_rank": 0,
            "compatible": True,
            "sticker": "🟢 Excelente",
            "short_sticker": "Excelente",
            "gpu_target": gname,
            "label": f"🟢 CPU / sin requisito de VRAM en {gname}",
            "detail": "Procesamiento CPU o servicio de red; duración y RAM dependen de la tarea.",
        }

    # A LoRA fits only with a declared base, not because its own file is small.
    if defn.get("category") == "lora":
        base_ids = list(defn.get("compatible_models") or [])
        if not base_ids and defn.get("parent_model"):
            base_ids = [defn["parent_model"]]
        bases = [d for d in DEFAULT_CATALOG if d["id"] in base_ids and d["category"] != "lora"]
        if bases:
            evaluations = [(d, evaluate_gpu_tier(d, target, ginfo)) for d in bases]
            base, best = min(evaluations, key=lambda pair: (pair[1]["tier_rank"], pair[0].get("min_vram_gb", 0)))
            return {**best, "requires_base": base_ids, "assessed_base_model_id": base["id"],
                    "label": f"{best['sticker']} · requiere base {base['id']} + adaptador (estimación)",
                    "detail": "Comprobar arquitectura, submodo, VRAM de la base, RAM y disco del conjunto."}
        return {"status": "candidate", "tier": "candidate", "tier_rank": 4, "compatible": False,
                "sticker": "🧪 Base desconocida", "short_sticker": "Sin base",
                "label": "LoRA sin modelo base exacto revisado", "detail": "No se infiere la base por el nombre.", "gpu_target": gname}

    if not target["has_gpu"] or vram <= 0.1:
        return {
            "status": "cpu_only",
            "tier": "incompatible",
            "tier_rank": 3,
            "compatible": False,
            "sticker": "🔴 Requiere GPU",
            "short_sticker": "Requiere GPU",
            "gpu_target": gname,
            "label": f"🔴 Requiere GPU (~{min_v:.0f} GB VRAM) · Estás en modo sin GPU",
            "detail": f"Pide mínimo ~{min_v:.0f} GB VRAM (óptimo {rec_v:.0f} GB)",
        }

    # 1. EXCELENTE: La GPU iguala o supera la VRAM recomendada
    if vram >= rec_v:
        hint_txt = f" · ~{l4_hint}" if l4_hint and "L4" in gname else ""
        return {
            "status": "ok",
            "tier": "excellent",
            "tier_rank": 0,
            "compatible": True,
            "sticker": "🟢 Excelente",
            "short_sticker": "Excelente",
            "gpu_target": gname,
            "label": f"🟢 Excelente en {gname}{hint_txt}",
            "detail": f"Holgado en VRAM (usa ~{min_v:.0f}-{rec_v:.0f} GB de {vram:.1f} GB){hint_txt}",
        }

    # 2. NORMAL: Cabe bien con CPU Offload inteligente (al menos un 12% de margen sobre min_vram)
    if vram >= min_v * 1.12:
        hint_txt = f" · ~{l4_hint}" if l4_hint and "L4" in gname else ""
        return {
            "status": "offload",
            "tier": "normal",
            "tier_rank": 1,
            "compatible": True,
            "sticker": "🟡 Normal",
            "short_sticker": "Normal",
            "gpu_target": gname,
            "label": f"🟡 Normal en {gname} (con CPU Offload){hint_txt}",
            "detail": f"Estable con CPU Offload (pide ~{min_v:.0f} GB, tienes {vram:.1f} GB){hint_txt}",
        }

    # 3. LENTO: Muy justo de VRAM (entre 95% y 112% del mínimo: corre pero intercambiando capas con RAM)
    if vram >= min_v * 0.95:
        return {
            "status": "tight",
            "tier": "slow",
            "tier_rank": 2,
            "compatible": True,
            "sticker": "🟠 Lento",
            "short_sticker": "Lento",
            "gpu_target": gname,
            "label": f"🟠 Lento en {gname} · Al límite de VRAM (usar 480p)",
            "detail": f"Muy justo ({vram:.1f} GB vs ~{min_v:.0f} GB mínimos); offload intenso",
        }

    # 4. NO CABE: Excede la VRAM de esta GPU
    suggested_gpu = "A100 40 GB" if min_v <= 35 else "A100 80 GB / H100"
    if min_v <= 20:
        suggested_gpu = "L4 (22.5 GB) o superior"
    return {
        "status": "incompatible",
        "tier": "incompatible",
        "tier_rank": 3,
        "compatible": False,
        "sticker": "🔴 No cabe",
        "short_sticker": "No cabe",
        "gpu_target": gname,
        "label": f"🔴 No cabe en {gname} · Pide ~{min_v:.0f} GB VRAM (usa {suggested_gpu})",
        "detail": f"Necesita ~{min_v:.0f} GB VRAM · Recomendado: {suggested_gpu}",
    }


def _matches_artifact_filters(
    rfilename: str,
    allow_patterns: Optional[List[str]],
    ignore_patterns: Optional[List[str]],
    has_model_index: bool,
) -> bool:
    if rfilename.startswith(".git"):
        return False
    low = rfilename.lower()
    # Ignorar siempre medios promocionales / demos / ONNX innecesarios salvo petición explícita
    default_skip_prefixes = ("assets/", "examples/", "media/", "samples/", "eval/")
    default_skip_exts = (".gif", ".mp4", ".webp", ".jpg", ".jpeg", ".png", ".onnx", ".onnx_data", ".gguf")
    if not allow_patterns:
        if low.startswith(default_skip_prefixes) or low.endswith(default_skip_exts):
            return False
        # Evitar descargar checkpoints monolíticos duplicados en la raíz si el repo ya es Diffusers modular
        if has_model_index and "/" not in rfilename and low.endswith((".safetensors", ".ckpt", ".pth", ".bin")):
            return False

    if allow_patterns:
        if not any(fnmatch.fnmatch(rfilename, pat) for pat in allow_patterns):
            return False
    if ignore_patterns:
        if any(fnmatch.fnmatch(rfilename, pat) for pat in ignore_patterns):
            return False
    return True


class CatalogManager:
    def __init__(self) -> None:
        self._custom_models: List[Dict[str, Any]] = self._load_custom_models()
        env_tok = (
            os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            or ""
        ).strip()
        self._hf_token: Optional[str] = env_tok or None
        self._hf_username: Optional[str] = None

    def get_hf_token(self) -> Optional[str]:
        return self._hf_token

    def hf_auth_status(self) -> Dict[str, Any]:
        tok = self._hf_token
        masked = None
        if tok:
            masked = f"{tok[:5]}...{tok[-4:]}" if len(tok) > 10 else "hf_***"
        return {
            "has_token": bool(tok),
            "masked_token": masked,
            "username": self._hf_username,
        }

    def set_hf_token(self, token: str) -> Dict[str, Any]:
        clean = (token or "").strip()
        if not clean:
            return self.clear_hf_token()
        if len(clean) < 8:
            raise ValueError("El token de Hugging Face parece demasiado corto (suele empezar por 'hf_').")

        username: Optional[str] = None
        try:
            who = HfApi(token=clean).whoami()
            if isinstance(who, dict):
                username = who.get("name") or who.get("fullname")
        except Exception as exc:
            msg = str(exc).lower()
            if "401" in msg or "invalid" in msg or "unauthorized" in msg:
                raise ValueError("Token rechazado por Hugging Face (401 no autorizado). Revisa que esté bien copiado.") from exc
            # Si estamos sin salida directa desde el servidor (ej. vista previa offline), guardarlo igualmente en sesión

        self._hf_token = clean
        self._hf_username = username
        os.environ["HF_TOKEN"] = clean
        return self.hf_auth_status()

    def clear_hf_token(self) -> Dict[str, Any]:
        self._hf_token = None
        self._hf_username = None
        os.environ.pop("HF_TOKEN", None)
        os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
        return self.hf_auth_status()

    def _load_custom_models(self) -> List[Dict[str, Any]]:
        if CUSTOM_MODELS_FILE.exists():
            try:
                rows = json.loads(CUSTOM_MODELS_FILE.read_text(encoding="utf-8"))
                if not isinstance(rows, list):
                    return []
                return [{**row, "validation_state": row.get("validation_state", "candidate")}
                        for row in rows if isinstance(row, dict) and row.get("id")]
            except Exception:
                return []
        return []

    def _save_custom_models(self) -> None:
        CUSTOM_MODELS_FILE.parent.mkdir(parents=True, exist_ok=True)
        CUSTOM_MODELS_FILE.write_text(
            json.dumps(self._custom_models, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def get_all_definitions(self) -> List[Dict[str, Any]]:
        return DEFAULT_CATALOG + self._custom_models

    def get_definition(self, model_id: str) -> Optional[Dict[str, Any]]:
        for item in self.get_all_definitions():
            if item["id"] == model_id:
                return item
        return None

    def model_dir(self, model_id: str) -> Path:
        return MODELS_DIR / model_id

    def marker_file(self, model_id: str) -> Path:
        return self.model_dir(model_id) / ".download_complete.json"

    def is_downloaded(self, model_id: str) -> bool:
        defn = self.get_definition(model_id)
        if not defn:
            return False
        if defn.get("builtin_no_download"):
            return True
        m_dir = self.model_dir(model_id)
        if not self.marker_file(model_id).exists():
            return False
        for art in defn.get("artifacts", []):
            sub = m_dir / art["subdir"]
            verify_file = art.get("verify_file") or art.get("filename")
            if verify_file:
                target = sub / verify_file
                if not target.exists() or target.stat().st_size == 0:
                    return False
            elif not sub.exists():
                return False
        if defn.get("category") == "image" and not image_recipe_files_complete(defn, m_dir / "base"):
            return False
        return True

    def disk_usage_gb(self, model_id: str) -> float:
        m_dir = self.model_dir(model_id)
        return round(_dir_size_bytes(m_dir) / (1024**3), 2)

    def compatibility_badge(
        self,
        defn: Dict[str, Any],
        gpu_info: Dict[str, Any],
        gpu_profile: str = "auto",
    ) -> Dict[str, Any]:
        return evaluate_gpu_tier(defn, gpu_profile_id=gpu_profile, gpu_info=gpu_info)

    def list_catalog(
        self,
        loaded_ids: Optional[List[str]] = None,
        gpu_profile: str = "auto",
        only_compatible: bool = False,
        detected_gpu: Optional[Dict[str, Any]] = None,
        assessment_context_value: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        loaded_set = set(loaded_ids or [])
        gpu_info = detected_gpu or query_gpu_info()
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        free_gb = round(shutil.disk_usage(MODELS_DIR).free / (1024**3), 2)

        matrix_targets = [
            ("T4", "T4"),
            ("L4", "L4"),
            ("A100_40", "A100 40G"),
            ("A100_80", "A100/H100 80G"),
        ]

        from studio.model_compatibility import assessment_context, assess_definition
        context = assessment_context_value or assessment_context(gpu_profile=gpu_profile, detected_gpu=gpu_info)
        items: List[Dict[str, Any]] = []
        for defn in self.get_all_definitions():
            mid = defn["id"]
            downloaded = self.is_downloaded(mid)
            actual_gb = self.disk_usage_gb(mid) if downloaded and not defn.get("builtin_no_download") else 0.0
            compat = assess_definition(defn, context, catalog=self)
            if only_compatible and not compat.get("compatible", True) and not downloaded and mid not in loaded_set:
                continue
            per_gpu_summary = []
            for gid, gshort in matrix_targets:
                ev = evaluate_gpu_tier(defn, gpu_profile_id=gid, gpu_info=gpu_info)
                per_gpu_summary.append(
                    {
                        "gpu_id": gid,
                        "gpu_short": gshort,
                        "tier": ev["tier"],
                        "sticker": ev["sticker"],
                        "short": ev["short_sticker"],
                    }
                )
            items.append(
                {
                    **defn,
                    "downloaded": downloaded,
                    "actual_disk_gb": actual_gb,
                    "loaded_in_memory": mid in loaded_set,
                    "free_disk_gb": free_gb,
                    "compatibility": compat,
                    "gpu_matrix": per_gpu_summary,
                }
            )

        # Ordenar priorizando: predeterminado / cargado / descargado y mejor rendimiento en la GPU elegida
        items.sort(
            key=lambda x: (
                0 if x.get("loaded_in_memory") else 1,
                0 if x.get("is_default") else 1,
                x["compatibility"].get("tier_rank", 2),
                float(x.get("estimated_gb") or 0.0),
            )
        )
        return items

    def search_huggingface_live(
        self,
        query: str = "",
        category: str = "video",
        gpu_profile: str = "auto",
        only_compatible: bool = True,
        detected_gpu: Optional[Dict[str, Any]] = None,
        limit: int = 18,
    ) -> Dict[str, Any]:
        """
        Consulta en tiempo real la API pública de Hugging Face (sin requerir token)
        para explorar modelos compatibles con Diffusers/Transformers y calcular su sticker de GPU.
        """
        from studio.model_discovery import model_discovery
        return model_discovery.search(
            query=query, category=category, gpu_profile=gpu_profile,
            only_compatible=only_compatible, detected_gpu=detected_gpu, limit=limit,
        )

    def add_custom_hf_model(
        self,
        name: str,
        repo_id: str,
        category: str = "lora",
        filename: Optional[str] = None,
        estimated_gb: float = 1.0,
        min_vram_gb: Optional[float] = None,
        adapter_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        # User-entered names/GB and remote tags are NOT a loader contract. Preserve
        # manual imports as visible, non-executable candidates, not guessed Qwen/Wan.
        from studio.model_discovery import clean_repo_id, clean_filename, CATEGORIES
        import math
        repo_id = clean_repo_id(repo_id)
        clean_fn = clean_filename(filename)
        if category not in CATEGORIES or category == "all":
            raise ValueError("Categoría de candidato no válida.")
        if not math.isfinite(float(estimated_gb)) or not 0 <= float(estimated_gb) <= 1000:
            raise ValueError("El tamaño declarado debe estar entre 0 y 1000 GB.")
        import hashlib
        slug = repo_id.lower().replace("/", "-").replace("_", "-").replace(".", "-")
        identity = json.dumps([repo_id.lower(), category, clean_fn], separators=(",", ":"))
        slug += "-" + hashlib.sha256(identity.encode()).hexdigest()[:10]
        for saved in self._custom_models:
            artifacts = saved.get("artifacts") or []
            if (saved.get("category") == category and len(artifacts) == 1
                    and str(artifacts[0].get("repo_id", "")).lower() == repo_id.lower()
                    and artifacts[0].get("filename") == clean_fn):
                return saved
        # Reuse an exact reviewed recipe, never substitute a single component for
        # a multi-repository pipeline or silently download another model.
        for definition in self.get_all_definitions():
            arts = definition.get("artifacts", [])
            if (len(arts) == 1 and definition["category"] == category
                    and arts[0].get("repo_id", "").lower() == repo_id.lower()
                    and arts[0].get("filename") == clean_fn
                    and not definition.get("is_custom")):
                return definition
        existing = self.get_definition(slug)
        if existing:
            return existing
        entry = {
            "id": slug, "name": name.strip()[:160] or repo_id, "repo_id": repo_id,
            "category": category, "engine": "unvalidated", "is_custom": True, "is_default": False,
            "validation_state": "candidate", "gpu_tested": False,
            "badge": "Candidato guardado · no ejecutable", "compatible_models": [],
            "description": "Importación manual pendiente de inspección y receta revisada. Los GB declarados no demuestran compatibilidad.",
            "estimated_gb": round(float(estimated_gb), 2), "declared_min_vram_gb": min_vram_gb,
            "min_vram_gb": None, "recommended_vram_gb": None,
            "artifacts": [{"role": "candidate", "repo_id": repo_id, "subdir": "candidate",
                           "filename": clean_fn, "verify_file": clean_fn or "model_index.json"}],
        }
        self._custom_models.append(entry)
        self._save_custom_models()
        return entry

    def check_disk_space(
        self,
        model_id: str,
        required_bytes_override: Optional[int] = None,
        free_bytes_override: Optional[int] = None,
    ) -> Dict[str, Any]:
        defn = self.get_definition(model_id)
        if not defn:
            raise ValueError(f"Modelo desconocido: {model_id}")
        if defn.get("builtin_no_download"):
            return {"ok": True, "missing_gb": 0.0, "suggestions": []}

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        free_bytes = (
            free_bytes_override
            if free_bytes_override is not None
            else shutil.disk_usage(MODELS_DIR).free
        )
        already_bytes = _dir_size_bytes(self.model_dir(model_id))
        est_bytes = (
            required_bytes_override
            if required_bytes_override is not None
            else int(float(defn.get("estimated_gb", 1.0)) * (1024**3))
        )
        needed_bytes = max(0, est_bytes - already_bytes)
        safety_bytes = int(DISK_SAFETY_MARGIN_GB * (1024**3))
        total_required = needed_bytes + safety_bytes

        if free_bytes >= total_required:
            return {
                "ok": True,
                "free_gb": round(free_bytes / (1024**3), 2),
                "required_gb": round(total_required / (1024**3), 2),
                "missing_gb": 0.0,
                "suggestions": [],
            }

        missing_gb = round((total_required - free_bytes) / (1024**3), 2)
        suggestions: List[Dict[str, Any]] = []
        for other in self.get_all_definitions():
            oid = other["id"]
            if oid == model_id or other.get("builtin_no_download"):
                continue
            used = _dir_size_bytes(self.model_dir(oid))
            if used > 0:
                suggestions.append(
                    {
                        "id": oid,
                        "name": other["name"],
                        "frees_gb": round(used / (1024**3), 2),
                    }
                )
        suggestions.sort(key=lambda x: x["frees_gb"], reverse=True)

        if suggestions:
            sug_str = ", ".join(f"«{s['name']}» (libera {s['frees_gb']} GB)" for s in suggestions)
            msg = (
                f"Espacio insuficiente en disco: faltan {missing_gb} GB libres "
                f"(quedan {free_bytes / (1024**3):.2f} GB y se necesitan ~{total_required / (1024**3):.2f} GB con margen). "
                f"Sugerencia para borrar: {sug_str}."
            )
        else:
            msg = (
                f"Espacio insuficiente en disco: faltan {missing_gb} GB libres "
                f"(quedan {free_bytes / (1024**3):.2f} GB y se necesitan ~{total_required / (1024**3):.2f} GB). "
                f"No hay otros modelos descargados; puedes limpiar archivos antiguos de salidas o subidas."
            )

        return {
            "ok": False,
            "free_gb": round(free_bytes / (1024**3), 2),
            "required_gb": round(total_required / (1024**3), 2),
            "missing_gb": missing_gb,
            "suggestions": suggestions,
            "message": msg,
        }

    def download_model(
        self,
        model_id: str,
        job: JobRecord,
        progress_range: Tuple[float, float] = (0.0, 100.0),
    ) -> Dict[str, Any]:
        with _get_model_download_lock(model_id):
            job.check_cancelled()
            return self._download_model_locked(model_id=model_id, job=job, progress_range=progress_range)

    def _download_model_locked(
        self,
        model_id: str,
        job: JobRecord,
        progress_range: Tuple[float, float] = (0.0, 100.0),
    ) -> Dict[str, Any]:
        p_min, p_max = float(progress_range[0]), float(progress_range[1])

        def _scale_pct(raw_pct: float) -> float:
            clamped = max(0.0, min(100.0, raw_pct))
            return round(p_min + (clamped / 100.0) * (p_max - p_min), 1)

        defn = self.get_definition(model_id)
        if not defn:
            raise ValueError(f"Modelo desconocido: {model_id}")
        if defn.get("validation_state") == "candidate":
            raise ValueError("MODEL_RECIPE_UNVALIDATED: candidato sin receta revisada. Inspecciona metadatos; no se descargarán pesos ni código remoto.")
        if defn.get("builtin_no_download"):
            job.log("Este motor no requiere descargar archivos.", progress=_scale_pct(100.0), step="Listo")
            return {"model_id": model_id, "downloaded_gb": 0.0}
        if self.is_downloaded(model_id):
            actual_gb = self.disk_usage_gb(model_id)
            job.log(
                f"«{defn['name']}» ya está verificado en disco ({actual_gb} GB).",
                progress=_scale_pct(100.0),
                step="Modelo ya en disco",
            )
            return {"model_id": model_id, "downloaded_gb": actual_gb}

        precheck = self.check_disk_space(model_id)
        if not precheck["ok"]:
            raise RuntimeError(precheck["message"])

        tok = self.get_hf_token()
        api = HfApi(token=tok)
        m_dir = self.model_dir(model_id)
        m_dir.mkdir(parents=True, exist_ok=True)

        job.log(
            f"Consultando metadatos y tamaño exacto en Hugging Face para «{defn['name']}»...",
            progress=_scale_pct(2.0),
            step="Consultando Hugging Face...",
        )

        plan_files: List[Dict[str, Any]] = []
        total_bytes = 0
        already_bytes = 0

        for art in defn.get("artifacts", []):
            job.check_cancelled()
            repo_id = art["repo_id"]
            subdir = m_dir / art["subdir"]
            subdir.mkdir(parents=True, exist_ok=True)
            single_fn = art.get("filename")
            allow_pats = art.get("allow_patterns")
            ignore_pats = art.get("ignore_patterns")
            direct_url = art.get("direct_url")

            if direct_url and single_fn:
                fsize = int(art.get("expected_bytes") or 0)
                local_target = subdir / single_fn
                exists_already = local_target.exists() and (
                    fsize == 0 or abs(local_target.stat().st_size - fsize) < 4096
                )
                total_bytes += fsize
                if exists_already:
                    already_bytes += local_target.stat().st_size
                plan_files.append(
                    {
                        "repo_id": repo_id,
                        "direct_url": direct_url,
                        "rfilename": single_fn,
                        "subdir": subdir,
                        "size": fsize,
                        "exists": exists_already,
                    }
                )
                continue

            info = api.model_info(repo_id=repo_id, files_metadata=True)
            siblings = info.siblings or []
            has_model_index = any(s.rfilename == "model_index.json" for s in siblings)

            for sib in siblings:
                rfilename = sib.rfilename
                if single_fn:
                    if rfilename != single_fn:
                        continue
                else:
                    if not _matches_artifact_filters(rfilename, allow_pats, ignore_pats, has_model_index):
                        continue

                fsize = int(getattr(sib, "size", None) or 0)
                local_target = subdir / rfilename
                exists_already = local_target.exists() and (
                    fsize == 0 or local_target.stat().st_size == fsize
                )
                total_bytes += fsize
                if exists_already:
                    already_bytes += fsize
                plan_files.append(
                    {
                        "repo_id": repo_id,
                        "rfilename": rfilename,
                        "subdir": subdir,
                        "size": fsize,
                        "exists": exists_already,
                    }
                )

        if not plan_files:
            raise RuntimeError("No se encontraron archivos descargables en el repositorio indicado.")

        exact_check = self.check_disk_space(model_id, required_bytes_override=total_bytes)
        if not exact_check["ok"]:
            raise RuntimeError(exact_check["message"])

        total_gb = total_bytes / (1024**3)
        job.log(
            f"Plan de descarga verificado: {len(plan_files)} archivo(s), peso real {total_gb:.2f} GB.",
            progress=_scale_pct(4.0),
            step=f"0 / {len(plan_files)} archivos (0.00 / {total_gb:.2f} GB)",
        )

        completed_files_bytes = already_bytes
        dl_clock = {
            "start_ts": time.time(),
            "net_bytes": 0,
            "last_ui_ts": 0.0,
        }

        for idx, item in enumerate(plan_files, start=1):
            job.check_cancelled()
            rfn = item["rfilename"]
            short_name = Path(rfn).name
            fsize = int(item["size"] or 0)

            if item["exists"]:
                raw_pct = (completed_files_bytes / total_bytes * 100.0) if total_bytes > 0 else (idx / len(plan_files) * 100.0)
                job.log(
                    f"[{idx}/{len(plan_files)}] Ya en disco: {short_name}",
                    progress=_scale_pct(min(99.0, max(4.0, raw_pct))),
                    step=f"[{idx}/{len(plan_files)}] Verificado {short_name}",
                )
                continue

            file_mb = fsize / (1024**2)
            base_raw_pct = (completed_files_bytes / total_bytes * 100.0) if total_bytes > 0 else ((idx - 1) / len(plan_files) * 100.0)
            job.log(
                f"[{idx}/{len(plan_files)}] Descargando {short_name} ({file_mb:.1f} MB)...",
                progress=_scale_pct(min(99.0, max(4.0, base_raw_pct))),
                step=f"[{idx}/{len(plan_files)}] {short_name} ({completed_files_bytes / (1024**3):.2f} / {total_gb:.2f} GB)",
            )

            file_progress = {
                "reconstructed": 0,
                "transfer": 0,
                "done": False,
            }

            def _emit_live_progress(force: bool = False) -> None:
                now = time.time()
                if not force and (now - dl_clock["last_ui_ts"] < 0.35):
                    return
                dl_clock["last_ui_ts"] = now

                est_file_done = max(file_progress["reconstructed"], file_progress["transfer"])
                if fsize > 0:
                    est_file_done = min(fsize, est_file_done)

                cur_total_done = completed_files_bytes + est_file_done
                raw_pct = (
                    (cur_total_done / total_bytes) * 100.0
                    if total_bytes > 0
                    else ((idx - 0.5) / len(plan_files)) * 100.0
                )
                dt = max(0.05, now - dl_clock["start_ts"])
                mbps = (dl_clock["net_bytes"] / (1024**2)) / dt
                done_gb = cur_total_done / (1024**3)
                rem_bytes = max(0, total_bytes - cur_total_done)
                bps = dl_clock["net_bytes"] / dt
                eta_str = f" · ETA ~{int(rem_bytes / bps)}s" if bps > 64 * 1024 else ""
                spd_str = f"{mbps:.1f} MB/s{eta_str}" if dl_clock["net_bytes"] > 0 else "Conectando con HF..."

                job.heartbeat(
                    progress=_scale_pct(min(99.2, max(4.0, raw_pct))),
                    step=f"[{idx}/{len(plan_files)}] {short_name} ({done_gb:.2f} / {total_gb:.2f} GB)",
                    speed_text=spd_str,
                )

            class _JobTqdm(std_tqdm):  # type: ignore
                def __init__(self, *args: Any, **kwargs: Any) -> None:
                    desc = str(kwargs.get("desc") or "")
                    self._is_xet_transfer = desc == "Downloading bytes"
                    kwargs["file"] = io.StringIO()
                    kwargs["disable"] = False
                    kwargs.pop("name", None)
                    super().__init__(*args, **kwargs)

                def update(self, n: Optional[int | float] = 1) -> Optional[bool]:
                    job.check_cancelled()
                    res = super().update(n)
                    inc = int(n or 0)
                    if inc > 0:
                        if self._is_xet_transfer:
                            file_progress["transfer"] += inc
                            dl_clock["net_bytes"] += inc
                        else:
                            prev_rec = file_progress["reconstructed"]
                            file_progress["reconstructed"] += inc
                            if file_progress["transfer"] == 0:
                                dl_clock["net_bytes"] += inc
                            elif file_progress["reconstructed"] > file_progress["transfer"]:
                                dl_clock["net_bytes"] += max(
                                    0,
                                    file_progress["reconstructed"] - max(prev_rec, file_progress["transfer"]),
                                )
                        _emit_live_progress(force=False)
                    return res

            def _pulse_watcher() -> None:
                while not file_progress["done"] and not job.cancel_requested:
                    time.sleep(1.5)
                    if not file_progress["done"]:
                        _emit_live_progress(force=True)

            watcher = threading.Thread(target=_pulse_watcher, daemon=True)
            watcher.start()
            try:
                if item.get("direct_url"):
                    target_fp = Path(item["subdir"]) / rfn
                    target_fp.parent.mkdir(parents=True, exist_ok=True)
                    part_fp = target_fp.with_suffix(target_fp.suffix + ".part")
                    with requests.get(str(item["direct_url"]), stream=True, timeout=60) as resp_dl:
                        resp_dl.raise_for_status()
                        with open(part_fp, "wb") as fh_out:
                            for chunk in resp_dl.iter_content(chunk_size=256 * 1024):
                                job.check_cancelled()
                                if not chunk:
                                    continue
                                fh_out.write(chunk)
                                c_len = len(chunk)
                                file_progress["transfer"] += c_len
                                dl_clock["net_bytes"] += c_len
                                _emit_live_progress(force=False)
                    part_fp.replace(target_fp)
                    downloaded_path = str(target_fp)
                else:
                    downloaded_path = hf_hub_download(
                        repo_id=item["repo_id"],
                        filename=rfn,
                        local_dir=str(item["subdir"]),
                        tqdm_class=_JobTqdm,
                        token=tok,
                    )
            finally:
                file_progress["done"] = True

            p_check = Path(downloaded_path)
            if not p_check.exists() or p_check.stat().st_size == 0:
                raise RuntimeError(f"Verificación fallida: {short_name} no existe o está vacío tras descargar.")

            completed_files_bytes += fsize if fsize > 0 else p_check.stat().st_size
            pct_done = (completed_files_bytes / total_bytes * 100.0) if total_bytes > 0 else (idx / len(plan_files) * 100.0)
            job.log(
                f"[{idx}/{len(plan_files)}] Completado: {short_name} ({completed_files_bytes / (1024**3):.2f} / {total_gb:.2f} GB)",
                progress=_scale_pct(min(99.5, max(4.0, pct_done))),
                step=f"[{idx}/{len(plan_files)}] {short_name} listo",
            )

        for art in defn.get("artifacts", []):
            hf_local_cache = m_dir / art["subdir"] / ".cache"
            if hf_local_cache.exists():
                shutil.rmtree(hf_local_cache, ignore_errors=True)

        actual_gb = self.disk_usage_gb(model_id)
        self.marker_file(model_id).write_text(
            json.dumps(
                {
                    "model_id": model_id,
                    "completed_at": time.time(),
                    "actual_disk_gb": actual_gb,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        if not self.is_downloaded(model_id):
            raise RuntimeError("Faltan archivos clave tras la descarga; no se marca como válido.")

        job.log(
            f"Descarga en disco verificada: «{defn['name']}» ({actual_gb:.2f} GB).",
            level="success",
            progress=_scale_pct(100.0),
            step="Archivos verificados en disco",
        )
        return {"model_id": model_id, "actual_disk_gb": actual_gb}

    def delete_model(
        self,
        model_id: str,
        unload_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        defn = self.get_definition(model_id)
        if not defn:
            raise ValueError(f"Modelo desconocido: {model_id}")
        if defn.get("builtin_no_download"):
            raise ValueError("Este motor está integrado y no ocupa espacio en disco.")

        if unload_callback:
            unload_callback(model_id)

        m_dir = self.model_dir(model_id)
        before_bytes = _dir_size_bytes(m_dir)
        if m_dir.exists():
            shutil.rmtree(m_dir)
        freed_gb = round(before_bytes / (1024**3), 2)
        free_now_gb = round(shutil.disk_usage(MODELS_DIR).free / (1024**3), 2)
        msg = f"«{defn['name']}» eliminado del disco ({freed_gb} GB liberados). Espacio libre actual: {free_now_gb} GB."
        colab_console_log(msg, icon="🗑️")
        return {
            "ok": True,
            "model_id": model_id,
            "freed_gb": freed_gb,
            "free_disk_gb": free_now_gb,
            "message": msg,
        }


catalog_manager = CatalogManager()


def search_huggingface_live(
    query: str = "",
    category: str = "video",
    gpu_profile: str = "auto",
    only_compatible: bool = False,
    detected_gpu: Optional[Dict[str, Any]] = None,
    limit: int = 18,
) -> Dict[str, Any]:
    return catalog_manager.search_huggingface_live(
        query=query,
        category=category,
        gpu_profile=gpu_profile,
        only_compatible=only_compatible,
        detected_gpu=detected_gpu,
        limit=limit,
    )

