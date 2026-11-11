"""Per-user production preferences for the MCP plan, kept across sessions.

These are the decisions an agent must not invent (fast/speed LoRAs, clip length,
aspect ratio, voices, ambience). They live in ``<data>/mcp_preferences.json`` so a
restarted Colab instance can reuse them without re-downloading anything.

Only preferences are stored: never model weights, tokens or media files.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from studio.config import DATA_ROOT

_PREF_LOCK = threading.RLock()
PREF_FILE: Path = DATA_ROOT / "mcp_preferences.json"

# Agent-facing guide: the question to ask once, the recommended default and why.
PREFERENCE_QUESTIONS: Dict[str, Dict[str, Any]] = {
    "use_fast_loras": {
        "label": "LoRAs de generación rápida (FusionX / Phantom / 6 pasos)",
        "recommended": True,
        "options": [True, False],
        "why": (
            "Puede reducir los pasos en recetas compatibles, pero la velocidad y calidad deben medirse "
            "en la GPU real; un LoRA no garantiza identidad ni tiempo de generación."
        ),
        "ask_once": "¿Quieres que use los LoRAs de generación rápida (FusionX Phantom 6 pasos) en los clips?",
    },
    "clip_duration_sec": {
        "label": "Duración de cada clip antes de unir y sincronizar labios",
        "recommended": 5.0,
        "options": [5.0, 6.0, 8.0],
        "why": "5 s es lo que el preset valida; una petición de 30 s se compone de 6 clips.",
        "ask_once": "¿Mantengo clips de 5 s (6 clips para 30 s) o los prefieres más largos?",
    },
    "aspect_ratio": {
        "label": "Formato del vídeo",
        "recommended": "16:9",
        "options": ["16:9", "9:16", "1:1"],
        "why": "16:9 es el bucket nativo de Wan 2.1 VACE en L4 (832×480).",
        "ask_once": "¿Formato horizontal 16:9, vertical 9:16 o cuadrado?",
    },
    "voice_pair": {
        "label": "Voces para el diálogo",
        "recommended": ["es-ES-AlvaroNeural", "es-ES-ElviraNeural"],
        "options": [
            ["es-ES-AlvaroNeural", "es-ES-ElviraNeural"],
            ["es-ES-XimenaNeural", "es-ES-AlvaroNeural"],
        ],
        "why": "Dos voces distintas permiten alternar turnos y sincronizar labios por personaje.",
        "ask_once": "¿Te sirven las voces por defecto (Álvaro/Elvira) para los dos personajes?",
    },
}


def _read() -> Dict[str, Any]:
    try:
        data = json.loads(PREF_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write(data: Dict[str, Any]) -> None:
    PREF_FILE.parent.mkdir(parents=True, exist_ok=True)
    PREF_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def get_preferences() -> Dict[str, Any]:
    with _PREF_LOCK:
        return _read()


def get_preference(key: str, default: Any = None) -> Any:
    prefs = get_preferences()
    return prefs[key] if key in prefs else default


def set_preferences(updates: Dict[str, Any]) -> Dict[str, Any]:
    """Store boolean/number/string preferences only; unknown keys are rejected."""
    clean: Dict[str, Any] = {}
    for key, value in (updates or {}).items():
        if key not in PREFERENCE_QUESTIONS:
            continue
        expected = PREFERENCE_QUESTIONS[key]
        if isinstance(expected["recommended"], bool):
            clean[key] = bool(value)
        elif isinstance(expected["recommended"], (int, float)) and not isinstance(value, bool):
            try:
                clean[key] = float(value)
            except (TypeError, ValueError):
                continue
        elif isinstance(expected["recommended"], list) and key == "voice_pair":
            if isinstance(value, (list, tuple)) and len(value) == 2 and all(str(v).strip() for v in value):
                clean[key] = [str(v).strip() for v in value]
        elif isinstance(value, str) and value.strip():
            clean[key] = value.strip()
    with _PREF_LOCK:
        data = _read()
        data.update(clean)
        _write(data)
        return data


def fast_loras_enabled(default: bool = True) -> bool:
    value = get_preference("use_fast_loras", default)
    return bool(value)


def pending_questions(keys: Optional[list] = None) -> list:
    """Questions still unanswered by this user (agent asks them once, then remembers)."""
    prefs = get_preferences()
    wanted = keys or list(PREFERENCE_QUESTIONS)
    return [
        {
            "key": key,
            "question": PREFERENCE_QUESTIONS[key]["ask_once"],
            "recommended": PREFERENCE_QUESTIONS[key]["recommended"],
            "options": PREFERENCE_QUESTIONS[key]["options"],
            "why": PREFERENCE_QUESTIONS[key]["why"],
        }
        for key in wanted
        if key in PREFERENCE_QUESTIONS and key not in prefs
    ]


def guidance_block() -> Dict[str, Any]:
    """Short, agent-readable contract: ask once, store, never re-ask."""
    return {
        "how_to_ask": (
            "Pregunta UNA sola vez, con la opción recomendada ya marcada, y guarda la respuesta con "
            "`studio_set_production_preferences`. En las siguientes tareas reutiliza lo guardado: no vuelvas a preguntar."
        ),
        "pending_questions": pending_questions(),
        "saved": get_preferences(),
        "recommended_env_profile": "production_fast" if fast_loras_enabled() else "production_quality",
    }
