"""Experimental LTX 2.3 voice graph and fixed adapter manifest."""
from __future__ import annotations

import math

LORA = ('voice_lora', 'AviadDahan/LTX-2.3-ID-LoRA-TalkVid-3K',
        '6eabedadefbcfa6c95092d50006629d9413102fd', 'lora_weights.safetensors',
        1157884304, 'e5af73441743b4852f228b03e444888dff3da80d2666033af2367ab7bda6d8b9')
LORA_LINK = 'studio-ltx23-id-lora-talkvid.safetensors'


def frames_for_audio(seconds: float, fps: int) -> int:
    if not math.isfinite(seconds) or not 1 <= seconds <= 20 or not 8 <= fps <= 30:
        raise ValueError('Audio completo: duración entre 1 y 20 s y fps entre 8 y 30')
    # probe_media_info reports hundredths of a second: account for rounding so
    # the final tail is never clipped at an 8n+1 boundary.
    safe_seconds = min(20.0, seconds + 0.01)
    return max(9, 8 * math.ceil((safe_seconds * fps - 1) / 8) + 1)


def add_voice_graph(graph: dict, mode: str, audio_name: str, lora_name: str = LORA_LINK) -> dict:
    if mode not in ('audio', 'voice_reference'):
        raise ValueError('Modo de voz desconocido')
    for name in (audio_name, lora_name):
        if not name or '/' in name or '\\' in name or name in ('.', '..'):
            raise ValueError('Nombre de archivo de voz no válido')
    def node(kind, **inputs): return {'class_type': kind, 'inputs': inputs}
    graph['30'] = node('LoadAudio', audio=audio_name)
    if mode == 'audio':
        graph['31'] = node('LTXVAudioVAEEncode', audio=['30', 0], audio_vae=['4', 0])
        graph['32'] = node('SolidMask', value=0.0, width=64, height=64)
        graph['33'] = node('SetLatentNoiseMask', samples=['31', 0], mask=['32', 0])
        graph['10']['inputs']['audio_latent'] = ['33', 0]
        # Do not leave an unused empty audio node in the graph.
        graph.pop('9', None)
    else:
        graph['31'] = node('StudioStrictVoiceLoRA', model=['1', 0], lora_name=lora_name, strength_model=1.0)
        graph['32'] = node('LTXVReferenceAudio', model=['31', 0], positive=['7', 0], negative=['7', 1],
                           reference_audio=['30', 0], audio_vae=['4', 0], identity_guidance_scale=3.0,
                           start_percent=0.0, end_percent=1.0)
        graph['11']['inputs'].update(model=['32', 0], positive=['32', 1], negative=['32', 2])
    return graph
