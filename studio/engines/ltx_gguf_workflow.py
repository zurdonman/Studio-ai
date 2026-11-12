"""Experimental ComfyUI LTX-2.3 GGUF graph construction.

Graph-only: no model download or GPU execution. A validator must run against the
pinned ComfyUI and GGUF plugin; valid node types are not inference validation.
"""
from __future__ import annotations

import math


SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"


def make_graph(*, prompt: str, negative_prompt: str, width: int, height: int,
               frames: int, fps: float, seed: int, image_name: str | None,
               model: str, text_encoder: str, connectors: str,
               video_vae: str, audio_vae: str, image_strength: float = 1.0) -> dict:
    if not prompt.strip():
        raise ValueError("El prompt no puede estar vacío")
    if width < 256 or height < 256 or width % 32 or height % 32 or width * height > 768 * 512:
        raise ValueError("Resolución experimental: múltiplos de 32, mínimo 256 y máximo 768×512 píxeles")
    if frames < 9 or frames > 601 or (frames - 1) % 8 or not math.isfinite(fps) or not 1 <= fps <= 30:
        raise ValueError("Frames deben cumplir 8n+1 y ser 9–601; fps entre 1 y 30")
    if not math.isfinite(image_strength) or not 0.5 <= image_strength <= 1.0:
        raise ValueError("La fidelidad del primer fotograma debe estar entre 0.5 y 1")
    if not 0 <= seed <= 2**63 - 1:
        raise ValueError("Semilla fuera de rango")
    for name in [model, text_encoder, connectors, video_vae, audio_vae, *([image_name] if image_name else [])]:
        if not isinstance(name, str) or not name or '/' in name or '\\' in name or name in ('.', '..'):
            raise ValueError("Nombres de archivos de modelo o imagen no válidos")
    def n(kind, **inputs):
        return {"class_type": kind, "inputs": inputs}
    graph = {
        "1": n("UnetLoaderGGUF", unet_name=model),
        "2": n("DualCLIPLoaderGGUF", clip_name1=text_encoder, clip_name2=connectors, type="ltxv"),
        "3": n("VAELoader", vae_name=video_vae),
        "4": n("VAELoader", vae_name=audio_vae),
        "5": n("CLIPTextEncode", text=prompt, clip=["2", 0]),
        "6": n("CLIPTextEncode", text=negative_prompt, clip=["2", 0]),
        "7": n("LTXVConditioning", positive=["5", 0], negative=["6", 0], frame_rate=fps),
        "8": n("EmptyLTXVLatentVideo", width=width, height=height, length=frames, batch_size=1),
        "9": n("LTXVEmptyLatentAudio", frames_number=frames, frame_rate=fps, batch_size=1, audio_vae=["4", 0]),
        "10": n("LTXVConcatAVLatent", video_latent=["8", 0], audio_latent=["9", 0]),
        "11": n("CFGGuider", model=["1", 0], positive=["7", 0], negative=["7", 1], cfg=1.0),
        "12": n("RandomNoise", noise_seed=seed),
        "13": n("KSamplerSelect", sampler_name="lcm"),
        "14": n("ManualSigmas", sigmas=SIGMAS),
        "15": n("SamplerCustomAdvanced", noise=["12", 0], guider=["11", 0], sampler=["13", 0], sigmas=["14", 0], latent_image=["10", 0]),
        "16": n("LTXVSeparateAVLatent", av_latent=["15", 0]),
        "17": n("VAEDecode", samples=["16", 0], vae=["3", 0]),
        "18": n("LTXVAudioVAEDecode", samples=["16", 1], audio_vae=["4", 0]),
        "19": n("CreateVideo", images=["17", 0], audio=["18", 0], fps=fps),
        "20": n("SaveVideo", video=["19", 0], filename_prefix="studio_gguf", format="auto", codec="auto"),
    }
    if image_name:
        graph["21"] = n("LoadImage", image=image_name)
        graph["22"] = n("LTXVImgToVideoInplace", vae=["3", 0], image=["21", 0], latent=["8", 0], strength=image_strength, bypass=False)
        graph["10"]["inputs"]["video_latent"] = ["22", 0]
    return graph
