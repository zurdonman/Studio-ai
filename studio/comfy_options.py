"""Allowlisted startup policies for the isolated, pinned ComfyUI runtimes.

These are Comfy policies, not independent on/off switches for two Diffusers
hooks. Automatic Comfy may partially offload and unload models as needed.
"""
from typing import Any

MODES = ('auto', 'lowvram', 'novram', 'highvram')


def options(mode: str = 'auto', evict_models: bool = False) -> tuple[str, bool]:
    if mode not in MODES or type(evict_models) is not bool:
        raise ValueError('Opciones de memoria Comfy no válidas.')
    if mode == 'highvram' and evict_models:
        raise ValueError('No se puede combinar retener modelos con descarga agresiva.')
    return mode, evict_models


def job_options(job: Any) -> tuple[str, bool]:
    return options(*getattr(job, 'comfy_options', ('auto', False)))


def flags(config: tuple[str, bool]) -> list[str]:
    mode, evict = options(*config)
    # Both pinned Comfy versions say --lowvram is ineffective with dynamic
    # VRAM. Disable dynamic VRAM explicitly for the requested legacy policy.
    mode_flags = {
        'auto': [],
        'lowvram': ['--disable-dynamic-vram', '--lowvram'],
        'novram': ['--novram'],
        'highvram': ['--highvram'],
    }
    return mode_flags[mode] + (['--disable-smart-memory'] if evict else [])
