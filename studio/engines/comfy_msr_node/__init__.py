"""Strict bridge to the pinned Licon MSR extension in the isolated runtime."""
import importlib.util
from pathlib import Path

import comfy.lora
import comfy.lora_convert
import comfy.sd

_source = Path(__file__).resolve().parent.parent.parent.parent / 'studio-msr-source' / 'nodes.py'
_spec = importlib.util.spec_from_file_location('studio_pinned_msr', _source)
_msr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_msr)


def _strict_load(model, clip, lora, strength_model, strength_clip, lora_metadata=None):
    if clip is not None or not strength_model:
        raise RuntimeError('MSR_LORA_INCOMPATIBLE: se necesita un modelo y fuerza no nula')
    converted = comfy.lora_convert.convert_lora(lora)
    mapping = comfy.lora.model_lora_keys_unet(model.model, {})
    patches = comfy.lora.load_lora(converted, mapping, log_missing=False)
    covered = set()
    for adapter in patches.values():
        covered.update(getattr(adapter, 'loaded_keys', ()))
    missing = set(converted) - covered
    if missing or not patches:
        raise RuntimeError('MSR_LORA_INCOMPATIBLE: tensores sin correspondencia: ' + ', '.join(sorted(missing)[:8]))
    loaded = model.clone()
    applied = set(loaded.add_patches(patches, strength_model))
    if set(patches) - applied:
        raise RuntimeError('MSR_LORA_INCOMPATIBLE: algunos parches no se aplicaron al modelo')
    if lora_metadata:
        loaded.set_attachments('lora_metadata', lora_metadata)
    return loaded, None


class StudioStrictMSRLoader(_msr.ComfyUILTX25MSRICLoRALoader):
    @classmethod
    def define_schema(cls):
        schema = super().define_schema()
        schema.node_id = 'StudioStrictMSRLoader'
        return schema

    @classmethod
    def execute(cls, model, lora_name, strength_model):
        # Comfy executes one prompt at a time in this private worker. Restore
        # the upstream function even if validation fails; no other server shares it.
        original = comfy.sd.load_lora_for_models
        try:
            comfy.sd.load_lora_for_models = _strict_load
            return super().execute(model, lora_name, strength_model)
        finally:
            comfy.sd.load_lora_for_models = original


NODE_CLASS_MAPPINGS = {
    'StudioStrictMSRLoader': StudioStrictMSRLoader,
    'ComfyUILTX25MSRMultiReferenceGuide': _msr.ComfyUILTX25MSRMultiReferenceGuide,
}
