"""Strict LoRA loader for the Studio LTX voice experiment (ComfyUI custom node)."""
import folder_paths
import comfy.lora
import comfy.lora_convert
import comfy.utils

class StudioStrictVoiceLoRA:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'model': ('MODEL',), 'lora_name': (folder_paths.get_filename_list('loras'),),
                             'strength_model': ('FLOAT', {'default': 1.0, 'min': 0.01, 'max': 2.0})}}
    RETURN_TYPES = ('MODEL',)
    FUNCTION = 'load'
    CATEGORY = 'Studio/voice'

    def load(self, model, lora_name, strength_model):
        path = folder_paths.get_full_path_or_raise('loras', lora_name)
        raw = comfy.utils.load_torch_file(path, safe_load=True)
        weights = comfy.lora_convert.convert_lora(raw)
        key_map = comfy.lora.model_lora_keys_unet(model.model, {})
        patches = comfy.lora.load_lora(weights, key_map, log_missing=False)
        covered = set()
        for adapter in patches.values():
            covered.update(getattr(adapter, 'loaded_keys', ()))
        missing = sorted(set(weights) - covered)
        if missing or not patches:
            raise RuntimeError('STUDIO_LORA_INCOMPATIBLE: tensores no reconocidos: ' + ', '.join(missing[:8]))
        patched = model.clone()
        applied = set(patched.add_patches(patches, strength_model))
        unapplied = set(patches) - applied
        if unapplied:
            raise RuntimeError('STUDIO_LORA_INCOMPATIBLE: parches no aplicados: ' + ', '.join(map(str, list(unapplied)[:8])))
        print(f'Studio: ID-LoRA verificado, {len(patches)} parches registrados', flush=True)
        return (patched,)

NODE_CLASS_MAPPINGS = {'StudioStrictVoiceLoRA': StudioStrictVoiceLoRA}
NODE_DISPLAY_NAME_MAPPINGS = {'StudioStrictVoiceLoRA': 'Studio ID-LoRA estricto'}
