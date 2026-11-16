"""Narrow adapter for the pinned GGUF loader's missing Wan-Animate-2 metadata.

The quantizer documents that a stock GGUF loader silently selects ordinary Wan.
We use the existing Apache-licensed loader, inject the missing model config, then
fail closed unless Comfy actually constructed WAN_Animate2. This private Comfy
worker is single-job and never accepts user-supplied graph nodes.
"""
import json
import nodes


class StudioWanAnimate2GGUF:
    @classmethod
    def INPUT_TYPES(cls):
        return nodes.NODE_CLASS_MAPPINGS['UnetLoaderGGUF'].INPUT_TYPES()

    RETURN_TYPES = ('MODEL',)
    FUNCTION = 'load'
    CATEGORY = 'Studio/private'

    def load(self, unet_name):
        loader_cls = nodes.NODE_CLASS_MAPPINGS['UnetLoaderGGUF']
        namespace = loader_cls.load_unet.__globals__
        original = namespace['gguf_sd_loader']

        def load_with_config(path):
            sd, extra = original(path)
            extra = dict(extra or {})
            metadata = dict(extra.get('metadata') or {})
            cfg = json.loads(metadata.get('config') or '{}')
            cfg.setdefault('transformer', {})['model_type'] = 'animate2'
            metadata['config'] = json.dumps(cfg)
            extra['metadata'] = metadata
            return sd, extra

        namespace['gguf_sd_loader'] = load_with_config
        try:
            model = loader_cls().load_unet(unet_name)[0]
        finally:
            namespace['gguf_sd_loader'] = original
        actual = type(model.model).__name__
        if actual != 'WAN_Animate2':
            raise RuntimeError(f'STUDIO_MOTION_WRONG_ARCH: esperado WAN_Animate2, recibido {actual}')
        return (model,)


NODE_CLASS_MAPPINGS = {'StudioWanAnimate2GGUF': StudioWanAnimate2GGUF}
