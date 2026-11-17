"""Narrow adapter for the pinned GGUF loader's missing Wan-Animate-2 metadata.

The quantizer documents that a stock GGUF loader silently selects ordinary Wan.
We use the existing Apache-licensed loader, inject the missing model config, then
fail closed unless Comfy actually constructed WAN_Animate2. This private Comfy
worker is single-job and never accepts user-supplied graph nodes.
"""
import json
import logging
import nodes


def _fix_gguf_new_empty(reader):
    """Normalize positional dimensions for the pinned GGUF tensor subclass.

    Wan's attention uses x.new_empty(batch, tokens, heads, dim). The pinned
    GGUF override accepts this, but records only `batch` as tensor_shape; later
    shape[1] then fails. Keep the original allocation and quantization metadata,
    changing only the way positional sizes are passed to it.
    """
    tensor_cls = getattr(reader, '__globals__', {}).get('GGMLTensor')
    if not isinstance(tensor_cls, type) or not callable(getattr(tensor_cls, 'new_empty', None)):
        raise RuntimeError('STUDIO_MOTION_GGUF_COMPAT_UNAVAILABLE: no se encontró GGMLTensor del cargador fijado')
    original = tensor_cls.new_empty
    if getattr(original, '_studio_motion_size_fix', False):
        return

    def new_empty(self, size, *args, **kwargs):
        if isinstance(size, int) and all(isinstance(value, int) for value in args):
            size, args = (size, *args), ()
        return original(self, size, *args, **kwargs)

    new_empty._studio_motion_size_fix = True
    tensor_cls.new_empty = new_empty
    logging.info('[StudioMotion] Compatibilidad GGUF new_empty activada para Wan Animate 2')


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
        _fix_gguf_new_empty(original)

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
