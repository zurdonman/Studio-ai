"""Prune a completed Studio graph, retaining only its model loader outputs.

Pinned Comfy internals, called synchronously on the prompt worker after execute.
This is NOT a general Comfy cache cleaner and must never run during execution.
"""
import gc

LOADERS = {
    '1': {'UnetLoaderGGUF', 'UNETLoader'},
    '2': {'DualCLIPLoaderGGUF', 'CLIPLoaderGGUF', 'CLIPLoader'},
    '3': {'VAELoader'},
    '4': {'VAELoader'},
}


def prune_completed(executor, prompt):
    if not isinstance(prompt, dict):
        raise ValueError('not a recognized Studio graph')
    classes = {key: node.get('class_type') for key, node in prompt.items() if isinstance(node, dict)}
    if classes.get('14') in ('WanAnimate2ToVideo', 'WanSCAILToVideo'):
        loaders = {'1': {'StudioWanAnimate2GGUF', 'UnetLoaderGGUF', 'UNETLoader'},
                   '2': {'CLIPLoader'}, '3': {'VAELoader'}, '4': {'CLIPVisionLoader'}}
        if classes['14'] == 'WanSCAILToVideo':
            loaders['20'] = {'CheckpointLoaderSimple'}
    else:
        loaders = LOADERS
    if any(classes.get(node) not in allowed for node, allowed in loaders.items()):
        raise ValueError('not a recognized Studio graph')
    outputs, objects = executor.caches.outputs, executor.caches.objects
    if not getattr(outputs, 'initialized', False) or not hasattr(outputs, 'cache_key_set'):
        raise RuntimeError('unsupported Comfy output cache')
    if not isinstance(outputs.cache, dict) or not isinstance(objects.cache, dict):
        raise RuntimeError('unsupported Comfy cache structure')
    keys = {outputs.cache_key_set.get_data_key(node) for node in loaders}
    if None in keys or len(keys) != len(loaders):
        raise RuntimeError('model loader cache keys missing or ambiguous')
    kept = sum(key in outputs.cache for key in keys)
    for key in list(outputs.cache):
        if key not in keys:
            del outputs.cache[key]
    # RAMPressureCache/LRU bookkeeping must agree with the retained dictionary.
    for attr in ('used_generation', 'timestamps', 'children'):
        data = getattr(outputs, attr, None)
        if isinstance(data, dict):
            for key in list(data):
                if key not in keys:
                    del data[key]
    outputs.subcaches.clear()
    objects.cache.clear()
    objects.subcaches.clear()
    # Drop the old DynamicPrompt (including user inputs) while preserving the
    # signature-keyed loader entries; set_prompt rebuilds these on the next job.
    for cache in (outputs, objects):
        cache.initialized = False
        for attr in ('dynprompt', 'cache_key_set', 'is_changed_cache'):
            if hasattr(cache, attr):
                setattr(cache, attr, None)
    gc.collect()
    return {'retained_loaders': kept, 'total_loaders': len(loaders)}
