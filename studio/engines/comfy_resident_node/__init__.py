"""Isolated Studio LTX resident mode; not enabled in general ComfyUI installs."""
import logging
from aiohttp import web
import execution
from server import PromptServer
from .core import prune_completed

VERSION = 1
status = {'version': VERSION, 'prompt_id': None, 'ok': False}
original_execute = execution.PromptExecutor.execute


def execute_and_prune(self, prompt, prompt_id, *args, **kwargs):
    status.update(prompt_id=prompt_id, ok=False, error='execution incomplete')
    result = original_execute(self, prompt, prompt_id, *args, **kwargs)
    if not self.success:
        return result
    try:
        summary = prune_completed(self, prompt)
        if len(prompt) == 1 and prompt.get('1', {}).get('class_type') == 'StudioResidentReset':
            import comfy.model_management as mm
            mm.unload_all_models()
            mm.soft_empty_cache()
        status.update(prompt_id=prompt_id, ok=True, error=None, **summary)
    except Exception as exc:
        logging.exception('Studio: failed to prune intermediate outputs')
        status.update(prompt_id=prompt_id, ok=False, error=str(exc)[:300])
    return result


execution.PromptExecutor.execute = execute_and_prune


@PromptServer.instance.routes.get('/studio-resident/status')
async def resident_status(request):
    return web.json_response(status)


class StudioResidentCleanup:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {}}
    RETURN_TYPES = ()
    FUNCTION = 'noop'
    CATEGORY = 'Studio'
    def noop(self):
        return ()


class StudioResidentReset:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {}}
    RETURN_TYPES = ()
    FUNCTION = 'reset'
    OUTPUT_NODE = True
    CATEGORY = 'Studio'
    def reset(self):
        return ()


NODE_CLASS_MAPPINGS = {'StudioResidentCleanup': StudioResidentCleanup, 'StudioResidentReset': StudioResidentReset}
