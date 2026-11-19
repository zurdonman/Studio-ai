"""World catalog with current configuration status."""
from studio.scene_catalog import WORLDS
from studio.engines.scene_jobs import runtime_status


def catalog():
    return [{**m, **runtime_status(m['id'])} for m in WORLDS]
