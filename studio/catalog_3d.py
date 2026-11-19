"""3D catalog with current configuration status."""
from studio.scene_catalog import CATALOG_3D
from studio.engines.models_3d import runtime_status
from studio.engines.scene_jobs import runtime_status as scene_status


def catalog():
    return [{**m, **(runtime_status(m['id']) if m['id'] in ('sf3d', 'triposr') else scene_status(m['id']))} for m in CATALOG_3D]
