import os
from pathlib import Path


def is_colab_env() -> bool:
    if os.environ.get("STUDIO_FORCE_LOCAL") == "1":
        return False
    return bool(
        os.environ.get("COLAB_RELEASE_TAG")
        or os.environ.get("COLAB_GPU") is not None
        or Path("/var/colab").exists()
    )


def get_data_root() -> Path:
    env_override = os.environ.get("STUDIO_DATA_DIR")
    if env_override:
        root = Path(env_override).expanduser().resolve()
    elif is_colab_env() and Path("/content").exists():
        root = Path("/content/studio_data")
    else:
        root = Path(__file__).resolve().parent.parent / "studio_data"
    return root


DATA_ROOT = get_data_root()
MODELS_DIR = DATA_ROOT / "models"
UPLOADS_DIR = DATA_ROOT / "uploads"
OUTPUTS_DIR = DATA_ROOT / "outputs"
IMAGES_OUT_DIR = OUTPUTS_DIR / "images"
VIDEOS_OUT_DIR = OUTPUTS_DIR / "videos"
AUDIO_OUT_DIR = OUTPUTS_DIR / "audio"
MONTAGES_OUT_DIR = OUTPUTS_DIR / "montages"
TEMP_DIR = DATA_ROOT / "temp"
STATE_DIR = DATA_ROOT / "state"


def ensure_directories() -> None:
    for d in (
        DATA_ROOT,
        MODELS_DIR,
        UPLOADS_DIR,
        OUTPUTS_DIR,
        IMAGES_OUT_DIR,
        VIDEOS_OUT_DIR,
        AUDIO_OUT_DIR,
        MONTAGES_OUT_DIR,
        TEMP_DIR,
        STATE_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)


ensure_directories()
