"""Fit motion references into the output canvas without cropping their content."""
from pathlib import Path

from PIL import Image, ImageOps


def fit_reference_image(source: Path, destination: Path, width: int, height: int) -> None:
    """Apply EXIF orientation and letterbox before Comfy's center resize."""
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image).convert('RGB')
        result = ImageOps.pad(image, (width, height), method=Image.Resampling.LANCZOS,
                              color=(127, 127, 127), centering=(0.5, 0.5))
        result.save(destination)


def fit_video_filter(width: int, height: int) -> str:
    """Return FFmpeg filter preserving source aspect; dimensions are prevalidated."""
    if not isinstance(width, int) or not isinstance(height, int) or width < 2 or height < 2 or width % 2 or height % 2:
        raise ValueError('MOTION_FRAMING_INVALID')
    return (f'scale={width}:{height}:force_original_aspect_ratio=decrease:force_divisible_by=2,'
            f'pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=0x7f7f7f,setsar=1')
