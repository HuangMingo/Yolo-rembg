from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from . import parameters


def _default_output_directory() -> Path:
    """Resolve a shared animal-output folder without a machine-specific path."""
    for parent in Path(__file__).resolve().parents:
        if parent.name.casefold() == "btl":
            return parent / "remove_background image" / "animals"
    return Path.cwd() / "output" / "animals"


@dataclass(frozen=True, slots=True)
class Settings:
    """Validated application settings sourced from config/parameters.py."""

    yolo_model: str = parameters.YOLO_MODEL
    # A low default helps retain open-vocabulary detections. It remains
    # configurable because score distributions differ between animal classes.
    confidence: float = parameters.YOLO_CONFIDENCE
    iou_threshold: float = parameters.YOLO_IOU
    device: str = parameters.YOLO_DEVICE
    allowed_classes: tuple[str, ...] = parameters.YOLO_CLASSES
    bbox_padding: float = parameters.BBOX_PADDING

    rembg_model: str = parameters.REMBG_MODEL
    alpha_matting: bool = parameters.ALPHA_MATTING
    alpha_foreground_threshold: int = parameters.ALPHA_FOREGROUND_THRESHOLD
    alpha_background_threshold: int = parameters.ALPHA_BACKGROUND_THRESHOLD
    alpha_erode_size: int = parameters.ALPHA_ERODE_SIZE

    output_directory: Path = field(
        default_factory=lambda: (
            Path(parameters.OUTPUT_DIRECTORY)
            if parameters.OUTPUT_DIRECTORY is not None
            else _default_output_directory()
        )
    )
    max_upload_mb: int = parameters.MAX_UPLOAD_MB
    max_image_pixels: int = parameters.MAX_IMAGE_PIXELS

    def __post_init__(self) -> None:
        if not 0 < self.confidence <= 1:
            raise ValueError("YOLO_CONFIDENCE must be in (0, 1].")
        if not 0 <= self.iou_threshold <= 1:
            raise ValueError("YOLO_IOU must be in [0, 1].")
        if not 0 <= self.bbox_padding <= 1:
            raise ValueError("BBOX_PADDING must be in [0, 1].")

