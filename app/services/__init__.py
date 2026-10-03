from .background_remover import BackgroundRemover
from .detector import AnimalDetector, Detection, TigerDetector
from .image_processor import (
    BatchItemResult,
    BatchProcessingResult,
    ImageProcessor,
    ProcessingError,
    ProcessingResult,
)

__all__ = [
    "BackgroundRemover",
    "BatchItemResult",
    "BatchProcessingResult",
    "AnimalDetector",
    "Detection",
    "ImageProcessor",
    "ProcessingError",
    "ProcessingResult",
    "TigerDetector",
]

