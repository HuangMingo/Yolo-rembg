from __future__ import annotations

from dataclasses import dataclass
import logging

from PIL import Image

from app.config import Settings

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Detection:
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


class AnimalDetector:
    """Open-vocabulary YOLO detector configured for supported animals."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        try:
            from ultralytics import YOLOWorld
        except ImportError as exc:
            raise RuntimeError("Thiếu ultralytics. Hãy chạy: pip install -r requirements.txt") from exc

        LOGGER.info("[LOAD] YOLO model=%s", settings.yolo_model)
        self.model = YOLOWorld(settings.yolo_model)
        self.model.set_classes(list(settings.allowed_classes))

    def detect(self, image: Image.Image) -> list[Detection]:
        LOGGER.info("[DETECT] image_size=%s", image.size)
        device = None if self.settings.device == "auto" else self.settings.device
        try:
            results = self.model.predict(
                source=image,
                conf=self.settings.confidence,
                iou=self.settings.iou_threshold,
                device=device,
                verbose=False,
            )
        except (RuntimeError, ValueError, OSError) as exc:
            raise RuntimeError(f"YOLO inference thất bại: {exc}") from exc

        detections: list[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                class_id = int(box.cls.item())
                class_name = str(result.names[class_id]).lower()
                confidence = float(box.conf.item())
                if class_name not in self.settings.allowed_classes:
                    continue
                coords = tuple(float(value) for value in box.xyxy[0].tolist())
                if len(coords) == 4 and coords[2] > coords[0] and coords[3] > coords[1]:
                    detections.append(Detection(class_name, confidence, coords))
        return sorted(detections, key=lambda item: item.area, reverse=True)


# Backward compatibility for code that imported the old project-specific name.
TigerDetector = AnimalDetector

