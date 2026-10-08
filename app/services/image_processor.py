from __future__ import annotations

from dataclasses import dataclass
import io
import json
import logging
from pathlib import Path
import time
from typing import Any, Callable, Sequence
from uuid import uuid4
import zipfile

from PIL import Image

from app.config import Settings
from app.services.detector import Detection
from app.utils.image_utils import center_without_resize, draw_detections, expand_bbox
from app.utils.validation import ImageValidationError, load_validated_image

LOGGER = logging.getLogger(__name__)


class ProcessingError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message

    def as_dict(self) -> dict[str, Any]:
        return {"success": False, "error_code": self.code, "message": self.message}


@dataclass(slots=True)
class DetectionResult:
    image: Image.Image
    preview: Image.Image
    detections: list[Detection]
    source_stem: str
    detection_method: str = "yolo"


@dataclass(slots=True)
class ProcessingResult:
    output: Image.Image
    preview: Image.Image
    metadata: dict[str, Any]
    saved_path: Path | None = None

    def png_bytes(self) -> bytes:
        buffer = io.BytesIO()
        self.output.save(buffer, format="PNG")
        return buffer.getvalue()


@dataclass(slots=True)
class BatchItemResult:
    source_name: str
    success: bool
    saved_path: Path | None = None
    metadata: dict[str, Any] | None = None
    error_code: str | None = None
    message: str | None = None

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "source_name": self.source_name,
            "success": self.success,
        }
        if self.saved_path is not None:
            result["saved_path"] = str(self.saved_path)
        if self.metadata is not None:
            result["metadata"] = self.metadata
        if self.error_code is not None:
            result["error_code"] = self.error_code
        if self.message is not None:
            result["message"] = self.message
        return result


@dataclass(slots=True)
class BatchProcessingResult:
    items: list[BatchItemResult]
    archive_path: Path | None

    @property
    def success_count(self) -> int:
        return sum(item.success for item in self.items)

    @property
    def failure_count(self) -> int:
        return len(self.items) - self.success_count

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": len(self.items),
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "archive_path": str(self.archive_path) if self.archive_path else None,
            "items": [item.as_dict() for item in self.items],
        }


class ImageProcessor:
    def __init__(self, settings: Settings, detector: Any, background_remover: Any) -> None:
        self.settings = settings
        self.detector = detector
        self.background_remover = background_remover

    def detect(self, source: str | Path | bytes) -> DetectionResult:
        try:
            image = load_validated_image(source, self.settings)
        except ImageValidationError as exc:
            raise ProcessingError("INVALID_IMAGE", str(exc)) from exc
        try:
            detections = self.detector.detect(image)
        except (RuntimeError, ValueError, OSError) as exc:
            raise ProcessingError("DETECTION_FAILED", str(exc)) from exc
        detection_method = "yolo"
        if not detections:
            # The input contract guarantees one animal per image. If YOLO's
            # confidence is too low, let BiRefNet segment the complete frame
            # instead of dropping an otherwise valid image.
            width, height = image.size
            detections = [
                Detection(
                    class_name="animal",
                    confidence=0.0,
                    bbox=(0.0, 0.0, float(width), float(height)),
                )
            ]
            detection_method = "full_image_fallback"
            LOGGER.warning(
                "[DETECT] no YOLO detection; using full image bbox=%s",
                detections[0].bbox,
            )
        source_stem = Path(source).stem if isinstance(source, (str, Path)) else "animal_result"
        return DetectionResult(
            image,
            draw_detections(image, detections),
            detections,
            source_stem,
            detection_method,
        )

    def process_detection(
        self,
        detected: DetectionResult,
        selected_index: int = 0,
        save_output: bool = False,
        save_directory: str | Path | None = None,
    ) -> ProcessingResult:
        started = time.perf_counter()
        if selected_index < 0 or selected_index >= len(detected.detections):
            raise ProcessingError("INVALID_SELECTION", "Đối tượng được chọn không hợp lệ.")
        detection = detected.detections[selected_index]
        try:
            padded_bbox = expand_bbox(detection.bbox, detected.image.size, self.settings.bbox_padding)
            LOGGER.info("[CROP] bbox=%s padded_bbox=%s", detection.bbox, padded_bbox)
            crop = detected.image.crop(padded_bbox)
            removed = self.background_remover.remove(crop)
            LOGGER.info("[POST_PROCESS] preserve soft alpha; trim empty boundary")
            output = center_without_resize(removed, detected.image.size)
            LOGGER.info("[CENTER] output_size=%s scale=unchanged", output.size)
        except ProcessingError:
            raise
        except (RuntimeError, ValueError, OSError, MemoryError) as exc:
            code = "OUT_OF_MEMORY" if isinstance(exc, MemoryError) else "BACKGROUND_REMOVAL_FAILED"
            raise ProcessingError(code, str(exc)) from exc

        elapsed = time.perf_counter() - started
        preview = draw_detections(detected.image, detected.detections, selected_index)
        metadata = {
            "success": True,
            "detection_method": detected.detection_method,
            "class_name": detection.class_name,
            "confidence": round(detection.confidence, 4),
            "bbox": [round(value, 2) for value in detection.bbox],
            "padded_bbox": list(padded_bbox),
            "original_size": list(detected.image.size),
            "crop_size": list(crop.size),
            "output_size": list(output.size),
            "resized": False,
            "processing_seconds": round(elapsed, 3),
        }
        saved_path = None
        if save_output:
            safe_stem = Path(detected.source_stem).name or "animal_result"
            target_directory = Path(save_directory) if save_directory is not None else self.settings.output_directory
            saved_path = target_directory / f"{safe_stem}.png"
            try:
                saved_path.parent.mkdir(parents=True, exist_ok=True)
                output.save(saved_path, format="PNG")
                LOGGER.info("[SAVE] path=%s", saved_path)
                metadata["saved_path"] = str(saved_path)
            except OSError as exc:
                raise ProcessingError("SAVE_FAILED", f"Không thể lưu ảnh: {exc}") from exc
        return ProcessingResult(output, preview, metadata, saved_path)

    def process_batch(
        self,
        sources: Sequence[str | Path],
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> BatchProcessingResult:
        """Process every uploaded image, continuing when an individual file fails."""
        if not sources:
            raise ProcessingError("EMPTY_BATCH", "Vui lòng chọn một thư mục có ảnh.")

        batch_id = f"batch_{time.strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
        batch_directory = self.settings.output_directory / batch_id
        items: list[BatchItemResult] = []
        used_stems: dict[str, int] = {}
        total = len(sources)

        for index, source in enumerate(sources, start=1):
            source_path = Path(source)
            source_name = source_path.name
            try:
                detected = self.detect(source_path)
                base_stem = Path(detected.source_stem).name or "animal_result"
                occurrence = used_stems.get(base_stem, 0) + 1
                used_stems[base_stem] = occurrence
                detected.source_stem = base_stem if occurrence == 1 else f"{base_stem}_{occurrence}"
                result = self.process_detection(
                    detected,
                    selected_index=0,
                    save_output=True,
                    save_directory=batch_directory,
                )
                items.append(
                    BatchItemResult(
                        source_name=source_name,
                        success=True,
                        saved_path=result.saved_path,
                        metadata=result.metadata,
                    )
                )
            except ProcessingError as exc:
                LOGGER.warning("[BATCH] source=%s code=%s message=%s", source_name, exc.code, exc.message)
                items.append(
                    BatchItemResult(
                        source_name=source_name,
                        success=False,
                        error_code=exc.code,
                        message=exc.message,
                    )
                )
            finally:
                if progress_callback is not None:
                    progress_callback(index, total, source_name)

        successful_paths = [item.saved_path for item in items if item.saved_path is not None]
        archive_path = None
        if successful_paths:
            batch_directory.mkdir(parents=True, exist_ok=True)
            report_path = batch_directory / "report.json"
            report = {
                "total": len(items),
                "success_count": sum(item.success for item in items),
                "failure_count": sum(not item.success for item in items),
                "items": [item.as_dict() for item in items],
            }
            try:
                report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
                archive_path = self.settings.output_directory / f"{batch_id}.zip"
                with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                    for saved_path in successful_paths:
                        archive.write(saved_path, arcname=saved_path.name)
                    archive.write(report_path, arcname=report_path.name)
                LOGGER.info("[SAVE] batch_archive=%s", archive_path)
            except OSError as exc:
                raise ProcessingError("SAVE_FAILED", f"Không thể tạo file ZIP: {exc}") from exc

        return BatchProcessingResult(items=items, archive_path=archive_path)

