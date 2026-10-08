from __future__ import annotations

import io
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import zipfile

from PIL import Image, ImageDraw

from app.config import Settings
from app.services.detector import Detection
from app.services.image_processor import ImageProcessor, ProcessingError
from app.utils.image_utils import center_without_resize, expand_bbox, resize_and_center
from app.utils.validation import ImageValidationError, load_validated_image


def image_bytes(size=(320, 180), fmt="JPEG") -> bytes:
    image = Image.new("RGB", size, "#e2e8f0")
    draw = ImageDraw.Draw(image)
    draw.rectangle((80, 20, 250, 170), fill="#d97706")
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


class FakeDetector:
    def __init__(self, detections):
        self.detections = detections

    def detect(self, _image):
        return self.detections


class FakeRemover:
    def remove(self, image):
        rgba = image.convert("RGBA")
        alpha = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(alpha)
        draw.ellipse((5, 3, image.width - 5, image.height - 3), fill=255)
        rgba.putalpha(alpha)
        return rgba


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings()

    def test_default_prompt_covers_all_requested_animals(self):
        self.assertEqual(self.settings.allowed_classes, ("animal",))

    def test_validation_decodes_real_image_content(self):
        image = load_validated_image(image_bytes(), self.settings)
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.size, (320, 180))
        with self.assertRaises(ImageValidationError):
            load_validated_image(b"not an image", self.settings)

    def test_bbox_padding_is_clamped(self):
        self.assertEqual(expand_bbox((0, 10, 100, 90), (120, 100), 0.1), (0, 2, 110, 98))

    def test_resize_preserves_ratio_and_transparency(self):
        source = Image.new("RGBA", (100, 50), (200, 100, 20, 255))
        result = resize_and_center(source, (224, 224), 0.9)
        self.assertEqual(result.mode, "RGBA")
        self.assertEqual(result.size, (224, 224))
        alpha_bbox = result.getchannel("A").getbbox()
        self.assertEqual(alpha_bbox[2] - alpha_bbox[0], 202)
        self.assertIn(alpha_bbox[3] - alpha_bbox[1], (100, 101))
        self.assertEqual(result.getpixel((0, 0))[3], 0)

    def test_center_without_resize_preserves_dimensions(self):
        source = Image.new("RGBA", (100, 50), (200, 100, 20, 255))
        result = center_without_resize(source, (320, 180))
        self.assertEqual(result.size, (320, 180))
        self.assertEqual(result.getchannel("A").getbbox(), (110, 65, 210, 115))

    def test_full_pipeline_selects_largest_by_default(self):
        detections = [
            Detection("tiger", 0.91, (40, 10, 280, 175)),
            Detection("tiger", 0.95, (10, 10, 50, 50)),
        ]
        processor = ImageProcessor(self.settings, FakeDetector(detections), FakeRemover())
        detected = processor.detect(image_bytes())
        result = processor.process_detection(detected)
        self.assertEqual(result.output.size, (320, 180))
        self.assertEqual(result.output.mode, "RGBA")
        self.assertFalse(result.metadata["resized"])
        self.assertEqual(result.metadata["detection_method"], "yolo")
        self.assertEqual(result.metadata["class_name"], "tiger")
        self.assertEqual(result.metadata["bbox"], [40, 10, 280, 175])

    def test_no_detection_falls_back_to_full_image(self):
        processor = ImageProcessor(self.settings, FakeDetector([]), FakeRemover())
        detected = processor.detect(image_bytes())

        self.assertEqual(detected.detection_method, "full_image_fallback")
        self.assertEqual(len(detected.detections), 1)
        self.assertEqual(detected.detections[0].confidence, 0.0)
        self.assertEqual(detected.detections[0].bbox, (0.0, 0.0, 320.0, 180.0))

        result = processor.process_detection(detected)
        self.assertEqual(result.metadata["detection_method"], "full_image_fallback")
        self.assertEqual(result.metadata["crop_size"], [320, 180])
        self.assertEqual(result.output.size, (320, 180))

    def test_save_keeps_source_stem_and_uses_png(self):
        detections = [Detection("tiger", 0.91, (40, 10, 280, 175))]
        with tempfile.TemporaryDirectory() as directory:
            settings = replace(self.settings, output_directory=Path(directory))
            processor = ImageProcessor(settings, FakeDetector(detections), FakeRemover())
            detected = processor.detect(image_bytes())
            # Bytes do not carry a name; emulate the name preserved by Gradio.
            detected.source_stem = "tiger_001"
            result = processor.process_detection(detected, save_output=True)
            expected = Path(directory) / "tiger_001.png"
            self.assertEqual(result.saved_path, expected)
            self.assertTrue(expected.is_file())
            with Image.open(expected) as saved:
                self.assertEqual(saved.mode, "RGBA")
                self.assertEqual(saved.size, (320, 180))

    def test_batch_processes_all_images_and_creates_zip(self):
        detections = [Detection("tiger", 0.91, (40, 10, 280, 175))]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_directory = root / "output"
            first_directory = root / "first"
            second_directory = root / "second"
            first_directory.mkdir()
            second_directory.mkdir()
            first_image = first_directory / "animal.jpg"
            second_image = second_directory / "animal.jpg"
            first_image.write_bytes(image_bytes())
            second_image.write_bytes(image_bytes())

            settings = replace(self.settings, output_directory=output_directory)
            processor = ImageProcessor(settings, FakeDetector(detections), FakeRemover())
            result = processor.process_batch([first_image, second_image])

            self.assertEqual(result.success_count, 2)
            self.assertEqual(result.failure_count, 0)
            self.assertIsNotNone(result.archive_path)
            self.assertTrue(result.archive_path.is_file())
            with zipfile.ZipFile(result.archive_path) as archive:
                self.assertEqual(
                    set(archive.namelist()),
                    {"animal.png", "animal_2.png", "report.json"},
                )

    def test_batch_uses_full_image_when_yolo_finds_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "animal.jpg"
            source.write_bytes(image_bytes())
            settings = replace(self.settings, output_directory=root / "output")
            processor = ImageProcessor(settings, FakeDetector([]), FakeRemover())

            result = processor.process_batch([source])

            self.assertEqual(result.success_count, 1)
            self.assertEqual(result.failure_count, 0)
            self.assertEqual(
                result.items[0].metadata["detection_method"],
                "full_image_fallback",
            )
            self.assertEqual(result.items[0].metadata["crop_size"], [320, 180])
            self.assertTrue(result.archive_path.is_file())

    def test_batch_continues_after_invalid_image(self):
        detections = [Detection("tiger", 0.91, (40, 10, 280, 175))]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid_image = root / "valid.jpg"
            invalid_image = root / "invalid.jpg"
            valid_image.write_bytes(image_bytes())
            invalid_image.write_bytes(b"not an image")

            settings = replace(self.settings, output_directory=root / "output")
            processor = ImageProcessor(settings, FakeDetector(detections), FakeRemover())
            result = processor.process_batch([invalid_image, valid_image])

            self.assertEqual(result.success_count, 1)
            self.assertEqual(result.failure_count, 1)
            self.assertEqual(result.items[0].error_code, "INVALID_IMAGE")
            self.assertTrue(result.items[1].success)
            self.assertIsNotNone(result.archive_path)


if __name__ == "__main__":
    unittest.main()

