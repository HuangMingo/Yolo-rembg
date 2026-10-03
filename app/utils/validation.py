from __future__ import annotations

import io
from pathlib import Path
from typing import BinaryIO

from PIL import Image, ImageOps, UnidentifiedImageError

from app.config import Settings


class ImageValidationError(ValueError):
    """Raised when input bytes do not decode into a supported image."""


SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP"}


def _read_bytes(source: str | Path | bytes | bytearray | BinaryIO) -> bytes:
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise ImageValidationError("Không tìm thấy tệp ảnh.")
        return path.read_bytes()
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    if hasattr(source, "read"):
        position = source.tell() if hasattr(source, "tell") else None
        data = source.read()
        if position is not None and hasattr(source, "seek"):
            source.seek(position)
        return data
    raise ImageValidationError("Kiểu dữ liệu ảnh không được hỗ trợ.")


def load_validated_image(
    source: str | Path | bytes | bytearray | BinaryIO,
    settings: Settings,
) -> Image.Image:
    data = _read_bytes(source)
    if not data:
        raise ImageValidationError("Tệp ảnh rỗng.")
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise ImageValidationError(f"Ảnh vượt quá giới hạn {settings.max_upload_mb} MB.")

    try:
        with Image.open(io.BytesIO(data)) as probe:
            image_format = probe.format
            probe.verify()
        if image_format not in SUPPORTED_FORMATS:
            raise ImageValidationError("Chỉ hỗ trợ JPG, JPEG, PNG và WEBP.")
        with Image.open(io.BytesIO(data)) as decoded:
            decoded.load()
            if decoded.width * decoded.height > settings.max_image_pixels:
                raise ImageValidationError("Độ phân giải ảnh quá lớn.")
            # YOLO and rembg receive the same orientation-corrected RGB pixels.
            return ImageOps.exif_transpose(decoded).convert("RGB")
    except ImageValidationError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageValidationError("Tệp bị hỏng hoặc không phải ảnh hợp lệ.") from exc

