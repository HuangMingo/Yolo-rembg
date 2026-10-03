from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


BBox = tuple[int, int, int, int]


def expand_bbox(bbox: Sequence[float], image_size: tuple[int, int], padding: float) -> BBox:
    x1, y1, x2, y2 = map(float, bbox)
    image_width, image_height = image_size
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Bounding box không hợp lệ.")
    pad_x = (x2 - x1) * padding
    pad_y = (y2 - y1) * padding
    expanded = (
        max(0, int(np.floor(x1 - pad_x))),
        max(0, int(np.floor(y1 - pad_y))),
        min(image_width, int(np.ceil(x2 + pad_x))),
        min(image_height, int(np.ceil(y2 + pad_y))),
    )
    if expanded[2] <= expanded[0] or expanded[3] <= expanded[1]:
        raise ValueError("Bounding box sau padding bị rỗng.")
    return expanded


def draw_detections(image: Image.Image, detections: Sequence[object], selected_index: int = 0) -> Image.Image:
    preview = image.convert("RGB").copy()
    draw = ImageDraw.Draw(preview)
    font = ImageFont.load_default()
    line_width = max(2, round(min(preview.size) / 220))
    for index, detection in enumerate(detections):
        color = "#22c55e" if index == selected_index else "#f59e0b"
        bbox = tuple(round(value) for value in detection.bbox)
        draw.rectangle(bbox, outline=color, width=line_width)
        label = f"{index + 1}. {detection.class_name} {detection.confidence:.1%}"
        text_box = draw.textbbox((bbox[0], bbox[1]), label, font=font)
        label_y = max(0, bbox[1] - (text_box[3] - text_box[1]) - 6)
        draw.rectangle(
            (bbox[0], label_y, bbox[0] + text_box[2] - text_box[0] + 8, bbox[1]),
            fill=color,
        )
        draw.text((bbox[0] + 4, label_y + 2), label, fill="black", font=font)
    return preview


def trim_transparent(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    bbox = rgba.getchannel("A").getbbox()
    if bbox is None:
        raise ValueError("Mask tách nền rỗng.")
    return rgba.crop(bbox)


def resize_and_center(
    image: Image.Image,
    target_size: tuple[int, int],
    fill_ratio: float,
) -> Image.Image:
    rgba = trim_transparent(image)
    target_width, target_height = target_size
    available_width = max(1, round(target_width * fill_ratio))
    available_height = max(1, round(target_height * fill_ratio))
    scale = min(available_width / rgba.width, available_height / rgba.height)
    new_size = (max(1, round(rgba.width * scale)), max(1, round(rgba.height * scale)))
    resized = rgba.resize(new_size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", target_size, (0, 0, 0, 0))
    offset = ((target_width - new_size[0]) // 2, (target_height - new_size[1]) // 2)
    canvas.alpha_composite(resized, offset)
    return canvas


def center_without_resize(image: Image.Image, canvas_size: tuple[int, int]) -> Image.Image:
    """Center visible foreground on a transparent canvas without scaling it."""
    foreground = trim_transparent(image)
    canvas_width, canvas_height = canvas_size
    if foreground.width > canvas_width or foreground.height > canvas_height:
        # Segmentation operates on a crop clamped to the source image, so this
        # is defensive only. Crop symmetrically instead of changing scale.
        left = max(0, (foreground.width - canvas_width) // 2)
        top = max(0, (foreground.height - canvas_height) // 2)
        foreground = foreground.crop(
            (left, top, min(foreground.width, left + canvas_width), min(foreground.height, top + canvas_height))
        )
    canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    offset = ((canvas_width - foreground.width) // 2, (canvas_height - foreground.height) // 2)
    canvas.alpha_composite(foreground, offset)
    return canvas


def pil_to_rgb_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"))


def rgb_array_to_pil(array: np.ndarray) -> Image.Image:
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("Expected an RGB array with shape (H, W, 3).")
    return Image.fromarray(array.astype(np.uint8), mode="RGB")

