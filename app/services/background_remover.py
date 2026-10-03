from __future__ import annotations

import io
import logging

from PIL import Image

from app.config import Settings

LOGGER = logging.getLogger(__name__)


class BackgroundRemover:
    """Reusable rembg session with alpha matting configured in one place."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        try:
            from rembg import new_session, remove
        except ImportError as exc:
            raise RuntimeError("Thiếu rembg. Hãy chạy: pip install -r requirements.txt") from exc
        self._remove = remove
        LOGGER.info("[LOAD] rembg model=%s", settings.rembg_model)
        self.session = new_session(settings.rembg_model)

    def remove(self, image: Image.Image) -> Image.Image:
        LOGGER.info("[REMOVE_BACKGROUND] crop_size=%s", image.size)
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        try:
            result = self._remove(
                buffer.getvalue(),
                session=self.session,
                alpha_matting=self.settings.alpha_matting,
                alpha_matting_foreground_threshold=self.settings.alpha_foreground_threshold,
                alpha_matting_background_threshold=self.settings.alpha_background_threshold,
                alpha_matting_erode_size=self.settings.alpha_erode_size,
                post_process_mask=False,
            )
            with Image.open(io.BytesIO(result)) as decoded:
                decoded.load()
                rgba = decoded.convert("RGBA")
        except (RuntimeError, ValueError, OSError) as exc:
            raise RuntimeError(f"Tách nền thất bại: {exc}") from exc
        if rgba.getchannel("A").getbbox() is None:
            raise RuntimeError("Tách nền thất bại: mask kết quả rỗng.")
        return rgba

