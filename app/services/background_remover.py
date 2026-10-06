from __future__ import annotations

import inspect
import logging
from threading import Lock

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
        self._lock = Lock()
        self._supports_decontaminate = "decontaminate" in inspect.signature(remove).parameters
        if settings.decontaminate and not settings.alpha_matting and not self._supports_decontaminate:
            raise RuntimeError(
                "Phiên bản rembg này chưa hỗ trợ decontaminate. "
                "Hãy nâng cấp rembg hoặc đặt DECONTAMINATE = False."
            )
        LOGGER.info("[LOAD] rembg model=%s", settings.rembg_model)
        self.session = new_session(settings.rembg_model)

    def remove(self, image: Image.Image) -> Image.Image:
        LOGGER.info("[REMOVE_BACKGROUND] crop_size=%s", image.size)
        options = {}
        if self.settings.decontaminate and not self.settings.alpha_matting:
            options["decontaminate"] = True
        try:
            # Follow rembg's PIL API and reuse one session. Avoid concurrent
            # calls into the shared model from different UI events.
            with self._lock:
                result = self._remove(
                    image.convert("RGB"),
                    session=self.session,
                    alpha_matting=self.settings.alpha_matting,
                    alpha_matting_foreground_threshold=self.settings.alpha_foreground_threshold,
                    alpha_matting_background_threshold=self.settings.alpha_background_threshold,
                    alpha_matting_erode_size=self.settings.alpha_erode_size,
                    post_process_mask=self.settings.post_process_mask,
                    **options,
                )
                if not isinstance(result, Image.Image):
                    raise RuntimeError("rembg không trả về ảnh PIL như mong đợi.")
                if result.size != image.size:
                    raise RuntimeError("rembg trả về kích thước ảnh không hợp lệ.")
                rgba = result.convert("RGBA")
        except (RuntimeError, ValueError, OSError) as exc:
            raise RuntimeError(f"Tách nền thất bại: {exc}") from exc
        if rgba.getchannel("A").getbbox() is None:
            raise RuntimeError("Tách nền thất bại: mask kết quả rỗng.")
        return rgba

