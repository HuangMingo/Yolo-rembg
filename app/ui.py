from __future__ import annotations

from functools import lru_cache
import json
import logging
from pathlib import Path
import gradio as gr

from app.config import Settings
from app.services import (
    AnimalDetector,
    BackgroundRemover,
    ImageProcessor,
    ProcessingError,
)

LOGGER = logging.getLogger(__name__)

APP_CSS = """
.app-title {text-align:center; margin-bottom:0.2rem}
.app-subtitle {text-align:center; color:#64748b; margin-bottom:1.2rem}
"""


@lru_cache(maxsize=1)
def get_processor(settings: Settings | None = None) -> ImageProcessor:
    """Models are initialized exactly once and reused by every request."""
    settings = settings or Settings()
    return ImageProcessor(
        settings, AnimalDetector(settings), BackgroundRemover(settings)
    )


def _detect(image_path: str | None, settings: Settings | None = None):
    if not image_path:
        raise gr.Error("Vui lòng chọn một ảnh.")
    try:
        result = get_processor(settings).detect(image_path)
    except ProcessingError as exc:
        raise gr.Error(exc.message) from exc
    if result.detection_method == "full_image_fallback":
        choices = ["1. Toàn bộ ảnh - YOLO không phát hiện"]
    else:
        choices = [
            f"{index + 1}. {item.class_name} - {item.confidence:.1%}"
            for index, item in enumerate(result.detections)
        ]
    return (
        result.preview,
        gr.Dropdown(choices=choices, value=choices[0], interactive=True),
        result,
    )


def _remove_background(
    detected,
    selection: str | None,
    settings: Settings | None = None,
):
    if detected is None:
        raise gr.Error("Hãy chạy YOLO trước.")
    selected_index = int(selection.split(".", 1)[0]) - 1 if selection else 0
    try:
        result = get_processor(settings).process_detection(
            detected, selected_index, save_output=True
        )
    except ProcessingError as exc:
        raise gr.Error(exc.message) from exc
    return (
        result.preview,
        result.output,
        json.dumps(result.metadata, ensure_ascii=False, indent=2),
    )


def _process_folder(
    image_paths: list[str] | None,
    settings: Settings | None = None,
    progress=gr.Progress(),
):
    if not image_paths:
        raise gr.Error("Vui lòng chọn một thư mục có ảnh.")

    def update_progress(completed: int, total: int, source_name: str) -> None:
        progress(
            completed / total, desc=f"Đang xử lý {source_name} ({completed}/{total})"
        )

    try:
        result = get_processor(settings).process_batch(
            image_paths, progress_callback=update_progress
        )
    except ProcessingError as exc:
        raise gr.Error(exc.message) from exc

    gallery = [
        (str(item.saved_path), item.source_name)
        for item in result.items
        if item.success and item.saved_path is not None
    ]
    report = json.dumps(result.as_dict(), ensure_ascii=False, indent=2)
    archive = str(result.archive_path) if result.archive_path else None
    if result.failure_count:
        gr.Warning(
            f"Hoàn thành với {result.failure_count} ảnh lỗi. Xem báo cáo để biết chi tiết."
        )
    return gallery, report, archive


def build_ui(
    output_directory: str | Path | None = None,
    settings: Settings | None = None,
) -> gr.Blocks:
    if settings is not None and output_directory is not None:
        raise ValueError(
            "Chỉ truyền settings hoặc output_directory, không truyền đồng thời cả hai."
        )
    resolved_settings = settings or (
        Settings(output_directory=Path(output_directory))
        if output_directory is not None
        else Settings()
    )

    def detect_from_ui(image_path: str | None):
        return _detect(image_path, resolved_settings)

    def remove_from_ui(detected, selection: str | None):
        return _remove_background(detected, selection, resolved_settings)

    def process_folder_from_ui(image_paths: list[str] | None, progress=gr.Progress()):
        return _process_folder(image_paths, resolved_settings, progress)

    with gr.Blocks(title="AnimalCut - Tách nền ảnh động vật") as demo:
        gr.Markdown("# AnimalCut", elem_classes="app-title")
        gr.Markdown(
            "YOLO-World định vị báo, cáo, sư tử, hổ và sói · BiRefNet tách nền · giữ nguyên kích thước",
            elem_classes="app-subtitle",
        )
        with gr.Tabs():
            with gr.Tab("Một ảnh"):
                state = gr.State()
                with gr.Row():
                    original = gr.Image(
                        type="filepath",
                        label="1. Ảnh gốc",
                        sources=["upload", "clipboard"],
                    )
                    detection_preview = gr.Image(
                        type="pil", label="2. Kết quả YOLO", interactive=False
                    )
                    output = gr.Image(
                        type="pil",
                        label="3. Ảnh tách nền",
                        image_mode="RGBA",
                        interactive=False,
                    )
                with gr.Row():
                    detect_button = gr.Button("Phát hiện động vật", variant="secondary")
                    object_selector = gr.Dropdown(
                        label="Chọn đối tượng", choices=[], interactive=True
                    )
                    remove_button = gr.Button("Tách nền", variant="primary")
                    reset_button = gr.Button("Đặt lại")
                metadata = gr.Code(
                    label="Thông tin xử lý", language="json", interactive=False
                )
                # download = gr.DownloadButton("Tải PNG", interactive=False)

                detect_button.click(
                    detect_from_ui,
                    inputs=original,
                    outputs=[detection_preview, object_selector, state],
                )
                remove_button.click(
                    remove_from_ui,
                    inputs=[state, object_selector],
                    outputs=[detection_preview, output, metadata],
                )
                reset_button.click(
                    lambda: (
                        None,
                        None,
                        None,
                        gr.Dropdown(choices=[], value=None),
                        None,
                        "",
                        None,
                    ),
                    outputs=[
                        original,
                        detection_preview,
                        output,
                        object_selector,
                        state,
                        metadata,
                    ],
                )

            with gr.Tab("Cả thư mục"):
                gr.Markdown(
                    "Chọn một thư mục ảnh. Chương trình sẽ tự chọn động vật lớn nhất trong từng ảnh "
                    "và tiếp tục xử lý nếu một ảnh bị lỗi."
                )
                folder_input = gr.File(
                    label="Thư mục ảnh",
                    file_count="directory",
                    file_types=["image"],
                    type="filepath",
                )
                with gr.Row():
                    batch_button = gr.Button(
                        "Tách nền toàn bộ thư mục", variant="primary"
                    )
                    batch_reset_button = gr.Button("Đặt lại")
                batch_gallery = gr.Gallery(
                    label="Ảnh đã tách nền",
                    columns=4,
                    object_fit="contain",
                    height="auto",
                )
                batch_report = gr.Code(
                    label="Báo cáo xử lý", language="json", interactive=False
                )
                batch_download = gr.DownloadButton(
                    "Tải toàn bộ kết quả (.zip)", interactive=False
                )

                batch_button.click(
                    process_folder_from_ui,
                    inputs=folder_input,
                    outputs=[batch_gallery, batch_report, batch_download],
                )
                batch_reset_button.click(
                    lambda: (None, [], "", None),
                    outputs=[folder_input, batch_gallery, batch_report, batch_download],
                )
    return demo


def launch(
    output_directory: str | Path | None = None,
    settings: Settings | None = None,
) -> None:
    """Launch the app and optionally override the directory used for PNG output."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    build_ui(output_directory=output_directory, settings=settings).launch(
        inbrowser=True, css=APP_CSS
    )
