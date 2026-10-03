# Cấu trúc dự án AnimalCut

## Tổng quan

AnimalCut là ứng dụng phát hiện động vật bằng YOLO-World, cắt vùng chứa đối tượng, tách nền bằng BiRefNet thông qua `rembg`, sau đó xuất ảnh PNG có nền trong suốt.

Luồng xử lý chính:

```text
Ảnh đầu vào
  → kiểm tra và sửa hướng EXIF
  → YOLO phát hiện động vật
  → mở rộng bounding box
  → cắt vùng đối tượng
  → BiRefNet/rembg tách nền
  → căn giữa trên canvas trong suốt
  → lưu ảnh PNG
```

## Cây thư mục

```text
Yolo rembg/
├── app/
│   ├── config/
│   │   ├── __init__.py
│   │   ├── parameters.py
│   │   └── settings.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── background_remover.py
│   │   ├── detector.py
│   │   └── image_processor.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── image_utils.py
│   │   └── validation.py
│   ├── __init__.py
│   └── ui.py
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py
├── weights/
│   └── clip/
│       └── ViT-B-32.pt
├── .gitignore
├── AGENTS.md,.md
├── CAU_TRUC_DU_AN.md
├── README.md
├── debug.log
├── requirements.txt
├── rules.md
├── run.py
└── yolov8s-worldv2.pt
```

Các thư mục `__pycache__/` và file `*.pyc` có thể xuất hiện sau khi chạy Python. Đây là cache được tạo tự động, không phải mã nguồn và có thể xóa an toàn khi chương trình đã dừng.

## Thư mục `app/`

Đây là mã nguồn chính của ứng dụng.

### `app/__init__.py`

Đánh dấu `app` là một Python package. File này hiện chỉ chứa mô tả ngắn về ứng dụng.

### `app/ui.py`

Xây dựng giao diện Gradio và kết nối thao tác của người dùng với pipeline xử lý.

Trách nhiệm chính:

- Hiển thị ảnh gốc, ảnh có bounding box và ảnh đã tách nền.
- Nhận ảnh do người dùng tải lên.
- Gọi YOLO để phát hiện động vật.
- Tạo danh sách cho phép chọn đối tượng khi ảnh có nhiều kết quả.
- Gọi bước tách nền và cung cấp file PNG để tải xuống.
- Nhận cả thư mục ảnh, hiển thị tiến độ, thư viện kết quả và file ZIP.
- Cache `ImageProcessor` để YOLO và BiRefNet không bị tải lại ở mỗi yêu cầu.

## Thư mục `app/config/`

Chứa toàn bộ cấu hình của ứng dụng.

### `app/config/parameters.py`

Là nơi người dùng chỉnh các tham số vận hành. Thông thường, đây là file duy nhất cần sửa khi muốn thử cấu hình khác.

Các nhóm tham số gồm:

- Đường dẫn thư mục lưu kết quả.
- Model, class, confidence, IoU và thiết bị chạy YOLO.
- Tỷ lệ mở rộng bounding box.
- Model tách nền.
- Các tham số alpha matting và erode.
- Giới hạn dung lượng và số pixel của ảnh đầu vào.

Ví dụ:

```python
BBOX_PADDING = 0.15
ALPHA_MATTING = True
ALPHA_ERODE_SIZE = 3
```

Sau khi sửa file này, cần khởi động lại chương trình.

### `app/config/settings.py`

Chuyển các hằng số trong `parameters.py` thành đối tượng `Settings` dùng chung trong chương trình.

File này còn:

- Tính thư mục kết quả mặc định nếu `OUTPUT_DIRECTORY` là `None`.
- Kiểm tra tính hợp lệ của confidence, IoU và bounding-box padding.
- Giúp truyền một cấu hình nhất quán cho detector, background remover và image processor.

### `app/config/__init__.py`

Xuất lớp `Settings` để các file khác có thể import ngắn gọn:

```python
from app.config import Settings
```

## Thư mục `app/services/`

Chứa các thành phần thực hiện pipeline xử lý ảnh.

### `app/services/detector.py`

Phụ trách phát hiện động vật bằng YOLO-World.

File này:

- Tải model YOLO.
- Thiết lập prompt/class từ cấu hình.
- Chạy inference trên ảnh PIL.
- Lấy class, confidence và bounding box.
- Loại bỏ kết quả không hợp lệ.
- Sắp xếp các đối tượng theo diện tích giảm dần; đối tượng lớn nhất đứng đầu.

`Detection` là cấu trúc dữ liệu biểu diễn một kết quả phát hiện.

### `app/services/background_remover.py`

Phụ trách tách nền bằng `rembg` và model BiRefNet.

File này:

- Tạo một session model có thể tái sử dụng.
- Chuyển vùng ảnh cần xử lý thành PNG trong bộ nhớ.
- Truyền các tham số alpha matting từ `Settings` vào `rembg`.
- Trả kết quả dưới dạng ảnh RGBA.
- Báo lỗi nếu model tạo ra mask rỗng.

### `app/services/image_processor.py`

Điều phối toàn bộ pipeline, nhưng không tự thực hiện inference.

Hai bước chính:

- `detect()`: kiểm tra ảnh, chạy detector và tạo ảnh xem trước có bounding box.
- `process_detection()`: chọn detection, mở rộng bounding box, crop, tách nền, căn giữa và lưu PNG.
- `process_batch()`: duyệt toàn bộ danh sách ảnh, tự chọn đối tượng lớn nhất, tiếp tục khi một ảnh lỗi và đóng gói kết quả thành ZIP.

File cũng định nghĩa:

- `ProcessingError`: lỗi có mã và thông báo rõ ràng.
- `DetectionResult`: kết quả của bước phát hiện.
- `ProcessingResult`: ảnh đầu ra, metadata và đường dẫn file đã lưu.
- `BatchItemResult`: trạng thái thành công hoặc lỗi của từng ảnh trong batch.
- `BatchProcessingResult`: tổng hợp toàn bộ batch và đường dẫn file ZIP.

### `app/services/__init__.py`

Xuất các class dịch vụ để phần còn lại của ứng dụng có thể import từ `app.services` thay vì import từng module riêng lẻ.

## Thư mục `app/utils/`

Chứa các hàm hỗ trợ không phụ thuộc trực tiếp vào model.

### `app/utils/validation.py`

Đọc và kiểm tra ảnh đầu vào.

Các kiểm tra gồm:

- File có tồn tại và có dữ liệu hay không.
- Dung lượng có vượt giới hạn không.
- Dữ liệu có giải mã thành ảnh hợp lệ không.
- Định dạng có thuộc JPEG, PNG hoặc WEBP không.
- Tổng số pixel có vượt giới hạn không.
- Sửa hướng ảnh theo EXIF và chuyển sang RGB trước khi inference.

### `app/utils/image_utils.py`

Chứa các phép xử lý ảnh dùng chung:

- Mở rộng bounding box và giới hạn tọa độ trong kích thước ảnh.
- Vẽ bounding box, nhãn và confidence lên ảnh xem trước.
- Cắt bỏ vùng trong suốt thừa.
- Resize mà vẫn giữ đúng tỷ lệ.
- Căn giữa foreground trên canvas RGBA mà không resize.
- Chuyển đổi giữa ảnh PIL và mảng NumPy RGB.

### `app/utils/__init__.py`

Đánh dấu `utils` là Python package và mô tả ngắn nhóm tiện ích.

## Thư mục `tests/`

Chứa kiểm thử tự động, không tải model thật nên có thể chạy nhanh.

### `tests/test_pipeline.py`

Kiểm tra các hành vi cốt lõi:

- Đọc ảnh hợp lệ và từ chối dữ liệu ảnh hỏng.
- Mở rộng bounding box đúng và không vượt biên.
- Giữ tỷ lệ ảnh và kênh alpha.
- Căn giữa ảnh mà không thay đổi kích thước.
- Chọn đối tượng lớn nhất theo mặc định.
- Dừng đúng khi không phát hiện động vật.
- Lưu PNG đúng tên và đúng chế độ RGBA.

Chạy test bằng:

```powershell
python -m unittest discover -v
```

### `tests/__init__.py`

Đánh dấu `tests` là Python package.

## Model và trọng số

### `yolov8s-worldv2.pt`

Checkpoint YOLO-World dùng để xác định vị trí động vật trong ảnh. Model này tạo bounding box, không trực tiếp tạo mask tách nền.

### `weights/clip/ViT-B-32.pt`

Trọng số CLIP được YOLO-World sử dụng để xử lý prompt văn bản như `animal`.

File này có kích thước lớn, vì vậy cần chú ý giới hạn dung lượng khi đưa repository lên GitHub.

Checkpoint BiRefNet không nằm trực tiếp trong cây thư mục này. `rembg` tải model ở lần chạy đầu tiên và lưu vào cache của môi trường người dùng.

## Các file ở thư mục gốc

### `run.py`

Điểm khởi chạy ứng dụng:

```powershell
python run.py
```

File tạo `Settings`, truyền cấu hình vào giao diện và khởi động Gradio.

### `requirements.txt`

Danh sách thư viện Python cần cài, gồm Gradio, NumPy, Pillow, rembg, Ultralytics và CLIP.

Cài đặt bằng:

```powershell
pip install -r requirements.txt
```

### `README.md`

Hướng dẫn tổng quan về mục tiêu dự án, cách cài đặt, chạy chương trình, cấu hình và kiểm thử.

### `rules.md`

Mô tả các nguyên tắc kỹ thuật của dự án, chẳng hạn trách nhiệm của YOLO, quy tắc bounding box, alpha mask, resize, output PNG, logging và testing.

### `AGENTS.md,.md`

File hướng dẫn dành cho công cụ hoặc AI agent làm việc với repository. Nội dung đặt ra mục tiêu, kiến trúc và các quy tắc cần tuân thủ khi chỉnh sửa dự án.

Tên file hiện có dấu phẩy (`AGENTS.md,.md`); nếu công cụ phát triển yêu cầu đúng tên chuẩn `AGENTS.md`, có thể cân nhắc đổi tên file.

### `.gitignore`

Khai báo các file không nên đưa vào Git, hiện gồm cache Python và thư mục kết quả `output/`.

### `debug.log`

File log chẩn đoán được tạo trong quá trình chạy hoặc mở ứng dụng. Nội dung hiện tại liên quan đến lỗi quyền truy cập của Crashpad trên Windows; file này không tham gia pipeline xử lý ảnh.

## Nơi cần chỉnh khi thay đổi chức năng

| Nhu cầu | File cần chỉnh |
|---|---|
| Đổi confidence, padding hoặc alpha matting | `app/config/parameters.py` |
| Đổi cách YOLO phát hiện đối tượng | `app/services/detector.py` |
| Đổi cách tách nền | `app/services/background_remover.py` |
| Đổi thứ tự pipeline hoặc cách lưu kết quả | `app/services/image_processor.py` |
| Đổi cách crop, resize hoặc căn giữa | `app/utils/image_utils.py` |
| Đổi kiểm tra ảnh đầu vào | `app/utils/validation.py` |
| Đổi giao diện và nút thao tác | `app/ui.py` |
| Thêm kiểm thử | `tests/test_pipeline.py` |

