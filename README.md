# AnimalCut - ứng dụng tách nền ảnh động vật

Ứng dụng triển khai pipeline được mô tả trong báo cáo Nhóm 13:

`Ảnh đầu vào → sửa EXIF → YOLO → bounding box có padding → crop → rembg/BiRefNet + alpha matting → căn giữa không resize → PNG RGBA`

## Vì sao dùng YOLO-World?

AnimalCut dùng YOLO-World với prompt mở `animal` để định vị ổn định nhiều loài mà không buộc YOLO phải phân loại chính xác từng loài. YOLO chỉ định vị đối tượng; mask tách nền do BiRefNet tạo.

## Cài đặt

Khuyến nghị Python 3.11 và môi trường ảo:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` cài CLIP trực tiếp từ kho Ultralytics dưới dạng ZIP, vì
YOLO-World cần mô-đun này khi đặt prompt văn bản; máy không cần cài Git.

Lần chạy đầu tiên cần Internet để tải `yolov8s-worldv2.pt` và checkpoint `birefnet-general`. Các lần sau dùng cache cục bộ. BiRefNet ưu tiên chất lượng biên nên tải và chạy chậm hơn U2Net.

## Chạy ứng dụng

```powershell
python run.py
```

Trình duyệt sẽ mở giao diện tại địa chỉ Gradio hiển thị trong terminal. Quy trình sử dụng:

1. Chọn ảnh JPG, JPEG, PNG hoặc WEBP.
2. Nhấn **Phát hiện động vật**.
3. Nếu ảnh có nhiều động vật, chọn đúng đối tượng trong danh sách.
4. Nhấn **Tách nền** rồi tải PNG.

### Xử lý cả thư mục

1. Mở tab **Cả thư mục**.
2. Chọn thư mục chứa ảnh JPG, JPEG, PNG hoặc WEBP.
3. Nhấn **Tách nền toàn bộ thư mục**.
4. Xem thư viện kết quả và báo cáo từng ảnh.
5. Nhấn **Tải toàn bộ kết quả (.zip)** để tải các PNG cùng `report.json`.

Với mỗi ảnh, chương trình tự chọn detection có diện tích lớn nhất. Nếu một ảnh hỏng hoặc không phát hiện được động vật, ảnh đó được ghi vào báo cáo và các ảnh còn lại vẫn tiếp tục xử lý. Mỗi lần chạy batch được lưu trong một thư mục riêng để không ghi đè kết quả cũ.

## Cấu hình

Các tham số nằm trong `app/config/settings.py` và có thể ghi đè bằng biến môi trường:

| Biến | Mặc định | Ý nghĩa |
|---|---:|---|
| `YOLO_MODEL` | `yolov8s-worldv2.pt` | Checkpoint YOLO-World |
| `YOLO_CLASSES` | `animal` | Prompt YOLO-World; có thể nhập nhiều prompt, phân tách bằng dấu phẩy |
| `YOLO_CONFIDENCE` | `0.10` | Ngưỡng phát hiện; nên hiệu chỉnh sau khi đánh giá từng bộ ảnh |
| `YOLO_IOU` | `0.45` | Ngưỡng IoU/NMS |
| `YOLO_DEVICE` | `auto` | `auto`, `cpu`, `0`, ... |
| `BBOX_PADDING` | `0.10` | Padding 10% quanh bounding box |
| `REMBG_MODEL` | `birefnet-general` | Model tách nền |
| `ALPHA_MATTING` | `true` | Bật alpha matting |
| `ALPHA_FOREGROUND_THRESHOLD` | `240` | Ngưỡng foreground |
| `ALPHA_BACKGROUND_THRESHOLD` | `10` | Ngưỡng background |
| `ALPHA_ERODE_SIZE` | `10` | Kích thước erode cho trimap |
| `OUTPUT_DIRECTORY` | tự tìm `BTL/remove_background image/animals` | Thư mục tự động lưu PNG |

Ví dụ đổi thư mục lưu kết quả:

```powershell
$env:OUTPUT_DIRECTORY="D:\duong-dan\thu-muc-ket-qua"
python run.py
```

## Kiểm thử

```powershell
python -m unittest discover -v
```

Các test lõi không tải model: kiểm tra ảnh thật/ảnh hỏng, bbox có padding, giữ tỉ lệ và alpha, luồng thành công, và dừng đúng khi không phát hiện động vật.

## Cấu trúc

```text
app/
  config/settings.py
  services/detector.py
  services/background_remover.py
  services/image_processor.py
  utils/image_utils.py
  utils/validation.py
  ui.py
tests/test_pipeline.py
run.py
```

Ảnh kết quả giữ nguyên kích thước ảnh gốc, chỉ dịch foreground vào giữa canvas trong suốt. Khi xử lý từ giao diện, kết quả được tự động lưu dưới dạng PNG với cùng phần tên, ví dụ `tiger_001.jpg` thành `tiger_001.png`.
