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

Với mỗi ảnh, chương trình tự chọn detection có diện tích lớn nhất. Nếu YOLO không phát hiện, chương trình dùng toàn bộ ảnh làm vùng dự phòng cho BiRefNet; báo cáo ghi `detection_method` là `full_image_fallback`. Nếu một ảnh hỏng hoặc tách nền thất bại, ảnh đó được ghi vào báo cáo và các ảnh còn lại vẫn tiếp tục xử lý. Mỗi lần chạy batch được lưu trong một thư mục riêng để không ghi đè kết quả cũ.

## Cấu hình

Chỉnh các tham số trong `app/config/parameters.py`; `Settings` đọc và kiểm tra các giá trị này khi khởi động. Sau khi chỉnh, cần khởi động lại ứng dụng hoặc phiên notebook để nạp cấu hình mới.

| Biến | Mặc định | Ý nghĩa |
|---|---:|---|
| `YOLO_MODEL` | `yolov8s-worldv2.pt` | Checkpoint YOLO-World |
| `YOLO_CLASSES` | `animal` | Prompt YOLO-World; có thể nhập nhiều prompt, phân tách bằng dấu phẩy |
| `YOLO_CONFIDENCE` | `0.10` | Ngưỡng phát hiện; nên hiệu chỉnh sau khi đánh giá từng bộ ảnh |
| `YOLO_IOU` | `0.45` | Ngưỡng IoU/NMS |
| `YOLO_DEVICE` | `auto` | `auto`, `cpu`, `0`, ... |
| `BBOX_PADDING` | `0.15` | Padding 15% quanh bounding box |
| `REMBG_MODEL` | `birefnet-general` | Model tách nền |
| `ALPHA_MATTING` | `true` | Bật alpha matting |
| `ALPHA_FOREGROUND_THRESHOLD` | `240` | Ngưỡng foreground |
| `ALPHA_BACKGROUND_THRESHOLD` | `10` | Ngưỡng background |
| `ALPHA_ERODE_SIZE` | `3` | Kích thước erode cho trimap |
| `POST_PROCESS_MASK` | `False` | Làm sạch mask bằng rembg; có thể mất lông mảnh |
| `DECONTAMINATE` | `False` | Làm sạch màu ám ở viền khi tắt alpha matting; cần rembg hỗ trợ |
| `OUTPUT_DIRECTORY` | tự tìm `BTL/remove_background image/animals` | Thư mục tự động lưu PNG |

Ví dụ đổi thư mục lưu kết quả:

```python
# Trong app/config/parameters.py
OUTPUT_DIRECTORY = r"D:\duong-dan\thu-muc-ket-qua"
```

### Tinh chỉnh viền theo rembg

Ứng dụng dùng API PIL theo [hướng dẫn rembg](https://github.com/danielgatis/rembg/blob/main/USAGE.md): tạo `new_session(REMBG_MODEL)` một lần rồi gọi `remove(image, session=session, ...)` cho mỗi vùng động vật. Kết quả được giữ ở dạng PNG RGBA, không đổi kích thước. Không cần cài thêm thư viện.

Cấu hình mặc định giữ `birefnet-general` và alpha matting để ước lượng độ trong suốt và màu foreground ở viền. `POST_PROCESS_MASK` mặc định tắt: rembg dùng morphology, làm mờ rồi threshold mask; bật có thể làm sạch cảnh sót nhưng cũng mất lông hoặc ria.

- Nếu còn mảng cảnh quanh viền: thử `POST_PROCESS_MASK = True` trên cùng một ảnh và so sánh với `False`.
- Nếu còn ám màu nền: giữ `ALPHA_MATTING = True` trước. Có thể thử riêng `ALPHA_MATTING = False` cùng `DECONTAMINATE = True`; rembg bỏ qua decontaminate khi alpha matting bật, nên ứng dụng chỉ truyền tùy chọn này khi tắt matting.
- Nếu matting nhận nhầm lông là foreground chắc chắn: thử tăng `ALPHA_FOREGROUND_THRESHOLD` từ 240 lên 245. Nếu còn nền mờ: thử tăng `ALPHA_BACKGROUND_THRESHOLD` từ 10 lên 20. Đây là giá trị thử nghiệm, cần kiểm tra mất lông trên ảnh thực tế; không phải thiết lập tốt nhất cho mọi ảnh.
- Giữ `0 <= background < foreground < 255`; ví dụ foreground 270 trong USAGE.md không phù hợp mask 8-bit vì không còn pixel foreground chắc chắn.

Tùy chọn `decontaminate` có trong mã rembg hiện tại nhưng có thể chưa có trong bản bạn cài. Ứng dụng kiểm tra API và báo lỗi rõ ràng nếu bật tùy chọn trên phiên bản không hỗ trợ, thay vì âm thầm bỏ qua. Trong trường hợp đó nâng cấp bằng `python -m pip install --upgrade "rembg[cpu]>=2.0.67,<3"`; nếu bản phát hành vẫn chưa hỗ trợ, dùng alpha matting hoặc tắt `DECONTAMINATE`.

Đánh giá kết quả trên nền trắng, đen và màu nổi ở mức zoom 100%, đặc biệt vùng lông, tai, đuôi và ria. Unit test xác minh API, session, alpha và lỗi; không thay thế đánh giá chất lượng mô hình trên ảnh thật.

## Kiểm thử

```powershell
python -m unittest discover -v
```

Các test lõi không tải model: kiểm tra ảnh thật/ảnh hỏng, bbox có padding, giữ tỉ lệ và alpha, luồng thành công, và fallback toàn ảnh khi YOLO không phát hiện động vật.

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
