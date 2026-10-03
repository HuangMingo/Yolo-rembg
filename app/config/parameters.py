"""Các tham số người dùng có thể chỉnh sửa cho toàn bộ ứng dụng."""

# Thư mục lưu kết quả. Đặt None để chương trình tự chọn thư mục mặc định.
OUTPUT_DIRECTORY = (
    r"D:\Documents\HỆ CƠ SỞ DỮ LIỆU ĐA PHƯƠNG TIỆN\BTL\remove_background image\wolf"
)

# YOLO
YOLO_MODEL = "yolov8s-worldv2.pt"  # Chon model chuong trinh se su dung
YOLO_CLASSES = ("animal",)
YOLO_CONFIDENCE = 0.10  # do tin cay cua model rang doi tuong no vua phat hien dung
# voi class duoc yeu cau
YOLO_IOU = 0.45
YOLO_DEVICE = "auto"
BBOX_PADDING = 0.15

# Tách nền
REMBG_MODEL = "birefnet-general"
ALPHA_MATTING = True
ALPHA_FOREGROUND_THRESHOLD = 240
ALPHA_BACKGROUND_THRESHOLD = 10
ALPHA_ERODE_SIZE = 3

# Giới hạn ảnh đầu vào
MAX_UPLOAD_MB = 20
MAX_IMAGE_PIXELS = 40_000_000
