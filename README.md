# Nhận diện CAPTCHA

## Dataset
+ Mô tả: 1000 ảnh, file nhị phân (.pbm), 4 số (0-9), chữ tách rời, không nhiễu
+ Note: Tập trung làm được trên dataset đơn giản này trước, rồi làm những cái phức tạp hơn sau

## Run
+ Clone project về
+ Mở terminal đúng path của project (sai path sẽ lỗi)
+ Tạo venv, tải thư viện trong `requirements.txt`
+ Chạy `python main.py --stage 0`
+ Input (đã có): Tập dữ liệu `dataset/raw`
+ Output: Trong folder `dataset` sẽ có thêm 2 folder: `processed` (kết quả của stage 2) và `segmented` (kết quả của stage 3)
+ Note: Stage chia theo 9 phần đã note trong docs

## Accuracy — Cách tính

Hệ thống đo **hai loại accuracy**:

| Metric | Ý nghĩa | Ví dụ |
|--------|----------|-------|
| **Per-character** | Tỉ lệ ký tự đơn đoán đúng / tổng ký tự | CAPTCHA "0824", đoán "0B24" → 3/4 = 75% |
| **Per-image** | Tỉ lệ CAPTCHA đoán đúng **toàn bộ** ký tự | Cùng ví dụ trên → 0/1 = 0% |

Per-image accuracy luôn ≤ per-character accuracy. Chi tiết xem docstring trong `model/train.py`.

## Kiến trúc CNN

Chi tiết lý do lựa chọn kiến trúc (số lớp Conv, Dropout, ...) được giải thích trong docstring đầu file `model/cnn.py`.

Tóm tắt: 3 Conv layers (32→64→128) + FC(512) + Dropout(0.5) là điểm cân bằng giữa complexity và generalization cho ảnh 28×28 với dataset nhỏ (~4000 samples).

## Tái sử dụng cho dataset khác

Các module được thiết kế **tách biệt theo chức năng**, có thể tái sử dụng cho dataset mới:

| Module | Hàm chính | Tái sử dụng? | Cần thay đổi gì? |
|--------|-----------|:---:|---|
| `datalayer/build_dataset.py` | `parse_labels`, `build_dataset`, `encode_labels` | **Có điều kiện** | `CHARSET` cần mở rộng nếu dataset có thêm ký tự (vd: chữ thường, ký tự đặc biệt). `parse_labels` cần viết lại nếu format label khác (vd: JSON thay vì tên file). |
| `model/cnn.py` | `CaptchaCNN`, `build_model` | **Có** | Thay `num_classes` khi khởi tạo. Nếu ảnh không phải 28×28, cần tính lại kích thước FC layer. |
| `model/train.py` | `train_model`, `evaluate`, `full_evaluation` | **Có** | Thay `chars_per_image` trong `full_evaluation` nếu CAPTCHA có số ký tự khác 4. |
| `model/baseline.py` | `run_knn`, `run_svm` | **Có** | Không — nhận numpy arrays (X, y) bất kỳ. |

**Để thêm dataset mới**, cần:
1. Viết hàm datalayer riêng (tương tự `pbm_1k_process.py`) để đưa ảnh vào `dataset/raw/<tên_dataset>/`
2. Gọi lại pipeline preprocess → segment → build_dataset → train với đường dẫn mới
3. Mở rộng `CHARSET` trong `build_dataset.py` nếu cần

## Ref
+ Docs tổng: [ML 2025.2](https://docs.google.com/document/d/1g3PKIR1HZzpv9pxYNPCW63b5PtAFzVbOIYlK6n1ih1c/edit?usp=sharing)
+ Dataset hiện tại: [CAPTCHA Dataset](https://cgi.cse.unsw.edu.au/~cs1511/17s1/assignments/captcha/captcha.html)
