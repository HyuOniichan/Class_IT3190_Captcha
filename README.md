# Nhận diện CAPTCHA

## Dataset
+ Mô tả: 1000 ảnh, file nhị phân (.pbm), 4 số (0-9), chữ tách rời, không nhiễu
+ Note: Tập trung làm được trên dataset đơn giản này trước, rồi làm những cái phức tạp hơn sau

## Setup
+ Clone project về
+ Mở terminal đúng path của project (sai path sẽ lỗi)
+ Tạo venv, tải thư viện trong `requirements.txt`
+ Chạy `python main.py --stage 0` để chạy toàn bộ pipeline
+ Note: Stage chia theo 9 phần đã note trong docs

## Updates
### 26/4/2026 - Huy
+ Stage 1, 2, 3
+ Input (đã có): Tập dữ liệu `dataset/raw`
+ Output: Trong folder `dataset` sẽ có thêm 2 folder: `processed` (kết quả của stage 2) và `sggmented` (kết quả của stage 3)

### 28/4/2026 - Hùng Anh
+ Stage 4, 5
+ Output: Trong folder `dataset` sẽ có thêm `meta`, `ready` (kết quả của stage 4), trong `model/saved` sẽ có model CNN sau khi train
+ So sánh model:

| Model | Config | Accuracy |
|:-:|:-:|:-:|
| KNN | k = 5 | 0.9613 |
| SVM | C=1.0 | 0.9650 |
| CNN | 3 Conv, 2 MaxPool, 2 FC, 20 epochs | 0.9663 |

+ Cần bổ sung: 
  + Chưa rõ accuracy tính như nào (tính trên số chữ đúng, số image đúng, ...)?
  + Lý do lựa chọn kiến trúc CNN hiện tại (ít lớp Conv hơn kết quả như nào? Bỏ dropout được không? ...)
  + Có thể sử dụng lại các hàm hiện tại cho các bộ dataset khác sau này không?
  + Còn lại ok

+ Update (28/4): 
  + Accuracy — Cách tính

Hệ thống đo **hai loại accuracy**:

| Metric | Ý nghĩa | Ví dụ |
|--------|----------|-------|
| **Per-character** | Tỉ lệ ký tự đơn đoán đúng / tổng ký tự | CAPTCHA "0824", đoán "0B24" → 3/4 = 75% |
| **Per-image** | Tỉ lệ CAPTCHA đoán đúng **toàn bộ** ký tự | Cùng ví dụ trên → 0/1 = 0% |

Per-image accuracy luôn $\le$ per-character accuracy. Chi tiết xem docstring trong `model/train.py`.

  + Kiến trúc CNN
    + Chi tiết lý do lựa chọn kiến trúc (số lớp Conv, Dropout, ...) được giải thích trong docstring đầu file `model/cnn.py`.
    + Tóm tắt: 3 Conv layers (32→64→128) + FC(512) + Dropout(0.5) là điểm cân bằng giữa complexity và generalization cho ảnh 28×28 với dataset nhỏ (~4000 samples).

  + Tái sử dụng cho dataset khác
    + Các module được thiết kế **tách biệt theo chức năng**, có thể tái sử dụng cho dataset mới:

| Module | Hàm chính | Tái sử dụng? | Cần thay đổi gì? |
|--------|-----------|:---:|---|
| `datalayer/build_dataset.py` | `parse_labels`, `build_dataset`, `encode_labels` | **Có điều kiện** | `CHARSET` cần mở rộng nếu dataset có thêm ký tự (vd: chữ thường, ký tự đặc biệt). `parse_labels` cần viết lại nếu format label khác (vd: JSON thay vì tên file). |
| `model/cnn.py` | `CaptchaCNN`, `build_model` | **Có** | Thay `num_classes` khi khởi tạo. Nếu ảnh không phải 28×28, cần tính lại kích thước FC layer. |
| `model/train.py` | `train_model`, `evaluate`, `full_evaluation` | **Có** | Thay `chars_per_image` trong `full_evaluation` nếu CAPTCHA có số ký tự khác 4. |
| `model/baseline.py` | `run_knn`, `run_svm` | **Có** | Không — nhận numpy arrays (X, y) bất kỳ. |

  + **Để thêm dataset mới**, cần:
    + Viết hàm datalayer riêng (tương tự `pbm_1k_process.py`) để đưa ảnh vào `dataset/raw/<tên_dataset>/`
    + Gọi lại pipeline preprocess → segment → build_dataset → train với đường dẫn mới
    + Mở rộng `CHARSET` trong `build_dataset.py` nếu cần


### 11/5/2026 - Minh Duy
+ Stage 6, 7
+ Update 2 file `model/inference.py`, `model/evaluate.py`
+ Thêm ảnh test ở trong folder `evaluation` và chạy luồng để đánh giá input
+ Input: 3 ảnh test trong `evaluation`, hoạt động ổn với chữ tách rời và không méo (1 ảnh), lỗi với ảnh dạng chữ dính liền (2 ảnh)
+ Output: In ra predicted text, và 2 ảnh `confusion_matrix.png`, `sample_errors.png`

+ Note:
  + Dataset hiện tại đang tập trung vào các image đơn giản (chữ tách rời), nên dạng chữ tách rời xử lý ổn, và dataset mới như chữ dính liền chưa predict được là đúng...
  + Sau khi xử lý xong 9 phần trong docs [1], sẽ quay lại nâng cấp dataset và build lại model sau

### 12/5/2026 - Huy
+ Refactor code và Stage 8 
+ Refactor
  + Thêm folder mới `predictors`, load 2 model class riêng `char_predictor.py` (predict từng chữ số) và `captcha_predictor.py` (predict cả image)
  + -> Sau này khi thử model mới sẽ lắp vào 2 class này
+ Stage 8
  + Web UI -> 2 folders `static` và `templates`
  + Viết backend `app.py` -> Xử lý 2 APIs: `/` để vẽ giao diện và `/predict` để predict captcha
  + Luồng hoạt động:
    + User upload image trên giao diện
    + Click "Predict" button
    + Gửi request gồm ảnh đến backend
    + Backend chạy luồng predict từ `captcha_predictor`
    + Trả về predicted text
    + Response về cho giao diện hiển thị

### 15/5/2026 - Huy
+ Giai đoạn 2 - Phần 1
+ Mới: Thêm folder `dataset_v2` gồm:
  + `raw`: Sẽ được gen từ dataset nhóm 1 [2] bằng phương pháp thêm nhiễu (đi kèm folder `meta` để chia train/test)
  + `kaggle_captcha`: Dataset nhóm 2 [3] (khoảng 1k image) 
+ -> Các stage sau có thể xử lý 1 trong 2 loại dataset, ưu tiên `kaggle_captcha`
+ Thêm file `datalayer/prepare_dataset_v2.py` -> Generate dataset mới từ dataset nhóm 1 -> Output: Folder `raw`
+ Note: Các stage sau nên follow theo filename với đuôi `_v2`. Ví dụ: Stage 2 ở folder `preprocess` -> Sẽ có file `general_preprocess_v2.py`
+ Note: Để ý cách chạy file, `python main.py --stage 1 --dataset 2` --> Thêm argument `--dataset 2` để chọn dataset nhóm 2.


## Ref
+ [1] Docs tổng: [ML 2025.2](https://docs.google.com/document/d/1g3PKIR1HZzpv9pxYNPCW63b5PtAFzVbOIYlK6n1ih1c/edit?usp=sharing)
+ [2] Dataset nhóm 1: [CAPTCHA Dataset](https://cgi.cse.unsw.edu.au/~cs1511/17s1/assignments/captcha/captcha.html) 
+ [3] Dataset nhóm 2: [CAPTCHA Images](https://www.kaggle.com/datasets/fournierp/captcha-version-2-images)
