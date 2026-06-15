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
  + Sau khi xử lý xong 9 phần trong docs, sẽ quay lại nâng cấp dataset và build lại model sau

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

### 1/6/2026 - Huy
+ Refactor toàn bộ source code
+ Cấu trúc folder:
```
project
|-- dataset
    |-- lv1_1k_pbm
    |-- lv2_1k_5digits
|-- models
    |-- cnn.py
    |-- knn.py
    |-- svm.py
    |-- utils.py
|-- output
    |-- dataset
        |-- lv1_1k_pbm
            |-- preprocessed
            |-- segmented
            |-- meta
            |-- build
        |-- lv2_1k_5digits
    |-- models
|-- preprocess
    |-- lv1.py
    |-- utils.py
|-- main.py
|-- README.md
|-- requirements.txt
```

+ Cấu trúc mã nguồn:
  + `main.py`
    + Là entrypoint của dự án.
    + Chạy pipeline theo tham số:
      + `--stage`: 0 = chạy pipeline hoàn chỉnh, 1 = chỉ tiền xử lý, 2 = chỉ lựa chọn mô hình.
      + `--dataset`: chọn dataset dể chạy (lv0_emnist, lv1_1k_pbm, lv2_1k_5digits).
      + `--dim_reduction_method`: hiện chỉ hỗ trợ `pca`.
    + Với `stage=0`, nếu dataset là `lv0_emnist` hoặc `lv1_1k_pbm` thì gọi:
      + `preprocess_emnist()` hoặc `preprocess_1k_pbm()`
      + `model_selection_pipeline(...)`

  + `preprocess/` sẽ gồm các giai đoạn tiền xử lý cho từng nhóm dataset:
    + `utils.py` - các hàm để hỗ trợ xử lý dữ liệu
    + `lv0.py` - pipeline hoàn chỉnh để xử lý dữ liệu cho dataset lv0
    + `lv1.py` - pipeline xử lý dữ liệu cho dataset lv1 
    + Xử lý cho từng dataset, và ghi ra output tương ứng. 
    + Ví dụ: Với dataset lv1 `lv1_1k_pbm` -> Output lưu tại: `output/dataset/lv1_1k_pbm/`
      + Bắt đầu với xử lý đơn giản -> Output: folder `preprocessed/`
      + Từ ảnh của một chuỗi số cắt thành ảnh của từng chữ cái -> Output: `segmented/`
      + Sau khi có tập ảnh thì chia train test và ghi metadata -> Output: `meta/`
      + Chuẩn bị dataset để sẵn sàng học -> Output: `build/`
  
  + `models/` tương ứng với giai đoạn **Model Selection**
    + Gồm các mô hình Học máy, load train/test data từ folder `build/`.
    + Khởi tạo class chung `BaseModelClass` (`base.py`), gồm các method chính:
      + `prepare()` - Chuẩn bị train/test dataset cho model
      + `run()` - Khởi tạo và chạy (huấn luyện) model bằng cách truyền các tham số yêu cầu của model vào, có thể lưu lại model.
      + `predict()` - Sử dụng model đã lưu để dự đoán giá trị cho ảnh đầu vào, trả về nhãn dự đoán và xác suất cho từng lớp (confidence).
    + Chuẩn bị class cho model và các siêu tham số tương ứng:
      + `knn.py` - KNN
        + k: range $[1, 25]$
        + metric: ["minkowski", "manhattan", "euclidean", "cosine"]
        + p = $2$ (khoảng cách Euclidean)
      + `decision_tree.py` - Decision Tree
        + max_depth: $[2, 10]$
      + `random_forest.py` - Random Forest
        + n_estimators: $[5, 10, 15, 20, 30, 50, 75, 100, 150]$
      + `svm.py` - SVM / LinearSVM
        + kernel: ["linear", "poly", "rbf", "sigmoid"]
        + C: $[0.1, 1.0, 2.0, 5.0, 10.0]$
      + `cnn.py` - CNN
        + Architecture:
          + Conv(1->32, 3) -> ReLU -> MaxPool(2)
          + Conv(32->64, 3) -> ReLU -> MaxPool(2)
          + Conv(64->128, 3) -> ReLU
          + Flatten -> FC(512) -> ReLU -> Dropout -> FC(num_classes)
          + After two 2x2 max-pools the spatial size is 5x5 (28->13->5 with valid padding inside each pool window), giving 128*5*5 = 3200 features before the FC layers.
        + Other hyperparams:
          + optimizer: ["Adam", "SGD"],
          + lr: $[10^{-2}, 10^{-3}, 10^{-4}]$,
          + batch_size: $[32, 64, 128]$,
          + epochs: $[5, 10, 20]$,
    + Các file trong thư mục `lv2` là quy trình xây dựng model để xử lý các dataset `lv2`.

  + `output/` sẽ chứa tất cả output của các phần sau khi
  + Github:  chạy `main.py`
    + `output/dataset` - Kết quả sau khi chạy Stage 1 (mới) - Preprocessing
    + `output/models` - Kết quả chạy Stage 2 - Model Selection

  + `weights/` gồm 2 files chính:
    + `save_models.py` - Chạy và lưu mô hình với các tham số tốt nhất (lựa chọn từ pha model selection)
    + `load_models.py` - Chạy thử các mô hình với ảnh được lựa chọn
    + Các mô hình được lưu tại `weights/<dataset_name>/<model_name>.joblib`

  + `scripts` tập hợp các file phụ để đo số liệu báo cáo, ví dụ như `benchmark.py` để đo dung lượng và thời gian chạy mô hình.

  + `web` chứa mã nguồn để chạy giao diện web:\
    + `index.html`, `style.css`, `script.js` để tạo giao diện web
    + `app.py` - file chính để khởi động server backend 
    + `predictor.py` - khởi tạo và lựa chọn mô hình dự đoán


## Ref
+ Docs tổng: [ML 2025.2](https://docs.google.com/document/d/1g3PKIR1HZzpv9pxYNPCW63b5PtAFzVbOIYlK6n1ih1c/edit?usp=sharing)
+ Github: [Class_IT3190_Captcha](https://github.com/HyuOniichan/Class_IT3190_Captcha/)
+ Dataset lv0 (lv0_emnist): [EMNIST Dataset](https://www.kaggle.com/datasets/crawford/emnist) 
+ Dataset lv1 (lv1_1k_pbm): [CAPTCHA Dataset](https://cgi.cse.unsw.edu.au/~cs1511/17s1/assignments/captcha/captcha.html) 
+ Dataset lv2 (lv2_1k_5digits): [CAPTCHA Dataset](https://www.kaggle.com/datasets/fournierp/captcha-version-2-images)
