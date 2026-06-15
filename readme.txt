# Hướng dẫn cài đặt và chạy chương trình

## 1. Chuẩn bị môi trường
1. Tải mã nguồn về máy.
2. Mở terminal tại thư mục dự án `IT3190-Group_14-Captcha_Recognition`.
3. Tạo môi trường ảo:
   `python -m venv .venv`
4. Kích hoạt môi trường ảo trên Windows:
   `.venv\Scripts\Activate`
5. Cài đặt phụ thuộc:
   `pip install -r requirements.txt`

## 2. Cấu trúc lệnh chính
Chạy chương trình bằng `main.py` với các tham số sau:
- `--stage`: Chọn giai đoạn chạy.
  - `0`: Chạy toàn bộ pipeline (tiền xử lý + lựa chọn mô hình) cho dataset hỗ trợ.
  - `1`: Chỉ chạy tiền xử lý.
  - `2`: Chỉ chạy lựa chọn mô hình trên dữ liệu đã được chuẩn bị.
- `--dataset`: Chọn dataset.
  - `lv0_emnist` (Nhóm dataset lv0)
  - `lv1_1k_pbm` (Nhóm dataset lv1)
  - `--dim_reduction_method`: Phương pháp giảm chiều (Hiện tại chỉ hỗ trợ `pca`).

## 3. Ví dụ chạy
### 3.1 Chạy toàn bộ pipeline cho dataset `lv0_emnist`
`python main.py --stage 0 --dataset lv0_emnist --dim_reduction_method pca`

### 3.2 Chỉ tiền xử lý dataset `lv1_1k_pbm`
`python main.py --stage 1 --dataset lv1_1k_pbm --dim_reduction_method pca`

### 3.3 Chỉ model selection sau khi đã tiền xử lý
`python main.py --stage 2 --dataset lv1_1k_pbm --dim_reduction_method pca`

## 4. Output tạo ra
- Kết quả tiền xử lý được lưu trong `output/dataset/<dataset>/...`
- Kết quả lựa chọn mô hình được lưu trong `output/models/<dataset>/...`

## 5. Lưu ý
- Với `stage=0`, chương trình hiện chỉ hỗ trợ tự động tiền xử lý và model selection cho `lv0_emnist` và `lv1_1k_pbm`.
- Nếu chọn `stage=2`, dữ liệu `train.npz` và `test.npz` phải có sẵn trong thư mục `output/dataset/<dataset>/build`.
- Nếu không cần giảm chiều, tham số `--dim_reduction_method` có thể đặt giá trị hợp lệ là `pca` hoặc bỏ qua nếu không cần.

## 6. Phần mềm và thư viện sử dụng
- Python 3.x
- OpenCV (`cv2`)
- NumPy
- scikit-learn
- PyTorch
