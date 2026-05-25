# Đánh giá Mô hình CRNN End-to-End Nhận diện CAPTCHA (Sequence-based)

Tệp báo cáo này đánh giá hiệu năng nhận diện CAPTCHA dạng chuỗi không qua phân đoạn (End-to-End) bằng mô hình CRNN (Convolutional Recurrent Neural Network) kết hợp với hàm mất mát CTC (Connectionist Temporal Classification) trên tập dữ liệu `dataset_v2`.

CRNN được chọn thay vì CNN vì sự kết hợp giữa BiLSTM và CTC Loss cho phép mô hình hóa CAPTCHA dưới dạng chuỗi đặc trưng để nhận diện các từ có độ dài biến đổi (4 hoặc 5 ký tự) mà không cần phân đoạn vật lý hay cố định kích thước đầu ra, do đó tốt hơn khi CAPTCHA có nhiều kí tự.

---

## 1. Mục tiêu thử nghiệm
* Đánh giá khả năng nhận dạng toàn bộ chuỗi ký tự CAPTCHA trực tiếp từ ảnh gốc (không cần cắt riêng lẻ từng ký tự - Segmentation-free).
* Tích hợp bộ tiền xử lý từ mô-đun trung tâm `preprocess` để đảm bảo tính nhất quán của hệ thống.
* Đo lường độ chính xác khớp từ (Word-level accuracy), độ chính xác ký tự (Character-level accuracy) và khoảng cách chỉnh sửa Levenshtein trung bình trên 3 tập nguồn khác nhau.

---

## 2. Kiến trúc mô hình CRNN
Hệ thống sử dụng mô hình kết hợp mạng tích chập (CNN) và mạng nơ-ron hồi quy (RNN) để xử lý dữ liệu dạng chuỗi:

* **CNN Feature Extractor**: Gồm 5 lớp tích chập (Convolutional) kết hợp Batch Normalization, ReLU và MaxPool. 
  * Định dạng đầu vào: Ảnh xám kích thước $32 \times 128 \times 1$ (Chiều cao $\times$ Chiều rộng $\times$ Số kênh).
  * Đầu ra: Bản đồ đặc trưng (Feature map) kích thước $256 \times 1 \times 32$.
* **Map to Sequence**: Squeeze chiều cao bản đồ đặc trưng về $256 \times 32$, sau đó chuyển đổi vị trí các trục thành chuỗi đầu vào dạng thời gian có độ dài $T = 32$ (mỗi bước thời gian tương ứng với một vùng cửa sổ trượt dọc theo chiều rộng ảnh CAPTCHA).
* **RNN Sequence Modeler**: Mạng LSTM 2 lớp hai chiều (Bidirectional LSTM) với kích thước ẩn (hidden size) là 128, đầu ra là đặc trưng chuỗi kích thước $256$.
* **Classifier & Decoding**:
  * Lớp tuyến tính (Linear) chiếu đặc trưng chuỗi về $37$ chiều (tương ứng với 36 ký tự chữ và số trong CHARSET + 1 lớp token blank đặc trưng cho CTC).
  * Giải mã Greedy CTC (Greedy Decoder): Tìm ký tự có xác suất lớn nhất tại mỗi bước thời gian, gộp các ký tự lặp liên tiếp và loại bỏ token blank để cho ra chuỗi dự đoán cuối cùng.

---

## 3. Cấu hình tiền xử lý tích hợp (Preprocessing Integration)
Để tuân thủ yêu cầu không tự xử lý bên ngoài, toàn bộ luồng tiền xử lý được gọi trực tiếp từ lớp chiến lược `Dataset2PreprocessingStrategy` trong mô-đun trung tâm `preprocess.strategies`.

### Siêu tham số tiền xử lý sử dụng:
* `size`: $(128, 32)$ (resize về định dạng chuẩn của mô hình CRNN).
* `maintain_aspect_ratio`: `False` (bắt buộc resize cố định chiều rộng về 128 để đồng bộ kích thước ma trận huấn luyện).
* `denoise_method`: `"median"` (kernel size = 3).
* `threshold_method`: `"adaptive"` (Gaussian threshold nhị phân đảo màu nền đen, chữ trắng để nổi bật nét chữ).
* `line_removal_kernel_size`: $3$ (khử các đường gạch nhiễu ngang dọc).
* `normalize`: `True` (chuẩn hóa giá trị điểm ảnh từ $[0, 255]$ sang $[0.0, 1.0]$ kiểu `float32`).

---

## 4. Thiết lập thực nghiệm (Methodology)
* **Tập dữ liệu**: 10,040 ảnh CAPTCHA lấy từ 3 thư mục:
  * `dataset_v2/kaggle_captcha` (ảnh nhãn 5 ký tự và 4 ký tự)
  * `dataset_v2/raw` (ảnh nhãn 4 ký tự sinh tự động có nhiễu)
  * `dataset_v2/processed` (ảnh đã lọc trước đó)
* **Phân chia dữ liệu**: 80% Train (`8,032` mẫu), 20% Validation (`2,008` mẫu).
* **Thuật toán huấn luyện**:
  * Chạy trên: CPU (Windows).
  * Số lượng Epochs: `20` epochs.
  * Kích thước Batch (Batch Size): `64`.
  * Bộ tối ưu hóa (Optimizer): Adam (Learning rate = `1e-3`).
  * Bộ giảm tỷ lệ học (Scheduler): ReduceLROnPlateau (giảm 0.5 lần nếu loss không cải thiện sau 3 epochs).
  * Hàm mất mát: `nn.CTCLoss(blank=0, zero_infinity=True)`.

---

## 5. Kết quả thực nghiệm

### 5.1. Quá trình huấn luyện
* **Thời gian huấn luyện**: `1167.5` giây (~19.5 phút).
* **Validation Loss tốt nhất**: đạt **`0.5875`** tại epoch thứ 14 (mô hình tối ưu đã được lưu tự động thành `crnn_model.pth`).
* Mức Loss giảm nhanh và ổn định, chứng tỏ mô hình học tốt và không bị overfit nghiêm trọng.

### 5.2. Chỉ số chính trên tập Validation (2,008 ảnh)

| Chỉ số đánh giá | Kết quả đạt được | Ý nghĩa |
| --- | --- | --- |
| **Word-level Accuracy** | **`60.06%`** | Tỷ lệ khớp chính xác 100% toàn bộ chuỗi CAPTCHA |
| **Character-level Accuracy** | **`84.26%`** | Tỷ lệ ký tự riêng lẻ dự đoán chính xác |
| **Average Edit Distance** | **`0.5876`** | Số thao tác chỉnh sửa trung bình để khớp nhãn đúng |

### 5.3. Chi tiết độ chính xác theo từng nguồn dữ liệu (Per-source Breakdown)

| Thư mục nguồn | Số mẫu | Word-level Accuracy | Character-level Accuracy | Avg Edit Distance |
| --- | --- | --- | --- | --- |
| **raw** | 783 | **76.12%** | 92.25% | 0.3538 |
| **kaggle_captcha** | 366 | **63.39%** | 89.11% | 0.5191 |
| **processed** | 859 | **44.00%** | 74.64% | 0.8291 |

**Nhận xét:**
1. **Hiệu năng xuất sắc trên raw và kaggle_captcha**: Đạt độ chính xác từ tương ứng là `76.12%` và `63.39%` chỉ sau 20 epochs trên CPU. Điều này chứng minh mô hình CRNN học các đặc trưng chuỗi rất mạnh mẽ mà không cần phải thực hiện bước cắt ký tự (segmentation) vốn cực kỳ bất ổn định trên ảnh nhiễu.
2. **Hiệu năng thấp hơn trên tập processed**: Tập `processed` chỉ đạt `44.00%` Word Accuracy. Lý do là vì tập này đã bị áp dụng một số phương pháp lọc nét đứt ở các giai đoạn trước, làm ký tự bị mất mát nét nghiêm trọng (nét mỏng, đứt đoạn), khiến mô hình CRNN gặp khó khăn hơn so với việc tự trích xuất đặc trưng từ ảnh gốc qua bộ tiền xử lý chuẩn hóa mới.

---

## 6. Lợi ích – Hạn chế

### 6.1. Lợi ích (Ưu điểm)
* **Xử lý CAPTCHA chuỗi nhiều ký tự và biến dạng**: Nhận diện tốt các CAPTCHA có độ dài chuỗi ký tự thay đổi (cả 4 và 5 ký tự) mà không cần thay đổi hay cấu hình lại kiến trúc mạng.
* **Không cần phân đoạn vật lý (Segmentation-free)**: Bỏ qua hoàn toàn việc tìm kiếm khoảng trắng hoặc cắt rời ký tự vật lý – bước cực kỳ bất ổn định và dễ thất bại khi chữ bị dính nhau, nghiêng hoặc chông chéo.
* **Độ chính xác**: Đạt tỷ lệ nhận dạng đúng toàn chuỗi cao so với phương pháp phân đoạn (tập Kaggle **`63.39%`**).

### 6.2. Hạn chế (Nhược điểm)
* **Thời gian huấn luyện lâu hơn**: Mô hình CRNN tuần tự yêu cầu nhiều thời gian tính toán hơn khi huấn luyện (huấn luyện trên CPU mất khoảng 20 phút cho 20 epochs), khuyến nghị nên sử dụng card đồ họa (GPU) để tối ưu.
* **Độ phức tạp và giám sát dữ liệu**: Đòi hỏi lượng dữ liệu gán nhãn lớn hơn (ít nhất hàng ngàn ảnh) để hàm mất mát CTC tự căn chỉnh chuỗi ký tự hiệu quả.
