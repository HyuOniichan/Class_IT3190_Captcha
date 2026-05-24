# Đánh giá Vertical Projection Profile Segmentation (VP)

## Phát hiện, thư mục kaggle có cả file 4 kí tự, đánh giá là không chuẩn và kết quả thực tế cao hơn

## 1) Mục tiêu

* Đánh giá hiệu năng phương pháp Vertical Projection Profile cho tách ký tự CAPTCHA trên datasetdataset_v2 gồm raw và kaggle_captcha.
* sử dụng cơ chế lọc bỏ ảnh segment có độ sáng cao >93% pixel trắng(ảnh nhiễu).
* Xác định bộ siêu tham số tối ưu cho mỗi loại dataset.
* Đánh giá bằng số ảnh cắt trên tổng kí tự cho từng ảnh.

## 2) Phương pháp Vertical Projection Profile

### 2.1. Nguyên lý hoạt động

* **Dữ liệu đầu vào:** ảnh nhị phân (đen–trắng), ký tự đen trên nền trắng.
* **Bước 1 – Chuẩn bị:** Ensure binary image, invert nếu cần (foreground = black).
* **Bước 2 – Tính projection:** Đếm số điểm đen (foreground) mỗi cột → vector 1D.
* **Bước 3 – Làm mịn projection:** Áp dụng Gaussian blur trên vector 1D.
* **Bước 4 – Tìm gap:** Phát hiện cột có projection < gap_ratio × max_projection.
* **Bước 5 – Cắt segment:** Tách các khoảng giữa các gap thành segment riêng biệt.
* **Bước 6 – Lọc bỏ segment sáng cao:** Loại bỏ segment nếu số pixel sáng (255) > 93%.
* **Bước 7 – Resize:** Chuẩn hóa kích thước mỗi segment về 28×28.

### 2.2. Ưu điểm

* **Dataset 1 (ảnh sạch):** Hoạt động rất tốt, không yêu cầu preprocessing phức tạp.
* **Nhiễu muối tiêu (salt & pepper):** Phương pháp projection robust với loại nhiễu này.
* **Tốc độ:** Nhanh, không cần contour detection, chỉ dùng projection 1D.

### 2.3. Hạn chế

* **Nhiễu gạch / đường:** Hay cắt sai khi có đường gạch ngang cắt ngang toàn bộ ảnh.
  * **Giải pháp:** Cần khử nhiễu/đường trước (preprocessing) mới dùng được.
* **Cắt nhầm khoảng trắng:** Khoảng trắng chứa nhiễu có thể được nhận dạng là segment riêng.
  * **Giải pháp:** Thêm chức năng xóa ký tự nếu số ô trắng > 93% → loại bỏ "ghost segments".

### 2.4. Cơ chế lọc bỏ ảnh sáng cao

Sau khi cắt segment, mỗi segment được:

1. **Binarize lại** (threshold = 127): Đảm bảo ảnh là binary (đen–trắng).
2. **Kiểm tra độ sáng:**
   - Đếm số pixel sáng (giá trị = 255).
   - Tính tỷ lệ: `white_ratio = count(pixel==255) / total_pixels`
   - Nếu `white_ratio > 0.93` → loại bỏ segment.
3. **Tác dụng:** Loại bỏ "ghost segments" (khoảng trắng, nền, nhiễu cao).
4. **Resize:** Chỉ những segment hợp lệ mới resize về 28×28.

## 3) Siêu tham số

| Tham số | Miền giá trị | Ý nghĩa |
| --- | --- | --- |
| `gap_ratio` | 0.08 – 0.20 | Ngưỡng phát hiện gap (% của max projection) |
| `min_width` | 4 – 8 | Chiều rộng tối thiểu (pixel) của segment |
| `smooth_kernel` | 5, 9, 11, 21 | Kích thước kernel Gaussian làm mịn projection |
| `max_white_ratio` | 0.93 (fixed) | Ngưỡng loại bỏ segment quá sáng |
| `char_size` | (28, 28) (fixed) | Kích thước output mỗi ký tự |

## 4) Phương pháp đánh giá

### 4.1. Dataset

* **Dataset v2 (raw):** `dataset_v2/raw` → ảnh tổng hợp nhiễu, 4 ký tự / ảnh.
* **Dataset v2 (Kaggle):** `dataset_v2/kaggle_captcha` → ảnh Kaggle, 5 ký tự / ảnh.

### 4.2. Chỉ số chính

$$\text{Correct Ratio} = \frac{\text{# ảnh tách đúng số ký tự kỳ vọng}}{\text{Tổng ảnh}}$$

* **Raw (v1 & v2):** Kỳ vọng = 4 ký tự.
* **Kaggle:** Kỳ vọng = 5 ký tự.

### 4.3. Chỉ số phụ

* `mean_segments`: Trung bình số segment/ảnh.
* `std_segments`: Độ lệch chuẩn (mục tiêu: thấp).
* `segment_distribution`: Phân bố số segment (hỗ trợ phân tích chi tiết).

## 5) Kết quả thực nghiệm

### 5.1. dataset v2 raw

**Top 5 cấu hình tốt nhất:**

| Rank | gap_ratio | min_width | smooth_kernel | Correct Ratio | Seg Count Mean | Std Dev |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.2 | 8 | 11 | **85.39%** | 3.861 | 0.884 |
| 2 | 0.2 | 6 | 11 | 85.15% | 3.863 | 0.885 |
| 3 | 0.2 | 4 | 11 | 85.10% | 3.865 | 0.888 |
| 4 | 0.2 | 8 | 9 | 83.87% | 3.892 | 0.892 |
| 5 | 0.2 | 6 | 9 | 83.61% | 3.895 | 0.895 |

**Nhận xét:**
* Cấu hình tối ưu: `gap_ratio=0.2, min_width=8, smooth_kernel=11`.

### 5.2. Dataset v2 Kaggle – Ảnh Kaggle 5 ký tự

**Top 5 cấu hình tốt nhất:**

| Rank | gap_ratio | min_width | smooth_kernel | Correct Ratio | Seg Count Mean | Std Dev |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.20 | 4 | 5 | **8.80%** | 3.269 | 1.118 |
| 2 | 0.20 | 6 | 5 | 8.59% | 3.265 | 1.113 |
| 3 | 0.20 | 8 | 5 | 8.42% | 3.264 | 1.111 |
| 4 | 0.20 | 4 | 9 | 4.18% | 2.765 | 1.296 |
| 5 | 0.20 | 6 | 9 | 4.18% | 2.765 | 1.296 |

**Nhận xét:**
* Cấu hình tối ưu: `gap_ratio=0.2, min_width=4, smooth_kernel=5`.
* **tỷ lệ 8.80% vẫn thấp** → chưa có ý nghĩa, cần cải thiện phương pháp hoặc cần preprocessing tốt hơn.

## 6) Ảnh hưởng của cơ chế lọc bỏ ảnh sáng cao (>93%)

### 6.1. So sánh trước và sau lọc

**Trước lọc (segment không qua kiểm tra sáng):**
* Nhiều "ghost segments" (khoảng trắng, nền, nhiễu) được chọn.
* Dataset v2 raw ~40%
* Dataset v2 kaggle 0.3% – 0.5%

**Sau lọc (loại bỏ segment với white_ratio > 0.93):**
* Ghost segments bị loại, chỉ ký tự thực (foreground tối) được giữ.
* Dataset v2 raw ~80%
* Dataset v2 kaggle ~8.8%

### 6.2. Tác dụng

* **Giảm false positive:** Loại bỏ segment không chứa ký tự thực.
* **Tăng precision:** Ít segment "sai", chỉ giữ segment có chứa foreground đặc.
* **Nhược điểm:** Có thể loại bỏ nhầm ký tự nhạt hoặc bị phủ nhiễu (foreground cũng sáng).
* **Tuning:** Có thể điều chỉnh ngưỡng 0.93 nếu cần (hiện tại cộng định).

## 7) Khuyến nghị

### 7.1. Khi nào dùng VP Segmentation

✅ **Dùng được:**
* Dataset v1 (ảnh sạch, PBM): hoạt động rất tốt (~80% tỷ lệ cắt đúng 4 ảnh).
* Ảnh với nhiễu muối tiêu (salt & pepper).

❌ **Không nên dùng:**
* Ảnh với nhiễu gạch / đường ngang rõ: VP có tỉ lệ cắt sai gộp nhiều kí tự làm 1 hoặc cắt riêng gạch.
* Ảnh chông chéo biến dạng.


### 7.2. Khi dùng VP Segmentation

1. **Trước:** Áp dụng preprocessing (median blur + morphological opening) để khử nhiễu.
2. **Chọn cấu hình mặc định:** `gap_ratio=0.2, min_width=8, smooth_kernel=11`.
3. **Kiểm tra:** Xác nhận segment sau khi lọc sáng (loại >93%) là hợp lệ.


