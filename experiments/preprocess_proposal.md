# Đề xuất tiền xử lý nâng cao cho CAPTCHA (Dataset 1 & 2)

## 1) Mục tiêu

* Tìm combo tiền xử lý giúp tăng tỉ lệ tách đúng 4 ký tự ở dataset 2 (ảnh nhiễu), đồng thời không làm giảm đáng kể hiệu năng ở dataset 1 (ảnh sạch).
* Chưa dùng mô hình deep learning; đánh giá dựa trên chất lượng segmentation.

## 2) Cơ sở lý thuyết các kỹ thuật

### 2.1. Bộ lọc nhiễu (Filter)

* **Gaussian Blur**

  * Nguyên lý: làm mượt bằng hàm Gaussian, làm giảm nhiễu tần số cao.
  * Ưu: đơn giản, tốc độ nhanh.
  * Nhược: làm mờ cạnh, có thể làm mất nét chữ nhỏ.

* **Median Filter**

  * Nguyên lý: thay mỗi điểm bằng trung vị của vùng lân cận.
  * Ưu: giảm nhiễu hạt (salt & pepper) rất tốt, giữ cạnh tốt hơn Gaussian.
  * Nhược: có thể gây mất chi tiết nếu kernel quá lớn.

* **Bilateral Filter**

  * Nguyên lý: làm mượt theo không gian và độ tương đồng màu, giữ cạnh.
  * Ưu: giữ cạnh tốt.
  * Nhược: chậm hơn, đôi khi không tốt với nhiễu hạt mạnh.

### 2.2. Morphology (Erosion / Dilation)

* **Opening (Erosion → Dilation)**

  * Tác dụng: xóa nhiễu chấm nhỏ và đối tượng nhỏ lẻ.
  * Phù hợp: giảm nhiễu dot mà ít ảnh hưởng nét chữ.

* **Closing (Dilation → Erosion)**

  * Tác dụng: nối các nét đứt, làm đầy các khe nhỏ.
  * Phù hợp: xử lý chữ bị đứt do đường gạch, nhưng có thể làm dày quá mức.

### 2.3. Xóa đường gạch (Line Removal)

* **Morphological line removal**

  * Nguyên lý: dùng kernel dài theo chiều ngang / dọc để loại bỏ các đường thẳng.
  * Ưu: đơn giản, nhanh.
  * Nhược: dễ xóa nhầm nét chữ nếu tham số không phù hợp.

* **Hough + Inpaint**

  * Quy trình: Canny → HoughLinesP → tạo mask đường → inpaint để phục hồi vùng bị đường che.
  * Ưu: phát hiện đường thẳng rõ ràng, xóa đường tự nhiên hơn.
  * Nhược: chậm hơn, không luôn cải thiện khi đường không rõ.

## 3) Phương pháp đánh giá (Methodology)

* **Dataset:**

  * Dataset 1: `dataset/raw/1k_pbm` (ảnh sạch)
  * Dataset 2: `dataset_v2/raw` (ảnh nhiễu)
* **Số lượng mẫu:** 200 ảnh / dataset
* **Chỉ số chính:**

  * **Segmentation success rate:** tỉ lệ ảnh tách đúng 4 ký tự
  * $acc_{seg} = \frac{\#(\mathrm{seg\_count} = 4)}{N}$
* **Chỉ số phụ:**

  * `seg_count_mean`: số ký tự trung bình
  * `contour_count_mean`: số contour trung bình
  * `contour_area_mean`: diện tích contour trung bình (kiểm soát mất nét)
  * `foreground_ratio_mean`: tỉ lệ cân bằng foreground/background

## 4) Kết quả thực nghiệm

### 4.1. Filter comparison

**Dataset 1 (ảnh sạch):**

| Method       | seg_success_rate |
| ------------ | ---------------: |
| gaussian_k3  |            0.995 |
| gaussian_k5  |            0.995 |
| median_k3    |            0.985 |
| median_k5    |             0.82 |
| bilateral_d5 |            0.995 |
| bilateral_d7 |            0.995 |

**Dataset 2 (ảnh nhiễu):**

| Method        | seg_success_rate |
| ------------- | ---------------: |
| gaussian_k3   |             0.65 |
| gaussian_k5   |             0.67 |
| **median_k3** |        **0.745** |
| median_k5     |             0.74 |
| bilateral_d5  |            0.605 |
| bilateral_d7  |             0.61 |

**Nhận xét:** Median k=3 là tốt nhất cho dataset 2, Gaussian và Bilateral kém hơn.

### 4.2. Morphology study

**Dataset 1:**

| Method        | seg_success_rate |
| ------------- | ---------------: |
| open_k2       |            0.995 |
| open_k3       |            0.985 |
| close_k2      |             0.93 |
| close_k3      |             0.61 |
| open_close_k2 |             0.94 |
| close_open_k2 |             0.98 |

**Dataset 2:**

| Method        | seg_success_rate |
| ------------- | ---------------: |
| **open_k2**   |         **0.76** |
| open_k3       |            0.605 |
| close_k2      |            0.595 |
| close_k3      |            0.535 |
| open_close_k2 |            0.745 |
| close_open_k2 |            0.745 |

**Nhận xét:** Opening k=2 là ổn định nhất, closing lớn (k=3) làm giảm mạnh hiệu năng.

### 4.3. Line removal

**Dataset 2:**

| Method        | seg_success_rate |
| ------------- | ---------------: |
| line_morph_k3 |            0.195 |
| line_morph_k5 |             0.01 |
| line_hough_s1 |             0.62 |
| line_hough_s2 |            0.605 |

**Nhận xét:** Hough + inpaint tốt hơn morph line removal, nhưng vẫn thấp hơn combo filter + morph.

### 4.4. Combo experiments

**Dataset 1:**

| Combo                            | seg_success_rate |
| -------------------------------- | ---------------: |
| combo_median_k3_open_k2          |            0.985 |
| combo_median_k3_open_k2_hough_s1 |            0.985 |
| combo_gaussian_k3_open_k2        |            0.995 |

**Dataset 2:**

| Combo                            | seg_success_rate |
| -------------------------------- | ---------------: |
| **combo_median_k3_open_k2**      |         **0.76** |
| combo_median_k3_open_k2_hough_s1 |             0.76 |
| combo_gaussian_k3_open_k2        |            0.695 |

**Nhận xét:** Combo median_k3 + open_k2 là tốt nhất trên dataset 2, giữ ổn định trên dataset 1.

## 5) Đề xuất cuối cùng

### 5.1. Combo mặc định (khuyến nghị)

* **Median k=3 + Opening k=2**
* Lý do: đạt seg_success cao nhất trên dataset 2 (0.76) và ít giảm trên dataset 1 (0.985).

### 5.2. Combo tùy chọn khi có nhiều đường gạch

* **Median k=3 + Opening k=2 + Hough line removal**
* Chỉ dùng nếu ảnh có line rõ, vì thời gian chậm hơn và không tăng kết quả trung bình.

### 5.3. Khi nào chọn Gaussian?

* Nếu ưu tiên dataset 1 (ảnh rất sạch) và dataset 2 không quan trọng.
* Gaussian k=3 + Opening k=2 giữ 0.995 cho dataset 1, nhưng giảm còn 0.695 cho dataset 2.
