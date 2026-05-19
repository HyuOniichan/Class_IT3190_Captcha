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

  * Dataset 1: `dataset/raw/1k_pbm` (ảnh sạch, 4 ký tự)
  * Dataset 2 (raw): `dataset_v2/raw` (ảnh nhiễu tổng hợp, 4 ký tự)
  * Dataset 2 (Kaggle): `dataset_v2/kaggle_captcha` (ảnh Kaggle, 5 ký tự)
* **Số lượng mẫu:** 200 ảnh / mỗi dataset
* **Chỉ số chính:**

  * **Segmentation success rate:** tỉ lệ ảnh tách đúng 4 ký tự
  * $acc_{seg} = \frac{\#(\mathrm{seg\_count} = 4)}{N}$
* **Chỉ số phụ:**

  * `seg_count_mean`: số ký tự trung bình
  * `contour_count_mean`: số contour trung bình
  * `contour_area_mean`: diện tích contour trung bình (kiểm soát mất nét)
  * `foreground_ratio_mean`: tỉ lệ cân bằng foreground/background

## 4) Kết quả thực nghiệm

### 4.1. Dataset 1 (ảnh sạch, 4 ký tự)

**Filter comparison**

| Method       | seg_success_rate |
| ------------ | ---------------: |
| gaussian_k3  |            0.995 |
| gaussian_k5  |            0.995 |
| median_k3    |            0.985 |
| median_k5    |             0.82 |
| bilateral_d5 |            0.995 |
| bilateral_d7 |            0.995 |

**Morphology study**

| Method        | seg_success_rate |
| ------------- | ---------------: |
| open_k2       |            0.995 |
| open_k3       |            0.985 |
| close_k2      |             0.93 |
| close_k3      |             0.61 |
| open_close_k2 |             0.94 |
| close_open_k2 |             0.98 |

**Combo experiments**

| Combo                            | seg_success_rate |
| -------------------------------- | ---------------: |
| combo_median_k3_open_k2          |            0.985 |
| combo_median_k3_open_k2_hough_s1 |            0.985 |
| combo_gaussian_k3_open_k2        |            0.995 |

**Nhận xét:** Dataset 1 ổn định nhất với Gaussian/Bilateral. Median k=3 vẫn đạt mức tốt và chấp nhận được.

### 4.2. Dataset 2 (raw, 4 ký tự)

**Filter comparison**

| Method        | seg_success_rate |
| ------------- | ---------------: |
| gaussian_k3   |            0.665 |
| gaussian_k5   |            0.675 |
| median_k3     |             0.74 |
| **median_k5** |        **0.745** |
| bilateral_d5  |            0.595 |
| bilateral_d7  |            0.625 |

**Morphology study**

| Method        | seg_success_rate |
| ------------- | ---------------: |
| **open_k2**   |         **0.76** |
| open_k3       |            0.595 |
| close_k2      |            0.575 |
| close_k3      |            0.535 |
| open_close_k2 |             0.75 |
| close_open_k2 |            0.745 |

**Line removal**

| Method        | seg_success_rate |
| ------------- | ---------------: |
| line_morph_k3 |              0.2 |
| line_morph_k5 |             0.01 |
| line_hough_s1 |             0.61 |
| line_hough_s2 |            0.595 |

**Combo experiments**

| Combo                            | seg_success_rate |
| -------------------------------- | ---------------: |
| **combo_median_k3_open_k2**      |        **0.755** |
| combo_median_k3_open_k2_hough_s1 |            0.755 |
| combo_gaussian_k3_open_k2        |             0.68 |

**Nhận xét:** Opening k=2 là ổn định nhất. Line removal bằng Hough có cải thiện so với morph line removal nhưng không vượt combo filter+morph.

### 4.3. Dataset 2 (Kaggle, 5 ký tự)

**Filter comparison**

| Method        | seg_success_rate |
| ------------- | ---------------: |
| gaussian_k3   |             0.06 |
| **gaussian_k5** |          **0.105** |
| median_k3     |            0.015 |
| median_k5     |            0.015 |
| bilateral_d5  |             0.01 |
| bilateral_d7  |            0.005 |

**Morphology study**

| Method        | seg_success_rate |
| ------------- | ---------------: |
| open_k2       |            0.025 |
| **open_k3**   |         **0.275** |
| close_k2      |            0.005 |
| close_k3      |              0.0 |
| open_close_k2 |            0.015 |
| close_open_k2 |             0.02 |

**Line removal**

| Method        | seg_success_rate |
| ------------- | ---------------: |
| **line_morph_k3** |        **0.105** |
| line_morph_k5 |             0.02 |
| line_hough_s1 |            0.015 |
| line_hough_s2 |             0.01 |

**Combo experiments**

| Combo                            | seg_success_rate |
| -------------------------------- | ---------------: |
| combo_median_k3_open_k2          |             0.02 |
| combo_median_k3_open_k2_hough_s1 |             0.03 |
| **combo_gaussian_k3_open_k2**    |              0.1 |

**Nhận xét:** Kaggle rất khó với pipeline hiện tại. Kết quả tốt nhất vẫn thấp (open_k3 = 0.275). Cần nghiên cứu lại threshold/segmentation cho dữ liệu này.

## 5) Đề xuất cuối cùng

### 5.1. Dataset 2 (raw) - combo mặc định (khuyến nghị)

* **Median k=3 + Opening k=2**
* Lý do: đạt seg_success cao nhất trên dataset 2 raw (0.755) và ít giảm trên dataset 1 (0.985).

### 5.2. Dataset 2 (raw) - tùy chọn khi có nhiều đường gạch

* **Median k=3 + Opening k=2 + Hough line removal**
* Chỉ dùng nếu ảnh có line rõ, vì thời gian chậm hơn và không tăng kết quả trung bình.

### 5.3. Dataset 1 (ảnh sạch)

* Gaussian hoặc Bilateral vẫn giữ mức cao nhất (0.995).
* Median k=3 vẫn ổn và phù hợp nếu muốn thống nhất pipeline với dataset 2 raw.

### 5.4. Dataset 2 (Kaggle)

* Pipeline hiện tại chưa phù hợp: seg_success thấp ở mọi cấu hình.
* Tạm thời tốt nhất là **open_k3** (0.275) hoặc **gaussian_k5** (0.105), nhưng cần cải tiến mạnh:
  * Điều chỉnh threshold (adaptive với tham số khác) và đảo màu phù hợp.
  * Tinh chỉnh segmentation: min_area, aspect ratio, min_height, và xử lý kí tự dính.
