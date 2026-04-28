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


## Ref
+ Docs tổng: [ML 2025.2](https://docs.google.com/document/d/1g3PKIR1HZzpv9pxYNPCW63b5PtAFzVbOIYlK6n1ih1c/edit?usp=sharing)
+ Dataset hiện tại: [CAPTCHA Dataset](https://cgi.cse.unsw.edu.au/~cs1511/17s1/assignments/captcha/captcha.html) 
