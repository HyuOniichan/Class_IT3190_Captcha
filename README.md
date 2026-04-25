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
+ Output: Trong folder `dataset` sẽ có thêm 2 folder: `processed` (kết quả của stage 2) và `sggmented` (kết quả của stage 3)
+ Note: Stage chia theo 9 phần đã note trong docs

## Ref
+ Docs tổng: [ML 2025.2](https://docs.google.com/document/d/1g3PKIR1HZzpv9pxYNPCW63b5PtAFzVbOIYlK6n1ih1c/edit?usp=sharing)
+ Dataset hiện tại: [CAPTCHA Dataset](https://cgi.cse.unsw.edu.au/~cs1511/17s1/assignments/captcha/captcha.html) 
