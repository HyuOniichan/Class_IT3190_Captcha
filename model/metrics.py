import numpy as np


def compute_image_accuracy(all_preds, all_labels, chars_per_image=4):
    """
    Per-image accuracy: nhóm các ký tự liên tiếp thành từng ảnh CAPTCHA gốc,
    chỉ tính đúng khi TẤT CẢ ký tự trong ảnh đều đúng.

    Parameters
    ----------
    all_preds : array-like, shape (N,)
    all_labels : array-like, shape (N,)
    chars_per_image : int
        Số ký tự trong mỗi ảnh CAPTCHA gốc (mặc định = 4 cho dataset 1k_pbm).

    Returns
    -------
    image_acc : float
        Tỉ lệ ảnh CAPTCHA được đoán đúng toàn bộ ký tự.
    n_images : int
        Số ảnh CAPTCHA đã đánh giá (bỏ phần dư nếu N không chia hết).
    """
    all_preds = np.asarray(all_preds)
    all_labels = np.asarray(all_labels)

    n = len(all_preds)
    n_images = n // chars_per_image
    if n_images == 0:
        return 0.0, 0

    usable = n_images * chars_per_image
    preds_grouped = all_preds[:usable].reshape(n_images, chars_per_image)
    labels_grouped = all_labels[:usable].reshape(n_images, chars_per_image)

    correct_images = np.all(preds_grouped == labels_grouped, axis=1).sum()
    return correct_images / n_images, n_images
