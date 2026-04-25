import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import cv2
import pytesseract

def autolabel_images(images_dir="0_dataset/raw/ctt_sis", limit_run=20):
    _counter = 0

    for img_name in os.listdir(images_dir):
        img = cv2.imread(os.path.join(images_dir, img_name))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )

        config = '--psm 7 -c tessedit_char_whitelist=0123456789'
        text = pytesseract.image_to_string(thresh, config=config)

        # cv2.imwrite("debug.png", thresh)

        print(f"{_counter}:", end=" ")
        
        _counter += 1
        if _counter > limit_run:
            break

        print(text)

if __name__ == "__main__":
    autolabel_images()