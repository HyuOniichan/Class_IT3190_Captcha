import os
import time
import random
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def scrape_images(url="https://ctt-sis.hust.edu.vn/Account/Login.aspx", save_folder="0_dataset/raw/ctt_sis", num_images=1000):
    os.makedirs(save_folder, exist_ok=True)

    session = requests.Session()

    for i in range(num_images):
        try:
            # Load page to get new captcha
            res = session.get(url)
            soup = BeautifulSoup(res.text, "html.parser")

            img = soup.select_one("img#ctl00_ctl00_contentPane_MainPanel_MainContent_ASPxCaptcha1_IMG")
            img_url = urljoin(url, img["src"])

            # Get image
            img_data = session.get(img_url).content

            file_path = os.path.join(save_folder, f"captcha_{i}.png")
            with open(file_path, "wb") as f:
                f.write(img_data)

            print(f"[{i}] Saved")

            # Delay to avoid ban IP
            time.sleep(random.uniform(0.5, 1.5))

        except Exception as e:
            print(f"[{i}] Error:", e)

if __name__ == "__main__":
    scrape_images()