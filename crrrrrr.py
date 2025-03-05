import os
import time
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

# 저장할 데이터셋 폴더 지정 (없으면 생성)
dataset_dir = "captcha_dataset2"
os.makedirs(dataset_dir, exist_ok=True)

# 크롬 옵션 설정 (디버깅 모드가 아니라 headless 모드로 진행할 수도 있음)
chrome_options = Options()
chrome_options.add_argument("--headless")
driver = webdriver.Chrome(options=chrome_options)

# 캡차 페이지 URL (매번 새로고침하면 다른 이미지가 표시됨)
url = "https://mwpt.mma.go.kr/caisBMHS/simpleCaptcha/0.06932339158397749.do"

num_images = 200  # 수집할 이미지 수 (원하는 개수로 조정)
for i in range(num_images):
    driver.get(url)
    time.sleep(0.5)  # 페이지가 완전히 로드될 시간 대기 (필요시 조정)
    try:
        # 페이지에 있는 img 태그를 찾음 (캡차 이미지가 유일하다고 가정)
        captcha_img = driver.find_element(By.TAG_NAME, "img")
        # 이미지의 src 속성 추출 (대부분 URL 또는 data:image 형식)
        src = captcha_img.get_attribute("src")
        # src가 URL인 경우 requests로 다운로드
        response = requests.get(src, stream=True)
        if response.status_code == 200:
            filename = os.path.join(dataset_dir, f"captcha_{i}.png")
            with open(filename, "wb") as f:
                f.write(response.content)
            print(f"Saved {filename}")
        else:
            print("Error downloading image")
    except Exception as e:
        print(f"오류 발생: {e}")

driver.quit()
