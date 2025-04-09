import glob
import CaptchaCracker as cc
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time

# ✅ 1. 기존 Chrome 창에 연결하기 (디버깅 모드로 실행된 Chrome 사용)
chrome_options = webdriver.ChromeOptions()
chrome_options.debugger_address = "localhost:9222"  # 기존 Chrome 창 사용

# ChromeDriver 실행 (새 창을 열지 않고 기존 창을 컨트롤)
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

print("✅ Selenium이 기존 Chrome 창을 제어합니다.")

# ✅ 2. 원하는 URL이 열려 있는 창으로 전환 (두 개의 URL을 모두 허용)
target_url = "https://mwpt.mma.go.kr/caisBMHS/dmem/dmem/mwgr/hyiy/moveHYBISTGNIlJaJH_P.do"

found = False

for handle in driver.window_handles:
    driver.switch_to.window(handle)
    current_url = driver.current_url

    if current_url.startswith(target_url):
        print(f"✅ 원하는 창을 찾았습니다! (URL: {current_url})")
        found = True
        break

if not found:
    print("❌ 원하는 창을 찾을 수 없습니다. 프로그램을 종료하지 않고 대기합니다.")
    while not found:
        for handle in driver.window_handles:
            driver.switch_to.window(handle)
            current_url = driver.current_url
            if current_url.startswith(target_url):
                print(f"✅ 원하는 창을 찾았습니다! (URL: {current_url})")
                found = True
                break
        time.sleep(1)

print("⏳ 프로그램이 실행 중입니다. F5를 눌러 보안문자를 갱신하세요.")

# ✅ 3. 🔥 TensorFlow 모델을 반복문 바깥에서 한 번만 로드 (⚡ 속도 최적화)
img_width = 80
img_height = 28
img_length = 5
img_char = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}
weights_path = r"C:\Users\HOME\AppData\Local\Programs\Python\Python39\weights.h5"

AM = cc.ApplyModel(weights_path, img_width, img_height, img_length, img_char)  # ✅ 모델을 한 번만 생성

while True:
    try:
        # 1️⃣ 기존 보안문자 요소 찾기
        captcha_element = driver.find_element(By.ID, "catpcha")

        # 2️⃣ F5로 새로고침될 때까지 기다리기 (기존 요소가 사라질 때까지, timeout=3초)
        WebDriverWait(driver, 3).until(EC.staleness_of(captcha_element))

        # 3️⃣ 새로고침 후 새로운 보안문자 요소 찾기 (timeout=3초)
        captcha_element = WebDriverWait(driver, 3).until(
            EC.presence_of_element_located((By.ID, "catpcha"))
        )

        # 4️⃣ 보안문자 이미지 캡처 (`screenshot_as_png` 사용) → 단 1회 실행됨!
        captcha_png = captcha_element.screenshot_as_png
        with open('munja.jpeg', 'wb') as file:
            file.write(captcha_png)
        print("✅ 캡차 이미지 저장 완료!")

        # 5️⃣ 캡차 해독 (🔥 반복문 내부에서 모델을 새로 만들지 않음!)
        target_img_path = 'munja.jpeg'
        pred = AM.predict(target_img_path)  # ✅ 여기서 기존 모델을 사용하여 예측만 수행
        print(f"🔍 예측된 보안문자: {pred}")

        # 6️⃣ 보안문자 입력 (id="answer"에 입력)
        inputElement = driver.find_element(By.ID, "answer")
        inputElement.clear()
        inputElement.send_keys(pred)
        print("✅ 보안문자 입력 완료!")

        print("⏳ 다음 F5 입력을 기다리는 중...")

    except Exception as e:
        print(f"❌ 오류 발생: {e}")