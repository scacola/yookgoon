from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import os
from PIL import Image
from io import BytesIO
import base64
# from apply_ocr import ApplyModel, CTCLayer
from urllib.parse import urljoin  # URL 결합을 위해 추가
# import pytesseract  # Tesseract OCR 라이브러리 추가
import CaptchaCracker as cc

# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# iframe 내부로 이동
def switch_to_iframe(iframe_id):
    print("현재 창:", driver.current_window_handle)
    try:
        WebDriverWait(driver, 10).until(EC.frame_to_be_available_and_switch_to_it((By.ID, iframe_id)))
        print("iframe 전환 성공!")
    except Exception as e:
        print(f"iframe 전환 실패: {e}")

# "입영 가능 일자 조회" 클릭
def click_by_text_wait(text):
    try:
        element = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, f"//a[contains(text(), '{text}')]"))
        )
        element.click()
        print("클릭 성공!")
    except Exception as e:
        print(f"오류 발생: {e}")




def switch_to_new_window():
    try:
        WebDriverWait(driver, 5).until(lambda d: len(d.window_handles) > 1)  # 새 창이 열릴 때까지 기다림
        new_window = driver.window_handles[-1]  # 마지막 창(새로 열린 창) 선택
        driver.switch_to.window(new_window)
        print("새 창으로 전환 성공!")
    except Exception as e:
        print(f"새 창 전환 실패: {e}")




def click_date(target_date):
    try:
        xpath = f"//a[contains(text(), '{target_date}')]"
        # element = WebDriverWait(driver, 5).until(
        #     EC.element_to_be_clickable((By.XPATH, xpath))
        # )
        element = driver.find_element(By.XPATH, xpath)
        print(element)
        if not element.is_displayed():
            print(f"{target_date} 날짜가 보이지 않습니다.")
            return False
        else:
            element.click()
            print(f"{target_date} 클릭 성공!")
            return True
    except Exception as e:
        print(f"클릭 실패: {e}")


def click_priority_dates(dates):
    for date in dates:
        try:
            is_clicked = click_date(date)
            if is_clicked:
                print(f"{date} 클릭 완료")
                break  # 클릭 성공 시 루프 종료
            else:
                print(f"{date} 클릭 실패")
                # 클릭 실패 시 다음 날짜로 넘어감
        except Exception as e:
            print(f"{date} 클릭 실패: {e}")
# 이미 열려있는 드라이버 사용
def capture_captcha_and_recognize(driver):
    # 1. 캡차 이미지 요소 찾기
    try:
        capthca_div = driver.find_element(By.ID, "catpcha")  # 캡차 이미지가 포함된 div 요소 선택 (필요에 따라 선택자 변경)
        captcha_img = capthca_div.find_element(By.TAG_NAME, "img")  # 캡차 이미지 요소 선택 (필요에 따라 선택자 변경)
        
        # 2. 이미지를 캡처하여 임시 파일로 저장
        temp_path = "temp_captcha.png"
        
        # JavaScript를 사용하여 이미지 획득 (src가 data:image 형식인 경우 대비)
        img_src = driver.execute_script('return arguments[0].getAttribute("src");', captcha_img)
        
        if img_src.startswith('data:image'):
            # Base64 인코딩된 이미지인 경우
            img_data = img_src.split(',')[1]
            with open(temp_path, "wb") as f:
                f.write(base64.b64decode(img_data))
        else:
            # 상대 경로인 경우 절대 경로로 변환
            if not img_src.startswith('http'):
                img_src = urljoin(driver.current_url, img_src)
            
            # 일반 URL인 경우 스크린샷으로 캡처
            captcha_img.screenshot(temp_path)
        
        # # 캡차 이미지 전처리: OCR 입력 크기에 맞게 조정
        # img = Image.open(temp_path).convert("L")  # 흑백 변환
        # img = img.resize((200, 50))  # OCR 모델 입력 크기와 동일하게 조정
        # img.save(temp_path)  # 전처리된 이미지를 다시 저장
        
        # # 3-1. Tesseract OCR을 사용하여 텍스트 인식
        # result = pytesseract.image_to_string(img, config='--psm 7')  # PSM 7: 단일 텍스트 라인 모드

        pred = AM.predict(temp_path)
        
        return pred  # 결과 문자열 반환 (양쪽 공백 제거)
    
    except Exception as e:
        print(f"오류 발생: {e}")
        return None

# 메인 코드: 현재 웹페이지의 캡차 인식하기
def recognize_current_captcha(driver):
    captcha_text = capture_captcha_and_recognize(driver)
    if (captcha_text):
        print(f"인식된 캡차: {captcha_text}")
        
        # 인식된 캡차 텍스트를 입력 필드에 넣기 (필요한 경우)
        try:
            captcha_input = driver.find_element(By.ID, "answer")  # 실제 입력 필드의 ID로 변경
            captcha_input.clear()
            captcha_input.send_keys(captcha_text)
            print("캡차 입력 성공")
        except Exception as e:
            print(f"캡차 입력 필드를 찾을 수 없음: {e}")
    else:
        print("캡차 인식 실패")

# # 캡차 입력창 포커스 상태
# def move_to_captcha_input():
#     try:
#         captcha_input = WebDriverWait(driver, 10).until(
#             EC.element_to_be_clickable((By.ID, "answer"))
#         )
#         # 요소로 마우스 이동 후 클릭까지 수행
#         ActionChains(driver).move_to_element(captcha_input).click().perform()
#         # 또는 직접 클릭
#         # captcha_input.click()
#         # 또는 JavaScript로 포커스 설정
#         # driver.execute_script("arguments[0].focus();", captcha_input)
#         print("캡차 입력창으로 이동 및 포커스 설정 성공!")
#     except Exception as e:
#         print(f"캡차 입력창으로 이동 실패: {e}")


# # 웹드라이버에서 엔터 키 감지
# def wait_for_enter():
#     print("웹페이지에서 엔터키를 누르면 종료됩니다.")
#     script = """
#     return new Promise(resolve => {
#         document.addEventListener('keydown', function handler(e) {
#             if (e.key === 'Enter') {
#                 document.removeEventListener('keydown', handler);
#                 resolve('엔터키가 눌렸습니다.');
#             }
#         });
#     });
#     """
#     try:
#         # JavaScript가 완료될 때까지 대기 (엔터키가 눌릴 때까지)
#         driver.execute_script(script)
#         print("엔터키가 감지되었습니다.")
#     except Exception as e:
#         print(f"키 감지 중 오류 발생: {e}")
#         input("콘솔에서 엔터를 누르면 종료됩니다.")

# "입영일자선택확인" 버튼 클릭
def click_confirm_button():
    try:
        confirm_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, '//*[@id="contents"]/div/form/table/tbody/tr[2]/th[6]/span/input'))
        )
        confirm_button.click()
        print("확인 버튼 클릭 성공!")
    except Exception as e:
        print(f"확인 버튼 클릭 실패: {e}")



if __name__ == "__main__":

    co = Options()
    co.add_experimental_option('debuggerAddress', '127.0.0.1:9222')
    driver = webdriver.Chrome(options=co)

    main_window = driver.current_window_handle

    img_width = 80
    img_height = 28
    img_length = 5
    img_char = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}
    weights_path = r".\weights.h5"

    AM = cc.ApplyModel(weights_path, img_width, img_height, img_length, img_char)  # ✅ 모델을 한 번만 생성

    # iframe 전환
    switch_to_iframe("main")

    print("현재 열린 모든 창:", driver.window_handles)

    # "입영 가능 일자 조회" 자동 클릭 부분 제거 (다음 줄 주석 처리)
    # click_by_text_wait("입영 가능 일자 조회")

    # 수동 클릭을 위한 안내 메시지
    initial_handles = driver.window_handles
    print("'입영 가능 일자 조회' 버튼을 수동으로 클릭해주세요.")
    print(f"현재 창 수: {len(initial_handles)}")
    num_handles = len(initial_handles)
    # 사용자가 수동으로 버튼을 클릭하여 새 창이 열릴 때까지 대기
    try:
        WebDriverWait(driver, 30).until(lambda d: len(d.window_handles) > len(initial_handles))
        print("새 창이 감지되었습니다.")
        # 새 창 (목록의 마지막 창)으로 전환
        new_window = driver.window_handles[-1]
        driver.switch_to.window(new_window)
        print("새 창으로 전환 성공!")
    except Exception as e:
        print(f"새 창 감지 실패: {e}")

    # 실제
    priority_dates = ["2025-05-26", "2025-05-27", "2025-05-19", "2025-05-20", "2025-06-09", "2025-06-02", "2025-04-21", "2025-04-28", "2025-06-23", "2025-07-07", "2025-12-08", "2025-12-15"]
    # # 테스트
    # priority_dates = [
    #     "2025-12-08", "2025-11-24",
    #     "2025-11-17", "2025-11-10", "2025-12-01", "2025-12-15", "2025-12-22"
    # ]


    # 우선순위 날짜 클릭 실행
    click_priority_dates(priority_dates)

    # 현재 열려있는 드라이버에서 캡차 인식 실행
    recognize_current_captcha(driver)

    # # 캡차 입력창으로 이동
    # move_to_captcha_input()

    # # 엔터 입력 대기
    # wait_for_enter()

    # 확인 버튼 클릭
    click_confirm_button()

    # 메인 창으로 초점 전환
    driver.switch_to.window(main_window)
    print("메인 창으로 전환 성공!")

    # iframe 찾기 및 전환
    iframe = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "main"))
    )
    driver.switch_to.frame(iframe)

    # iframe 내부에서 버튼 찾기 및 클릭
    button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, '//*[@id="contents"]/div/form[1]/p[2]/span[1]/a'))
    )
    button.click()
    print("버튼 클릭 성공!")