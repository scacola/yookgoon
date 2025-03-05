from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import time




co = Options()
co.add_experimental_option('debuggerAddress', '127.0.0.1:9222')
driver = webdriver.Chrome(options=co)

print("현재 열린 모든 창:", driver.window_handles)
# iframe 내부로 이동
def switch_to_iframe(iframe_id):
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

# iframe 전환
switch_to_iframe("main")

print("현재 열린 모든 창:", driver.window_handles)
# "입영 가능 일자 조회" 클릭
click_by_text_wait("입영 가능 일자 조회")
time.sleep(1)
# 현재 열린 모든 창 핸들 확인
print("현재 열린 모든 창:", driver.window_handles)
# 새 창 (목록의 마지막 창)으로 전환
new_window = driver.window_handles[-1]
driver.switch_to.window(new_window)
print("새 창으로 전환 성공!")


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
        formatted_date = f"{target_date[:4]}-{target_date[4:6]}-{target_date[6:]}"
        xpath = f"//a[contains(text(), '{formatted_date}')]"

        element = WebDriverWait(driver, 1).until(
            EC.element_to_be_clickable((By.XPATH, xpath))
        )
        element.click()
        print(f"{formatted_date} 클릭 성공!")
    except Exception as e:
        print(f"클릭 실패: {e}")

click_date("20250408")
