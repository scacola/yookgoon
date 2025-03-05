import os
import re
from paddleocr import PaddleOCR

# 커스텀 딕셔너리 파일 경로 (0~9가 한 줄에 하나씩 있어야 함)
custom_dict_path = r"C:\Users\hyungwoo\Desktop\hwcode\digit_dict.txt"

# PaddleOCR 초기화: 숫자만 인식하고 최대 텍스트 길이 5로 제한
ocr = PaddleOCR(lang="en",
                use_angle_cls=True,
                rec_char_dict_path=custom_dict_path,
                max_text_length=5)

# 데이터셋 폴더 경로
dataset_dir = r"C:\Users\hyungwoo\Desktop\hwcode\captcha_dataset2"

# 디렉토리 내 파일 순회 (예: captcha_0.png ~ captcha_199.png)
for filename in os.listdir(dataset_dir):
    if filename.startswith("captcha_") and filename.endswith(".png"):
        file_path = os.path.join(dataset_dir, filename)
        print(f"Processing file: {filename}")
        try:
            result = ocr.ocr(file_path, cls=True)
        except Exception as e:
            print(f"Error during OCR for file {filename}: {e}")
            continue  # 오류가 있으면 해당 파일 건너뜀

        recognized_text = ""
        for item in result:
            # 예상 구조: [box, (text, score)]
            try:
                if isinstance(item, (list, tuple)):
                    if len(item) == 2:
                        box, (text, score) = item
                        recognized_text += text
                    elif len(item) == 1 and isinstance(item[0], (list, tuple)) and len(item[0]) == 2:
                        box, (text, score) = item[0]
                        recognized_text += text
                    else:
                        print("Unexpected format in file:", filename, "item:", item)
                else:
                    print("Unexpected type in file:", filename, "item:", item)
            except Exception as e:
                print("Error processing OCR result for file:", filename, "Error:", e)
                recognized_text = ""
                break

        # 정규표현식으로 숫자만 남기기
        recognized_text = re.sub(r'\D', '', recognized_text)

        if len(recognized_text) == 5:
            new_filename = recognized_text + ".png"
            new_file_path = os.path.join(dataset_dir, new_filename)
            try:
                os.rename(file_path, new_file_path)
                print(f"Renamed {filename} to {new_filename}")
            except Exception as e:
                print(f"Error renaming file {filename}: {e}")
        else:
            print(f"File {filename}: recognized text is not 5 digits: {recognized_text}")
