import CaptchaCracker as cc

img_width = 80
img_height = 28
img_length = 5
img_char = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}
weights_path = r".\weights.h5"

AM = cc.ApplyModel(weights_path, img_width, img_height, img_length, img_char)  # ✅ 모델을 한 번만 생성

pred = AM.predict("./temp_captcha.png")
print(pred)
