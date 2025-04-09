import glob
import CaptchaCracker as cc


img_path_list = glob.glob(r'C:/Users/HOME/AppData/Local/Programs/Python/Python39/sample1/*.png')    #학습 데이터 이미지 경로 (파일명이 숫자와 같아야함)
img_width = 80 #이미지 넓이
img_height = 28 #이미지 높이
CM = cc.CreateModel(img_path_list, img_width, img_height)   # 학습모델 생성
model = CM.train_model(epochs=1031)  # 반복 학습 시작
model.save_weights('C:/Users/HOME/AppData/Local/Programs/Python/Python39/weights.h5')  #학습 결과 가중치 저장