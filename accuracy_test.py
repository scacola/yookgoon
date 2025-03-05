import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

##############################
# CTCLayer: CTC Loss를 계산하는 커스텀 레이어
##############################
class CTCLayer(layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype=tf.int64)
        input_length = tf.cast(tf.shape(y_pred)[1], dtype=tf.int64)
        label_length = tf.cast(tf.shape(y_true)[1], dtype=tf.int64)
        input_length = input_length * tf.ones((batch_len, 1), dtype=tf.int64)
        label_length = label_length * tf.ones((batch_len, 1), dtype=tf.int64)
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)
        return y_pred

##############################
# ApplyModel 클래스: 학습된 OCR 모델을 불러와 예측 수행
##############################
class ApplyModel:
    def __init__(self, weights_path,
                 img_width=200, 
                 img_height=50, 
                 max_length=5, 
                 characters={'0','1','2','3','4','5','6','7','8','9'}):
        self.img_width = img_width
        self.img_height = img_height
        self.max_length = max_length
        self.characters = characters
        
        # 최신 tf.keras.layers.StringLookup 사용
        self.char_to_num = tf.keras.layers.StringLookup(
            vocabulary=sorted(self.characters), num_oov_indices=0, mask_token=None
        )
        self.num_to_char = tf.keras.layers.StringLookup(
            vocabulary=self.char_to_num.get_vocabulary(), mask_token=None, invert=True
        )
        self.model = self.build_model()
        self.model.load_weights(weights_path)
        # 예측을 위한 모델: 전체 모델의 첫 번째 입력(이미지)와 "dense2" 층의 출력만 사용
        self.prediction_model = keras.models.Model(
            self.model.input[0], self.model.get_layer("dense2").output
        )

    def build_model(self):
        input_img = layers.Input(shape=(self.img_width, self.img_height, 1), name="image", dtype="float32")
        labels = layers.Input(name="label", shape=(None,), dtype="float32")
        x = layers.Conv2D(32, (3,3), activation="relu", kernel_initializer="he_normal",
                          padding="same", name="Conv1")(input_img)
        x = layers.MaxPooling2D((2,2), name="pool1")(x)
        x = layers.Conv2D(64, (3,3), activation="relu", kernel_initializer="he_normal",
                          padding="same", name="Conv2")(x)
        x = layers.MaxPooling2D((2,2), name="pool2")(x)
        new_shape = ((self.img_width // 4), (self.img_height // 4) * 64)
        x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
        x = layers.Dense(64, activation="relu", name="dense1")(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)
        x = layers.Dense(len(self.characters) + 1, activation="softmax", name="dense2")(x)
        output = CTCLayer(name="ctc_loss")(labels, x)
        model = keras.models.Model(inputs=[input_img, labels], outputs=output, name="ocr_model_v1")
        model.compile(optimizer=keras.optimizers.Adam())
        return model

    def encode_single_sample(self, img_path):
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {img_path}")
        img = tf.io.read_file(img_path)
        img = tf.io.decode_png(img, channels=1)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [self.img_height, self.img_width])
        img = tf.transpose(img, perm=[1, 0, 2])
        return {"image": img}

    def decode_batch_predictions(self, pred):
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :self.max_length]
        output_text = []
        for res in results:
            res = tf.strings.reduce_join(self.num_to_char(res + 1)).numpy().decode("utf-8")
            output_text.append(res)
        return output_text

    def predict(self, target_img_path):
        sample = self.encode_single_sample(target_img_path)
        img = tf.reshape(sample["image"], shape=[1, self.img_width, self.img_height, 1])
        pred_val = self.prediction_model.predict(img)
        pred = self.decode_batch_predictions(pred_val)[0]
        return pred

##############################
# get_train_image_size: 학습 폴더의 첫 이미지를 읽어 크기 결정
##############################
def get_train_image_size(train_folder):
    train_images = sorted(glob.glob(os.path.join(train_folder, "*.png")))
    if not train_images:
        raise ValueError("학습 폴더에 이미지가 없습니다.")
    first_img_bytes = tf.io.read_file(train_images[0])
    first_img = tf.io.decode_png(first_img_bytes, channels=1)
    return int(first_img.shape[1]), int(first_img.shape[0])  # width, height

##############################
# test_model 함수: 테스트 데이터셋의 예측 결과와 정답, 오답을 출력 및 전체 정확도 계산
##############################
def test_model(weights_path, test_folder, train_folder):
    # 학습 폴더의 첫 이미지 크기를 가져와 동일한 크기로 사용
    img_width, img_height = get_train_image_size(train_folder)
    print(f"테스트에 사용할 이미지 크기: width={img_width}, height={img_height}")

    # ApplyModel 인스턴스 생성 (학습 시 사용한 이미지 크기와 동일하게)
    model_app = ApplyModel(weights_path, img_width=img_width, img_height=img_height, max_length=5)
    test_images = sorted(glob.glob(os.path.join(test_folder, "*.png")))
    total = len(test_images)
    correct = 0
    wrong_list = []  # 오답 저장 리스트
    for img_path in test_images:
        ground_truth = os.path.splitext(os.path.basename(img_path))[0]
        prediction = model_app.predict(img_path)
        if prediction == ground_truth:
            correct += 1
        else:
            wrong_list.append((os.path.basename(img_path), ground_truth, prediction))
        print(f"파일: {os.path.basename(img_path)} | 정답: {ground_truth} | 예측: {prediction}")
    accuracy = correct / total * 100 if total > 0 else 0
    print(f"\n전체 테스트 이미지: {total}, 정확도: {accuracy:.2f}% ({correct}/{total})")
    if wrong_list:
        print("\n오답 목록:")
        for fname, truth, pred in wrong_list:
            print(f"파일: {fname} | 정답: {truth} | 예측: {pred}")

if __name__ == "__main__":
    weights_path = r"C:\Users\hyungwoo\Desktop\hwcode\ocr_model_weights.weights.h5"
    test_folder = r"C:\Users\hyungwoo\Desktop\hwcode\captcha_dataset2"
    train_folder = r"C:\Users\hyungwoo\Desktop\hwcode\captcha_dataset2"
    test_model(weights_path, test_folder, train_folder)
