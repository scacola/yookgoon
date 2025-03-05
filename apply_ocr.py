import os
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
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")
        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
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
                 max_length=6,
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
        # 예측을 위한 모델: 전체 모델에서 첫 번째 입력(이미지)과 dense2 층 출력만 사용
        self.prediction_model = keras.models.Model(
            self.model.input[0], self.model.get_layer(name="dense2").output
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
        # 이미지 파일 경로가 올바른지 확인
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

if __name__ == "__main__":
    weights_path = r"C:\Users\hyungwoo\Desktop\hwcode\ocr_model_weights.weights.h5"
    model_app = ApplyModel(weights_path)
    # 예측할 이미지 파일 경로를 올바르게 지정하세요.
    test_img = r"C:\Users\hyungwoo\Desktop\hwcode\captcha_dataset\00978.png"
    prediction = model_app.predict(test_img)
    print("예측 결과:", prediction)
