import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

##############################
# CTC Loss Layer
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
# CreateModel 클래스: 학습용 모델 생성 및 학습
##############################
class CreateModel:
    def __init__(self, train_img_list, img_width=200, img_height=50):
        self.img_width = img_width
        self.img_height = img_height
        # 이미지 경로 리스트와 파일명에서 라벨 추출 (예: "24121")
        self.images = sorted(train_img_list)
        self.labels = [os.path.splitext(os.path.basename(img))[0] for img in self.images]
        # 문자 집합 구성 (파일명에 포함된 모든 문자)
        self.characters = set(char for label in self.labels for char in label)
        self.max_length = max(len(label) for label in self.labels)
        # 문자 → 정수 매핑 (Keras의 StringLookup 사용)
        self.char_to_num = layers.experimental.preprocessing.StringLookup(
            vocabulary=sorted(self.characters), num_oov_indices=0, mask_token=None
        )
        # 정수 → 문자 매핑
        self.num_to_char = layers.experimental.preprocessing.StringLookup(
            vocabulary=self.char_to_num.get_vocabulary(), mask_token=None, invert=True
        )
    
    def encode_single_sample(self, img_path, label):
        # 이미지 전처리: 읽기, 그레이스케일, 정규화, 리사이즈, 전치
        img = tf.io.read_file(img_path)
        img = tf.io.decode_png(img, channels=1)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [self.img_height, self.img_width])
        img = tf.transpose(img, perm=[1, 0, 2])
        # 라벨 문자열을 문자 단위로 분할하여 정수로 변환
        label = self.char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
        return {"image": img, "label": label}
    
    def build_model(self):
        input_img = layers.Input(shape=(self.img_width, self.img_height, 1), name="image", dtype="float32")
        labels = layers.Input(name="label", shape=(None,), dtype="float32")
        
        # Convolutional Block 1
        x = layers.Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv1")(input_img)
        x = layers.MaxPooling2D((2, 2), name="pool1")(x)
        # Convolutional Block 2
        x = layers.Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv2")(x)
        x = layers.MaxPooling2D((2, 2), name="pool2")(x)
        
        # Reshape: 다운샘플링에 따라 feature map의 크기가 줄어듦 (여기서는 4배)
        new_shape = ((self.img_width // 4), (self.img_height // 4) * 64)
        x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
        x = layers.Dense(64, activation="relu", name="dense1")(x)
        x = layers.Dropout(0.2)(x)
        
        # RNN 계층: Bidirectional LSTM 2개
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)
        
        # 출력층: 문자 집합 크기 + 1 (빈 토큰)
        x = layers.Dense(len(self.characters) + 1, activation="softmax", name="dense2")(x)
        output = CTCLayer(name="ctc_loss")(labels, x)
        
        model = keras.models.Model(inputs=[input_img, labels], outputs=output, name="ocr_model_v1")
        model.compile(optimizer=keras.optimizers.Adam())
        return model
    
    def split_data(self, images, labels, train_size=0.7, shuffle=True):
        size = len(images)
        indices = np.arange(size)
        if shuffle:
            np.random.shuffle(indices)
        train_samples = int(size * train_size)
        x_train = images[indices[:train_samples]]
        y_train = labels[indices[:train_samples]]
        x_valid = images[indices[train_samples:]]
        y_valid = labels[indices[train_samples:]]
        return x_train, x_valid, y_train, y_valid
    
    def train_model(self, epochs=100, earlystopping=True):
        batch_size = 16
        # 데이터를 numpy 배열로 변환
        images_np = np.array(self.images)
        labels_np = np.array(self.labels)
        # 여기서는 70%를 학습, 30%를 검증 (140개, 60개로 분할)
        x_train, x_valid, y_train, y_valid = self.split_data(images_np, labels_np, train_size=0.7)
        
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_dataset = train_dataset.map(self.encode_single_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        
        validation_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
        validation_dataset = validation_dataset.map(self.encode_single_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        validation_dataset = validation_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        
        model = self.build_model()
        
        callbacks = []
        if earlystopping:
            callbacks.append(keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True))
        
        history = model.fit(train_dataset, validation_data=validation_dataset, epochs=epochs, callbacks=callbacks)
        return model, history

##############################
# ApplyModel 클래스: 학습된 모델로 예측 수행
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
        
        self.char_to_num = layers.experimental.preprocessing.StringLookup(
            vocabulary=sorted(self.characters), num_oov_indices=0, mask_token=None
        )
        self.num_to_char = layers.experimental.preprocessing.StringLookup(
            vocabulary=self.char_to_num.get_vocabulary(), mask_token=None, invert=True
        )
        self.model = self.build_model()
        self.model.load_weights(weights_path)
        # 예측을 위한 모델: 이미지 입력 -> dense2 출력
        self.prediction_model = keras.models.Model(
            self.model.get_layer(name="image").input, self.model.get_layer(name="dense2").output
        )
    
    def build_model(self):
        input_img = layers.Input(shape=(self.img_width, self.img_height, 1), name="image", dtype="float32")
        labels = layers.Input(name="label", shape=(None,), dtype="float32")
        x = layers.Conv2D(32, (3,3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv1")(input_img)
        x = layers.MaxPooling2D((2,2), name="pool1")(x)
        x = layers.Conv2D(64, (3,3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv2")(x)
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
            res = tf.strings.reduce_join(self.num_to_char(res+1)).numpy().decode("utf-8")
            output_text.append(res)
        return output_text
    
    def predict(self, target_img_path):
        sample = self.encode_single_sample(target_img_path)
        img = tf.reshape(sample["image"], shape=[1, self.img_width, self.img_height, 1])
        pred_val = self.prediction_model.predict(img)
        pred = self.decode_batch_predictions(pred_val)[0]
        return pred

##############################
# Main: 학습 및 테스트 수행
##############################
def main():
    # 데이터셋 경로: hwcode/captcha_dataset 폴더 내의 모든 png 파일 (총 200개)
    dataset_dir = r"C:\Users\hyungwoo\Desktop\hwcode\captcha_dataset"
    all_images = sorted(glob.glob(os.path.join(dataset_dir, "*.png")))
    if len(all_images) < 200:
        print("총 이미지 수가 200개 미만입니다.")
        return

    # 전체 200개 중 70% (약 140개)를 학습용, 나머지 60개를 테스트용으로 분할
    train_images = all_images[:140]
    test_images = all_images[140:200]

    # 학습
    print("학습 시작...")
    model_creator = CreateModel(train_images, img_width=200, img_height=50)
    model, history = model_creator.train_model(epochs=100, earlystopping=True)
    weights_path = os.path.join(dataset_dir, "ocr_model_weights.h5")
    model.save_weights(weights_path)
    print("모델 가중치 저장 완료:", weights_path)

    # 테스트: 테스트 이미지에 대해 예측 및 정답과 비교
    print("테스트 시작...")
    # ApplyModel 초기화: 학습 시 사용한 문자 집합(0~9)
    model_app = ApplyModel(weights_path, img_width=200, img_height=50, max_length=5, characters={'0','1','2','3','4','5','6','7','8','9'})
    correct = 0
    for img_path in test_images:
        # 파일명이 정답
        true_label = os.path.splitext(os.path.basename(img_path))[0]
        pred_label = model_app.predict(img_path)
        print(f"파일: {os.path.basename(img_path)} | 정답: {true_label} | 예측: {pred_label}")
        if pred_label == true_label:
            correct += 1
    print(f"정확도: {correct/len(test_images)*100:.2f}% ({correct}/{len(test_images)})")

if __name__ == "__main__":
    main()
