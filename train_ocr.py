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
# CreateModel 클래스: 학습용 OCR 모델 생성 및 학습
##############################
class CreateModel:
    def __init__(self, train_img_pattern):
        # glob 패턴으로 모든 PNG 파일 불러오기 (파일명이 정답으로 저장됨)
        self.images = sorted(glob.glob(train_img_pattern))
        if not self.images:
            raise ValueError("매칭된 파일이 없습니다. 경로 패턴을 확인하세요.")
        # 첫 이미지의 원본 크기를 사용
        first_image_path = self.images[0]
        first_img_bytes = tf.io.read_file(first_image_path)
        first_img = tf.io.decode_png(first_img_bytes, channels=1)
        # 이미지 크기 추출 (이미지들이 모두 동일 크기라고 가정)
        self.img_height, self.img_width = first_img.shape[:2]
        print(f"입력 이미지 크기: width={self.img_width}, height={self.img_height}")
        
        # 파일명에서 확장자 제거 후 라벨 추출 (예: "24121")
        self.labels = [os.path.splitext(os.path.basename(img))[0] for img in self.images]
        # 학습 데이터에 사용되는 문자 집합 구성 (파일명에 포함된 모든 문자)
        self.characters = set(char for label in self.labels for char in label)
        # 라벨 최대 길이 (예: 캡차는 보통 5자리)
        self.max_length = max(len(label) for label in self.labels)
        print(f"사용되는 문자: {sorted(self.characters)}")
        print(f"최대 라벨 길이: {self.max_length}")
        # 문자→정수 매핑 (최신 tf.keras.layers.StringLookup 사용)
        self.char_to_num = tf.keras.layers.StringLookup(
            vocabulary=sorted(self.characters), num_oov_indices=0, mask_token=None
        )
        # 정수→문자 매핑
        self.num_to_char = tf.keras.layers.StringLookup(
            vocabulary=self.char_to_num.get_vocabulary(), mask_token=None, invert=True
        )
    
    def encode_single_sample(self, img_path, label):
        # 이미지 읽기 및 전처리: 그레이스케일, 정규화, 리사이즈 없이 원본 크기 사용
        img = tf.io.read_file(img_path)
        img = tf.io.decode_png(img, channels=1)
        img = tf.image.convert_image_dtype(img, tf.float32)
        # 전치: time dimension이 width가 되도록 (RNN 입력을 위한 전처리)
        img = tf.transpose(img, perm=[1, 0, 2])
        # 라벨을 문자 단위로 분할 후 정수 시퀀스로 변환
        label = self.char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
        return {"image": img, "label": label}
    
    def build_model(self):
        # 모델 입력 정의: 이미지 크기는 위에서 결정한 원본 크기 사용
        input_img = layers.Input(shape=(self.img_width, self.img_height, 1), name="image", dtype="float32")
        labels = layers.Input(name="label", shape=(None,), dtype="float32")
        # CNN feature extractor
        x = layers.Conv2D(32, (3,3), activation="relu", kernel_initializer="he_normal",
                          padding="same", name="Conv1")(input_img)
        x = layers.MaxPooling2D((2,2), name="pool1")(x)
        x = layers.Conv2D(64, (3,3), activation="relu", kernel_initializer="he_normal",
                          padding="same", name="Conv2")(x)
        x = layers.MaxPooling2D((2,2), name="pool2")(x)
        # 다운샘플링 후 reshape: width와 height는 2번의 pooling으로 각각 /2² = /4됨
        new_shape = ((self.img_width // 4), (self.img_height // 4) * 64)
        x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
        x = layers.Dense(64, activation="relu", name="dense1")(x)
        x = layers.Dropout(0.2)(x)
        # RNN 계층: 보편적으로 많이 사용하는 Bidirectional LSTM 설정
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)
        # 출력층: 문자 집합 크기 + 1 (빈 토큰 포함)
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
        x_train, y_train = images[indices[:train_samples]], labels[indices[:train_samples]]
        x_valid, y_valid = images[indices[train_samples:]], labels[indices[train_samples:]]
        return x_train, x_valid, y_train, y_valid
    
    def train_model(self, epochs=100, earlystopping=True):
        batch_size = 16
        images_np = np.array(self.images)
        labels_np = np.array(self.labels)
        # 전체 200개 이미지 중 70% (약 140개)를 학습용, 나머지 30% (약 60개)를 검증용으로 분할
        x_train, x_valid, y_train, y_valid = self.split_data(images_np, labels_np, train_size=0.7)
        
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_dataset = train_dataset.map(self.encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
        train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        validation_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
        validation_dataset = validation_dataset.map(self.encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
        validation_dataset = validation_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        model = self.build_model()
        
        callbacks = []
        if earlystopping:
            callbacks.append(keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True))
        
        history = model.fit(train_dataset, validation_data=validation_dataset, epochs=epochs, callbacks=callbacks)
        return model, history

##############################
# 모델 학습 후 저장
##############################
def train_and_save_model():
    # 모든 PNG 파일을 대상으로 함 (파일명이 정답으로 되어 있음)
    train_img_pattern = r"C:\Users\hyungwoo\Desktop\hwcode\captcha_dataset\*.png"
    model_creator = CreateModel(train_img_pattern)
    model, history = model_creator.train_model(epochs=100, earlystopping=False)
    save_path = r"C:\Users\hyungwoo\Desktop\hwcode\ocr_model_weights.weights.h5"
    model.save_weights(save_path)
    print("모델 가중치 저장 완료:", save_path)

if __name__ == "__main__":
    train_and_save_model()
