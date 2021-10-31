# 패키지 import

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# MobileNetV2
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
# 추출한 2D feature map의 차원을 다운 샘플링하여 연산량을 감소시키고 주요한 특징 벡터를 추출 - Average Pooling : 각 filter에서 다루는 이미지 패치에서 모든 값의 평균을 반환
from tensorflow.keras.layers import AveragePooling2D
# 2차원 데이터로 이루어진 추출된 특징을 Dense Layer 에서 학습하기 위해 1차원 데이터로 변경
from tensorflow.keras.layers import Flatten
# 입력과 출력을 모두 연결(Fully Connected Layer)
from tensorflow.keras.layers import Dense
# 망의 크기가 커질 경우 Overfitting 문제를 피하기 위해 학습을 할 때 일부 뉴런을 생략하여 학습
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Model  # 학습 및 추론 기능을 가진 객체로 Layer를 모델링
# Nadam(Nesterov-accelerated Adaptive Moment Estimation) : NAG(Nesterov Accelarated Gradient) + Adam(Adaptive Moment Estimation)
# NAG(Nesterov accelarated gradient)
# - momentum 이 이동시킬 방향으로 미리 이동해서 gradient 를 계산(불필요한 이동을 줄이는 효과) - 정확도 개선
# - momentum : 경사 하강법에 관성을 더해주는 것으로, 매번 계산된 기울기에 과거 이동했던 방향을 기억하면서 그 방향으로 일정 값을 추가적으로 더해주는 방식
# Adam(Adaptive Moment Estimation) : momentum + RMSProp (정확도와 보폭 크기 개선)
# - RMSProp : Adagrad 의 보폭 민감도를 보완한 방법(보폭 크기 개선)
# - Adagrad : 변수의 업데이트가 잦으면 학습률을 작게하여 이동 보폭을 조절하는 방법(보폭 크기 개선)
# optimizer : 모델을 컴파일하기 위해 필요한 최적화 알고리즘
from tensorflow.keras.optimizers import Nadam

# sklearn : 머신러닝 분석을 할 때 유용하게 사용할 수 있는 라이브러리(머신러닝 모듈로 구성)
from sklearn.preprocessing import LabelBinarizer  # 레이블 이진화
from sklearn.model_selection import train_test_split  # 학습 dataset 과 테스트 dataset 분리
from sklearn.metrics import classification_report  # 분류 지표 텍스트

# 파이썬 OpenCV가 제공하는 기능 중 복잡하고 사용성이 떨어지는 부분을 보완(이미지 또는 비디오 스트림 파일 처리 등)
from imutils import paths
import matplotlib.pyplot as plt  # 데이터를 차트나 그래프로 시각화할 수 있는 라이브러리
import numpy as np  # 파이썬 행렬 수식 및 수치 계산 처리 모듈
import os  # 운영체제 기능 모듈

dataset = './data'
plot = 'result_plot.jpg'
model_name = 'mask_detector.h5'

init_learning_rate = 1e-4  # 초기 학습률 0.001
epochs = 20  # dataset 학습 횟수
batch_size = 32
print("데이터셋 로딩")
Img_Paths = list(paths.list_images(dataset))
data = []  # 데이터의 목록
labels = []  # 레이블 목록

for Img_Path in Img_Paths:

    label = Img_Path.split(os.path.sep)[-2]
    image = load_img(Img_Path,
                     target_size=(224, 224))  # mobilenetv2는 224*224로 해야 함

    image = img_to_array(image)  # 이미지를 배열로 변환
    image = preprocess_input(image)
    # 모델에 맞는 형식에 이미지를 맞추기 위한 함수
    data.append(image)
    # append함수를 이용하여 data 목록에 추가
    labels.append(label)  # append함수를 이용하여 labels 목록에 추가
    # numpy 배열로 변환
data = np.array(data, dtype="float32")

labels = np.array(labels)
# 레이블 이진화
label_binarizer = LabelBinarizer()  # LabelBinarizer 객체 생성
# 레이블 이진화(fit : 평균과 표준편차 계산, transform : 정규화)
labels = label_binarizer.fit_transform(labels)
labels = to_categorical(labels)  # 이진화된 레이블 One-Hot Encoding 처리
# 이진화된 라벨을 one-hot vector로 바꿈

(trainX, testX, trainY, testY) = train_test_split(data,
                                                  labels,
                                                  test_size=0.2,
                                                  stratify=labels,
                                                  random_state=42)
# 학습 데이터 80% 테스트 데이터 20%로 랜덤하게 분리

image_data_generator = ImageDataGenerator(rotation_range=20,
                                          width_shift_range=0.2,
                                          height_shift_range=0.2,
                                          shear_range=0.2,
                                          zoom_range=0.2,
                                          horizontal_flip=True,
                                          fill_mode="nearest")
# rotation_range : 지정된 각도 범위내에서 임의로 원본 이미지를 회전
# width_shift_range : 지정된 수평 방향 이동 범위내에서 임의로 원본 이미지를 이동
# height_shift_range : 지정된 수직 방향 이동 범위내에서 임의로 원본 이미지를 이동
# shear_range : 지정된 밀림 강도 범위내에서 임의로 원본 이미지를 변경
# zoom_range : 지정된 확대/축소 범위내에서 임의로 원본 이미지를 확대/축소
# horizontal_flip : 수평 방향으로 뒤집기
# fill_mode : 이미지를 회전, 이동하거나 축소할 때 생기는 공간을 채우는 방식
# nearest : 가장 가까운 값

input_model = MobileNetV2(weights="imagenet",
                          include_top=False,
                          input_tensor=Input(shape=(224, 224, 3)))
# weights : 로드할 가중치 파일(imagenet : 사전 학습된 ImageNet 가중치)
# include_top : 네트워크 상단의 Fully-Connected Layer 포함 여부(default : True)
# input_tensor : 모델에 대한 이미지 입력으로 사용할 Keras tensor
# input_shape : 입력 이미지 해상도가 (224, 224, 3)이 아닌 모델을 사용하려는 경우 지정(include_top 이 False 인 경우)

output_model = input_model.output
# 로드한 mobilenetV2의 출력을 사용

output_model = Conv2D(32, (5, 5), padding="same",
                      activation="relu")(output_model)
# Conv2D(Convulution filter 크기,convulution filter (행,열), padding(경계처리방법)="same"(출력 이미지 크기 = 입력이미지 크기))
# activation(활성화 함수)="relu"(활성화 함수)

output_model = AveragePooling2D(pool_size=(5, 5), strides=1,
                                padding="same")(output_model)
# Conv2D의 평균 풀링 작업
# pool_size => poolling을 적용할 필터의 크기 strides => stride의 간격
# 2차원 데이터로 이루어진 추출된 특징을 Dense Layer 에서 학습하기 위해 1차원 데이터로 변경
output_model = Flatten(name="flatten")(output_model)
output_model = Dense(32, activation="relu")(output_model)
output_model = Dense(64, activation="relu")(output_model)
output_model = Dropout(0.5)(output_model)
output_model = Dense(32, activation="relu")(output_model)
output_model = Dense(2, activation="softmax")(output_model)
# softmax 함수는 마지막 단계에서 출력값에 대한 정규화를 해주는 함수다.

model = Model(inputs=input_model.input, outputs=output_model)
# inputs => 모델 입력 / outputs => 모델 출력

# 기존 MobileNetV2 모델의 모든 Layer 를 반복하고 고정(첫 번째 학습 과정 동안 업데이트하지 않기 위함)
for layer in input_model.layers:
    layer.trainable = False

optimizer = Nadam(lr=init_learning_rate, decay=init_learning_rate / epochs)
# Nadam optimizer 설정
# lr : 학습률(default : 0.001)
# beta_1 : 첫 번째 모멘트 추정치에 대한 감소율(default : 0.9)
# beta_2 : 두 번째 모멘트 추정치에 대한 감소율(default : 0.999)
# epsilon : 학습 속도(default : 1e-7)
# decay : 업데이트마다 적용되는 학습률의 감소율
# optimizer : 모델을 컴파일하기 위해 필요한 최적화 알고리즘
print("모델 컴파일")
model.compile(loss="binary_crossentropy",
              optimizer=optimizer,
              metrics=["accuracy"])
# Model.compile() : 모델 컴파일
# optimizer : 적용할 optimizer
# loss : 오차(loss : 예측값과 실제값 간의 차이)를 표현
# - binary_crossentropy : 카테고리가 2개인 경우
# - categorical_crossentropy : 카테고리가 3개 이상인 경우
# metrics : 학습이 잘 이루어지는지 판단(평가)하는 기준 목록
# - accuracy : 정확도
# - mse : 평균 제곱근 오차(Mean Squared Error)

print("[모델 학습]")
train = model.fit(
    image_data_generator.flow(trainX, trainY, batch_size=batch_size),
    steps_per_epoch=len(trainX) // batch_size,  # batch_size = 32
    validation_data=(testX, testY),
    validation_steps=len(testX) // batch_size,
    epochs=epochs)
# Model.fit() : 모델 학습
# 첫 번째 인자 : 입력 데이터
# - ImageDataGenerator.flow() : 랜덤하게 변형된 학습 dataset 생성
# steps_per_epoch : ImageDataGenerator 로부터 얼마나 많은 sample 을 추출할 것인지
# validation_data : 한 번의 epoch 를 실행할 때마다 학습 결과를 확인할 Validation 데이터
# validation_steps : 학습 종료 전 검증할 단계(sample)의 총 개수

# epochs : 전체 dataset에 대해 학습할 횟수

print("[모델 평가]")
# Model.predict() : 테스트 입력 dataset 에 대한 모델의 출력값 확인
predict = model.predict(testX, batch_size=batch_size)
# numpy.argmas : 최대값 index 반환
# - axis : 계산할 기준(0 : 열, 1 : 행)
predict_index = np.argmax(predict, axis=1)
# 분류 결과 출력
print(
    classification_report(testY.argmax(axis=1),
                          predict_index,
                          target_names=label_binarizer.classes_))

print("[모델 저장]")
# Model.save() : 모델 아키텍처 및 가중치 저장
# - save_format : 저장 형식
model.save(model_name, save_format="h5")

# 학습 오차 및 정확도 그래프
n = epochs
plt.style.use("ggplot")  # style.use() : 스타일 적용
plt.figure()  # figure() : 새로운 figure 생성
# plot() : 그래프 그리기
# 첫 번째 인자 : X 축 데이터
# - np.arange() : 인자로 받는 값 만큼 1씩 증가하는 1차원 배열 생성
# 두 번째 인자 : Y 축 데이터
# - Model.fit.history : 학습 오차 및 정확도, 검증 오차 및 정확도
# label : 범례
# epoch 마다 학습 오차
plt.plot(np.arange(0, n), train.history["loss"], label="train_loss")
# epoch 마다 검증 오차
plt.plot(np.arange(0, n), train.history["val_loss"], label="val_loss")
# epoch 마다 학습 정확도
plt.plot(np.arange(0, n), train.history["accuracy"], label="train_accuracy")
# epoch 마다 검증 정확도
plt.plot(np.arange(0, n), train.history["val_accuracy"], label="val_accuracy")
plt.title("Training Loss and Accuracy")  # title() : 제목
plt.xlabel("epoch")  # xlabel() : X 축 레이블
plt.ylabel("Loss / Accuracy")  # ylabel() : Y 축 레이블
plt.legend(loc="lower left")  # legend() : 범례 위치
plt.savefig(plot)  # 그래프 이미지 파일로 저장
