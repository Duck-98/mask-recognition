# 사용 모듈 선언
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import FPS
import imutils
import numpy as np
import cv2

# 얼굴 인식을 위한 모델 파일 로드
print("prototxt, caffemodel 로딩 중...")
net = cv2.dnn.readNet('models/deploy.prototxt', 'models/res10_300x300_ssd_iter_140000.caffemodel')
# 학습 시킨 마스크 착용 여부 판단 모델 로드
print("mask_dector 모델 로딩 중...")
model = load_model('mask_detector.h5')
# 마스크 착용 여부 판단을 위한 내장캠 실행
print("웹캠 로딩 중...")
cap = cv2.VideoCapture(0)

# 얼굴 인식 함수
def Mask_dect(img, net, model):
    # 프레임의 크기
    h, w = img.shape[:2]
    # 이미지 생성을 위한 cv2 모듈 활용
    blob = cv2.dnn.blobFromImage(img, scalefactor=1., size=(300, 300), mean=(104., 177., 123.))
    # 로드한 모델에 생성한 이미지 적용
    net.setInput(blob)
    # 얼굴 인식을 위한 모델 네트워크 실행
    dets = net.forward()
    
    # 얼굴, 좌표, 확률 배열 생성
    faces = []
    locations = []
    predicts = []

    # 얼굴 인식을 위한 반복
    for i in range(0, dets.shape[2]):
        # 얼굴 인식 확률을 추출
        confidence = dets[0, 0, i, 2]
        # 최소 확률보다 큰 경우 실행
        if confidence < 0.5:
            continue
        
        # box 의 위치 계산
        box = dets[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x1, y1, x2, y2) = box.astype("int")
        # box가 전체 이미지 내에 있는 지 좌표 확인
        (x1, y1) = (max(0, x1), max(0, y1))
        (x2, y2) = (min(w - 1, x2), min(h - 1, y2))
        # 인식된 얼굴의 전처리
        face = img[y1:y2, x1:x2]
        face_in = cv2.resize(face, (224, 224))
        face_in = cv2.cvtColor(face_in, cv2.COLOR_BGR2RGB)
        face_in = img_to_array(face_in)
        face_in = preprocess_input(face_in)
        # 인식된 얼굴과 좌표를 배열에 저장
        faces.append(face_in)
        locations.append((x1, y1, x2, y2))
    # 인식된 얼굴이 하나 이상이라면 실행
    if len(faces) > 0:
        # 얼굴 목록을 np 배열로 변환 후 확률 계산
        face_result = np.array(faces, dtype="float32")
        predicts = model.predict(face_result, batch_size=32)
    # 좌표와 확률 값을 반환
    return (locations, predicts)

# fps 초기화
fps = FPS().start()

# 캠이 실행 중이면 True
while cap.isOpened():
    # 비디오 프레임 읽어오기
    ret, img = cap.read()
    # 비디오의 좌우 반전
    img = cv2.flip(img, 1)
    # 프레임이 없을 시 중단
    if not ret:
        break
    # 프레임 resize
    img = imutils.resize(img, width=1500)
    # 읽어온 비디오 프레임과 얼굴 인식 모델, 마스크 착용 여부 판단 모델로 Mask_dect 함수 실행 후 좌표와 확률 반환받기
    (loc, pre) = Mask_dect(img, net, model)
    # 인식된 좌표와 확률 수 만큼 for문 반복
    for (box, pre) in zip(loc, pre):
        # 얼굴 좌표 저장
        (x1, y1, x2, y2) = box
        # 마스크인지 노마스크인지 확률 저장
        (mask, nomask) = pre
        # 마스크가 노마스크보다 크다면 label에 마스크 저장, 반대의 경우 노마스크 저장
        label = "Mask" if mask > nomask else "No Mask"
        # label이 마스크이고 확률이 70% 이상이면 녹색 컬러 출력
        if label == "Mask" and max(mask, nomask) * 100 >= 70:
            color = (0, 255, 0)
        # label이 노마스크이고 확률이 70% 이상이면 빨간 컬러 출력
        elif label == "No Mask" and max(mask, nomask) * 100 >= 70:
            color = (0, 0, 255)
        # 어느 label이건 70% 이상이 아니라면 황색 컬러 출력
        else:
            color = (0, 255, 255)
        # 추출된 확률과 label 저장
        result = "{}: {: .2f}%".format(label, max(mask, nomask) * 100)
        # result에 저장된 텍스트를 출력
        cv2.putText(img, result, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.80, color, 2, lineType=cv2.LINE_AA)
        # 인식된 얼굴 좌표에 맞추어 사각형 실선 출력
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)
    # 비디오 프레임을 출력
    cv2.imshow("Mask_img", img)
    # q키가 입력되면 중단
    if cv2.waitKey(1) == ord("q"):
        break
    # fps 업데이트
    fps.update()
# 반복문이 종료되면 fps 정지
fps.stop()
# 출력된 시간과 fps 출력
print("<시간 : {:.2f}초>".format(fps.elapsed()))
print("<FPS : {:.2f}>".format(fps.fps()))
# 비디오 종료
cap.release()

