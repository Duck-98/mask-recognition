# 필요한 패키지 import
from imutils.video import FPS
import numpy as np # 파이썬 행렬 수식 및 수치 계산 처리 모듈
import cv2 # opencv 모듈
import imutils # 파이썬 OpenCV가 제공하는 기능 중 복잡하고 사용성이 떨어지는 부분을 보완(이미지 또는 비디오 스트림 파일 처리 등)
import time # 시간 처리 모듈
import argparse # 명령행 파싱(인자를 입력 받고 파싱, 예외처리 등) 모듈
import face_recognition # 얼굴 특성 정보 추출(얼굴 인식) 모듈

# 실행을 할 때 인자값 추가
ap = argparse.ArgumentParser() # 인자값을 받을 인스턴스 생성
# 입력받을 인자값 등록
ap.add_argument("-i", "--input", type=str, help="input 비디오 경로")
ap.add_argument("-o", "--output", type=str, help="output 비디오 경로") # 비디오 저장 경로
# 입력받은 인자값을 args에 저장
args = vars(ap.parse_args())

# input 비디오 경로가 제공되지 않은 경우 webcam
if not args.get("input", False):
    print("[webcam 시작]")
    vs = cv2.VideoCapture(0)

# input 비디오 경로가 제공된 경우 video
else:
    print("[video 시작]")
    vs = cv2.VideoCapture(args["input"])

# fps 정보 초기화
fps = FPS().start()

writer = None
(w, h) = (None, None)

# 비디오 스트림 프레임 반복
while True:
    # 프레임 읽기
    ret, frame = vs.read()

    # 읽은 프레임이 없는 경우 종료
    if args["input"] is not None and frame is None:
        break

    # 프레임 resize
    frame = imutils.resize(frame, width=400)

    # face_recognition.face_locations(이미지(numpy 배열), 모델) : 이미지에서 사람 얼굴의 bounding boxes 반환
    face_locations = face_recognition.face_locations(frame, model='hog') # hog(기본값) : 비교적 덜 정확하지만 cpu에서도 빠름(gpu를 사용 가능한 경우 cnn)

    # 얼굴 번호
    number = 0

    # 얼굴 인식 목록 수 만큼 반복
    for face_location in face_locations:
        (y1, x2, y2, x1) = face_location # 인식된 얼굴 좌표

        cv2.putText(frame, "Face[{}]".format(number + 1), (x1 - 5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2) # 얼굴 번호 출력
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # bounding box 출력

        number = number + 1 # 얼굴 번호 증가

    # 프레임 resize
    frame = imutils.resize(frame, width=500)

    # 프레임 출력
    cv2.imshow("Face Recognition", frame)
    key = cv2.waitKey(1) & 0xFF
    
    # 'q' 키를 입력하면 종료
    if key == ord("q"):
        break
    
    # fps 정보 업데이트
    fps.update()

    # video 설정
    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 25, (frame.shape[1], frame.shape[0]), True)

    # 비디오 저장
    if writer is not None:
        writer.write(frame)

# fps 정지 및 정보 출력
fps.stop()
print("[재생 시간 : {:.2f}초]".format(fps.elapsed()))
print("[FPS : {:.2f}]".format(fps.fps()))

# 종료
vs.release()
cv2.destroyAllWindows()