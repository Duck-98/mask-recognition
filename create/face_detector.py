# 사용 모듈 선언
from face_recognition.api import face_locations
import cv2
import imutils
import os
import face_recognition

# 사진 속에서 추출한 얼굴 이미지 저장
def image_save():
    # im_path 에서 확장자 추출
    im_path_splits = os.path.splitext(im_path)
    # 확장자 추출해낸 문자열에서 _로 구분해 split
    num = im_path_splits[0].split('_')
    # 이미지 파일의 숫자와 확장자를 결합해 새 이미지 파일 저장 경로 지정
    no_mask_im_path = "data/no-mask/no_mask_" + num[1] + im_path_splits[1]

    # 이미지 저장
    cv2.imwrite(no_mask_im_path, fc_image)
    print('저장 경로 -> ', no_mask_im_path)

# 경로 내에 있는 이미지 로드
images = [os.path.join("image/face", i) for i in os.listdir("image/face") if os.path.isfile(os.path.join("image/face", i))]

# 이미지 수만큼 반복하여 수행
for j in range(len(images)):
    # 얼굴 유무 판별
    fc_test = False
    # 얼굴 이미지 파일
    im_path = images[j]
    # 파일 읽어오기
    fc_image = cv2.imread(im_path)
    # 파일 속 얼굴 box 형태의 좌표로 인식해 저장 
    boundingboxs = face_recognition.face_locations(fc_image, model = 'hog')
    
    # 인식된 얼굴 수 만큼 반복
    for location in boundingboxs:
        # 얼굴 좌표
        (y1, x2, y2, x1) = location
        fc_test = True

    # 인식된 얼굴이 없을 경우
    if fc_test == False:
        print("인식 불가 -> ", im_path)
        print("종료...")
        break
    
    # 인식된 얼굴만큼 이미지 자르기
    fc_image = fc_image[y1 - 5 : y2 + 5, x1 - 5 : x2 + 5]
    # 자른 이미지 저장 메서드 실행
    image_save()

print("완료...")