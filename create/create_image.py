# 사용 모듈 선언
import cv2
import os
from face_recognition.api import face_landmarks, face_locations
import numpy as np
import face_recognition
from PIL import Image

# 추출한 얼굴 이미지 리스트 생성, join으로 새경로 설정, listdir로 디렉터리 리스트 조회, isfile로 파일의 유무 확인
images = [os.path.join("data/no-mask", fl) for fl in os.listdir("data/no-mask") if os.path.isfile(os.path.join("data/no-mask", fl))]
# 씌울 마스크 색상별 이미지 경로
white = "image/mask-image/white_mask.png"
black = "image/mask-image/black_mask.png"
blue = "image/mask-image/blue_mask.png"

# 마스크 쓴 이미지 생성하는 클래스 선언
class CreateMaskImage:
    # 마스크 영역 생성을 위한 콧등과 턱 포인트
    KEY_FE = ('nose_bridge', 'chin')
    # 인식 불가 파일의 카운트
    count = 0

    # 변수 초기화 메서드
    def __init__(self, fc_path, ms_path, model='hog'):
        self.fc_path = fc_path
        self.ms_path = ms_path
        self.model = model

    # 세팅
    def setting(self):
        # fc_path 경로에 있는 이미지 파일 로드 후 np 배열로 로드
        image_fl = face_recognition.load_image_file(self.fc_path)
        # 로드된 배열 형태의 이미지 파일에서 얼굴 형태를 바운딩 박스로 반환
        face_location = face_recognition.face_locations(image_fl, model=self.model)
        # 얼굴 바운딩 박스에서 특징의 위치 목록 저장
        face_landmarks = face_recognition.face_landmarks(image_fl, face_location)
        # np 배열을 이미지로 변환
        self._face_image = Image.fromarray(image_fl)
        # 씌울 마스크 이미지 파일 (흰색, 검은색, 파란색) 로드
        self._mask_image = Image.open(self.ms_path)
        # 얼굴 인식 유무
        found_face = False

        # 얼굴 특징 목록 수만큼 반복
        for face_landmark in face_landmarks:
            # 특징이 콧등과 턱인지 여부
            skip = False

            # 콧등과 턱 두 번 반복
            for facial_feature in self.KEY_FE:
                # 콧등, 턱 아닌 경우
                if facial_feature not in face_landmark:
                    skip = True
                    break
            # 스킵 진행
            if skip == True:
                continue

            # 콧등, 턱 있는 경우
            found_face = True
            # 마스크 생성 메서드 실행
            self.mask_face(face_landmark)

        # 마스크 생성 메서드 실행 후 콧등, 턱 있는 경우 이미지 파일 저장 메서드 실행
        if found_face:
            self._save()
        # 인식 불가한 경우
        else:
            # 인식하지 못한 파일 카운트 +1
            CreateMaskImage.count += 1
            print('얼굴 인식 불가')
            print(self.fc_path)

    # 얼굴 이미지에 마스크 이미지 생성 메서드
    def mask_face(self, face_landmark: dict):
        # 콧등 영역
        nose_bridge = face_landmark['nose_bridge']
        # 콧등에서 4분의 1 지점에 마스크 상단을 놓기 위한 포인트 지정
        nose_point = nose_bridge[len(nose_bridge) * 1 // 4]
        # 콧등 포인트를 np 배열로 변환 후 저장
        nose = np.array(nose_point)

        # 턱 영역
        chin = face_landmark['chin']
        # 턱의 길이 저장
        chin_len = len(chin)
        # 턱의 아랫부분 중간에 마스크를 씌우기 위한 포인트 지정
        chin_bottom_point = chin[chin_len // 2]
        # 지정한 포인트를 배열로 변환 후 저장
        chin_bottom = np.array(chin_bottom_point)
        # 턱의 왼쪽 부분에 마스크 이미지 왼쪽을 맞추기 위한 포인트 지정
        chin_left_point = chin[chin_len // 8]
        # 턱의 오른쪽 부분에 마스크 이미지 오른쪽을 맞추기 위한 포인트 지정
        chin_right_point = chin[chin_len * 7 // 8]

        # 마스크 이미지의 크기를 불러와 저장
        width = self._mask_image.width
        height = self._mask_image.height
        # 마스크의 기본 너비 지정
        width_ratio = 1.2
        # 벡터 공간을 계산하는 함수 사용해 마스크 높이 지정
        mask_height = int(np.linalg.norm(nose - chin_bottom))

        # 보다 입체적으로 마스크 입히기 위해 왼쪽 오른쪽 나누어 진행
        # 마스크 왼쪽 영역
        # 이미지 추출 crop 메서드를 사용해 너비는 반으로 높이는 완전히 추출
        mask_left_image = self._mask_image.crop((0, 0, width // 2, height))
        # get_distane 메서드를 사용해 왼쪽의 너비 반환
        mask_left_width = self.get_distance(chin_left_point, nose_point, chin_bottom_point)
        # 지정한 마스크 기본 너비를 적용
        mask_left_width = int(mask_left_width * width_ratio)
        # 왼쪽의 너비와 마스크 높이를 이용해 왼쪽 영역 resize 진행
        mask_left_image = mask_left_image.resize((mask_left_width, mask_height))

        # 마스크 오른쪽 영역
        # 이하 왼쪽 영역과 동일
        mask_right_image = self._mask_image.crop((width // 2, 0, width, height))
        mask_right_width = self.get_distance(chin_right_point, nose_point, chin_bottom_point)
        mask_right_width = int(mask_right_width * width_ratio)
        mask_right_image = mask_right_image.resize((mask_right_width, mask_height))

        # resize 진행한 왼쪽 오른쪽 영역 합치기
        # 지정한 마스크 너비 저장
        size = (mask_left_image.width + mask_right_image.width, mask_height)
        # 새로운 이미지 생성을 위한 함수 사용 모드로 'RGBA' 사용
        mask_image = Image.new('RGBA', size)
        # 0, 0 에 왼쪽 마스크 영역 합치기
        mask_image.paste(mask_left_image, (0, 0), mask_left_image)
        # 왼쪽 영역 너비, 0 에 오른쪽 마스크 영역 합치기
        mask_image.paste(mask_right_image, (mask_left_image.width, 0), mask_right_image)

        # 마스크 회전을 위한 연산 함수 arctan2 실행
        # [1] = y 좌표, [0] = x 좌표 이용해 각도 계산
        angle = np.arctan2(chin_bottom_point[1] - nose_point[1], chin_bottom_point[0] - nose_point[0])
        # rotate 메서드(각도, expand) 사용해 출력 크기에 맞게 조절하여 마스크 이미지 회전
        rotated_mask_image = mask_image.rotate(angle, expand=True)

        # 마스크 이미지의 중앙 위치 계산
        center_x = (nose_point[0] + chin_bottom_point[0]) // 2
        center_y = (nose_point[1] + chin_bottom_point[1]) // 2

        # 마스크 이미지의 오프셋 설정
        # 마스크 왼쪽 영역과 오른쪽 영역의 편차
        offset = mask_image.width // 2 - mask_left_image.width
        # 이미지의 각도 계산
        radian = angle * np.pi / 180
        # 계산된 각도와 오프셋으로 사인 코사인을 이용해 마스크 이미지의 좌표 설정
        box_x = center_x + int(offset * np.cos(radian)) - rotated_mask_image.width // 2
        box_y = center_y + int(offset * np.sin(radian)) - rotated_mask_image.height // 2

        # 변경된 마스크 이미지와 계산한 좌표를 이용해 얼굴 이미지에 마스크 이미지 합치기
        self._face_image.paste(mask_image, (box_x, box_y), mask_image)

    # 이미지 저장 메서드
    def _save(self):
        # splitext 함수로 가져온 이미지 파일의 확장자 명 추출
        im_splits = os.path.splitext(self.fc_path)
        # * 원래 파일 명 no_mask_000.jpg 에서 split한 0번째 인덱스 no_mask_000을 다시 split
        num = im_splits[0].split('_')
        # 세 번째에 있는 번호 값과 추출한 확장자명을 mask_ 뒤에 붙여서 경로와 파일명 설정
        ms_im_path = 'data/mask/mask_' + num[2] + im_splits[1]
        # 설정한 파일명으로 save 메서드를 사용해 저장
        self._face_image.save(ms_im_path)
        print('저장 경로 :', ms_im_path)

    # 정적 메서드 선언
    @staticmethod
    # 너비 계산 메서드
    def get_distance(point, line_point1, line_point2):
        # 마스크 끝 좌표, 코 좌표, 턱 좌표를 이용해 세 좌표 사이의 거리 구하는 공식
        distance = np.abs((line_point2[1] - line_point1[1]) * point[0] + (line_point1[0] - line_point2[0]) * point[1] + (line_point2[0] - line_point1[0]) * line_point1[1] + (line_point1[1] - line_point2[1]) * line_point1[0]) / \
                   np.sqrt((line_point2[1] - line_point1[1]) * (line_point2[1] - line_point1[1]) + (line_point1[0] - line_point2[0]) * (line_point1[0] - line_point2[0]))
        return int(distance)

# 변수 선언과 CreateMaskImage 클래스에 변수 넘겨주며 setting 메서드 실행하는 메서드
def create_mask(image, ms_image):
    # 로드한 얼굴 이미지 경로
    im_path = image
    # 씌울 마스크 이미지 경로
    ms_path = ms_image
    # 모델 선언
    model = 'hog'
    CreateMaskImage(im_path, ms_path, model).setting()

# 마스크 생성 프로그램 실행 메서드
def image_classifi():
    # 얼굴 이미지 수 만큼 반복
    for i in range(len(images)):
        # 400개는 흰색의 마스크 이미지 적용
        if i < 400:
            create_mask(images[i], white)
        # 401~800개는 검은색 마스크 이미지 적용
        elif i < 800:
            create_mask(images[i], black)
        # 801~1000개는 파란색 마스크 이미지 적용
        elif i < 999:
            create_mask(images[i], blue)
        # 마지막 실행 시 완료 메세지와 인식 불가 카운트 메세지 출력
        else:
            create_mask(images[i], blue)
            print("완료...")
            print("인식 불가 이미지 : 1000/", CreateMaskImage.count)
# 실행 메서드
image_classifi()