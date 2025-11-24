import csv
import copy #이거는 그대로 복사를 하는 빙식이다 이것을 한이유는 원래의 이미지 데이터를 유지하면서 복사본 영상에 하는 방식이기 때문
# Q: 이게 필수 일까?
import argparse # 터미넬에서 실행을 할때 -- 이것과 같이 뒤에 뭔가 붙이는것을 해석해주는 모듈
import itertools # [[x,y], [x,y]] 형태의 좌표 데이터를 [x, y, x, y]로 한 줄로 펴줌.
from collections import Counter
from collections import deque # 이거는 리스트에서 길이 제한을 두는것 -> 영원히 손가락이 어디로 움직이는 알 필요가 없어서(16프레임만 알면 되니까)

import cv2 as cv# 컴퓨터 비전
import numpy as np
import mediapipe as mp #이거는 그냥 핵심 모듈

from utils import CvFpsCalc# Computer Vision Frames Per Second Calculator// 프레임을 계산하기 위한것 
from model import KeyPointClassifier
# keypoint_classification_EN.ipynb에서 만든 'keypoint_classifier.tflite' 이 파잉을 불러와서 정답인지 아닌지 해주는 역활 
from model import PointHistoryClassifier
# point_history_classification.ipynb여기에서 만들어진 point_history_classifier.tflite이 파잉을 사용하여 과거 16프레임의 뒈적 데이터를 넘겨줘서 움직임의 종류를 알려줌


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)# 사용하는 카메라의 ID
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')# 사진 모드로 할지 선택(비디오가 아닌 매프레임을 사진으로 인식하자는 의미)
    # parser.add_argument('--use_static_image_mode', action='store_false') # 느리지만 정확히
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)# 신뢰도의 수준(이거는 멈춰 있을때의 신뢰도를 의미함)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)# 손추적 민감도 (그니까 이거는 따라가는 것을 의미한다)

    args = parser.parse_args()

    return args


def main():
    # 위에서 정한 설정들 가져오기
    args = get_args() # 일단 터미멀에서 킴

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    # 카메라 키기
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # ai모델 가져오기
    mp_hands = mp.solutions.hands# 손을 찾는 모델
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,# 아까 동영상 혹은 사진 모드 가져오기
        max_num_hands=2,# 손 인식 개수
        min_detection_confidence=min_detection_confidence, # 인지 민감도
        min_tracking_confidence=min_tracking_confidence,# 추적 민감도
    )

    keypoint_classifier = KeyPointClassifier()# 모양 판독기

    point_history_classifier = PointHistoryClassifier() # 움직임 판독기

    # csv 파일을 가져오는 역활 ###############################################################
    # 족보 느낌으로 가져오는 방식
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
    with open(
            'model/point_history_classifier/point_history_classifier_label.csv',
            encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]

    # FPS 지정
    # 이것을 하는 이유는 컴터마다 다르기 때문에 지정을 하는 것이다
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # 이게 16프레임을 저장하는 방식
    history_length = 16
    point_history = deque(maxlen=history_length)

    # Finger gesture history ################################################
    finger_gesture_history = deque(maxlen=history_length)# 16개 중에서 가장 많이 나온것을 정답으로 한다

    #  ########################################################################
    mode = 0

    while True:
        fps = cvFpsCalc.get() #현재 프레임 측정

        #  탈출키 
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode) # 0~9 숫자키 누르면 데이터 수집 모드 변경 

        # Camera capture #####################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # 좌우 반전
        debug_image = copy.deepcopy(image)# 이게 복사본을 만드는것 그림을 그리기 위해

        # Detection implementation #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # ################################################################
        image.flags.writeable = False
        # 이미 복사본이 있으니 굳이 새거로 계속 만들필요가 없으니 빠르게 기존의 것에서 하라는 의미
        # 메모리 아낌 ,속도하고도
        results = hands.process(image)
        # 21개의 관절이 어디에 있는지 확인 하기
        #results.multi_hand_landmarks: 손가락 관절 좌표 리스트 (가장 중요!)
        # results.multi_handedness: 왼손인지 오른손인지 정보
        image.flags.writeable = True
        # 이제부터는 다시 수정 하는것을 허요하느것
        # 이제 화면에서 그림을 그림
        # ################################################################

        # ################################################################
        if results.multi_hand_landmarks is not None:# 만약에 손이 있다면
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,# 2개있을수도 있으니 for문으로 순서 지정
                                                  results.multi_handedness):
                # 박스 계산: 손을 감싸는 네모칸 좌표를 구함
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                
                # 좌표 변환: 0.0~1.0 비율 좌표를 -> 실제 픽셀 좌표(예: 1920, 1080)로 바꿈
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # 전처리 1 (모양용): 손목을 (0,0)으로 만들고 크기를 1로 맞춤 (정적 AI용)
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                
                # 전처리 2 (움직임용): 과거 궤적 데이터를 AI가 좋아하는 형태로 바꿈 (동적 AI용)
                pre_processed_point_history_list = pre_process_point_history(debug_image, point_history)


                # 만약 키보드(0~9)를 눌렀다면, 지금 좌표를 CSV 파일에 저장!
                logging_csv(number, mode, pre_processed_landmark_list, pre_processed_point_history_list)
            
                # AI에게 물어봄: "이 손 모양(pre_processed_landmark_list)이 주먹이야 가위야?"
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                # 결과: hand_sign_id에는 0, 1, 2 같은 숫자가 들어감 (0:주먹, 1:보자기...)

                # 만약 손 모양이 '2번(검지 펴기)'이라면?
                if hand_sign_id == 2:
                    # 검지 끝(8번 관절)의 좌표를 궤적 리스트에 추가!
                    point_history.append(landmark_list[8])
                else:
                    # 다른 손 모양이면 궤적을 저장하지 않음 (0, 0을 넣어서 끊어버림)
                    point_history.append([0, 0])
                # 근데 왜 2만 측정을 하는거지?
                # 다른거는 정의가 안되어 있나?


                # 어떻게 움직였는지 확인하고자함
                finger_gesture_id = 0# 기본 정지 생태
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (history_length * 2):
                    # AI에게 물어봄: "이 궤적(pre_processed...)이 왼쪽이야 오른쪽이야?"
                    finger_gesture_id = point_history_classifier(
                        pre_processed_point_history_list)

                # 결과 안정화: 최근 16번의 예측 결과 중 가장 많이 나온 걸 최종 답으로 결정
                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(finger_gesture_history).most_common()

                ##################################################
                # 여기가 이제 그리는 부분
                debug_image = draw_bounding_rect(use_brect, debug_image, brect) # 박스 그리기
                debug_image = draw_landmarks(debug_image, landmark_list) # 손 뼈대와 관절 점 그리ㅣ
                debug_image = draw_info_text( #이게 결과 텍스트 쓰는것
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id],
                    point_history_classifier_labels[most_common_fg_id[0][0]],
                )
        # 마무리 출력
        else:
            point_history.append([0, 0])

        # 손가락이 지나간 자리에 선 그리기
        debug_image = draw_point_history(debug_image, point_history)
        # FPS속도 정보 표시
        debug_image = draw_info(debug_image, fps, mode, number)

        # Screen reflection #############################################################
        # cv.imshow('Hand Gesture Recognition', debug_image)
        window_name = 'Hand Gesture Recognition'
        cv.namedWindow(window_name, cv.WINDOW_NORMAL)
        cv.setWindowProperty(window_name, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
        cv.imshow(window_name, debug_image)

    cap.release()
    cv.destroyAllWindows()


def select_mode(key, mode):
    number = -1# 아무것도 없을때
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n 기본모드 -> 그냥 관전 모드. 저장안함
        mode = 0
    if key == 107:  # k -> 정적 제스처 데이터 수집(모양)
        mode = 1
    if key == 104:  # h -> 동적인 데이터 모집 (움직임)
        mode = 2
    return number, mode


def calc_bounding_rect(image, landmarks): #네모박스를 계산 하는 코드
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):# 이거는 내 컴터의 비율을 가져와서 그 다음에 실제 px로 바까주는 방식
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list): # 이걸로 추적하는 방식인가?
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def pre_process_point_history(image, point_history): #이거는 모델에게 주기전에 가공을 하는 코드
    # 1. 기준점(0,0)으로 초기화
    # 2. 크기 정규화
    # 3. 1차원 리스트로 펴기
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

    # Convert to a one-dimensional list
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history


def logging_csv(number, mode, landmark_list, point_history_list): #실제 파일에 저장하는 코드
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    if mode == 2 and (0 <= number <= 9):
        csv_path = 'model/point_history_classifier/point_history.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])
    return


def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Thumb
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (255, 255, 255), 2)

        # Index finger
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (255, 255, 255), 2)

        # Middle finger
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (255, 255, 255), 2)

        # Ring finger
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (255, 255, 255), 2)

        # Little finger
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (255, 255, 255), 2)

        # Palm
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (255, 255, 255), 2)

    # Key Points
    for index, landmark in enumerate(landmark_point):
        if index == 0:  # 手首1
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:  # 手首2
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:  # 親指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:  # 親指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:  # 親指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5:  # 人差指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:  # 人差指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:  # 人差指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:  # 人差指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:  # 中指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:  # 中指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:  # 中指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:  # 中指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:  # 薬指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:  # 薬指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:  # 薬指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:  # 薬指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:  # 小指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:  # 小指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:  # 小指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:  # 小指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image


def draw_info_text(image, brect, handedness, hand_sign_text,
                   finger_gesture_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    if finger_gesture_text != "":
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                   cv.LINE_AA)

    return image


def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (152, 251, 152), 2)

    return image


def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)

    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image


if __name__ == '__main__':
    main()
