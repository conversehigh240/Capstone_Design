import tensorflow as tf
import tensorflow_hub as hub
import cv2
from matplotlib import pyplot as plt
import numpy as np
import urllib
import yt_dlp
import time
import math

#model = hub.load('https://tfhub.dev/google/movenet/multipose/lightning/1')
#movenet = model.signatures['serving_default']

model = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
movenet = model.signatures['serving_default']
input_size = 256

EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

def loop_through_people(frame, keypoints_with_scores, edges, confidence_threshold):
    #for person in keypoints_with_scores:
    #    draw_connections(frame, person, edges, confidence_threshold)
    #    draw_keypoints(frame, person, confidence_threshold)
    draw_connections(frame, keypoints_with_scores, edges, confidence_threshold)
    draw_keypoints(frame, keypoints_with_scores, confidence_threshold)


def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 6, (0,255,0), -1)


def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    #print("np.multiply: ", np.multiply(keypoints, [y,x,1]))
    #print("keypoints: ", keypoints)
    #print("shaped: ", shaped)

    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):      
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 4)

def get_YouTube_stream_url(youtube_url):
    ydl_opts = {}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(youtube_url, download=False)
        best_format = info_dict['formats'][40]
        return best_format['url']
    

def get_keypoints(frame):
    # 이미지 전처리
    img = frame.copy()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 384, 640)
    img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), input_size, input_size)
    input_img = tf.cast(img, dtype=tf.int32)

    # keypoint 예측
    results = movenet(input_img)
    #keypoints = results['output_0'].numpy()[:,:,:51].reshape((6,17,3))
    keypoints = results['output_0'].numpy()[:,0,:51].reshape((17,3))

    return keypoints
    

# def calculate_similarity(keypoints1, keypoints2):
#     # 유클리디안 거리를 사용하여 두 keypoint 세트간의 유사도 계산
#     distance = np.sqrt(np.sum((keypoints1[:,:2] - keypoints2[:,:2]) ** 2, axis=1))
#     similarity = np.exp(-np.mean(distance))
#     return similarity


def crop_keypoints(keypoints):
    # 키포인트에서 x, y 좌표 추출
    x_coordinates = keypoints[:, 0]
    y_coordinates = keypoints[:, 1]

    # 키포인트 중 최소, 최대 좌표값 계산
    min_x, min_y = np.min(x_coordinates), np.min(y_coordinates)
    max_x, max_y = np.max(x_coordinates), np.max(y_coordinates)

    # bounding box 크기 계산
    bbox_width = max_x - min_x
    bbox_height = max_y - min_y

    # 좌표 중점 계산
    center_x = (min_x + max_x) // 2
    center_y = (min_y + max_y) // 2

    # 적절한 크기의 bounding box 생성 (예: 폭과 높이에 패딩 추가)
    padding = 20
    bbox_x = max(0, center_x - bbox_width // 2 - padding)
    bbox_y = max(0, center_y - bbox_height // 2 - padding)
    #bbox_width = min(image.shape[1] - bbox_x, bbox_width + 2 * padding)
    #bbox_height = min(image.shape[0] - bbox_y, bbox_height + 2 * padding)

    # 키포인트를 bounding box 크기에 맞게 조정
    keypoints[:, 0] -= bbox_x
    keypoints[:, 1] -= bbox_y
    
    return keypoints


def flatten_keypoints(keypoints):
    # 키포인트에서 x, y 좌표 추출
    x_coordinates = keypoints[:, 0]
    y_coordinates = keypoints[:, 1]

    # 키포인트를 평평하게 펴기 (신뢰도 정보는 제외)
    flat_keypoints = np.concatenate([x_coordinates, y_coordinates])

    return flat_keypoints


def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    
    similarity = dot_product / norm_product
    return similarity


def calculate_pose_similarity(keypoints1, keypoints2):
    keypoints1 = crop_keypoints(keypoints1)
    keypoints2 = crop_keypoints(keypoints2)

    # 키포인트를 평평하게 펴기
    flat_keypoints1 = flatten_keypoints(keypoints1)
    flat_keypoints2 = flatten_keypoints(keypoints2)

    # 코사인 유사성 계산
    similarity = math.sqrt(2 * (1 - cosine_similarity(flat_keypoints1, flat_keypoints2)))

    return similarity


# def calculate_pose_similarity_weighted(keypoints1, keypoints2):
#     keypoints1 = crop_keypoints(keypoints1)
#     keypoints2 = crop_keypoints(keypoints2)




#def get_angle(p1, p2, p3):
#    import math
#    angle = math.degrees(math.atan2(p3[1]-p2[1], p3[0]-p2[0]) - math.atan2(p1[1]-p2[1], p1[0]-p2[0]))
#    return 360 - angle if angle > 180 else abs(angle)


# def calculate_angle_diff(youtube_keypoints, webcam_keypoints):
#     angle_diff = 0

#     # 모범 keypoints
#     ex_nose = youtube_keypoints[0]
#     ex_eye_l, ex_eye_r = youtube_keypoints[1], youtube_keypoints[2]
#     ex_ear_l, ex_ear_r = youtube_keypoints[3], youtube_keypoints[4]
#     ex_shoulder_l, ex_shoulder_r = youtube_keypoints[5], youtube_keypoints[6]
#     ex_elbow_l, ex_elbow_r = youtube_keypoints[7], youtube_keypoints[8]
#     ex_wrist_l, ex_wrist_r = youtube_keypoints[9], youtube_keypoints[10]
#     ex_hip_l, ex_hip_r = youtube_keypoints[11], youtube_keypoints[12]
#     ex_knee_l, ex_knee_r = youtube_keypoints[13], youtube_keypoints[14]
#     ex_ankle_l, ex_ankle_r = youtube_keypoints[15], youtube_keypoints[16]
    
#     # 사용자 keypoints
#     user_nose = webcam_keypoints[0]
#     user_eye_l, user_eye_r = webcam_keypoints[1], webcam_keypoints[2]
#     user_ear_l, user_ear_r = webcam_keypoints[3], webcam_keypoints[4]
#     user_shoulder_l, user_shoulder_r = webcam_keypoints[5], webcam_keypoints[6]
#     user_elbow_l, user_elbow_r = webcam_keypoints[7], webcam_keypoints[8]
#     user_wrist_l, user_wrist_r = webcam_keypoints[9], webcam_keypoints[10]
#     user_hip_l, user_hip_r = webcam_keypoints[11], webcam_keypoints[12]
#     user_knee_l, user_knee_r = webcam_keypoints[13], webcam_keypoints[14]
#     user_ankle_l, user_ankle_r = webcam_keypoints[15], webcam_keypoints[16]

#     # 왼쪽 팔꿈치 각도 체크하기
#     ex_elbow_l_angle = get_angle(ex_shoulder_l, ex_elbow_l, ex_wrist_l)
#     user_elbow_l_angle = get_angle(user_shoulder_l, user_elbow_l, user_wrist_l)
#     angle_diff += abs(ex_elbow_l_angle - user_elbow_l_angle)

#     # 오른쪽 팔꿈치 각도 체크하기
#     ex_elbow_r_angle = get_angle(ex_shoulder_r, ex_elbow_r, ex_wrist_r)
#     user_elbow_r_angle = get_angle(user_shoulder_r, user_elbow_r, user_wrist_r)
#     angle_diff += abs(ex_elbow_r_angle - user_elbow_r_angle)

#     # 왼쪽 무릎 각도 체크하기
#     ex_knee_l_angle = get_angle(ex_hip_l, ex_knee_l, ex_ankle_l)
#     user_knee_l_angle = get_angle(user_hip_l, user_knee_l, user_ankle_l)
#     angle_diff += abs(ex_knee_l_angle - user_knee_l_angle)

#     # 오른쪽 무릎 각도 체크하기
#     ex_knee_r_angle = get_angle(ex_hip_r, ex_knee_r, ex_ankle_r)
#     user_knee_r_angle = get_angle(user_hip_r, user_knee_r, user_ankle_r)
#     angle_diff += abs(ex_knee_r_angle - user_knee_r_angle)

#     return angle_diff


def angle_similarity(angle_difference, mean=0, std_dev=30):
    from scipy.stats import norm
    
    # 정규분포 함수를 사용하여 각도 차이를 유사도로 변환
    similarity = norm.pdf(angle_difference, loc=mean, scale=std_dev)
    return similarity


def main(youtube_url):

    youtube_stream_url = get_YouTube_stream_url(youtube_url)
    youtube_cap = cv2.VideoCapture(youtube_stream_url)

    webcam_cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

    prev_time = 0
    fps = 5


    while youtube_cap.isOpened():

        ret_youtube, frame_youtube = youtube_cap.read()
        current_time = time.time() - prev_time

        # 유튜브에서 프레임 읽기
        if not ret_youtube:
            print("Failed to read youtube video")
            break

        # 웹캠에서 프레임 읽기
        ret_webcam, frame_webcam = webcam_cap.read()
        if not ret_webcam:
            print("Failed to read webcam")
            break

        #fps 조절
        if current_time > 1./fps:
            prev_time = time.time()

            # keypoint 추출
            youtube_keypoints = get_keypoints(frame_youtube)
            loop_through_people(frame_youtube, youtube_keypoints, EDGES, 0.1)
            webcam_keypoints = get_keypoints(frame_webcam)
            loop_through_people(frame_webcam, webcam_keypoints, EDGES, 0.1)

            # 상태 정보 받아오기
            # angle_diff = calculate_angle_diff(youtube_keypoints, webcam_keypoints)

            # 유사도 계산
            similarity = calculate_pose_similarity(youtube_keypoints, webcam_keypoints)
            print("Pose similarity:", similarity)

        # 결과 표시
        cv2.putText(frame_webcam, f"Similarity: {similarity:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Frame', frame_youtube)
        cv2.imshow('Webcam', frame_webcam)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    youtube_cap.release()
    webcam_cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    youtube_url = "https://www.youtube.com/watch?v=OFibSNpw2hE"
    main(youtube_url)