import tensorflow as tf
import tensorflow_hub as hub
import cv2
from matplotlib import pyplot as plt
import numpy as np
import urllib
import yt_dlp
import time

model = hub.load('https://tfhub.dev/google/movenet/multipose/lightning/1')
movenet = model.signatures['serving_default']

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
    for person in keypoints_with_scores:
        draw_connections(frame, person, edges, confidence_threshold)
        draw_keypoints(frame, person, confidence_threshold)


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
    # img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 384, 640)
    input_img = tf.cast(img, dtype=tf.int32)

    # keypoint 예측
    results = movenet(input_img)
    keypoints = results['output_0'].numpy()[:,:,:51].reshape((6,17,3))

    return keypoints
    

def calculate_similarity(keypoints1, keypoints2):
    # 유클리디안 거리를 사용하여 두 keypoint 세트간의 유사도 계산
    distance = np.sqrt(np.sum((keypoints1[:,:2] - keypoints2[:,:2]) ** 2, axis=1))
    similarity = np.exp(-np.mean(distance))
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

            # 유사도 계산
            similarity = calculate_similarity(youtube_keypoints, webcam_keypoints)
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
    youtube_url = "https://www.youtube.com/watch?v=hpJIcy0syxw"
    main(youtube_url)