import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import math

model = hub.load("https://www.kaggle.com/models/google/movenet/frameworks/TensorFlow2/variations/multipose-lightning/versions/1")
movenet = model.signatures['serving_default']

webcam_cap = None

def initialize_webcam():
    global webcam_cap
    webcam_cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    webcam_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
    webcam_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1280)

# Dictionary that maps from joint names to keypoint indices.
KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

# Maps bones to a matplotlib color name.
KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): (255, 0, 255),
    (0, 2): (0, 255, 255),
    (1, 3): (255, 0, 255),
    (2, 4): (0, 255, 255),
    (0, 5): (255, 0, 255),
    (0, 6): (0, 255, 255),
    (5, 7): (255, 0, 255),
    (7, 9): (255, 0, 255),
    (6, 8): (0, 255, 255),
    (8, 10): (0, 255, 255),
    (5, 6): (255, 255, 0),
    (5, 11): (255, 0, 255),
    (6, 12): (0, 255, 255),
    (11, 12): (255, 255, 0),
    (11, 13): (255, 0, 255),
    (13, 15): (255, 0, 255),
    (12, 14): (0, 255, 255),
    (14, 16): (0, 255, 255)
}

def draw_on_video(frame, keypoints, confidence_threshold):
    draw_connections(frame, keypoints, confidence_threshold)
    draw_keypoints(frame, keypoints, confidence_threshold)

def draw_keypoints(frame, keypoints, confidence_threshold):
    height, width, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [height, width, 1]))
    
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 3, (255, 20, 147), -1)

def draw_connections(frame, keypoints, confidence_threshold):
    height, width, channel = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [height, width, 1]))

    for edge, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):      
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 4)
    

def get_keypoints_webcam(frame):
    # 이미지 전처리
    img = frame.copy()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 256, 256)
    input_img = tf.cast(img, dtype=tf.int32)

    # keypoint 예측
    results = movenet(input_img)
    #keypoints = results['output_0'].numpy()[:,:,:51].reshape((6,17,3))
    keypoints = results['output_0'].numpy()[:,:,:51].reshape((6,17,3))

    return keypoints[0]

def L2_normalize(array):
    norm = np.linalg.norm(array, axis=1)
    normalized_array = array / norm[:, np.newaxis]
    return normalized_array


def score_similarity(similarity):
    # scaled = ((similarity + 1) / 2) * 100
    scaled = math.sqrt(2 * (1 - similarity)) * 100
    return np.round(scaled, 2)


def calculate_pose_similarity(keypoints_ytb, keypoints_web):

    keypoints_ytb = keypoints_ytb[:, :-1]
    keypoints_web = keypoints_web[:, :-1]

    # x_ytb = keypoints_ytb[:, 1]
    # y_ytb = keypoints_ytb[:, 0]

    # center_x_ytb = np.mean(x_ytb)
    # center_y_ytb = np.mean(y_ytb)

    # x_web = keypoints_web[:, 1]
    # y_web = keypoints_web[:, 0]

    # center_x_web = np.mean(x_web)
    # center_y_web = np.mean(y_web)

    # x_trans = center_x_web - center_x_ytb
    # y_trans = center_y_web - center_y_ytb
    
    # keypoints_ytb[:, 1] += x_trans
    # keypoints_ytb[:, 0] += y_trans

    # norm_ytb = L2_normalize(keypoints_ytb)
    # norm_web = L2_normalize(keypoints_web)

    similarity = cosine_similarity(keypoints_ytb, keypoints_web)
    average = np.mean(similarity)
    score = score_similarity(average)

    return score


def webcam_similarity(id):

    file_path = 'static/keypoints/' + id + '.npy'

    youtube_keypoints = np.load(file_path)
    
    while True:
        for i in range(len(youtube_keypoints)):

            # 웹캠에서 프레임 읽기
            ret_webcam, frame_webcam = webcam_cap.read()
            if not ret_webcam:
                print("Failed to read webcam")
                break

            # keypoint 추출
            keypoints = get_keypoints_webcam(frame_webcam)
            draw_on_video(frame_webcam, keypoints, 0.1)
            similarity = calculate_pose_similarity(youtube_keypoints[i], keypoints)
            cv2.putText(frame_webcam, f"Similarity: {similarity}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)

            ret, buffer = cv2.imencode('.jpg', frame_webcam)
            frame_webcam = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_webcam + b'\r\n')


            # 'q' 키를 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # webcam_cap.release()
    # cv2.destroyAllWindows()


def webcam():

    while True:

        # 웹캠에서 프레임 읽기
        ret_webcam, frame_webcam = webcam_cap.read()
        if not ret_webcam:
            print("Failed to read webcam")
            break

        # keypoint 추출
        keypoints = get_keypoints_webcam(frame_webcam)
        draw_on_video(frame_webcam, keypoints, 0.1)


        ret, buffer = cv2.imencode('.jpg', frame_webcam)
        frame_webcam = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_webcam + b'\r\n')

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # webcam_cap.release()
    # cv2.destroyAllWindows()


def test():

    file_path = 'static/keypoints/Yu20j5jGHTc.npy'
    youtube_keypoints = np.load(file_path)

    test_path = 'static/keypoints/exXly1KGEgM.npy'
    test_keypoints = np.load(test_path)

    video_path = 'static/exXly1KGEgM.mp4'
    youtube_cap = cv2.VideoCapture(video_path)

    
    while True:

        for i in range(len(youtube_keypoints)):

            ret_youtube, frame_youtube = youtube_cap.read()

            # 유튜브에서 프레임 읽기
            if not ret_youtube:
                print("Failed to read youtube video")
                break

            # 유사도 계산
            similarity = calculate_pose_similarity(youtube_keypoints[i], test_keypoints[i])
            cv2.putText(frame_youtube, f"Similarity: {similarity}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)

            ret, buffer = cv2.imencode('.jpg', frame_youtube)
            frame_youtube = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_youtube + b'\r\n')


            # 'q' 키를 누르면 종료
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        youtube_cap.release()
        cv2.destroyAllWindows()