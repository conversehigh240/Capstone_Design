import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
import yt_dlp
import subprocess
import os

model = hub.load("https://www.kaggle.com/models/google/movenet/frameworks/TensorFlow2/variations/multipose-lightning/versions/1")
movenet = model.signatures['serving_default']
youtube_keypoints = []

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
    

def get_keypoints_youtube(frame):
    # 이미지 전처리
    img = frame.copy()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 384, 640)
    input_img = tf.cast(img, dtype=tf.int32)

    # keypoint 예측
    results = movenet(input_img)
    keypoints = results['output_0'].numpy()[:,:,:51].reshape((6,17,3))

    return keypoints[0]

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


def download(youtube_url):
    ydl_opts = {
        'format':'best',
        'outtmpl':'static/input.mp4'
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])


def download_audio(youtube_url):
    ydl_opts = {
        'format':'bestaudio',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'm4a',
            'preferredquality': '192',
        }],
        'outtmpl':'static/output_audio'
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])


def save_youtube(id):

    youtube_url = "https://www.youtube.com/watch?v=" + id
    thumbnail_path = 'static/thumbnails/' + id

    download(youtube_url)
    download_audio(youtube_url)
    subprocess.run(['yt-dlp', '--skip-download', '--write-thumbnail', '-o', thumbnail_path, youtube_url])

    input_path = 'static/input.mp4'
    output_path = 'static/output_video.mp4'
    keypoint_path = 'static/keypoints/' + id
    youtube_cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output = cv2.VideoWriter(output_path, fourcc, youtube_cap.get(cv2.CAP_PROP_FPS), (int(youtube_cap.get(3)), int(youtube_cap.get(4))))
    
    while youtube_cap.isOpened():

        ret, frame = youtube_cap.read()

        # 유튜브에서 프레임 읽기
        if not ret:
            break

        # keypoint 추출
        keypoints = get_keypoints_youtube(frame)
        draw_on_video(frame, keypoints, 0.1)
        youtube_keypoints.append(keypoints)
        
        output.write(frame)

    youtube_cap.release()
    output.release()
    cv2.destroyAllWindows()

    np.save(keypoint_path, youtube_keypoints)


if __name__ == "__main__":
    with open('list.txt') as f:
        for id in f:
            save_youtube(id)
            output_path = 'static/' + id + '.mp4'
            subprocess.run(['ffmpeg', '-i', 'static/output_video.mp4', '-i', 'static/output_audio.m4a',
                            '-c:v', 'copy', '-c:a', 'aac', '-strict', 'experimental', 'static/output_combined.mp4'])
            subprocess.run(['ffmpeg', '-i', 'static/output_combined.mp4', '-c:v', 'h264', output_path])

            os.remove('static/input.mp4')
            os.remove('static/output_video.mp4')
            os.remove('static/output_audio.m4a')
            os.remove('static/output_combined.mp4')