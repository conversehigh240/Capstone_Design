from flask import Flask, render_template, request, Response, redirect, url_for, send_file
import werkzeug.utils
import webcam
import yt_dlp
import requests
from io import BytesIO

app = Flask(__name__)

video_list = [
    {"id": "IT94xC35u6k",
     "thumbnail":"static/thumbnails/IT94xC35u6k.webp",
     "title":"20 min Fat Burning Workout for TOTAL BEGINNERS (Achievable, No Equipment)"},
    {"id": "Cw-Wt4xKD2s",
     "thumbnail":"static/thumbnails/Cw-Wt4xKD2s.webp",
     "title":"15 MIN HAPPY DANCE WORKOUT - burn calories and smile / No Equipment I Pamela Reif"},
    {"id": "kVUKL8Ro1l4",
     "thumbnail":"static/thumbnails/kVUKL8Ro1l4.webp",
     "title":"24 MIN TABATA PARTY - HIIT Home Workout - No Equipment, No Repeat, Sweaty Workout with Tabata Songs"},
    {"id": "Yu20j5jGHTc",
     "thumbnail":"static/thumbnails/Yu20j5jGHTc.webp",
     "title":"20 Minute Full Body Workout (No Equipment)"},
    {"id": "HIFWIE44Ffs",
     "thumbnail":"static/thumbnails/HIFWIE44Ffs.webp",
     "title":"Quick Full Body Burn || No-Equipment || Nike Training Club"},
    {"id": "ExJSrqVMOhQ",
     "thumbnail":"static/thumbnails/ExJSrqVMOhQ.webp",
     "title":"15 MINUTE ABS & ARMS WORKOUT | NIKE TRAINING CLUB"},
    {"id": "Kl9-cYPNuWQ",
     "thumbnail":"static/thumbnails/Kl9-cYPNuWQ.webp",
     "title":"20-Min HIT: Lower Body Strength | Nike Training Club"},
    {"id": "RbCCLGMnmW8",
     "thumbnail":"static/thumbnails/RbCCLGMnmW8.webp",
     "title":"All-Around Athleticism + Core Strength | Nike Training Club"},
    {"id": "HeolReSa5ic",
     "thumbnail":"static/thumbnails/HeolReSa5ic.webp",
     "title":"20 MIN BOOTY WORKOUT // No Equipment | Pamela Reif"},
    {"id": "GLy2rYHwUqY",
     "thumbnail":"static/thumbnails/GLy2rYHwUqY.webp",
     "title":"Total Body Yoga | Deep Stretch | Yoga With Adriene"},
    {"id": "g_tea8ZNk5A",
     "thumbnail":"static/thumbnails/g_tea8ZNk5A.webp",
     "title":"15 Min. Full Body Stretch | Daily Routine for Flexibility, Mobility & Relaxation | DAY 7"},
    {"id": "UEEsdXn8oG8",
     "thumbnail":"static/thumbnails/UEEsdXn8oG8.webp",
     "title":"Wake Up Yoga | 11-Minute Morning Yoga Practice"},
    {"id": "C2HX2pNbUCM",
     "thumbnail":"static/thumbnails/C2HX2pNbUCM.webp",
     "title":"30 MIN FULL BODY WORKOUT || At-Home Pilates (No Equipment)"},
    {"id": "2MoGxae-zyo",
     "thumbnail":"static/thumbnails/2MoGxae-zyo.webp",
     "title":"Do This Everyday To Lose Weight | 2 Weeks Shred Challenge"},
    {"id": "p-uUnrCdhR8",
     "thumbnail":"static/thumbnails/p-uUnrCdhR8.webp",
     "title":"15 MIN BOOTY WORKOUT, LOW IMPACT - knee friendly, no squats, no jumps / No Equipment I Pamela Reif"},
    {"id": "2pLT-olgUJs",
     "thumbnail":"static/thumbnails/2pLT-olgUJs.webp",
     "title":"Get Abs in 2 WEEKS | Abs Workout Challenge"},
]

app.before_request_funcs = [(None, webcam.initialize_webcam())]

@app.route('/')
def index():
    return render_template('search.html', video_list=video_list)


@app.route('/result/<id>')
def result(id):
    return render_template('result.html', video_id=id)


@app.route('/process', methods=['POST'])
def process():
    url = request.form['youtube_url']
    ydl_opts = {}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=False)
        video_id = info_dict.get("id")
        video_title = info_dict.get('title')

    path = "http://163.180.160.36:5001/process/" + video_id
    video_response = requests.get(path)

    keypoints_path = 'http://163.180.160.36:5001/static/keypoints/' + video_id + '.npy'
    key_response = requests.get(keypoints_path)
    file_content = BytesIO(key_response.content)
    file_path = 'static/keypoints/' + video_id + '.npy'
    with open(file_path, 'wb') as file:
        file.write(file_content.getvalue())

    thumbnail_path = 'http://163.180.160.36:5001/static/thumbnails/' + video_id + '.webp'
    thumb_response = requests.get(thumbnail_path)
    thumb_content = BytesIO(thumb_response.content)
    thumb_path = 'static/thumbnails/' + video_id + '.webp'
    with open(thumb_path, 'wb') as f:
        f.write(thumb_content.getvalue())

    video_list.append({"id": video_id, "thumbnail":thumb_path, "title":video_title},)

    return redirect(url_for('result', id = video_id))
    

@app.route('/get_video/<id>')
def get_video(id):
    path = 'http://163.180.160.36:5001/static/' + id + '.mp4'
    return redirect(path)


@app.route('/get_similarity/<id>')
def get_similarity(id):
    return Response(webcam.webcam_similarity(id), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get_webcam')
def get_webcam():
    return Response(webcam.webcam(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/test')
def test():
    return Response(webcam.test(), mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__ == '__main__':
    app.run(host = '0.0.0.0', port=6001)