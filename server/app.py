from flask import Flask
import movenet
import subprocess
import os
import json
import numpy

app = Flask(__name__)


@app.route('/process/<id>', methods=['GET'])
def process(id):
    movenet.save_youtube(id)
    output_path = 'static/' + id + '.mp4'
    subprocess.run(['ffmpeg', '-i', 'static/output_video.mp4', '-i', 'static/output_audio.m4a',
                    '-c:v', 'copy', '-c:a', 'aac', '-strict', 'experimental', 'static/output_combined.mp4'])
    subprocess.run(['ffmpeg', '-i', 'static/output_combined.mp4', '-c:v', 'h264', output_path])

    os.remove('static/input.mp4')
    os.remove('static/output_video.mp4')
    os.remove('static/output_audio.m4a')
    os.remove('static/output_combined.mp4')
    return "http://163.180.160.36:5001/" + output_path



if __name__ == '__main__':
    app.run(host = '0.0.0.0', port=5001)