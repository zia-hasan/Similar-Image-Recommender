from __future__ import absolute_import, division, print_function
import os
from uuid import uuid4
from flask import Flask, request, render_template, send_from_directory
import ann
import sys
from config import *
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
sys.dont_write_bytecode=True

app = Flask(__name__)

ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    return render_template("upload.html")

@app.route("/similar", methods=["POST"])
def upload():
    target = os.path.join(APP_ROOT, UPLOAD_FOLDER)
    print(target)
    if not os.path.isdir(target):
            os.mkdir(target)
    else:
        print("Couldn't create upload directory: {}".format(target))
    print(request.files.getlist("file"))
    for upload in request.files.getlist("file"):
        print(upload)
        print("{} is the file name".format(upload.filename))
        filename = upload.filename
        if allowed_file(filename):
            destination = "/".join([target, filename])
            print ("Accept incoming file:", filename)
            print ("Save it to:", destination)
            upload.save(destination)
            uploaded_image = UPLOAD_FOLDER+filename
            similar_images = ann.find_similar_images(uploaded_image)
            similar_images = [image.split("/")[1] for image in similar_images]
            list_images = [filename]+ similar_images 
            print(list_images)
    return render_template("similar.html", image_names=list_images)

@app.route('/upload/<filename>')
def send_uploaded(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/similar/<filename>')
def send_similar(filename):
    return send_from_directory(DATASET_PATH, filename)


if __name__ == "__main__":
    # app.run(port=8080, debug=True)
    app.run(host= '0.0.0.0', port=8080)