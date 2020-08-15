from flask import Flask, render_template, request, flash, redirect, url_for
import cv2
import numpy
from werkzeug.utils import secure_filename
import os
import glob
import image_process_training as ipt

upload_folder = "static/image/"

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = upload_folder

def save_file(file):
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

def get_label(image):
    label = ipt.test(image)

    return label

def resize_image(img):
    scale_percent = 50 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    
    return resized

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/upload", methods=['POST'])
def upload():
    path = glob.glob('static/image/*')

    for f in path:
        os.remove(f)

    file = request.files['makanan']
    save_file(file)
    images = [cv2.imread(img) for img in glob.glob('static/image/*')]
    label = get_label(images[0])

    if len(label.split(' ')) == 2:
        for_link = '_'.join(label.split(' '))
        link = "https://en.wikipedia.org/wiki/" + for_link
    else:
        link = "https://en.wikipedia.org/wiki/" + label

    return render_template("page.html", label=label, filename = file.filename, label_for_link = link)

@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(debug=True)