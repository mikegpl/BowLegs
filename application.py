import os

from flask import Flask, flash, request, redirect
from werkzeug.utils import secure_filename

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(ROOT_DIR, 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename('photo_' + file.filename)
            if not os.path.exists(UPLOAD_FOLDER):
                os.makedirs(UPLOAD_FOLDER)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(request.url)
    return '''
   <!doctype html>
   <head>
        <title>Bones angle counter</title>
        <link rel="stylesheet" href='./static/css/main.css' />
   </head>
   <div class="container">
       <h1 class="title">Bones angle counter</h1>
       <form class="form-container" method=post enctype=multipart/form-data>
         <span class="photo">Upload a photo: </span>
         <input class="photo__input" type=file name=file>
         <input class="photo__upload custom-button" type=submit value=Upload>
       </form>
       <div class="result">
            <span class="result__title">Angle between bones: </span>
            <span class="result__angle">Upload a picture to get the angle</span>
       </div>
   </div>
   '''


from flask import send_from_directory


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)
