import os
from flask import Flask, flash, request, redirect, url_for
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
      if 'file' not in request.files or 'file2' not in request.files:
         flash('No file part')
         return redirect(request.url)
      file = request.files['file']
      file2 = request.files['file2']
      # if user does not select file, browser also
      # submit an empty part without filename
      if file.filename == '' or file2.filename == '':
         flash('No selected file')
         return redirect(request.url)
      if file and allowed_file(file.filename) and file2 and allowed_file(file2.filename):
         filename = secure_filename('photo_' + file.filename)
         filename2 = secure_filename('mask_' + file2.filename)
         if not os.path.exists(UPLOAD_FOLDER):
            os.makedirs(UPLOAD_FOLDER)
         file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
         file2.save(os.path.join(app.config['UPLOAD_FOLDER'], filename2))
         return redirect(request.url)
   return '''
   <!doctype html>
   <title>Bones angle counter</title>
   <h1>Bones angle counter</h1>
   <form method=post enctype=multipart/form-data>
     <span>Photo: </span>
     <input type=file name=file>
     <span>Mask :</span>
     <input type=file name=file2>
     <input type=submit value=Upload>
   </form>
   '''

from flask import send_from_directory

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)