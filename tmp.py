from flask import Flask
import os
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
import json

import cv2
import tensorflow as tf
import numpy as np
app = Flask(__name__)

@app.route("/")
def hello_world():
    return '<p>Hello, World!</p><img class="logo" src="../_static/flask-icon.png" alt="Logo">'

    
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# CATEGORIES = ["covid", "normal"]
# #
# def prepare(filepath):
#     IMG_SIZE = 70
#     img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
#     ret,thresh1 = cv2.threshold(img_array,85,255,cv2.THRESH_BINARY)
#     thresh1 = thresh1/255.0

#     new_array = cv2.resize(thresh1, (IMG_SIZE, IMG_SIZE))
#     return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


UPLOAD_FOLDER = 'uploads_cnn'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}



def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=[ 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))


            
            # model = tf.keras.models.load_model("Final_Model")
            # img_tensor =[prepare('uploads_cnn/'+filename)]

            # prediction = np.argmax(model.predict(img_tensor), axis=-1)
            # classP = model.predict_classes(img_tensor)[0][0]
            # result1 = (prediction*100 , classP) 
            # result2 = (CATEGORIES[int(prediction)])
            # print(result1)
            # print(result2)

            return json.dumps({'result':str(filename)}), 200, {'ContentType':'application/json'}  
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

if __name__ == "__main__":
    app.run()
