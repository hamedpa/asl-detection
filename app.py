from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
import json
import os
from pathlib import Path

import cv2
import tensorflow as tf
from PIL import Image
import numpy as np
from skimage import transform

app = Flask(__name__)


UPLOAD_FOLDER = './uploads_cnn'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
labels1 = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25, 'del': 26, 'nothing': 27, 'space': 28,'s':29}


def predictImage(imagePath,model,labels):
  np_image = Image.open(imagePath)
  np_image = np.array(np_image).astype('float32')/255
  np_image = transform.resize(np_image, (224, 224, 3))
  np_image = np.expand_dims(np_image, axis=0)
  pred = model.predict(np_image)
  pred = np.argmax(pred,axis=1)
  labels = dict((v,k) for k,v in labels.items())
  pred = [labels[k] for k in pred]
  return pred[:5]


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=[ 'GET','POST'])
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
            file.save(os.path.join('uploads_cnn/', filename))


            
            model_mobilenet = tf.keras.models.load_model('saved_model7/MobileNet')


            return json.dumps({'result':str(predictImage(os.path.join('uploads_cnn/', filename),model_mobilenet,labels1))}), 200, {'ContentType':'application/json'}  
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