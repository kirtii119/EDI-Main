#!/usr/bin/env python
import cv2
import json
import numpy as np
import face_classifier
import tensorflow as tf

from flask import Flask, render_template,  request
from keras.models import model_from_json, load_model

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Load Haarcascade File
face_detector = cv2.CascadeClassifier("ml_folder/haarcascade_frontalface_default.xml")

# Load the Model and Weights
model = load_model('ml_folder/video.h5')
speech_model = tf.keras.models.load_model("speech_model.h5")
# model.load_weights('ml_folder/model.h5')
# model._make_predict_function()


analyticsDict = {
  "angry": 0,
  "disgust": 0,
  "fear": 0,
  "happy": 0,
  "sad": 0,
  "surprise": 0,
  "neutral": 0
}




@app.route('/')
def index():
    return render_template('index.html')


@app.route('/uploade', methods=['POST', 'GET'])
def upload_file():
    if request.method == 'POST':
        # f.save("somefile.jpeg")
        # f = request.files['file']

        f = request.files['file'].read()
        npimg = np.fromstring(f, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)
        face_properties = face_classifier.classify(img, face_detector, model)
        # label = face_properties[0]['label']
        # if (label == "happy" or label == "happy" or label == "happy" or label == "happy" or label == "happy" or label == "happy" )
        if (len(face_properties)!=0 ):
            analyticsDict[face_properties[0]['label']] +=1
        return json.dumps(face_properties)

@app.route('/finish', methods=['GET'])
def finish_file():
    print(analyticsDict)
    confidence = 0
    nervous = 0
    neutral = 0

    for key in analyticsDict:
        if key == "happy":
            confidence += analyticsDict[key]
        if key == "sad":
            nervous += analyticsDict[key]
        if key == "angry":
            nervous += analyticsDict[key]
        if key == "neutral":
            neutral += analyticsDict[key]
        if key == "disgust":
            nervous += analyticsDict[key]
        if key == "fear":
            nervous += analyticsDict[key]
        if key == "surprise":
            nervous += analyticsDict[key]
    
    total = confidence + nervous + neutral
    confidence = ((confidence + neutral)/total) * 100
    neutral = (neutral/total) * 100
    nervous = (nervous /total) * 100

    mainstr = "confidence: " + str(confidence) + "%\t nervousness: " + str(nervous) + "%\t neutral: " + str(neutral)
    print(mainstr)  
    print(analyticsDict)

    file1 = open("face_result.txt", "a")
    file1.write(mainstr)
    file1.write("\n")
    file1.close()

    for key in analyticsDict:
        analyticsDict[key] = 0
    print(analyticsDict)
    
    return "done"


if __name__ == '__main__':

    # Run the flask app
    app.run()