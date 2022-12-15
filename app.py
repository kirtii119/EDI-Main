#!/usr/bin/env python
import cv2
import json
import numpy as np
import face_classifier
import tensorflow as tf

from flask import Flask, render_template,  request
from keras.models import model_from_json, load_model
from flask import Flask, jsonify, request, render_template
import pandas as pd
import utility_functions
from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver
from selenium.webdriver.chrome.service import Service


df = pd.read_csv("Question Dataset.csv")
pd.set_option('display.max_colwidth', None)

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Load Haarcascade File
face_detector = cv2.CascadeClassifier("ml_folder/haarcascade_frontalface_default.xml")

# Load the Model and Weights
model = load_model('ml_folder/video.h5')
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
    return render_template("success.html")


@app.route('/interview', methods=['POST'])
def interview():
    return render_template('interview.html')

@app.route('/start', methods=['POST'])
def start():
    #start video interview from frontend 


    file = open('report_file.txt', 'r', encoding='utf-8', errors='ignore')
    data = file.readlines()
    text = []
    for line in data:
        word = line.split()
        text.append(word)

    index = 0
    for i in range(len(text)):
        if len(text[i]) > 0:
            if text[i][0] == 'Skills:':
                index = i
                break

    skills = text[index + 1]
    print(skills)
    options = webdriver.ChromeOptions()
    options.add_argument("headless")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), chrome_options=options)
    # driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

    count = 1
    interview_score = 0

    interview_score = utility_functions.ask_first_question(df, count, driver)
    print(interview_score)
    
    return render_template("success.html", interview_score=interview_score, questions=utility_functions.asked)


if __name__ == '__main__':

    # Run the flask app
    app.run()


