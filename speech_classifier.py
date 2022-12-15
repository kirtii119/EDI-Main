from __future__ import division, print_function
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv
import os 
import shutil
import librosa
import tensorflow as tf
speech_model = tf.keras.models.load_model("speech_model.h5")

# try:
#     shutil.rmtree('songs')
# except:
#     print("unable to delete previous audio data or no song folder is present")

# try: 
#     os.mkdir("songs")
# except: 
#     print("directry is already present")

speech_ouput = []
dict1 = {0:'Fear',1:'Happiness',2:'Neutral',3:'Sadness'}

def extract_mfcc(filename): #
    y, sr = librosa.load(filename, duration=3, offset=0.5) #load audio file
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0) #extract mfcc features
    return mfcc

def audio_classifier():
                filename = "candidate_answer.wav"
                rate = 22050           # samples per second
                T = 4                 # sample duration (seconds)
                n = int(rate*T)        # number of samples
                t = np.arange(n)/rate  # grid of time values

                # f = 440.0              # sound frequency (Hz)
                # wavio.write("sine24.wav", x, rate, sampwidth=3)

                freq = 44100  # audio sampling rate
                x = np.sin(2*np.pi * freq * t) #generate sine wave
                wv.write(filename, x, freq, sampwidth=4) #write audio file
                
                feature = extract_mfcc(filename) #extract mfcc features 
                
                feature = np.array([feature]) #reshape to 3d array
                feature = feature.reshape(1, 40, 1)
                
                audio_result = speech_model.predict(feature) #predicting 
                audio_result = np.argmax(audio_result)

                
                audio_result = dict1[audio_result]
                speech_ouput.append(audio_result)
                print(speech_ouput)
                return audio_result