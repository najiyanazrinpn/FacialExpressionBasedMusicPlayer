import random
from turtle import onclick, width
from pandas import wide_to_long
import streamlit as st

import tensorflow as tf
from tensorflow.keras.models import load_model

import cv2
import numpy as np

from pygame import mixer

from IPython.display import display, Javascript
# from js2py import eval_js
from base64 import b64decode

from PIL import Image, ImageOps


from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import matplotlib.pyplot as plt

from streamlit_player import st_player

##########################################################################

def facecrop():  
    facedata = 'haarcascade_frontalface_default.xml'
    cascade = cv2.CascadeClassifier(facedata)

    img = cv2.imread('picture.jpg')

    try:
    
        minisize = (img.shape[1],img.shape[0])
        miniframe = cv2.resize(img, minisize)

        faces = cascade.detectMultiScale(miniframe)

        for f in faces:
            x, y, w, h = [ v for v in f ]
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

            sub_face = img[y:y+h, x:x+w]

            
            cv2.imwrite('capture.jpg', sub_face)
            #print ("Writing: " + image)

    except Exception as e:
        st.write(e)

#######################################################################


def emotion_analysis(emotions):
    objects = ('angry', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    y_pos = np.arange(len(objects))
    
    plt.bar(y_pos, emotions, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('percentage')
    plt.title('emotion')
    
    plt.show()

    return (objects[np.argmax(emotions)])

########################################################################

st.write("""
          # Intelligent Music Player
          """
          )
emotion_model = tf.keras.models.load_model('emotion_model.hdf5')


run = st.checkbox('Check')
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)
i=0



while run:
    _, frame = camera.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)
    i+=1
    if i==1:
        if(st.button('Capture')):
            cv2.imwrite('picture.jpg', frame)
            img = cv2.imread('picture.jpg')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imwrite('picture.jpg', img)
            
            img = cv2.imread('picture.jpg')
            facecrop()

            file = 'capture.jpg'
            true_image = image.load_img(file)
            imag = image.load_img(file, color_mode="grayscale", target_size=(48, 48))

            x = image.img_to_array(imag)
            x = np.expand_dims(x, axis = 0)

            x /= 255

            custom = emotion_model.predict(x)

            img = Image.open("capture.jpg")
            st.image(img, width=200)

            myemotion = emotion_analysis(custom[0])

            st.write(myemotion)
            
            

            if(myemotion=='angry'):

                songs = ["angry1.mp3", "angry2.mp3", "angry3.mp3", "angry4.mp3"]

                random.shuffle(songs)
                s=songs[0]

                song = open("angry\\"+s, 'rb')
                audio_bytes = song.read()
                st.audio(audio_bytes, format='audio/ogg')
            elif(myemotion=='fear'):
                songs = ["fear1.mp3", "fear2.mp3", "fear3.mp3", "fear4.mp3"]

                random.shuffle(songs)
                s=songs[0]

                song = open("fear\\"+s, 'rb')
                audio_bytes = song.read()
                st.audio(audio_bytes, format='audio/ogg')
            elif(myemotion=='happy'):
                songs = ["happy1.mp3", "happy2.mp3", "happy3.mp3", "happy4.mp3"]

                random.shuffle(songs)
                s=songs[0]

                song = open("happy\\"+s, 'rb')
                audio_bytes = song.read()
                st.audio(audio_bytes, format='audio/ogg')
            elif(myemotion=='sad'):
                songs = ["sad1.mp3", "sad2.mp3", "sad3.mp3", "sad4.mp3"]

                random.shuffle(songs)
                s=songs[0]

                song = open("sad\\"+s, 'rb')
                audio_bytes = song.read()
                st.audio(audio_bytes, format='audio/ogg')
                
                
  
# for playing note.wav file



else:
    st.write('Stopped')




###############################################################################