import cv2
import os
from PIL import Image
import numpy as np

detector = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
recogniser = cv2.face.LBPHFaceRecognizer_create()

recogniser.read('./recogniser.yml')

camera = cv2.VideoCapture(0)

while (1>0):
     _ ,image=camera.read()
     
     grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
     faces = detector.detectMultiScale(grey, 1.2,5)
     for(x,y,w,h) in faces:
        # Create rectangle
        cv2.rectangle(image, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 4)
        #predict faceID of the face
        Id, confidence = recogniser.predict(grey[y:y+h,x:x+w])
        if(Id == 0):
            Id = "Iga {0:.2f}%".format(round(100 - confidence, 2))
        if(Id == 1):
            Id = "Bagja {0:.2f}%".format(round(100 - confidence, 2))
        # add text
        cv2.rectangle(image, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1)
        cv2.putText(image, str(Id), (x,y-40), 0, 1, (255,255,255), 3)

    # show the video feed with recognised faces
     cv2.imshow('Result',image) 

    # If 'q' is pressed, close program
     if  cv2.waitKey(100) & 0xFF == ord('q') :
        break


camera.release()
cv2.destroyAllWindows()
