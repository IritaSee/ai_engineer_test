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
     
     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   
    # Get all face from the video frame
     faces = detector.detectMultiScale(image, 1.2,5)

    # For each face in faces
     for(x,y,w,h) in faces:

        # Create rectangle around the face
        cv2.rectangle(image, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 4)

        # Recognize the face belongs to which ID
        Id, confidence = recogniser.predict(image[y:y+h,x:x+w])

        # Check the ID if exist 
        if(Id == 0):
            Id = "Iga {0:.2f}%".format(round(100 - confidence, 2))
        if(Id == 1):
            Id = "Bagja {0:.2f}%".format(round(100 - confidence, 2))

        # Put text describe who is in the picture
        cv2.rectangle(image, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1)
        cv2.putText(image, str(Id), (x,y-40), 0, 1, (255,255,255), 3)

    # Display the video frame with the bounded rectangle
     cv2.imshow('Result',image) 

    # If 'q' is pressed, close program
     if  cv2.waitKey(100) & 0xFF == ord('q') :
        break

# Stop the camera
camera.release()

# Close all windows
cv2.destroyAllWindows()
