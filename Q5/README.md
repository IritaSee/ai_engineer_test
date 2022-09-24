# Artificial Intelligence Engineer Employee Test: Q5) Face recognition using Haar Cascade

## Overview
In this folder there are several files having their own functions, such as:
* `detect_face.py` to see the face recognition result using built in webcam. Screenshot of this script running is `result.png`
* `detect_face_grey.py` same with `detect_face` but in greyscale. Screenshot of this script running is `result_grey.png`
* `haarcascade_frontalface_default.xml` a Stump-based 24x24 adaboost frontal face detector configuration. Created by Rainer Lienhart.
* `Q5_Haar_Cascade_face_recognition.ipynb` contains initial setup and training function
* `recogniser.yml` contains histogram data for each faces 
* `sample.jpeg` and `sample2.jpg` face sample included for demo.
## How to use
Just like Q4, external library I used in these answers are available in `requirements.txt`, simply open `Q5_Haar_Cascade_face_recognition.ipynb` and run the first cell. Once we are ready with all of the libraries, follow this instruction.
1) Open `Q5_Haar_Cascade_face_recognition.ipynb`, directions are continued in the notebook.
2) Make sure we have `recogniser.yml` and `haarcascade_frontalface_default.xml` since the detection will not work without these two files. 
3) Run `detect_face.py` or `detect_face_grey.py` using terminal, the GUI will stuck if they are being runned using jupyter notebook. That is why they are being seperated.