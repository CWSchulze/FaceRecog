"""
Create new dataset by using the webcam
"""

import cv2
import os

new_dataset_name = 'MyUserName'
picture_to_take = 300
# defining the size of images
(width, height) = (250,250)

datasets_subfolder = 'datasets'
haar_file = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')

path = os.path.join(datasets_subfolder, new_dataset_name)
if not os.path.isdir(path):
    os.makedirs(path)

#'0' is used for my webcam,
# if you've any other camera
#  attached use '1' like this
face_cascade = cv2.CascadeClassifier(haar_file)
webcam = cv2.VideoCapture(0)


count = 1
while count < picture_to_take:
    (_, im) = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = gray[y:y + h, x:x + w]
        # width = int(height/h*w)
        face_resize = cv2.resize(face, (width, height))
        cv2.imwrite('% s/% s.png' % (path, count), face_resize)
    count += 1

    cv2.imshow('OpenCV', im)
    key = cv2.waitKey(10)
    if key == 27:
        break
