"""
Try to recognize person visible in webcam
"""

import cv2
import numpy
import os

# defining the size of images
(width, height) = (250,250)

datasets_subfolder = 'datasets'
haar_file = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')

# Part 1: Create fisherRecognizer
print('Recognizing Face Please Be in sufficient Lights...')

# Create a list of images and a list of corresponding names
(images, lables, names, id) = ([], [], {}, 0)
for (subdirs, dirs, files) in os.walk(datasets_subfolder):
    for subdir in dirs:
        if subdir == 'christian_webcam2':
            continue
        names[id] = subdir
        subjectpath = os.path.join(datasets_subfolder, subdir)
        for filename in os.listdir(subjectpath):
            path = subjectpath + '/' + filename
            lable = id
            images.append(cv2.imread(path, 0))
            lables.append(int(lable))
        id += 1

# Create a Numpy array from the two lists above
errors = [lis for lis in zip(images, lables) if lis[0].shape != (100,100)]
(images, lables) = [numpy.array(lis) for lis in [images, lables]]

# OpenCV trains a model from the images
# model = cv2.face.FisherFaceRecognizer_create() ; good_value = 8000
# model = cv2.face.EigenFaceRecognizer_create() ; good_value = 3000
model = cv2.face.LBPHFaceRecognizer_create() ; good_value = 90
model.train(images, lables)

# Part 2: Use recognizer on camera stream
face_cascade = cv2.CascadeClassifier(haar_file)
webcam = cv2.VideoCapture(0)
while True:
    (_, im) = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        #width = int(height/h*w)
        face_resize = cv2.resize(face, (width, height))
        # Try to recognize the face
        prediction = model.predict(face_resize)

        if good_value/prediction[1]*100 > 100:
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(im, f'{int(good_value/prediction[1]*100)}% {names[prediction[0]]}',
                       (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
        else:
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(im, f'{int(good_value/prediction[1]*100)}% {names[prediction[0]]}',
                       (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))

    cv2.imshow('OpenCV', im)

    key = cv2.waitKey(10)
    if key == 27:
        break
