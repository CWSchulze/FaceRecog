"""
Generate new dataset from photos folder
"""

import cv2
import numpy as np
import os

photos_subfolder = r'C:\Users\Christian\Pictures\iCloud Photos\Downloads\2020'
photos_subfolder = r'C:\Users\Christian\Pictures\iphone\104APPLE'
# defining the size of images
(width, height) = (250,250)

found_faces_subfolder = 'output'

datasets_subfolder = 'datasets'
face_haar_file = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
eye_haar_file = os.path.join(cv2.data.haarcascades, 'haarcascade_eye.xml')

def rotate_image(image, image_center, angle):
  #image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def rotate_face(x,y,w,h, gray):
    face = gray[y:y + h, x:x + w]
    face_center = (x+w/2, y+h/2)
    
    # detect eyes to rotate the image
    detected_eyes = eyes_cascade.detectMultiScale(face, 1.25, 8)
    # width = int(height/h*w)
    if len(detected_eyes) == 2:
        detected_eyes = sorted(detected_eyes,key=lambda x:x[0])
        eye_placement = []
        for (eye_x, eye_y, eye_w, eye_h) in detected_eyes:
            cv2.rectangle(im, (eye_x + x, eye_y + y), (eye_x + eye_w + x, eye_y + eye_h + y), (255, 255, 0), 2)
            eye_center = ( x + eye_x + eye_w/2, y + eye_y + eye_h/2 )
            eye_radius = int( (eye_w + eye_h)*0.25 )
            eye_placement.append((eye_center, eye_radius))
        
        leftEyeCenter = eye_placement[0][0]
        rightEyeCenter = eye_placement[1][0]

        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX)) #- 180
        if angle > -45 and angle < 45:
            # only rotate in reasonable range
            rotated_gray = rotate_image(gray, face_center, angle)
            return face, rotated_gray[y:y + h, x:x + w]
        else:
            return None, None
    return None, None #face, gray



# Part 1: Create fisherRecognizer
face_cascade = cv2.CascadeClassifier(face_haar_file)
eyes_cascade = cv2.CascadeClassifier(eye_haar_file)
print('Recognizing Face Please Be in sufficient Lights...')

# Create a list of images and a list of corresponding names
(images, lables, names, id) = ([], [], {}, 0)
for (subdirs, dirs, files) in os.walk(datasets_subfolder):
    for subdir in dirs:
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
(images, lables) = [np.array(lis) for lis in [images, lables]]

# OpenCV trains a model from the images
# model = cv2.face.FisherFaceRecognizer_create() ; good_value = 8000
# model = cv2.face.EigenFaceRecognizer_create() ; good_value = 3000
model = cv2.face.LBPHFaceRecognizer_create() ; good_value = 400
model.train(images, lables)

# Part 2: Use recognizer on camera stream
for filename in os.listdir(photos_subfolder):
    if os.path.isfile(os.path.join(photos_subfolder, filename)):
        full_filename = os.path.join(photos_subfolder, filename)
        im = cv2.imread(full_filename)
        if im is None:
            continue
        #(_, im) = webcam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 8, minSize=(
            int(min(gray.shape[:2])*0.1), int(min(gray.shape[:2])*0.1)))
        for index, (x, y, w, h) in enumerate(faces):
            #face = gray[y:y + h, x:x + w]
            face, rotated_gray = rotate_face(x,y,w,h, gray)
            #width = int(height/h*w)
            if face is None:
                continue
            face_resize = cv2.resize(rotated_gray, (width, height))
            # Try to recognize the face
            prediction = model.predict(face_resize)

            if good_value/prediction[1]*100 > good_value:
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(im, f'{int(good_value/prediction[1]*100)}% {names[prediction[0]]}',
                        (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, w/100, (0, 255, 0), int(max(1,width/50)))
            else:
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(im, f'{int(good_value/prediction[1]*100)}% {names[prediction[0]]}',
                        (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, w/100, (0, 255, 0), int(max(1,width/50)))
            #cv2.imshow('OpenCV', im)
            #key = cv2.waitKey(10000)
            if not os.path.isdir(os.path.join(found_faces_subfolder)):
                os.makedirs(os.path.join(found_faces_subfolder))
            cv2.imwrite(os.path.join(found_faces_subfolder,
                                        f'{index}_'+filename), face_resize)








