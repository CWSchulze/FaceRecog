"""
Create new dataset by using the webcam
"""

import cv2
import os
import numpy as np

new_dataset_name = 'MyUserName2'
picture_to_take = 300
# defining the size of images
(width, height) = (250,250)

datasets_subfolder = 'datasets'
face_haar_file = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
eye_haar_file = os.path.join(cv2.data.haarcascades, 'haarcascade_eye.xml')

path = os.path.join(datasets_subfolder, new_dataset_name)
if not os.path.isdir(path):
    os.makedirs(path)

#'0' is used for my webcam,
# if you've any other camera
#  attached use '1' like this
face_cascade = cv2.CascadeClassifier(face_haar_file)
eyes_cascade = cv2.CascadeClassifier(eye_haar_file)
webcam = cv2.VideoCapture(0)


def rotate_image(image, image_center, angle):
  #image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result
  
def rotateImage(image, angle):
    row,col = image.shape
    center=tuple(np.array([row,col])/2)
    rot_mat = cv2.getRotationMatrix2D(center,angle,1.0)
    new_image = cv2.warpAffine(image, rot_mat, (col,row))
    return new_image

    

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


count = 1
while count < picture_to_take:
    (_, im) = webcam.read()
    gray_ = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    normalised_image = np.zeros((width, height))
    gray = cv2.normalize(gray_, normalised_image, 0, 255, cv2.NORM_MINMAX)
    faces = face_cascade.detectMultiScale(gray, 1.3, 8)
    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face, gray = rotate_face(x,y,w,h, gray)
        if face is None:
            continue
        face_resize = cv2.resize(face, (width, height))
        cv2.imwrite('% s/% s.png' % (path, count), face_resize)
        count += 1

    cv2.imshow('OpenCV', im)
    key = cv2.waitKey(10)
    if key == 27:
        break
