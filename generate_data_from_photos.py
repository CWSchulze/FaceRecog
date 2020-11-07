"""
Generate new dataset from photos folder
"""

import cv2
import os

photos_subfolder = 'photos'
# defining the size of images
(width, height) = (250,250)

datasets_subfolder = 'datasets'
haar_file = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')

face_cascade = cv2.CascadeClassifier(haar_file)

def handle_face(im, face_index, face_resize, name):
    if name is None:
        cv2.imshow('OpenCV', face_resize)
        key = cv2.waitKey(100000)
        if key == 27:
            return
        name = chr(key)
    if not os.path.isdir(os.path.join(datasets_subfolder, name)):
        os.makedirs(os.path.join(datasets_subfolder, name))
    foto_index_string = ""
    foto_index = 0
    while os.path.isfile(os.path.join(datasets_subfolder, name, f'{face_index}_' + foto_index_string + '.jpg')):
        foto_index += 1
        foto_index_string = str(foto_index)
        
    cv2.imwrite(os.path.join(datasets_subfolder, name,
                                f'{face_index}_'+filename), face_resize)

def handle_photo(full_filename, name, handle_face=handle_face):
    im = cv2.imread(full_filename)
    #(_, im) = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.25, 10, minSize=(
        int(min(gray.shape[:2])*0.05), int(min(gray.shape[:2])*0.05)))
    # portrait:
    # faces = face_cascade.detectMultiScale(gray, 1.05, 4, minSize=(
    #     int(min(gray.shape[:2])*0.6), int(min(gray.shape[:2])*0.6)))
    for face_index, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = gray[y:y + h, x:x + w]
        #width = int(height/h*w)
        face_resize = cv2.resize(face, (width, height))
        handle_face(im, face_index, face_resize, name)


# for dirname in os.listdir(photos_subfolder):
#     for filename in os.listdir(os.path.join(photos_subfolder, dirname)):
#         if os.path.isfile(filename) and os.path.splitext(filename).upper()=='.JPG':
#             full_filename = os.path.join(photos_subfolder, dirname, filename)
#             handle_photo(full_filename, dirname)

dirname = r"C:\Users\Christian\Pictures\iCloud Photos\Downloads\2020"
#dirname = r"C:\Users\Christian\Pictures\iphone\103APPLE"
for filename in os.listdir(dirname):
    full_filename = os.path.join(dirname, filename)
    if os.path.isfile(full_filename) and os.path.splitext(full_filename)[1].upper()=='.JPG':
        handle_photo(full_filename, None)