import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'Imageattendence'
images = []
classNames = []
myList = os.listdir(path)
print(myList)

for cls in myList:
    curImg = cv2.imread(f'{path}/{cls}')
    images.append(curImg)
    classNames.append(os.path.splitext(cls)[0])
print(classNames)

def findEncodeings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendence(name):
    with open('Attendence.csv','r+') as f:
        mydataList = f.readlines()
        nameList = []
        for line in mydataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtstring = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtstring}')


encodeListKnown = findEncodeings(images)
print('Encodeing Complete')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgs = cv2.resize(img,(0,0),None,0.25,0.25)
    imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)

    faceCurframe  = face_recognition.face_locations(imgs)
    try:
        encodeCurrFrame = face_recognition.face_encodings(imgs,faceCurframe)

        for encodeFace, faceLoc in zip(encodeCurrFrame, faceCurframe):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            # print(faceDis)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                # print(name)
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                markAttendence(name)

        cv2.imshow('Webcam', img)
        cv2.waitKey(1)

    except KeyboardInterrupt:
        print("Program Terminated")
        # quit(-1)




