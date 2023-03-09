import cv2
import numpy as np
import face_recognition

imgkholi = face_recognition.load_image_file('image/kholi.jpg')
imgkholi = cv2.cvtColor(imgkholi, cv2.COLOR_BGR2RGB)
imgvirat = face_recognition.load_image_file('image/dhoni.jpg')
imgvirat = cv2.cvtColor(imgvirat, cv2.COLOR_BGR2RGB)

facekholi = face_recognition.face_locations(imgkholi)[0]
encodekholi = face_recognition.face_encodings(imgkholi)[0]
cv2.rectangle(imgkholi,(facekholi[3],facekholi[0]),(facekholi[1],facekholi[2]),(255,0,255),2)

facevirat = face_recognition.face_locations(imgvirat)[0]
encodevirat= face_recognition.face_encodings(imgvirat)[0]
cv2.rectangle(imgvirat,(facevirat[3],facevirat[0]),(facevirat[1],facevirat[2]),(255,0,255),2)


result = face_recognition.compare_faces([encodekholi],encodevirat)
facedis = face_recognition.face_distance([encodekholi],encodevirat)
print(result,facedis)
cv2.putText(imgvirat,f'{result} {round(facedis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)


cv2.imshow('kholi', imgkholi)
cv2.imshow('virat', imgvirat)
cv2.waitKey(0)