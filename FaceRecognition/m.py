from unittest import result
import cv2 as cv
import face_recognition

img = cv.imread('G:/MyImages/ElonMusk.webp')
rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
cv.imshow('Elon', img)
img_encoding = face_recognition.face_encodings(rgb_img)[0]

# img2 = cv.imread('G:/MyImages/ElonMusk2.webp')
# rgb_img2 = cv.cvtColor(img2, cv.COLOR_BGR2RGB)
# cv.imshow('Elon2', img2)
# img_encoding2 = face_recognition.face_encodings(rgb_img2)[0]


img2 = cv.imread('G:/MyImages/murat2.jpg')
rgb_img2 = cv.cvtColor(img2, cv.COLOR_BGR2RGB)
cv.imshow('murat2', img2)
img_encoding2 = face_recognition.face_encodings(rgb_img2)[0]

result = face_recognition.compare_faces([img_encoding], img_encoding2)
print('Result: ', result)


cv.waitKey(0)
cv.destroyAllWindows()