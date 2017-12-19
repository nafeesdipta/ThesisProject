import numpy as np
import cv2
from matplotlib import pyplot as plt#cv2

#image 1
img1 = cv2.imread('face1.jpg',0)
img2 = cv2.imread("face2.png",0)
cv2.imshow('Image1',img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
print img1
#Classifiers
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
img = cv2.imread('face1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


#Laplitian Face 1
lap1 = cv2.Laplacian(img1,cv2.CV_64F)
lap1 = np.uint8(np.absolute(lap1))
cv2.imshow("Face 1 Lap",lap1)
cv2.waitKey(0)

#image 2
cv2.imshow('Image2',img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Laplitian Face 2
lap2 = cv2.Laplacian(img2,cv2.CV_64F)
lap2 = np.uint8(np.absolute(lap2))
cv2.imshow("Face 2 Lap",lap2)
cv2.waitKey(0)
print img2
#img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#hist = cv2.calcHist(img,[0],None,[256],[0,256])
'''
#matplotlib
plt.figure()
plt.title("Messi Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.plot(hist)
plt.xlim([0, 256])
plt.show()
cv2.waitKey(0)
'''

#i is for channel. 0 means gray scale. 0 1 2 means R G B
color = ('b','g','r')
plt.title("Face 1")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
for i,col in enumerate(color):
    his = cv2.calcHist(img1,[i],None,[256],[0,256])
    plt.plot(his,color = col)
    plt.xlim([0,256])
plt.show()

color = ('b','g','r')
plt.title("Face 2")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
for i,col in enumerate(color):
    his = cv2.calcHist(img2,[i],None,[256],[0,256])
    plt.plot(his,color = col)
    plt.xlim([0,256])
plt.show()