import cv2
import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt#cv2
import pca
from pylab import *
import pickle, pprint


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

img = cv2.imread('D2.png')
#cv2.imshow('img',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
minisize = (img.shape[1],img.shape[0])
miniframe = cv2.resize(img, minisize)
faces = face_cascade.detectMultiScale(miniframe)


for f in faces:
        x, y, w, h = [ v for v in f ]
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,255))
        #Eye
        #roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        sub_face = img[y:y+h, x:x+w]
        face_file_name = "face_crop.jpg"
        cv2.imwrite(face_file_name, sub_face)

img1 = cv2.imread(face_file_name)
#cv2.imshow('Cropped',img1)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


X = imread(face_file_name);
cv2.imwrite('test.pgm',X)
allfiles = os.listdir(os.getcwd())
imlist=[filename for filename in allfiles if filename[-4:] in [".pgm"]]
im = array(Image.open(imlist[0])) # open one image to get size
m,n = im.shape[0:2] # get the size of the images
imnbr = list(imlist) # get the number of images

# create matrix to store all flattened images
immatrix = asarray([array(Image.open(im)).flatten() for im in imlist],'f')

# perform PCA
V,S,immean = pca.pca(immatrix)

# show some images (mean and 7 first modes)
figure()
gray()
subplot(2,4,1)
#imshow(immean.reshape(m,n))
'''
for i in range(7):
  subplot(2,4,i+2)
  imshow(V[i].reshape(m,n))

show()
'''


f = open('pca_face.pkl','wb')
pickle.dump(immean,f)
pickle.dump(V,f)
f.close()

# load mean and principal components
with open('pca_face.pkl', 'rb') as f:
  immean = pickle.load(f)
  V = pickle.load(f)

pkl_file = open('pca_face.pkl', 'rb')
data1= pickle.load(pkl_file)
pprint.pprint(data1)