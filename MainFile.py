import cv2
import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt#cv2
import skimage
from skimage.io import imread
from skimage import io, color, img_as_ubyte
from skimage.feature import greycomatrix, greycoprops
from sklearn.metrics.cluster import entropy


def getper(a,b):
    maxn= max(a,b)
    per = abs(a-b)/float(maxn)
    per = 100-(per*100)
    return per

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier('Mouth.xml')

img = cv2.imread('twin1.jpg')
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

img22 = cv2.imread('twin2.jpg')
cv2.imshow('img',img22)
cv2.waitKey(0)
cv2.destroyAllWindows()


minisize = (img.shape[1],img.shape[0])
miniframe = cv2.resize(img, minisize)
faces = face_cascade.detectMultiScale(miniframe)


for f in faces:
        x, y, w, h = [ v for v in f ]
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,255))
        #Eye
        #roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_color)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        sub_face = img[y:y+h, x:x+w]
        face_file_name = "face_crop.jpg"
        cv2.imwrite(face_file_name, sub_face)
        #
        img2 = cv2.imread(face_file_name)
        eyes = eye_cascade.detectMultiScale(roi_color)
        i=0
        for (ex, ey, ew, eh) in eyes:
            i += 1

            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            sub_eye = img2[ey:ey + eh, ex:ex + ew]
            if(i==1):
               face_file_eye_left = "left_eye" + str(i) + ".jpg"
               cv2.imwrite(face_file_eye_left, sub_eye)
            else:
                face_file_eye_right = "right_eye" + str(i) + ".jpg"
                cv2.imwrite(face_file_eye_right, sub_eye)

        mouth = mouth_cascade.detectMultiScale(roi_color)
        for (ex, ey, ew, eh) in mouth:
            cv2.rectangle(roi_color, (ex, ey), (ex + eh, ey + ew), (0, 255, 0), 2)
            sub_mouth = img2[ey:ey + eh, ex:ex + ew]
            face_file_mouth= "face_mouth.jpg"
            cv2.imwrite(face_file_mouth, sub_mouth)


img1= cv2.imread(face_file_name)
cv2.imshow("face_crop.jpg",img1)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
#Histogram
color = ('b','g','r')
plt.title("Face 1")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
for i,col in enumerate(color):
    his = cv2.calcHist(img1, [i], None, [256], [0,256])
    plt.plot(his,color = col)
    plt.xlim([0,256])
plt.show()

'''
#print img1
#
'''
img3= cv2.imread(face_file_eye_left)
cv2.imshow("Left_Eye.jpg",img3)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
'''
img5= cv2.imread(face_file_mouth)
cv2.imshow("Mouth.jpg",img5)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
# 3126 10543 1979 |
# 1237 8476 965   |
# 693 7780 570    |

#Sir's Daughter
#8483 10416 5223
#8944 7009 5250

photo = Image.open(face_file_eye_left) #your image
photo = photo.convert('RGB')

width = photo.size[0] #define W and H
height = photo.size[1]

for y in range(0, height): #each pixel has coordinates
    row = ""
    R1 = 0
    G1 = 0
    B1 = 0
    for x in range(0, width):

        RGB = photo.getpixel((x,y))
        R1 = R1 + RGB[0]  #now you can use the RGB value
        G1 = G1 + RGB[1]
        B1 = B1 + RGB[2]
print 'Printing Left Eye RGB Value'
print str(R1)+" "+str(G1)+" "+str(B1)

#
'''
img4 = cv2.imread(face_file_eye_left)
cv2.imshow("Left_Eye.jpg",img4)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

photo = Image.open(face_file_mouth) #your image
photo = photo.convert('RGB')

width = photo.size[0] #define W and H
height = photo.size[1]

for y in range(0, height): #each pixel has coordinates
    row = ""
    RM1 = 0
    GM1 = 0
    BM1 = 0
    for x in range(0, width):

        RGB = photo.getpixel((x,y))
        RM1 = RM1 + RGB[0]  #now you can use the RGB value
        GM1 = GM1 + RGB[1]
        BM1 = BM1 + RGB[2]
print 'Printing Mouth RGB Value'
print str(RM1)+" "+str(GM1)+" "+str(BM1)

'''

#
color = ('b','g','r')
plt.title("Face 1 Left Eye")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
for i,col in enumerate(color):
    his = cv2.calcHist(img3, [i], None, [256], [0,256])
    plt.plot(his,color = col)
    plt.xlim([0,256])
plt.show()
#print img3
'''

'''
color = ('b','g','r')
plt.title("Face 1 Right Eye")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
for i,col in enumerate(color):
    his = cv2.calcHist(img4, [i], None, [256], [0,256])
    plt.plot(his,color = col)
    plt.xlim([0,256])
plt.show()
'''
#print img4
#

im = imread(face_file_name,as_grey=True)
im = skimage.img_as_ubyte(im)
im /= 32
g = skimage.feature.greycomatrix(im, [1], [0], levels=8, symmetric=False, normed=True)

contrast1 = skimage.feature.greycoprops(g, 'contrast')[0][0]

energy1 = skimage.feature.greycoprops(g, 'energy')[0][0]

homogeneity1 = skimage.feature.greycoprops(g, 'homogeneity')[0][0]

correlation1 = skimage.feature.greycoprops(g, 'correlation')[0][0]

dissimilarity1 = skimage.feature.greycoprops(g, 'dissimilarity')[0][0]

ASM1 = skimage.feature.greycoprops(g, 'ASM')[0][0]

print('contrast is: ', contrast1)
print('energy is: ', energy1)
print('homogeneity is: ', homogeneity1)
print('correlation is: ', correlation1)
print('dissimilarity is: ', dissimilarity1)
print('ASM is: ', ASM1)

distances = [1, 2, 3]
angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
properties = ['energy1', 'homogeneity1']

print 'Entropy1 is : ' + str(entropy(im))
print np.set_printoptions(precision=4)

GLCM1 = contrast1*1000+energy1*1000+homogeneity1*1000+dissimilarity1*1000+ASM1*1000+correlation1*1000
os.remove(face_file_name)
os.remove(face_file_eye_left)
os.remove(face_file_eye_right)
os.remove(face_file_mouth)

############

minisize = (img22.shape[1],img22.shape[0])
miniframe = cv2.resize(img22, minisize)
faces2 = face_cascade.detectMultiScale(miniframe)


for f in faces2:
        x, y, w, h = [ v for v in f ]
        cv2.rectangle(img22, (x,y), (x+w,y+h), (255,255,255))
        #Eye
        #roi_gray = gray[y:y + h, x:x + w]
        roi_color = img22[y:y + h, x:x + w]
        eyes2 = eye_cascade.detectMultiScale(roi_color)
        for (ex, ey, ew, eh) in eyes2:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        sub_face = img22[y:y+h, x:x+w]
        face_file_name2 = "face_crop2.jpg"
        cv2.imwrite(face_file_name2, sub_face)
        #
        img23 = cv2.imread(face_file_name2)
        #eyes = eye_cascade.detectMultiScale(roi_color)
        i=0
        for (ex, ey, ew, eh) in eyes2:
            i += 1

            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            sub_eye = img23[ey:ey + eh, ex:ex + ew]
            if(i==1):
               face_file_eye_left2 = "left_eye2" + str(i) + ".jpg"
               cv2.imwrite(face_file_eye_left2, sub_eye)
            else:
                face_file_eye_right2 = "right_eye2" + str(i) + ".jpg"
                cv2.imwrite(face_file_eye_right2, sub_eye)

        mouth2 = mouth_cascade.detectMultiScale(roi_color)
        for (ex, ey, ew, eh) in mouth2:
            cv2.rectangle(roi_color, (ex, ey), (ex + eh, ey + ew), (0, 255, 0), 2)
            sub_mouth = img23[ey:ey + eh, ex:ex + ew]
            face_file_mouth2 = "face_mouth2.jpg"
            cv2.imwrite(face_file_mouth2, sub_mouth)


img11 = cv2.imread(face_file_name2)
cv2.imshow("face_crop.jpg",img11)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
#Histogram
color = ('b','g','r')
plt.title("Face 1")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
for i,col in enumerate(color):
    his = cv2.calcHist(img1, [i], None, [256], [0,256])
    plt.plot(his,color = col)
    plt.xlim([0,256])
plt.show()

'''
#print img1
#
'''
img3= cv2.imread(face_file_eye_left)
cv2.imshow("Left_Eye.jpg",img3)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
'''
img5= cv2.imread(face_file_mouth)
cv2.imshow("Mouth.jpg",img5)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
# 3126 10543 1979 |
# 1237 8476 965   |
# 693 7780 570    |

#Sir's Daughter
#8483 10416 5223
#8944 7009 5250

photo = Image.open(face_file_eye_left2) #your image
photo = photo.convert('RGB')

width = photo.size[0] #define W and H
height = photo.size[1]

for y in range(0, height): #each pixel has coordinates
    row = ""
    R2 = 0
    G2 = 0
    B2 = 0
    for x in range(0, width):

        RGB = photo.getpixel((x,y))
        R2 = R2 + RGB[0]  #now you can use the RGB value
        G2 = G2 + RGB[1]
        B2 = B2 + RGB[2]
print 'Printing Left Eye 2 RGB Value'
print str(R2)+" "+str(G2)+" "+str(B2)

#
'''
img44 = cv2.imread(face_file_eye_left2)
cv2.imshow("Left_Eye.jpg",img44)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

photo = Image.open(face_file_mouth2) #your image
photo = photo.convert('RGB')

width = photo.size[0] #define W and H
height = photo.size[1]

for y in range(0, height): #each pixel has coordinates
    row = ""
    RM2 = 0
    GM2 = 0
    BM2 = 0
    for x in range(0, width):

        RGB = photo.getpixel((x,y))
        RM2 = RM2 + RGB[0]  #now you can use the RGB value
        GM2 = GM2 + RGB[1]
        BM2 = BM2 + RGB[2]
print 'Printing Mouth RGB Value'
print str(RM2)+" "+str(GM2)+" "+str(BM2)

'''

#
color = ('b','g','r')
plt.title("Face 1 Left Eye")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
for i,col in enumerate(color):
    his = cv2.calcHist(img3, [i], None, [256], [0,256])
    plt.plot(his,color = col)
    plt.xlim([0,256])
plt.show()
#print img3
'''

'''
color = ('b','g','r')
plt.title("Face 1 Right Eye")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
for i,col in enumerate(color):
    his = cv2.calcHist(img4, [i], None, [256], [0,256])
    plt.plot(his,color = col)
    plt.xlim([0,256])
plt.show()
'''
#print img4
#
im = imread(face_file_name2,as_grey=True)
im = skimage.img_as_ubyte(im)
im = im/32

g = skimage.feature.greycomatrix(im, [1], [0], levels=8, symmetric=False, normed=True)
contrast2 = skimage.feature.greycoprops(g, 'contrast')[0][0]
energy2 = skimage.feature.greycoprops(g, 'energy')[0][0]
homogeneity2 = skimage.feature.greycoprops(g, 'homogeneity')[0][0]
correlation2 = skimage.feature.greycoprops(g, 'correlation')[0][0]
dissimilarity2 = skimage.feature.greycoprops(g, 'dissimilarity')[0][0]

ASM2 = skimage.feature.greycoprops(g, 'ASM')[0][0]

print('contrast is: ', contrast2)
print('energy is: ', energy2)
print('homogeneity is: ', homogeneity2)
print('correlation is: ', correlation2)
print('dissimilarity is: ', dissimilarity2)
print('ASM is: ', ASM2)

distances = [1, 2, 3]
angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
properties = ['energy2', 'homogeneity2']

print 'Entropy2 is : ' + str(entropy(im))
print np.set_printoptions(precision=4)
GLCM2 = contrast2*1000+energy2*1000+homogeneity2*1000+dissimilarity2*1000+ASM2*1000+correlation2*1000

######
RT1 = R1+G1+B1
RT2 = R2+G2+B2

RMT1 = RM1+GM1+BM1
RMT2 = RM2+GM2+BM2

per_eyes = getper(RT1,RT2)
print "Percentage for Eyes :", getper(RT1,RT2)

per_lips = getper(RMT1,RMT2)
print "Percentage for Lips :", getper(RMT1,RMT2)

print "Percentage for faces contrast :", getper(contrast1,contrast2)

print "Percentage for faces energy :", getper(energy1,energy2)

print "Percentage for faces homogeneity :", getper(homogeneity1,homogeneity2)

print "Percentage for faces dissimilarity :", getper(dissimilarity1,dissimilarity2)

print "Percentage for faces ASM :", getper(ASM1,ASM2)

print "Percentage for faces correlation :", getper(correlation1,correlation2)


os.remove(face_file_name2)
os.remove(face_file_eye_left2)
os.remove(face_file_eye_right2)
os.remove(face_file_mouth2)