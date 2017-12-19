import numpy as np
import matplotlib as plt
import skimage
from skimage.io import imread
from skimage import io, color, img_as_ubyte
from skimage.feature import greycomatrix, greycoprops
from sklearn.metrics.cluster import entropy

im = imread('D2.png',as_grey=True)

im = skimage.img_as_ubyte(im)
im /= 32

g = skimage.feature.greycomatrix(im, [1], [0], levels=8, symmetric=False, normed=True)

contrast = skimage.feature.greycoprops(g, 'contrast')[0][0]

energy = skimage.feature.greycoprops(g, 'energy')[0][0]

homogeneity= skimage.feature.greycoprops(g, 'homogeneity')[0][0]

correlation=skimage.feature.greycoprops(g, 'correlation')[0][0]

dissimilarity=skimage.feature.greycoprops(g, 'dissimilarity')[0][0]

ASM = skimage.feature.greycoprops(g, 'ASM')[0][0]

print('contrast is: ', contrast)

print('energy is: ', energy)

print('homogeneity is: ', homogeneity)

print('correlation is: ', correlation)

print('dissimilarity is: ', dissimilarity)

print('ASM is: ', ASM)

distances = [1, 2, 3]
angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
properties = ['energy', 'homogeneity']


print 'Entropy is : ' + str(entropy(im))

print np.set_printoptions(precision=4)

'''
('contrast is: ', 0.23299548308912638)
  "%s to %s" % (dtypeobj_in, dtypeobj))
('energy is: ', 0.33722506961179582)
('homogeneity is: ', 0.90580400365500391)
('correlation is: ', 0.96672961961267267)
('dissimilarity is: ', 0.19580257794425471)
('ASM is: ', 0.11372074757468055)
Entropy is : 1.91075725468
None
'''