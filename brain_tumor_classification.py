#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri Jan 25 12:14:08 2019

@author: jai
"""

import cv2
import math
import h5py
import numpy as np
import pandas as pd
from PIL import Image
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

#Extracting data
f = h5py.File('1.mat')

f['cjdata']['PID'].value
f['cjdata']['label'].value

tumorImage = f['cjdata']['image'].value
tumorBorder = f['cjdata']['tumorBorder']
tumorMask = f['cjdata']['tumorMask']

plt.imshow(tumorMask)
plt.imshow(tumorImage)

#Tumor Highlight

f = h5py.File('600.mat')
tumorImage = f['cjdata']['image'].value
tumorBorder = f['cjdata']['tumorBorder']
tumorMask = f['cjdata']['tumorMask']


plt.imshow(tumorImage, cmap='gray')
plt.imshow(tumorMask, cmap='jet', alpha=0.1)    


#Cropping Tumor

array = f['cjdata']['tumorBorder'].value

x = array[0][::2]
y = array[0][1::2]

plt.plot(y, x)

x_max = math.ceil(max(x))
x_min = math.floor(min(x))
y_max = math.ceil(max(y))
y_min = math.floor(min(y))

crop_rect_x = [x_min, x_min, x_max, x_max, x_min]
crop_rect_y = [y_min, y_max, y_max, y_min, y_min]
plt.plot(crop_rect_y, crop_rect_x)

width = x_max - x_min
height = y_max - y_min


# Testing Data
if width > 300:
    excess_width = width - 300
    if excess_width % 2 == 0:
        x_max = x_max - excess_width / 2
        x_min = x_min - excess_width / 2
            



crop = tumorImage[x_min: x_max, y_min: y_max]   
plt.imshow(crop, cmap='gray')
plt.imshow(tumorImage)
#crop_scaled = tumorImage[(x_min-10): (x_max+10), (y_min-10): (y_max+10)]
#plt.imshow(crop_scaled, cmap='gray')

array = (np.random.rand(100, 200)*256).astype(np.uint8)

img = Image.fromarray(data).convert('L')
plt.imshow(img)

plt.imshow(tumorImage)

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    
gray = rgb2gray(array)   

cv2.imshow('hey', img)

        
data = tumorImage
data = data.astype(np.uint8)
data = data / 255
plt.imshow(data)

# Finding range and mean of cropped tumor shape

sum_0 = 0
sum_1 = 0
max_0 = 0
max_1 = 0
min_0 = 1000
min_1 = 1000
range_min = 0
range_max = 0
for image in croppedTumorImages:
    sum_0 += image.shape[0]
    sum_1 += image.shape[1]
    if image.shape[0] > max_0:
        max_0 = image.shape[0]
    if image.shape[1] > max_1:
        max_1 = image.shape[1]    
    if image.shape[0] < min_0:
        min_0 = image.shape[0]
    if image.shape[1] < min_1:
        min_1 = image.shape[1]    
    if image.max() > range_max:
        range_max = image.max()
    if image.min() < range_min:
        range_min = image.min()

print(sum_0/3064, sum_1/3064)

plt.imshow(croppedTumorImages[20])


tumorImage = tumorImage.astype('uint8')
pic = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)


info = np.iinfo(croppedTumorImages[20].dtype) # Get the information of the incoming image type
data = croppedTumorImages[20].astype(np.float64) / info.max # normalize the data to 0 - 1
data = 255 * data # Now scale by 255
img = data.astype(np.uint8)
plt.imshow(img)


# Scaling Images
res = cv2.resize(img, dsize=(75, 75), interpolation=cv2.INTER_CUBIC)
plt.imshow(res)


# Creating dataset

PIDs = [0] * 3064
labels = [0] * 3064
tumorImages = [0] * 3064
tumorBorders = [0] * 3064
tumorMasks = [0] * 3064
croppedTumorImages = [0] * 3064
scaledTumorImages = [0] * 3064


for i in range(1, 3065):
    f = h5py.File(str(i) + '.mat')
    labels[i-1] = math.floor(f['cjdata']['label'].value[0][0])
    PIDs[i-1] = f['cjdata']['PID'].value    
    tumorImages[i-1] = f['cjdata']['image'].value    
    tumorBorders[i-1] = f['cjdata']['tumorBorder']    
    tumorMasks[i-1] = f['cjdata']['tumorMask']
    
    array = f['cjdata']['tumorBorder'].value
    x = array[0][::2]
    y = array[0][1::2]
    
    x_max = math.ceil(max(x))
    x_min = math.floor(min(x))
    y_max = math.ceil(max(y))
    y_min = math.floor(min(y))
    
    tumorImage = f['cjdata']['image'].value
    croppedTumorImages[i-1] = tumorImage[x_min: x_max, y_min: y_max]

    scaledTumorImages[i-1] = cv2.resize(croppedTumorImages[i-1], 
                     dsize=(75, 75), interpolation=cv2.INTER_CUBIC)
    
    f.close()
    

columns = ['PID', 'label', 'tumorImage', 'tumorBorder', 'tummorMask', 
           'croppedTumorImage', 'scaledTumorImage']

data = pd.DataFrame({'PID':PIDs, 'label': labels, 'tumorImage': tumorImages, 
                     'tumorBorder': tumorBorders, 'tummorMask': tumorMasks, 
                     'croppedTumorImage': croppedTumorImages, 'scaledTumorImage': scaledTumorImages})    

# Export as CSV
        
data.to_csv('dataframe.csv', index=False)

# Import CSV

data = pd.read_csv('dataframe.csv')

# Feature Extraction

# HOG Descriptor
    
winSize = (75,75)
blockSize = (10,10)
blockStride = (5,5)
cellSize = (10,10)
nbins = 9
derivAperture = 1
winSigma = -1.
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection = 1
nlevels = 64
signedGradients = True
 
hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,
                        winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels)

check = scaledTumorImages[1].astype('uint8')
descriptor = hog.compute(check)

scaledTumorImagesUINT8 = [0] * 3064

for i in range(1, 3065):
    scaledTumorImagesUINT8[i-1] = scaledTumorImages[i-1].astype('uint8')           


HOGDescriptor = [0] * 3064

for i in range(1, 3065):
    #xarr=np.squeeze(np.array(scaledTumorImagesUINT8[i-1]).astype(np.float32))
    v = hog.compute(scaledTumorImagesUINT8[i-1])
    arr = np.array(v)
    flat_arr = arr.ravel()
    HOGDescriptor[i-1] = flat_arr


# PCA Values

PCAvalues = [0] * 3064

for i in range(1, 3065):
     xarr=np.squeeze(np.array(scaledTumorImagesUINT8[i-1]).astype(np.float32))
     m,v=cv2.PCACompute(xarr, mean=None)
     arr= np.array(v)
     flat_arr= arr.ravel()
     PCAvalues[i-1] = flat_arr
     

# Prepare training and testing data

X_train, X_test, y_train, y_test = train_test_split(HOGDescriptor, labels, 
                                                    test_size=0.10, random_state=10)

X_train = np.asarray(X_train).flatten()
y_train = y_train.astype(int)
y_train = np.asarray(y_train) 
y_train = np.float32(y_train)

# SVM Classifier
from sklearn import svm

clf = svm.SVC()
clf.fit(X_train, y_train)  

y_pred = clf.predict(X_test)



# Random Forest Classifier

classifier = RandomForestClassifier(n_estimators=100, random_state=0)  
classifier.fit(X_train, y_train)  
y_pred = classifier.predict(X_test)


# Test accuracy

y_train.count(1)
y_train.count(2)
y_train.count(3)

acc = 0
for i in range(1, 308):
    if(y_pred[i-1]==y_test[i-1]):
        acc += 1

print((acc/307)*100)


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
