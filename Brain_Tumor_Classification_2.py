#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 18:21:19 2019

@author: adit
"""

import matplotlib.pyplot as plt
import h5py
import math
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from keras.utils import to_categorical
from PIL import Image

PIDs = [0] * 3064
labels = [0] * 3064
tumorImages = [0] * 3064
tumorBorders = [0] * 3064
tumorMasks = [0] * 3064
croppedTumorImages = [0] * 3064
scaledTumorImages = [0] * 3064


"""
f = h5py.File('200.mat')
tumorImages[200] = f['cjdata']['image'].value
tumorBorders[200] = f['cjdata']['tumorBorder']
tumorMasks[200] = f['cjdata']['tumorMask']


plt.imshow(tumorImages[100], cmap='gray')
plt.imshow(tumorMasks[100], cmap='gray', alpha=0.1)    

plt.imsave('test2.png', tumorImages[200])



img = Image.open('test2.png').convert('L')  # convert image to 8-bit grayscale
WIDTH, HEIGHT = img.size

data = list(img.getdata()) # convert image data to a list of integers
# convert that to 2D list (list of lists of integers)
data = [data[offset:offset+WIDTH] for offset in range(0, WIDTH*HEIGHT, WIDTH)]

array_2 = np.array(data)
"""



output_array = np.zeros((3064,28,28), dtype=np.int64)


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
                     dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
    
    
    something = 'scaledTumorImageSmall'+str(i)
    # Comment out after first iteration
    plt.imsave(something+'.png', scaledTumorImages[i-1])
    
    img = Image.open(something+'.png').convert('L')  # convert image to 8-bit grayscale
    WIDTH, HEIGHT = img.size
    
    
    data = list(img.getdata()) # convert image data to a list of integers
    # convert that to 2D list (list of lists of integers)
    data = [data[offset:offset+WIDTH] for offset in range(0, WIDTH*HEIGHT, WIDTH)]
    output_array[i-1] = np.array(data)
    
    f.close()
    


# Comment out after first iteration
"""    
columns = ['PID', 'label', 'tumorImage', 'tumorBorder', 'tummorMask', 
           'croppedTumorImage', 'scaledTumorImage']

data = pd.DataFrame({'PID':PIDs, 'label': labels, 'tumorImage': tumorImages, 
                     'tumorBorder': tumorBorders, 'tummorMask': tumorMasks, 
                     'croppedTumorImage': croppedTumorImages, 'scaledTumorImage': scaledTumorImages})    
    
data.to_csv('dataframe_new.csv', index=False)
"""

# Import CSV

data = pd.read_csv('dataframe_new.csv')


X_train, X_test, y_train, y_test = train_test_split(output_array, labels, test_size=0.2)
    

# Find the unique numbers from the train labels
classes = np.unique(y_train)
nClasses = len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)

# Data Preprocessing
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.
X_test = X_test / 255.


train_Y_one_hot = to_categorical(y_train)
test_Y_one_hot = to_categorical(y_test)


print('Original label:', y_train[2])
print('After conversion to one-hot:', train_Y_one_hot[2])

from tensorflow.keras import layers, models

import keras
from keras import Sequential,Input,Model
from layers import Dense, Dropout, Flatten
import Conv2D, MaxPooling2D
from layers.normalization import BatchNormalization
from layers.advanced_activations import LeakyReLU

"""
batch_size = 64
epochs = 20
num_classes = nClasses


tumor_model = Sequential()
tumor_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(75,75,1),padding='same'))
tumor_model.add(LeakyReLU(alpha=0.1))
tumor_model.add(MaxPooling2D((2, 2),padding='same'))
tumor_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
tumor_model.add(LeakyReLU(alpha=0.1))
tumor_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
tumor_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
tumor_model.add(LeakyReLU(alpha=0.1))                  
tumor_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
tumor_model.add(Flatten())
tumor_model.add(Dense(128, activation='linear'))
tumor_model.add(LeakyReLU(alpha=0.1))                  
tumor_model.add(Dense(4, activation='softmax'))

tumor_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

tumor_model.summary()

tumor_model_train = tumor_model.fit(X_train, train_Y_one_hot, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(X_test, test_Y_one_hot))

test_eval = tumor_model.evaluate(X_test, test_Y_one_hot, verbose=0)

print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])

accuracy = tumor_model_train.history['acc']
val_accuracy = tumor_model_train.history['val_acc']
loss = tumor_model_train.history['loss']
val_loss = tumor_model_train.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
"""

# Adding dropout layer to prevent overfitting

batch_size = 64
epochs = 20
num_classes = nClasses

tumor_model = Sequential()
tumor_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',padding='same',input_shape=(28, 28,1)))
tumor_model.add(LeakyReLU(alpha=0.1))
tumor_model.add(MaxPooling2D((2, 2),padding='same'))
tumor_model.add(Dropout(0.25))
tumor_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
tumor_model.add(LeakyReLU(alpha=0.1))
tumor_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
tumor_model.add(Dropout(0.25))
tumor_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
tumor_model.add(LeakyReLU(alpha=0.1))                  
tumor_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
tumor_model.add(Dropout(0.4))
tumor_model.add(Flatten())
tumor_model.add(Dense(128, activation='linear'))
tumor_model.add(LeakyReLU(alpha=0.1))           
tumor_model.add(Dropout(0.3))
tumor_model.add(Dense(4, activation='softmax'))

tumor_model.summary()

tumor_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

tumor_model_train_dropout = tumor_model.fit(X_train, train_Y_one_hot, batch_size=batch_size,epochs=epochs,
                                            verbose=1,validation_data=(X_test, test_Y_one_hot))

test_eval = tumor_model.evaluate(X_test, test_Y_one_hot, verbose=1)

print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])


accuracy = tumor_model_train_dropout.history['acc']
val_accuracy = tumor_model_train_dropout.history['val_acc']
loss = tumor_model_train_dropout.history['loss']
val_loss = tumor_model_train_dropout.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


predicted_classes = tumor_model.predict(X_test)

predicted_classes = np.argmax(np.round(predicted_classes),axis=1)

correct = np.where(predicted_classes==y_test)[0]
print("Found %d correct labels" % len(correct))
for i, correct in enumerate(correct[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(X_test[correct].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], y_test[correct]))
    plt.tight_layout()
    
    
incorrect = np.where(predicted_classes!=y_test)[0]
print("Found %d incorrect labels" % len(incorrect))

from sklearn.metrics import classification_report
target_names = ["Class {}".format(i) for i in range(4)]
print(classification_report(y_test, predicted_classes, target_names=target_names))


### Adversarial Attack
def do_eval(preds, x_set, y_set, report_key, is_adv=None):
    acc = model_eval(sess, x, y, preds, x_set, y_set, args=eval_params)
    setattr(report, report_key, acc)
    if is_adv is None:
      report_text = None
    elif is_adv:
      report_text = 'adversarial'
    else:
      report_text = 'legitimate'
    if report_text:
      print('Test accuracy on %s examples: %0.4f' % (report_text, acc))

import tensorflow as tf

from cleverhans.loss import CrossEntropy
#from cleverhans.dataset import MNIST
from cleverhans.utils_tf import model_eval
from cleverhans.train import train
from cleverhans.attacks import FastGradientMethod
from cleverhans.utils import AccuracyReport, set_log_level
#from cleverhans.model_zoo.basic_cnn import ModelBasicCNN
from cleverhans import initializers
from cleverhans.model import Model
from cleverhans.utils_keras import KerasModelWrapper
from tensorflow.keras import layers,models

fgsm_params = {
      'eps': 0.3,
      'clip_min': 0.,
      'clip_max': 1.
  }

with tf.Session() as sess:
    # Initialize the Fast Gradient Sign Method (FGSM) attack object and
    # graph
    x = tf.placeholder(tf.float32, shape=(None, 75, 75, 1))
    y = tf.placeholder(tf.float32, shape=(None, 3))

    wrap = KerasModelWrapper(tumor_model)
    fgsm = FastGradientMethod(wrap, sess=sess)
    adv_x = fgsm.generate(x, **fgsm_params)
    preds_adv = model.get_logits(adv_x)

    # Evaluate the accuracy of the model on adversarial examples
    do_eval(preds_adv, X_test, y_test, 'clean_train_adv_eval', True)

    # Calculate training error
    if testing:
      do_eval(preds_adv, X_train, y_train, 'train_clean_train_adv_eval')

    print('Repeating the process, using adversarial training')

    # Create a new model and train it to be robust to FastGradientMethod
    model2 = ModelBasicCNN('model2', nb_classes, nb_filters)
    fgsm2 = FastGradientMethod(model2, sess=sess)

    def attack(x):
        return fgsm2.generate(x, **fgsm_params)

    loss2 = CrossEntropy(model2, smoothing=label_smoothing, attack=attack)
    preds2 = model2.get_logits(x)
    adv_x2 = attack(x)

    if not backprop_through_attack:
    # For the fgsm attack used in this tutorial, the attack has zero
    # gradient so enabling this flag does not change the gradient.
    # For some other attacks, enabling this flag increases the cost of
    # training, but gives the defender the ability to anticipate how
    # the atacker will change their strategy in response to updates to
    # the defender's parameters.
        adv_x2 = tf.stop_gradient(adv_x2)
    preds2_adv = model2.get_logits(adv_x2)

    def evaluate2():
        # Accuracy of adversarially trained model on legitimate test inputs
        do_eval(preds2, x_test, y_test, 'adv_train_clean_eval', False)
        # Accuracy of the adversarially trained model on adversarial examples
        do_eval(preds2_adv, x_test, y_test, 'adv_train_adv_eval', True)

  # Perform and evaluate adversarial training
    train(sess, loss2, x_train, y_train, evaluate=evaluate2,
          args=train_params, rng=rng, var_list=model2.get_params())

  # Calculate training errors
    if testing:
        do_eval(preds2, x_train, y_train, 'train_adv_train_clean_eval')
        do_eval(preds2_adv, x_train, y_train, 'train_adv_train_adv_eval')


