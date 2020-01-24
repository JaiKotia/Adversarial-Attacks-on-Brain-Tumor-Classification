#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 22:23:03 2019

@author: jai
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
from keras.utils import to_categorical
from PIL import Image


PIDs = [0] * 3064
labels = [0] * 3064
output_array = np.zeros((3064,32,32), dtype=np.int64)

for i in range(1, 3065):
    f = h5py.File(str(i) + '.mat')
    labels[i-1] = math.floor(f['cjdata']['label'].value[0][0])
    PIDs[i-1] = f['cjdata']['PID'].value
    something = 'scaledTumorImageSmall'+str(i)
    img = Image.open(something+'.png').convert('L')
    WIDTH, HEIGHT = img.size    
    data = list(img.getdata())
    data = [data[offset:offset+WIDTH] for offset in range(0, WIDTH*HEIGHT, WIDTH)]
    output_array[i-1] = np.array(data)
    f.close()
    
X_train, X_test, y_train, y_test = train_test_split(output_array, labels, test_size=0.2)   

# Data Preprocessing
classes = np.unique(y_train)
nClasses = len(classes)

X_train = X_train.reshape(-1, 32, 32, 1)
X_test = X_test.reshape(-1, 32, 32, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.
X_test = X_test / 255.

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_train = y_train[:, 1:]
y_test = y_test[:, 1:]

from keras import layers, models
from cleverhans.utils_keras import residual_network
from keras import optimizers
from cleverhans.attacks import FastGradientMethod




def load_dataset():
    train_dataset = h5py.File('datasets/train_signs.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_signs.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples) (m, n_y)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[0]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y


def forward_propagation_for_predict(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters
    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3'] 
                                                           # Numpy Equivalents:
    Z1 = tf.add(tf.matmul(W1, X), b1)                      # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)                                    # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)                     # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)                                    # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)                     # Z3 = np.dot(W3,Z2) + b3
    
    return Z3

def predict(X, parameters):
    
    W1 = tf.convert_to_tensor(parameters["W1"])
    b1 = tf.convert_to_tensor(parameters["b1"])
    W2 = tf.convert_to_tensor(parameters["W2"])
    b2 = tf.convert_to_tensor(parameters["b2"])
    W3 = tf.convert_to_tensor(parameters["W3"])
    b3 = tf.convert_to_tensor(parameters["b3"])
    
    params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2,
              "W3": W3,
              "b3": b3}
    
    x = tf.placeholder("float", [12288, 1])
    
    z3 = forward_propagation_for_predict(x, params)
    p = tf.argmax(z3)
    
    sess = tf.Session()
    prediction = sess.run(p, feed_dict = {x: X})
        
    return prediction

from urllib.request import urlopen,urlretrieve
from PIL import Image
from tqdm import tqdm_notebook
from sklearn.utils import shuffle
from keras.models import load_model
from sklearn.datasets import load_files   
from keras.utils import np_utils
from glob import glob
from keras import applications
from keras.preprocessing.image import ImageDataGenerator 
from keras import optimizers
from keras.models import Sequential,Model,load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,GlobalAveragePooling2D
from keras.callbacks import TensorBoard,ReduceLROnPlateau,ModelCheckpoint



base_model = applications.resnet50.ResNet50(weights= None, 
                                            input_shape= (32, 32, 1))

print(base_model.input)


x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.7)(x)
predictions = Dense(3, activation= 'softmax')(x)
model = Model(inputs = base_model.input, outputs = predictions)

from keras.optimizers import SGD, Adam
# sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
adam = Adam(lr=0.0001)
base_model.compile(optimizer= adam, loss='categorical_crossentropy', metrics=['accuracy'])

base_model.fit(X_train, y_train, epochs = 10, batch_size = 64, verbose=1)


from cleverhans.attacks import FastGradientMethod
from cleverhans.compat import flags
from cleverhans.dataset import MNIST
from cleverhans.utils import AccuracyReport
from cleverhans.utils_keras import cnn_model
from cleverhans.utils_keras import KerasModelWrapper


from tensorflow import flags
FLAGS = flags.FLAGS

NB_EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = .001

from tensorflow.keras import layers, Model
from keras.layers import Input, Flatten, Dense


def mnist_tutorial(train_start=0, train_end=60000, test_start=0,
                   test_end=10000, nb_epochs=NB_EPOCHS, batch_size=BATCH_SIZE,
                   learning_rate=LEARNING_RATE, testing=False,
                   label_smoothing=0.1):
  """
  MNIST CleverHans tutorial
  :param train_start: index of first training set example
  :param train_end: index of last training set example
  :param test_start: index of first test set example
  :param test_end: index of last test set example
  :param nb_epochs: number of epochs to train model
  :param batch_size: size of training batches
  :param learning_rate: learning rate for training
  :param testing: if true, training error is calculated
  :param label_smoothing: float, amount of label smoothing for cross entropy
  :return: an AccuracyReport object
  """
  
  # Object used to keep track of (and return) key accuracies
  report = AccuracyReport()

  '''
  # Set TF random seed to improve reproducibility
  tf.set_random_seed(1234)
  
  # Force TensorFlow to use single thread to improve reproducibility
  config = tf.ConfigProto(intra_op_parallelism_threads=1,
                          inter_op_parallelism_threads=1)
  '''
  
  if keras.backend.image_data_format() != 'channels_last':
    raise NotImplementedError("this tutorial requires keras to be configured to channels_last format")

  # Create TF session and set as Keras backend session
  #sess = tf.Session(config=config)
  sess = tf.Session()
  keras.backend.set_session(sess)

  # Obtain Image Parameters
  img_rows, img_cols, nchannels = 32, 32, 1
  nb_classes = y_train.shape[1]

  # Label smoothing
  #y_train -= label_smoothing * (y_train - 1. / nb_classes)

  # Define Keras model
  #model = ResNet50(weights= None, input_shape=(32, 32, 1))
  
 
  
  print("Defined Keras model.")

  # To be able to call the model in the custom loss, we need to call it once
  # before, see https://github.com/tensorflow/tensorflow/issues/23769
  #model(model.input)

  # Initialize the Fast Gradient Sign Method (FGSM) attack object
  wrap = KerasModelWrapper(base_model)
  fgsm = FastGradientMethod(wrap, sess=sess)
  fgsm_params = {'eps': 0.3,
                 'clip_min': 0.,
                 'clip_max': 1.}

  adv_acc_metric = get_adversarial_acc_metric(base_model, fgsm, fgsm_params)
  
  base_model.compile(optimizer = opt, loss='categorical_crossentropy', 
                metrics=['accuracy', adv_acc_metric])

  base_model.fit(X_train, y_train, epochs = 10,
            validation_data=(X_test, y_test),
            batch_size = 64, verbose=1)

  # Evaluate the accuracy on legitimate and adversarial test examples
  _, acc, adv_acc = model.evaluate(X_test, y_test,
                                   batch_size=batch_size,
                                   verbose=1)
  report.clean_train_clean_eval = acc
  report.clean_train_adv_eval = adv_acc
  print('Test accuracy on legitimate examples: %0.4f' % acc)
  print('Test accuracy on adversarial examples: %0.4f\n' % adv_acc)

  # Calculate training error
  if testing:
    _, train_acc, train_adv_acc = model.evaluate(X_train, y_train,
                                                 batch_size=batch_size,
                                                 verbose=1)
    report.train_clean_train_clean_eval = train_acc
    report.train_clean_train_adv_eval = train_adv_acc

  print("Repeating the process, using adversarial training")
  # Redefine Keras model
  model_2 = cnn_model(img_rows=img_rows, img_cols=img_cols,
                      channels=nchannels, nb_filters=64,
                      nb_classes=nb_classes)
  model_2(model_2.input)
  wrap_2 = KerasModelWrapper(model_2)
  fgsm_2 = FastGradientMethod(wrap_2, sess=sess)

  # Use a loss function based on legitimate and adversarial examples
  adv_loss_2 = get_adversarial_loss(model_2, fgsm_2, fgsm_params)
  adv_acc_metric_2 = get_adversarial_acc_metric(model_2, fgsm_2, fgsm_params)
  model_2.compile(
      optimizer=keras.optimizers.Adam(learning_rate),
      loss=adv_loss_2,
      metrics=['accuracy', adv_acc_metric_2]
  )

  # Train an MNIST model
  model_2.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=nb_epochs,
              validation_data=(X_test, y_test),
              verbose=1)

  # Evaluate the accuracy on legitimate and adversarial test examples
  _, acc, adv_acc = model_2.evaluate(X_test, y_test,
                                     batch_size=batch_size,
                                     verbose=1)
  report.adv_train_clean_eval = acc
  report.adv_train_adv_eval = adv_acc
  print('Test accuracy on legitimate examples: %0.4f' % acc)
  print('Test accuracy on adversarial examples: %0.4f\n' % adv_acc)

  # Calculate training error
  if testing:
    _, train_acc, train_adv_acc = model_2.evaluate(X_train, y_train,
                                                   batch_size=batch_size,
                                                   verbose=1)
    report.train_adv_train_clean_eval = train_acc
    report.train_adv_train_adv_eval = train_adv_acc

  return report


def get_adversarial_acc_metric(model, fgsm, fgsm_params):
  def adv_acc(y, _):
    # Generate adversarial examples
    x_adv = fgsm.generate(model.input, **fgsm_params)
    # Consider the attack to be constant
    x_adv = tf.stop_gradient(x_adv)

    # Accuracy on the adversarial examples
    preds_adv = model(x_adv)
    return keras.metrics.categorical_accuracy(y, preds_adv)

  return adv_acc


def get_adversarial_loss(model, fgsm, fgsm_params):
  def adv_loss(y, preds):
    # Cross-entropy on the legitimate examples
    cross_ent = keras.losses.categorical_crossentropy(y, preds)

    # Generate adversarial examples
    x_adv = fgsm.generate(model.input, **fgsm_params)
    # Consider the attack to be constant
    x_adv = tf.stop_gradient(x_adv)

    # Cross-entropy on the adversarial examples
    preds_adv = model(x_adv)
    cross_ent_adv = keras.losses.categorical_crossentropy(y, preds_adv)

    return 0.5 * cross_ent + 0.5 * cross_ent_adv

  return adv_loss


def main(argv=None):
  mnist_tutorial(nb_epochs=20,
                 batch_size=64,
                 learning_rate=0.01)
  tf.app.run()

import keras

main()

