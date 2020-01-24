#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 00:39:14 2019

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



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
from tensorflow import keras

from cleverhans.attacks import FastGradientMethod
from cleverhans.compat import flags
from cleverhans.dataset import MNIST
from cleverhans.utils import AccuracyReport
from cleverhans.utils_keras import cnn_model
from cleverhans.utils_keras import KerasModelWrapper

FLAGS = flags.FLAGS

NB_EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = .001


def run_attack(train_start=0, train_end=60000, test_start=0,
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

  # Set TF random seed to improve reproducibility
  tf.set_random_seed(1234)
  # Force TensorFlow to use single thread to improve reproducibility
  config = tf.ConfigProto(intra_op_parallelism_threads=1,
                          inter_op_parallelism_threads=1)

  if keras.backend.image_data_format() != 'channels_last':
    raise NotImplementedError("this tutorial requires keras to be configured to channels_last format")

  # Create TF session and set as Keras backend session
  sess = tf.Session(config=config)
  keras.backend.set_session(sess)


  # Define Keras model
  model = cnn_model(img_rows=32, img_cols=32,
                    channels=1, nb_filters=64,
                    nb_classes=3)
  print("Defined Keras model.")

  # To be able to call the model in the custom loss, we need to call it once
  # before, see https://github.com/tensorflow/tensorflow/issues/23769
  model(model.input)

  # Initialize the Fast Gradient Sign Method (FGSM) attack object
  wrap = KerasModelWrapper(model)
  fgsm = FastGradientMethod(wrap, sess=sess)
  fgsm_params = {'eps': 0.3,
                 'clip_min': 0.,
                 'clip_max': 1.}

  adv_acc_metric = get_adversarial_acc_metric(model, fgsm, fgsm_params)
  model.compile(
      optimizer=keras.optimizers.Adam(learning_rate),
      loss='categorical_crossentropy',
      metrics=['accuracy', keras.losses.mean_squared_error, adv_acc_metric]
  )

  # Train the model
  model.fit(X_train, y_train,
            batch_size=batch_size,
            epochs=nb_epochs,
            validation_data=(X_test, y_test),
            verbose=1)

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
  model_2 = cnn_model(img_rows=32, img_cols=32,
                      channels=1, nb_filters=64,
                      nb_classes=3)
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
nb_classes=FLAGS.nb_classes, batch_size=FLAGS.batch_size,
                 learning_rate=FLAGS.learning_rate,
                 nb_epochs=FLAGS.nb_epochs, holdout=FLAGS.holdout,
                 data_aug=FLAGS.data_aug, nb_epochs_s=FLAGS.nb_epochs_s,
                 lmbda=FLAGS.lmbda, aug_batch_size=FLAGS.data_aug_batch_size
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
  run_attack()

from cleverhans.utils_keras import cnn_model

tf.reset_default_graph()

main()
