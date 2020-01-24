#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 23:25:30 2019

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

X_check = X_test
y_check = y_test

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import functools
import logging
import numpy as np
from six.moves import xrange
import tensorflow as tf

from cleverhans.attacks import FastGradientMethod
from cleverhans.utils_tf import jacobian_graph, jacobian_augmentation
from cleverhans.compat import flags
from cleverhans.dataset import MNIST
from cleverhans.initializers import HeReLuNormalInitializer
from cleverhans.loss import CrossEntropy
from cleverhans.model import Model
from cleverhans.train import train
from cleverhans.utils import set_log_level
from cleverhans.utils import TemporaryLogLevel
from cleverhans.utils import to_categorical
from cleverhans.utils_tf import model_eval, batch_eval

from cleverhans.model_zoo.basic_cnn import ModelBasicCNN

FLAGS = flags.FLAGS

NB_CLASSES = 3
BATCH_SIZE = 128
LEARNING_RATE = .001
NB_EPOCHS = 10
HOLDOUT = 150
DATA_AUG = 6
NB_EPOCHS_S = 10
LMBDA = .1
AUG_BATCH_SIZE = 512


def setup_tutorial():
  """
  Helper function to check correct configuration of tf for tutorial
  :return: True if setup checks completed
  """

  # Set TF random seed to improve reproducibility
  tf.set_random_seed(1234)

  return True


def prep_bbox(sess, x, y, X_train, y_train, X_test, y_test,
              nb_epochs, batch_size, learning_rate,
              rng, nb_classes=3, img_rows=32, img_cols=32, nchannels=1):
  """
  Define and train a model that simulates the "remote"
  black-box oracle described in the original paper.
  :param sess: the TF session
  :param x: the input placeholder for MNIST
  :param y: the ouput placeholder for MNIST
  :param x_train: the training data for the oracle
  :param y_train: the training labels for the oracle
  :param x_test: the testing data for the oracle
  :param y_test: the testing labels for the oracle
  :param nb_epochs: number of epochs to train model
  :param batch_size: size of training batches
  :param learning_rate: learning rate for training
  :param rng: numpy.random.RandomState
  :return:
  """

  # Define TF model graph (for the black-box model)
  nb_filters = 64
  model = ModelBasicCNN('model1', 3, nb_filters)
  loss = CrossEntropy(model, smoothing=0.1)
  predictions = model.get_logits(x)
  print("Defined TensorFlow model graph.")

  # Train an MNIST model
  train_params = {
      'nb_epochs': nb_epochs,
      'batch_size': batch_size,
      'learning_rate': learning_rate
  }
  train(sess, loss, X_train, y_train, args=train_params, rng=rng)

  # Print out the accuracy on legitimate data
  eval_params = {'batch_size': batch_size}
  accuracy = model_eval(sess, x, y, predictions, X_test, y_test,
                        args=eval_params)
  print('Test accuracy of black-box on legitimate test '
        'examples: ' + str(accuracy))

  return model, predictions, accuracy


class ModelSubstitute(Model):
  def __init__(self, scope, nb_classes, nb_filters=200, **kwargs):
    del kwargs
    Model.__init__(self, scope, nb_classes, locals())
    self.nb_filters = nb_filters

  def fprop(self, x, **kwargs):
    del kwargs
    my_dense = functools.partial(
        tf.layers.dense, kernel_initializer=HeReLuNormalInitializer)
    with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
      y = tf.layers.flatten(x)
      y = my_dense(y, self.nb_filters, activation=tf.nn.relu)
      y = my_dense(y, self.nb_filters, activation=tf.nn.relu)
      logits = my_dense(y, self.nb_classes)
      return {self.O_LOGITS: logits,
              self.O_PROBS: tf.nn.softmax(logits=logits)}


def train_sub(sess, x, y, bbox_preds, x_sub, y_sub, nb_classes,
              nb_epochs_s, batch_size, learning_rate, data_aug, lmbda,
              aug_batch_size, rng, img_rows=32, img_cols=32,
              nchannels=1):
  """
  This function creates the substitute by alternatively
  augmenting the training data and training the substitute.
  :param sess: TF session
  :param x: input TF placeholder
  :param y: output TF placeholder
  :param bbox_preds: output of black-box model predictions
  :param x_sub: initial substitute training data
  :param y_sub: initial substitute training labels
  :param nb_classes: number of output classes
  :param nb_epochs_s: number of epochs to train substitute model
  :param batch_size: size of training batches
  :param learning_rate: learning rate for training
  :param data_aug: number of times substitute training data is augmented
  :param lmbda: lambda from arxiv.org/abs/1602.02697
  :param rng: numpy.random.RandomState instance
  :return:
  """
  # Define TF model graph (for the black-box model)
  model_sub = ModelSubstitute('model_s', nb_classes)
  preds_sub = model_sub.get_logits(x)
  loss_sub = CrossEntropy(model_sub, smoothing=0)

  print("Defined TensorFlow model graph for the substitute.")

  # Define the Jacobian symbolically using TensorFlow
  grads = jacobian_graph(preds_sub, x, nb_classes)

  # Train the substitute and augment dataset alternatively
  for rho in xrange(data_aug):
    print("Substitute training epoch #" + str(rho))
    train_params = {
        'nb_epochs': nb_epochs_s,
        'batch_size': batch_size,
        'learning_rate': learning_rate
    }
    with TemporaryLogLevel(logging.WARNING, "cleverhans.utils.tf"):
      train(sess, loss_sub, x_sub, to_categorical(y_sub, nb_classes),
            init_all=False, args=train_params, rng=rng,
            var_list=model_sub.get_params())

    # If we are not at last substitute training iteration, augment dataset
    if rho < data_aug - 1:
      print("Augmenting substitute training data.")
      # Perform the Jacobian augmentation
      lmbda_coef = 2 * int(int(rho / 3) != 0) - 1
      x_sub = jacobian_augmentation(sess, x, x_sub, y_sub, grads,
                                    lmbda_coef * lmbda, aug_batch_size)

      print("Labeling substitute training data.")
      # Label the newly generated synthetic points using the black-box
      y_sub = np.hstack([y_sub, y_sub])
      x_sub_prev = x_sub[int(len(x_sub)/2):]
      eval_params = {'batch_size': batch_size}
      bbox_val = batch_eval(sess, [x], [bbox_preds], [x_sub_prev],
                            args=eval_params)[0]
      # Note here that we take the argmax because the adversary
      # only has access to the label (not the probabilities) output
      # by the black-box model
      y_sub[int(len(x_sub)/2):] = np.argmax(bbox_val, axis=1)

  return model_sub, preds_sub


def mnist_blackbox(train_start=0, train_end=60000, test_start=0,
                   test_end=10000, nb_classes=NB_CLASSES,
                   batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
                   nb_epochs=NB_EPOCHS, holdout=HOLDOUT, data_aug=DATA_AUG,
                   nb_epochs_s=NB_EPOCHS_S, lmbda=LMBDA,
                   aug_batch_size=AUG_BATCH_SIZE):
  """
  MNIST tutorial for the black-box attack from arxiv.org/abs/1602.02697
  :param train_start: index of first training set example
  :param train_end: index of last training set example
  :param test_start: index of first test set example
  :param test_end: index of last test set example
  :return: a dictionary with:
           * black-box model accuracy on test set
           * substitute model accuracy on test set
           * black-box model accuracy on adversarial examples transferred
             from the substitute model
  """

  # Set logging level to see debug information
  set_log_level(logging.DEBUG)

  # Dictionary used to keep track and return key accuracies
  accuracies = {}

  # Perform tutorial setup
  assert setup_tutorial()

  # Create TF session
  sess = tf.Session()
  
  X_test, y_test = X_check, y_check

  # Initialize substitute training set reserved for adversary
  x_sub = X_test[:holdout]
  y_sub = np.argmax(y_test[:holdout], axis=1)

  # Redefine test set as remaining samples unavailable to adversaries
  X_test = X_test[holdout:]
  y_test = y_test[holdout:]

  # Obtain Image parameters
  img_rows, img_cols, nchannels = 32, 32, 1
  nb_classes = 3

  # Define input TF placeholder
  x = tf.placeholder(tf.float32, shape=(None, 32, 32,
                                        1))
  y = tf.placeholder(tf.float32, shape=(None, 3))

  # Seed random number generator so tutorial is reproducible
  rng = np.random.RandomState([2019, 4, 30])

  # Simulate the black-box model locally
  # You could replace this by a remote labeling API for instance
  print("Preparing the black-box model.")
  prep_bbox_out = prep_bbox(sess, x, y, X_train, y_train, X_test, y_test,
                            nb_epochs, batch_size, learning_rate,
                            rng, nb_classes, img_rows, img_cols, nchannels)
  model, bbox_preds, accuracies['bbox'] = prep_bbox_out

  # Train substitute using method from https://arxiv.org/abs/1602.02697
  print("Training the substitute model.")
  train_sub_out = train_sub(sess, x, y, bbox_preds, x_sub, y_sub,
                            nb_classes, nb_epochs_s, batch_size,
                            learning_rate, data_aug, lmbda, aug_batch_size,
                            rng, img_rows, img_cols, nchannels)
  model_sub, preds_sub = train_sub_out

  # Evaluate the substitute model on clean test examples
  eval_params = {'batch_size': batch_size}
  acc = model_eval(sess, x, y, preds_sub, X_test, y_test, args=eval_params)
  accuracies['sub'] = acc

  # Initialize the Fast Gradient Sign Method (FGSM) attack object.
  fgsm_par = {'eps': 0.3, 'ord': np.inf, 'clip_min': 0., 'clip_max': 1.}
  fgsm = FastGradientMethod(model_sub, sess=sess)

  # Craft adversarial examples using the substitute
  eval_params = {'batch_size': batch_size}
  x_adv_sub = fgsm.generate(x, **fgsm_par)

  # Evaluate the accuracy of the "black-box" model on adversarial examples
  accuracy = model_eval(sess, x, y, model.get_logits(x_adv_sub),
                        X_test, y_test, args=eval_params)
  print('Test accuracy of oracle on adversarial examples generated '
        'using the substitute: ' + str(accuracy))
  accuracies['bbox_on_sub_adv_ex'] = accuracy

  return accuracies


def main(argv=None):

  mnist_blackbox()
  tf.app.run()
  

tf.reset_default_graph()  
main()