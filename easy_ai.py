
"""

"easy_ai.py"

Wrapper library for easy use of keras. Targeted towards those with little to no experience with Python and
machine learning.

"""

import tensorflow as tf
import keras as K
import numpy as np

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class Input(object):

  def __init__(self, num_neurons):
    self.num_neurons = num_neurons

class Dense(object):

  def __init__(self, num_neurons, actv = "sigmoid", prev = None):
    self.num_neurons = num_neurons
    self.actv = actv
    self.prev = prev

  def k_mask(self):
    return K.layers.Dense(units = self.num_neurons, input_shape = (self.prev.num_neurons, ), activation = self.actv)

class Network(object):
  """uses Layer classes to create a functional network"""

  def __init__(self, layers = [], cost = "mse"):
    #layers = [Input(784), Dense(100), Dense(10)]
    self.layers = tuple(layers)
    self.k_layers = []
    for prev, layer in zip(self.layers, self.layers[1:]):
      layer.prev = prev
      self.k_layers.append(layer.k_mask())
    self.k_model = K.Sequential(self.k_layers)
    self.cost = cost

  def train(self, x, y, lr = 3.0, epochs = 60, batch_size = 10, momentum = 0.0, nesterov = False):
    optimizer = K.optimizers.SGD(lr = lr, momentum = momentum, nesterov = nesterov)
    self.k_model.compile(optimizer = optimizer, loss = self.cost, metrics = ["accuracy"])
    self.k_model.fit(x, y, epochs = epochs, batch_size = batch_size, validation_split = 0.2, verbose = 2)

def load_mnist():
  (x_train, y_train), (x_test, y_test) = K.datasets.mnist.load_data()

  x_train = (x_train.reshape(x_train.shape[0], -1) / 255).astype("float32")
  y_train = K.utils.to_categorical(y_train, 10).astype("float32")

  x_test = (x_test.reshape(x_test.shape[0], -1) / 255).astype("float32")
  y_test = K.utils.to_categorical(y_test, 10).astype("float32")

  return (x_train, y_train), (x_test, y_test)

test = Network([Input(784), Dense(100), Dense(10)])
(x_train, y_train), (x_test, y_test) = load_mnist()
test.train(x_train, y_train)