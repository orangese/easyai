
"""

"easy_ai.py"

Wrapper library for easy use of keras. Targeted towards those with little to no experience with Python and
machine learning.

"""

import tensorflow as tf
import keras as K

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class Input(object):
  """input layer, no keras mask"""

  def __init__(self, num_neurons):
    self.num_neurons = num_neurons

class Dense(object):
  """wrapper for keras' Dense class"""

  def __init__(self, num_neurons, actv = "sigmoid", prev = None):
    self.num_neurons = num_neurons
    self.actv = actv
    self.prev = prev
    self.k_mask = None

  def update_mask(self):
    self.k_mask = K.layers.Dense(units = self.num_neurons, input_shape = (self.prev.num_neurons, ),
                                 activation = self.actv)

class Network(object):
  """uses wrapper classes to create a functional network-- wrapper for Sequential"""

  def __init__(self, layers = [], cost = "crossentropy"):
    self.layers = tuple(layers)
    self.cost = cost
    self.link_layers()

  def __setattr__(self, name, value):
    """tweaked so that only the layer/model attributes can be changed"""
    if hasattr(self, name) and not (name in "k_model, k_layers, layers"):
      raise ValueError("cannot modify attribute {0}".format(name))
    super().__setattr__(name, value)
    if name == "layers": self.link_layers()

  def link_layers(self):
    """links wrapper objects and creates a wrapper keras Sequential object"""
    self.k_layers = []
    for prev, layer in zip(self.layers, self.layers[1:]):
      layer.prev = prev
      layer.update_mask()
      self.k_layers.append(layer.k_mask)
    self.k_model = K.Sequential(self.k_layers)

  def add_layer(self, layer, position = None):
    """adds a layer"""
    new_layers = list(self.layers)
    if not position: position = len(self.layers)
    new_layers.insert(position, layer)
    self.layers = new_layers

  def rm_layer(self, position = None, layer = None):
    """removes a layer"""
    assert position or layer, "position or layer arguments must be provided"
    new_layers = list(self.layers)
    if layer: new_layers.remove(layer)
    elif position: del new_layers[position]
    self.layers = new_layers

  def train(self, x, y, lr = 3.0, epochs = 60, batch_size = 10):
    """trains and compiles k_model-- simplified version of keras' compile and fit"""
    optimizer = K.optimizers.SGD(lr = lr)
    self.k_model.compile(optimizer = optimizer, loss = self.cost, metrics = ["accuracy"])
    self.k_model.fit(x, y, epochs = epochs, batch_size = batch_size, validation_split = 0.2, verbose = 2)

  def evaluate(self, x, y):
    """wrapper for Sequential evaluate"""
    return self.k_model.evaluate(x, y)

  def predict(self, x):
    """wrapper for Sequential predict"""
    return self.k_model.predict(x)

def load_mnist():
  """loads MNIST data for use by a keras Sequential object"""
  (x_train, y_train), (x_test, y_test) = K.datasets.mnist.load_data()

  x_train = (x_train.reshape(x_train.shape[0], -1) / 255).astype("float32")
  y_train = K.utils.to_categorical(y_train, 10).astype("float32")

  x_test = (x_test.reshape(x_test.shape[0], -1) / 255).astype("float32")
  y_test = K.utils.to_categorical(y_test, 10).astype("float32")

  return (x_train, y_train), (x_test, y_test)

if __name__ == "__main__":
  test = Network([Input(784), Dense(30), Dense(10)])
  (x_train, y_train), (x_test, y_test) = load_mnist()
  test.train(x_train, y_train)
