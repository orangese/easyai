
"""

"easyai.core.py"

Contains core functionality for easyai: the NN class.

"""

# ERROR HANDLING
def suppress_tf_warnings():
  """
  Suppresses tensorflow warnings. Does not work if tensorflow is outdated.
  """
  import os
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

  import warnings
  warnings.simplefilter(action = "ignore", category = FutureWarning)

  import tensorflow as tf
  try:
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
  except AttributeError:
    tf.logging.set_verbosity(tf.logging.ERROR)
  # compatible with tensorflow == 1.14.0 and tensorflow-gpu == 1.8.0

suppress_tf_warnings()

# IMPORTS
from typing import Union, List
from time import time

import keras as K
import numpy as np

# FRAMEWORK
class Static_Interface(object):
  """
  Static interface for other programs. An object of this class cannot be created.
  """

  def __init__(self):
    """
    As Static_Interface objects should not be created, __init__ throws a NotImplementedError.

    :raises NotImplementedError
    """
    raise NotImplementedError("class is static")

class Abstract_Layer(Static_Interface):
  """
  Abstract class that acts as the base for all layer classes. Should not be implemented.
  """

  def __init__(self, num_neurons: int , actv: str):
    """
    As Abstract_Layer objects should not be created, __init__ throws a NotImplementedError.

    :raises NotImplementedError
    """
    self.num_neurons = num_neurons
    self.actv = actv
    self.prev = None
    self.k_model = None
    raise NotImplementedError("abstract class should not be implemented")

  def __str__(self):
    try:
      return "{0} {1} layer: {2} neurons".format(self.__class__.__name__, self.actv, self.num_neurons)
    except AttributeError:
      return "{0} layer: {1} neurons".format(self.__class__.__name__, self.num_neurons)

  def __repr__(self):
    return self.__str__()

  def update_mask(self):
    """
     Creates keras mask. This mask will be used for training and all computations.

    :raises AssertionError if self.prev is not initialized.
    """
    assert self.prev, "self.prev must be initialized"
    raise NotImplementedError("cannot be implemented. How did you even call this function?")

class Network_Interface(object):
  """Interface for all network-like classes."""

  def __init__(self):
    """
    Initialization for interfaces should not be used.

    :raises NotImplementedError: class is an interface
    """
    raise NotImplementedError("class is an interface")

# NEURAL NETWORK IMPLEMENTATION
class NN(Network_Interface):
  """
  Uses easyai layer objects to create a functional keras model.
  """

  def __init__(self, *layers, cost: str = "binary_crossentropy"):
    """
    Initializes NN (neural network) object.

    :param layers: layers of the network. Expects easyai core layer objects.
    :param cost: cost used by the neural network. Default is categorical_crossentropy.
    """
    if layers is None:
      layers = []
    self.layers = layers
    self.link_layers()

    self.cost = cost
    self.is_trained = False

  def link_layers(self):
    """ Links wrapper objects and creates a wrapper keras Sequential object."""
    self.k_layers = []
    for prev, layer in zip(self.layers, self.layers[1:]):
      layer.prev = prev
      layer.update_mask()
      if isinstance(layer.k_model, list):
        for mask in layer.k_model:
          self.k_layers.append(mask)
      else:
        self.k_layers.append(layer.k_model)
    self.k_model = K.Sequential(self.k_layers)

  def add_layer(self, layer: Abstract_Layer, position: int = None):
    """Adds a layer and creates a new keras object.

    :param layer: layer to be added. Should be instance of easyai core layer classes.
    :param position: position at which to insert new layer. Uses Python's list "insert".
    """
    new_layers = list(self.layers)
    if not position:
      position = len(self.layers)
    new_layers.insert(position, layer)
    self.__init__(new_layers)

  def rm_layer(self, position: int = None, layer: Abstract_Layer = None):
    """Removes a layer and creates a new keras object.

    :param position: position at which layer should be removed. Recommended instead of `layer`.
    :param layer: layer that should be removed. Not recommended due to possible duplicate layers.
    """
    assert position or layer, "position or layer arguments must be provided"
    new_layers = list(self.layers)
    if layer:
      new_layers.remove(layer)
    elif position:
      del new_layers[position]
    self.__init__(new_layers)

  def train(self, x: np.ndarray, y: np.ndarray, lr: float = 0.1, epochs: int = 1, batch_size: int = 10):
    """Trains and compiles this NN object.  Only SGD is used.

    :param x: input data. For example, if classifying an image, `x` would the pixel vectors.
    :param y: labels. For example, if classifying an image, `y` would be the image labels.
     The `y` data should be comprised of one-hot encodings.
    :param lr:  learning rate used in SGD. Default is 3.0.
    :param epochs: number of epochs. Default is 1.
    :param batch_size:  minibatch size. Default is 10.
    """
    optimizer = K.optimizers.SGD(lr = lr)
    metrics = ["categorical_accuracy"] if self.layers[-1].num_neurons > 2 else ["binary_accuracy"]
    # fixes weird keras feature in which "accuracy" metric causes unexpected results if using "binary_crossentropy"
    self.k_model.compile(optimizer = optimizer, loss = self.cost, metrics = metrics)
    self.k_model.fit(x, y, epochs = epochs, batch_size = batch_size, validation_split = 0.2, verbose = 2)
    self.is_trained = True

  def evaluate(self, x: np.ndarray, y: np.ndarray, verbose: bool = True) -> list:
    """Evaluates this NN object using test data.

    :param x: inputs. See train documentation for more information.
    :param y: labels. See train documentation for more information.
    :param verbose: if true, this function gives more information about the evaluation process.
    :return: evaluation.
    """
    start = time()
    evaluation = self.k_model.evaluate(x, y, verbose = 2)
    result = "Test evaluation\n - {0}s".format(round(time() - start))
    for name, evaluation in zip(self.k_model.metrics_names, evaluation):
      result += (" - test_{0}: {1}".format(name, round(evaluation, 4)))
    if verbose:
      print (result)
    return evaluation

  def predict(self, x: np.ndarray) -> np.ndarray:
    """Predicts the labels given input data.

    :param x: input data.
    :return: prediction.
    """
    return self.k_model.predict(x)

  def summary(self, advanced: bool = False) -> str:
    """
    Summary of model.

    :param advanced: if true, print advanced information.
    :return: summary of this NN object.
    """
    alphabet = list(map(chr, range(97, 123)))
    result = "Network summary: \n"
    result += "  1. Layers: \n"
    for layer in self.layers:
      result += "    {0}. {1}\n".format(alphabet[self.layers.index(layer)], layer)
    result += "  2. Trained: {0}\n".format(self.is_trained)
    if advanced and self.is_trained:
      result += "Advanced: \n"
      result += "    1. Cost function: {0}\n".format(self.cost)
    return result

  def save(self, filename: str):
    """
    Saves self as hdf5 file.

    :param filename: hdf5 file to save to.
    """
    self.k_model.save(filename)
