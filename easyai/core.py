
"""

"easyai.core.py"

Wrapper program for easy use of keras. Targeted towards those with little to no experience with Python and
machine learning.

"""

from time import time
from typing import Union

import keras as K
import numpy as np

# SUPPORT
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

class Error_Handling(Static_Interface):
  """
  Static class for error handling.
  """

  @staticmethod
  def suppress_tf_warnings():
    """
    Suppresses tensorflow warnings. Does not work if tensorflow is outdated.
    """
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf
    try:
      tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    except AttributeError:
      tf.logging.set_verbosity(tf.logging.ERROR)
    # compatible with tensorflow == 1.14.0 and tensorflow-gpu == 1.8.0

# CORE LAYERS
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
    self.k_mask = None
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

class Input(Abstract_Layer):
  """
  MLP input layer with no activation, bias, or weights of its own.  No keras mask.
  """

  def __init__(self, neurons: Union[int, tuple]):
    """
    Initializes Input object. Please flatten data before using this object.

    :param neurons: number of neurons in input layer. Should be of type int if NN is MLP;
                    if NN is ConvNN, it should be a tuple of ints in the format (num_cols, num_rows, num_channels)
                    or (num_cols, num_rows) if num_channels == 1.
    """
    self.num_neurons = neurons
    if isinstance(self.num_neurons, tuple):
      if len(self.num_neurons) == 2:
        self.is_3d = False
        self.num_neurons = (*self.num_neurons, 1) # if user only gives two dimensions, assume num_channels = 1
      else:
        self.is_3d = True
    else:
      self.is_3d = False
    self.k_mask = None # just to be explicit

class Dense(Abstract_Layer):
  """
  MLP layer (aka a dense layer).  Has a keras mask.
  """

  def __init__(self, num_neurons: int , actv: str = "sigmoid"):
    """
    Initializes Dense object.

    :param num_neurons: number of neurons in this layer.
    :param actv: activation function of this layer. Default is sigmoid.
    """
    self.num_neurons = num_neurons
    self.actv = actv
    self.prev = None # previous layer-- used to link layers together to create keras sequential object in NN
    self.k_mask = None

  def update_mask(self):
    assert self.prev, "self.prev must be initialized"
    if hasattr(self.prev, "is_3d") and not isinstance(self.prev, Input):
      self.k_mask = [K.layers.Flatten(), K.layers.Dense(units = self.num_neurons, activation = self.actv)]
    else:
      self.k_mask = K.layers.Dense(units = self.num_neurons, activation = self.actv,
                                   input_shape = (self.prev.num_neurons, ),)

# CONV NET LAYERS
class Conv(Abstract_Layer):
  """
  Convolutional layer.  Has a keras mask, stride = 1, padding = "valid".
  """

  def __init__(self, filter_size: tuple, num_filters: int, actv: str = "sigmoid"):
    """
    Initializes Conv object.  No padding and a stride of 1 is assumed.

    :param filter_size: size of the local receptive field (filter).
    :param num_filters: number of filters used. Other sources call this parameter "number of feature maps".
    :param actv: activation function of this layer. Default is sigmoid.
    """
    self.filter_size = filter_size
    self.num_filters = num_filters
    self.actv = actv
    self.prev = None
    self.k_mask = None
    self.is_3d = False # default, will be changed in update_mask

  def __str__(self):
    return "Conv {0} layer: {1} filters, filter size of {2}".format(self.actv, self.num_filters, self.filter_size)

  def __repr__(self):
    return self.__str__()

  def update_mask(self):
    assert self.prev, "self.prev must be initialized"
    self.is_3d = self.prev.is_3d
    if self.is_3d:
      self.k_mask = K.layers.Conv3D(filters = self.num_filters, kernel_size = self.filter_size,
                                    activation = self.actv, input_shape = self.prev.num_neurons)

    else:
      self.k_mask = K.layers.Conv2D(filters = self.num_filters, kernel_size = self.filter_size,
                                    activation = self.actv, input_shape = self.prev.num_neurons)

class Pooling(Abstract_Layer):
  """
  Pooling layer.  Has a keras mask, stride = 1, pooling type = "max pooling".
  """

  def __init__(self, pool_size: tuple):
    """
    Initializes Pooling object.  Only max pooling is implemented, stride = None.

    :param pool_size: size of pool.
    """
    self.pool_size = pool_size
    self.prev = None
    self.k_mask = None
    self.is_3d = False # default, will be changed in other functions

  def __str__(self):
    return "(Max) pooling layer: pool size of {0}".format(self.pool_size)

  def update_mask(self):
    assert self.prev, "self.prev must be initialized"
    self.is_3d = self.prev.is_3d
    if self.is_3d:
      self.k_mask = K.layers.MaxPool3D(pool_size = self.pool_size)
    else:
      self.k_mask = K.layers.MaxPool2D(pool_size = self.pool_size)

# NEURAL NETWORK IMPLEMENTATION
class NN(object):
  """
  Uses easyai layer objects to create a functional keras model.
  """

  def __init__(self, layers: list = None, cost: str = "binary_crossentropy"):
    """
    Initializes NN (neural network) object.

    :param layers: layers of the network. Expects easyai core layer objects.
    :param cost: cost used by the neural network. Default is categorical_crossentropy.
    """
    if layers is None:
      layers = []
    self.layers = tuple(layers)
    self.link_layers()

    self.cost = cost
    self.is_trained = False

  def link_layers(self):
    """ Links wrapper objects and creates a wrapper keras Sequential object."""
    self.k_layers = []
    for prev, layer in zip(self.layers, self.layers[1:]):
      layer.prev = prev
      layer.update_mask()
      if isinstance(layer.k_mask, list):
        for mask in layer.k_mask:
          self.k_layers.append(mask)
      else:
        self.k_layers.append(layer.k_mask)
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
