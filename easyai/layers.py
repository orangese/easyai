
"""

"easyai.layers.py"

Part of core easyai functionality: provides simplified syntax for keras layers, which by nature requires a loss in
flexibility and diversity.

"""

from easyai.core import *

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