
"""

"easyai.layers.py"

Part of core easyai functionality: provides simplified syntax for keras layers, which by nature requires a loss in
flexibility and diversity.

"""

from easyai.core import *

# CORE LAYERS
class Input(Abstract_Layer):
  """
  MLP input layer with no activation, bias, or weights of its own.  No keras mask.
  """

  def __init__(self, input_shape: Union[int, tuple]):
    """
    Initializes Input object. Please flatten data before using this object.

    :param input_shape: number of neurons in input layer. Should be of type int if NN is MLP;
                        if NN is ConvNN, it should be a tuple of ints in the format (num_cols, num_rows, num_channels)
                        or (num_cols, num_rows) if num_channels == 1.
    """
    self.output_shape = input_shape
    if isinstance(self.output_shape, tuple):
      if len(self.output_shape) == 2:
        self.is_3d = False
        self.output_shape = (*self.output_shape, 1) # if user only gives two dimensions, assume num_channels = 1
      else:
        self.is_3d = True
    else:
      self.is_3d = False
    self.k_model = None # just to be explicit

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
    self.k_model = None

    self.output_shape = (self.num_neurons, )

  def update_mask(self):
    assert self.prev is not None, "self.prev must be initialized"
    if isinstance(self.prev, Conv):
      self.k_model = [K.layers.Flatten(), K.layers.Dense(units = self.num_neurons, activation = self.actv)]
    else:
      self.k_model = K.layers.Dense(units = self.num_neurons, activation = self.actv)

# CONV NET LAYERS
class Conv(Abstract_Layer):
  """
  Convolutional layer.  Has a keras mask, padding = "valid".
  """

  def __init__(self, filter_size: tuple, num_filters: int, actv: str = "sigmoid", strides: int = 1):
    """
    Initializes Conv object.  No padding and a stride of 1 is assumed.

    :param filter_size: size of the local receptive field (filter).
    :param num_filters: number of filters used. Other sources call this parameter "number of feature maps".
    :param actv: activation function of this layer. Default is sigmoid.
    :param strides: stride of this layer. Default is 1.
    """
    self.filter_size = filter_size
    self.num_filters = num_filters
    self.actv = actv
    self.strides = strides # not going to be used by class, just for NST

    self.prev = None

  def __str__(self):
    return "Conv {0} layer: {1} filters, filter size of {2}".format(self.actv, self.num_filters, self.filter_size)

  def __repr__(self):
    return self.__str__()

  def update_mask(self):
    assert self.prev is not None, "self.prev must be initialized"
    self.k_model = K.layers.Conv2D(filters = self.num_filters, strides = (self.strides, self.strides),
                                   kernel_size = self.filter_size, activation = self.actv)

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
    self.k_model = None
    self.is_3d = False # default, will be changed in other functions

  def __str__(self):
    return "(Max) pooling layer: pool size of {0}".format(self.pool_size)

  def update_mask(self):
    assert self.prev is not None, "self.prev must be initialized"
    self.k_model = K.layers.MaxPool2D(pool_size = self.pool_size)
