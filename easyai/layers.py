
"""

"easyai.layers.py"

Part of core easyai functionality: provides simplified syntax for keras layers, which by nature requires a loss in
flexibility and diversity.

"""

from easyai.core import *

# CORE LAYERS
class Input(AbstractLayer):
  """
  MLP input layer with no activation, bias, or weights of its own.  No keras mask.
  """

  def __init__(self, *input_shape):
    """
    Initializes Input object. Please flatten data before using this object.

    :param input_shape: shape of the input. Either an int (for MLP networks) or a tuple (for conv networks).
    """
    if len(input_shape) > 2:
      self.is_3d = True
      self.output_shape = input_shape
    else:
      self.is_3d = False
      self.output_shape = (input_shape, 1)
    self.num_neurons = self.output_shape
    self.k_model = None # just to be explicit

class FC(AbstractLayer):
  """
  MLP layer (aka a FC or dense layer).  Has a keras mask.
  """

  def __init__(self, num_neurons: int , actv: str = "sigmoid"):
    """
    Initializes FC object.

    :param num_neurons: number of neurons in this layer.
    :param actv: activation function of this layer. Default is sigmoid.
    """
    self.num_neurons = num_neurons
    self.actv = actv
    self.prev = None # previous layer-- used to link layers together to create keras sequential object in NN
    self.k_model = None

    self.output_shape = (self.num_neurons, )

  def train_init(self):
    if isinstance(self.prev, Conv):
      self.k_model = [keras.layers.Flatten(), keras.layers.Dense(units = self.num_neurons, activation = self.actv)]
    else:
      self.k_model = keras.layers.Dense(units = self.num_neurons, activation = self.actv)

# CONV NET LAYERS
class Conv(AbstractLayer):
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

  def train_init(self):
    self.k_model = keras.layers.Conv2D(filters = self.num_filters, strides = (self.strides, self.strides),
                                   kernel_size = self.filter_size, activation = self.actv)

class Pooling(AbstractLayer):
  """
  Pooling layer.  Has a keras mask, stride = 1, pooling type = "max pooling".
  """

  def __init__(self, pool_width, pool_height):
    """
    Initializes Pooling object.  Only max pooling is implemented, stride = None.

    :param pool_width: width of pool.
    :param pool_height: height of pool.
    """
    self.pool_size = (pool_width, pool_height)
    self.prev = None
    self.k_model = None
    self.is_3d = False # default, will be changed in other functions

  def __str__(self):
    return "(Max) pooling layer: pool size of {0}".format(self.pool_size)

  def train_init(self):
    self.k_model = keras.layers.MaxPool2D(pool_size = self.pool_size)
