
"""

"easyai.advanced_layers.py"

Custom keras layers. Does not use easyai API. Not recommended for use by easyai users.

"""

import tensorflow as tf

from easyai.core import *

# CONV LAYERS
class Normalize(K.layers.Layer):
  """Image-normalizing input layer."""

  def __init__(self, **kwargs):
    super(Normalize, self).__init__(**kwargs)

  def build(self, input_shape):
    pass

  def compute_output_shape(self, input_shape):
    return input_shape

  def call(self, x, mask = None):
    return x / 255.0

class Denormalize(K.layers.Layer):
  """Reverses effect of Normalize input layer."""

  def __init__(self, **kwargs):
    super(Denormalize, self).__init__(**kwargs)

  def build(self, input_shape):
    pass

  def compute_output_shape(self, input_shape):
    return input_shape

  def call(self, x, mask=None):
    return (x + 1) * 127.5

class Instance_Norm(K.layers.Layer):
  """Instance normalization."""

  def __init__(self, **kwargs):
    super(Instance_Norm, self).__init__(**kwargs)
    self.epsilon = 1e-4

  def compute_output_shape(self, input_shape):
    return input_shape

  def call(self, x, mask = None):
    mean, variance = tf.nn.moments(x, axis = [1, 2], keep_dims = True)
    return tf.div(tf.subtract(x, mean), tf.sqrt(tf.add(variance, self.epsilon)))

class Reflection_Padding2D(K.layers.Layer):
  """Reflection padding."""

  def __init__(self, padding = (1, 1), **kwargs):
    self.padding = tuple(padding)
    self.input_spec = [K.layers.InputSpec(ndim = 4)]
    super(Reflection_Padding2D, self).__init__(**kwargs)

  def compute_output_shape(self, input_shape):
    return input_shape[0], input_shape[1] + 2 * self.padding[0], input_shape[2] + 2 * self.padding[1], input_shape[3]

  def call(self, x, mask = None):
    width_pad, height_pad = self.padding
    return tf.pad(x, padding = [[0, 0], [height_pad, height_pad], [width_pad, width_pad], [0, 0]], mode = "REFLECT")

class NST(Static_Interface):
  """Image transform NST layers."""

  @staticmethod
  def conv_norm_block(filters, filter_size, strides = (1, 1), norm = "batch", include_relu = True):
    def _conv_norm_block(x):
      a = K.layers.Conv2D(filters, filter_size, strides)(x)
      if norm == "batch":
        a = K.layers.BatchNormalization()(a)
      elif norm == "instance":
        a = Instance_Norm()(a)
      if include_relu:
        a = K.layers.Activation("relu")(a)
      return a
    return _conv_norm_block

  @staticmethod
  def conv_res_block(filters, filter_size, norm = "batch"):
    def _conv_res_block(x):
      identity = K.layers.convolutional.Cropping2D(cropping = ((2, 2), (2, 2)))(x) # 2 is the number of conv layers
      a = NST.conv_norm_block(filters, filter_size, norm = norm, include_relu = True)(x)
      a = NST.conv_norm_block(filters, filter_size, norm = norm, include_relu = False)(a)
      return K.layers.merge.add([identity, a])
    return _conv_res_block

  @staticmethod
  def img_transform_net(img_width, img_height, norm = "batch"):
    x = K.layers.Input(shape = (img_width, img_height, 3))
    x = Normalize()(x)

    a = Reflection_Padding2D((40, 40))(x)
    a = NST.conv_norm_block(32, (9, 9), norm = norm)(a)
    a = NST.conv_norm_block(64, (3, 3), strides = (2, 2), norm = norm)(a)
    a = NST.conv_norm_block(128, (3, 3), strides = (2, 2), norm = norm)(a)

    for i in range(5):
      a = NST.conv_res_block(128, (3, 3), norm = norm)(x)

    a = NST.conv_norm_block(64, (3, 3), strides = (0.5, 0.5), norm = norm)(a)
    a = NST.conv_norm_block(32, (3, 3), strides=(0.5, 0.5), norm = norm)(a)
    a = NST.conv_norm_block(3, (9, 9), norm = norm)(a)

    y = Denormalize()(a)

    return K.Model(inputs = x, outputs = y)