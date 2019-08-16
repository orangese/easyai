
"""

"easyai.advanced.py"

Custom keras layers. Does not use easyai API. Not recommended for use by easyai users.

"""

import importlib

import tensorflow as tf

from easyai.core import *

# CONV LAYERS
class Noisy_Normalize(K.layers.Layer):
  """Noisy image-normalizing input layer. Used for Fast NST."""

  def __init__(self, noise, **kwargs):
    self.noise = noise
    super(Noisy_Normalize, self).__init__(**kwargs)

  def build(self, input_shape):
    pass

  def compute_output_shape(self, input_shape):
    return input_shape

  def call(self, x, mask = None):
    noise_image = np.random.uniform(0, 1.0, size = K.backend.int_shape(x)[1:])
    return (noise_image * self.noise + x / 255. * (1.0 - self.noise)) / 255.

class Denormalize(K.layers.Layer):
  """Reverses effect of Normalize input layer."""

  def __init__(self, **kwargs):
    super(Denormalize, self).__init__(**kwargs)

  def build(self, input_shape):
    pass

  def compute_output_shape(self, input_shape):
    return input_shape

  def call(self, x, mask = None):
    return (x + 1) * 127.5

class VGG_Normalize(K.layers.Layer):
  """Custom normalization for VGG network."""

  def __init__(self, net, **kwargs):
    self.net = net
    super(VGG_Normalize, self).__init__(**kwargs)

  def build(self, input_shape):
    pass

  def compute_output_shape(self, input_shape):
    return input_shape

  def call(self, x, mask = None):
    self.k_net_module = importlib.import_module("keras.applications.{0}".format(self.net))
    return self.k_net_module.preprocess_input(x)

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
    return tf.pad(x, paddings = [[0, 0], [height_pad, height_pad], [width_pad, width_pad], [0, 0]], mode = "REFLECT")

# NST transform net
class NST_Transform_Net(Network_Interface):
  """Image transform NST layers. Implementation of network described in
  https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16.pdf. See 
  https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16Supplementary.pdf for exact network structure."""

  def __init__(self, num_rows = 512):
    self.num_rows = num_rows
    self.num_cols = None

  def train_init(self, img, noise = 0.6):
    width, height = img.size
    self.num_cols = self.num_cols = int(width * self.num_rows / height)
    self.model_init(noise = noise)

  def model_init(self, noise):
    self.k_model = NST_Transform_Net.img_transform_net(self.num_rows, self.num_cols, noise = noise)
    # assumes img is a PIL image
    print ("Loaded NST transform net")

  @staticmethod
  def conv_norm_block(filters, filter_size, strides = (1, 1), norm = "batch", include_relu = True, transpose = False):
    def _conv_norm_block(x):
      if transpose:
        a = K.layers.Conv2DTranspose(filters, filter_size, strides = strides)(x)
      else:
        a = K.layers.Conv2D(filters, filter_size, strides = strides)(x)
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
      a = NST_Transform_Net.conv_norm_block(filters, filter_size, norm = norm, include_relu = True)(x)
      a = NST_Transform_Net.conv_norm_block(filters, filter_size, norm = norm, include_relu = False)(a)
      return K.layers.merge.add([identity, a])
    return _conv_res_block

  @staticmethod
  def img_transform_net(img_width, img_height, noise = 0.6, norm = "batch"):
    x = K.layers.Input(shape = (img_width, img_height, 3))
    a = Noisy_Normalize(noise = noise)(x)

    a = Reflection_Padding2D((40, 40))(a)
    a = NST_Transform_Net.conv_norm_block(32, (9, 9), norm = norm)(a)
    a = NST_Transform_Net.conv_norm_block(64, (3, 3), strides = (2, 2), norm = norm)(a)
    a = NST_Transform_Net.conv_norm_block(128, (3, 3), strides = (2, 2), norm = norm)(a)

    for i in range(5):
      a = NST_Transform_Net.conv_res_block(128, (3, 3), norm = norm)(a)

    a = NST_Transform_Net.conv_norm_block(64, (3, 3), norm = norm, strides = (2, 2), transpose = True)(a)
    a = NST_Transform_Net.conv_norm_block(32, (3, 3), norm = norm, strides = (2, 2), transpose = True)(a)
    a = NST_Transform_Net.conv_norm_block(3, (9, 9), norm = norm, strides = (1, 1), transpose = True)(a)

    y = Denormalize()(a)

    return K.Model(inputs = x, outputs = y)