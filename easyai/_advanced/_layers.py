
"""

"easyai._advanced._layers"

Advanced keras layers. Not for use by easyai users-- does not use easyai API.

"""

import keras
import tensorflow as tf

# CONV LAYERS
class Normalize(keras.layers.Layer):
  """Image-normalizing input layer. Used for Fast NST."""

  def __init__(self, noise, **kwargs):
    self.noise = noise
    super(Normalize, self).__init__(**kwargs)

  def build(self, input_shape):
    pass

  def call(self, x, mask = None):
    return x / 255.

  def compute_output_shape(self, input_shape):
    return input_shape

class VGGNormalize(keras.layers.Layer):
  """VGG normalization layer."""

  def __init__(self, **kwargs):
    super(VGGNormalize, self).__init__(**kwargs)

  def build(self, input_shape):
    pass

  def call(self, x, mask = None):
    return keras.applications.vgg19.preprocess_input(x) # preprocessing is the same for VGG16 and VGG19

  def compute_output_shape(self, input_shape):
    return input_shape

class Denormalize(keras.layers.Layer):
  """Converts tanh range to RGB range."""

  def __init__(self, **kwargs):
    super(Denormalize, self).__init__(**kwargs)

  def build(self, input_shape):
    pass

  def call(self, x, mask = None):
    return (x + 1) * 127.5 # converting tanh range (-1, 1) to RGB range (0, 255)

  def compute_output_shape(self, input_shape):
    return input_shape

class InstanceNorm(keras.layers.Layer):
  """Instance normalization."""

  def __init__(self, **kwargs):
    super(InstanceNorm, self).__init__(**kwargs)
    self.epsilon = 1e-4

  def compute_output_shape(self, input_shape):
    return input_shape

  def call(self, x, mask = None):
    mean, variance = tf.nn.moments(x, axes = [1, 2], keep_dims = True)
    return tf.div(tf.subtract(x, mean), tf.sqrt(tf.add(variance, self.epsilon)))

class ReflectionPadding2D(keras.layers.Layer):
  """Reflection padding."""

  def __init__(self, padding = (1, 1), **kwargs):
    self.padding = tuple(padding)
    self.input_spec = [keras.layers.InputSpec(ndim = 4)]
    super(ReflectionPadding2D, self).__init__(**kwargs)

  def compute_output_shape(self, input_shape):
    return input_shape[0], input_shape[1] + 2 * self.padding[0], input_shape[2] + 2 * self.padding[1], input_shape[3]

  def call(self, x, mask = None):
    width_pad, height_pad = self.padding
    return tf.pad(x, paddings = [[0, 0], [height_pad, height_pad], [width_pad, width_pad], [0, 0]], mode = "REFLECT")