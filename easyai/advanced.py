
"""

"easyai.advanced.py"

Custom keras layers. Does not use easyai API. Not recommended for use by easyai users.

"""

import tensorflow as tf

from easyai.core import *

# CONV LAYERS
class Noisy_Normalize(keras.layers.Layer):
  """Noisy image-normalizing input layer. Used for Fast NST."""

  def __init__(self, noise, **kwargs):
    self.noise = noise
    super(Noisy_Normalize, self).__init__(**kwargs)

  def build(self, input_shape):
    pass

  def call(self, x, mask = None):
    noise_image = np.random.uniform(0, 1.0, size = K.int_shape(x)[1:])
    return tf.cast(noise_image * self.noise + ((x / 255.) * (1.0 - self.noise)), tf.float32)

  def compute_output_shape(self, input_shape):
    return input_shape

class Denormalize(keras.layers.Layer):
  """Reverses non-stochastic effect of Normalize input layer (noise is preserved)."""

  def __init__(self, **kwargs):
    super(Denormalize, self).__init__(**kwargs)

  def build(self, input_shape):
    pass

  def call(self, x, mask = None):
    return (x + 1) * 127.5

  def compute_output_shape(self, input_shape):
    return input_shape

class VGG_Normalize(keras.layers.Layer):
  """Custom normalization for VGG network."""

  def __init__(self, **kwargs):
    super(VGG_Normalize, self).__init__(**kwargs)

  def build(self, input_shape):
    pass

  def call(self, x, mask = None):
    x1 = keras.applications.vgg19.preprocess_input(x[0, ...])
    x2 = keras.applications.vgg19.preprocess_input(x[1, ...])
    return K.concatenate([x1, x2], axis = 0)
    # return keras.applications.vgg19.preprocess_input(K.concatenate([x[0], x[1]], axis = 0))
    # VGG19 and VGG16 have the same input preprocessing

  def compute_output_shape(self, input_shape):
    return input_shape

class Instance_Norm(keras.layers.Layer):
  """Instance normalization."""

  def __init__(self, **kwargs):
    super(Instance_Norm, self).__init__(**kwargs)
    self.epsilon = 1e-4

  def compute_output_shape(self, input_shape):
    return input_shape

  def call(self, x, mask = None):
    mean, variance = tf.nn.moments(x, axis = [1, 2], keep_dims = True)
    return tf.div(tf.subtract(x, mean), tf.sqrt(tf.add(variance, self.epsilon)))

class Reflection_Padding2D(keras.layers.Layer):
  """Reflection padding."""

  def __init__(self, padding = (1, 1), **kwargs):
    self.padding = tuple(padding)
    self.input_spec = [keras.layers.InputSpec(ndim = 4)]
    super(Reflection_Padding2D, self).__init__(**kwargs)

  def compute_output_shape(self, input_shape):
    return input_shape[0], input_shape[1] + 2 * self.padding[0], input_shape[2] + 2 * self.padding[1], input_shape[3]

  def call(self, x, mask = None):
    width_pad, height_pad = self.padding
    return tf.pad(x, paddings = [[0, 0], [height_pad, height_pad], [width_pad, width_pad], [0, 0]], mode = "REFLECT")

# NST TRANSFORM NET
class NST_Transform(Network_Interface):
  """Image transform NST layers. Implementation of network described in
  https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16.pdf. See 
  https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16Supplementary.pdf for exact network structure."""

  def __init__(self, num_rows = 512):
    self.num_rows = num_rows
    self.num_cols = None

  def train_init(self, target_size, coef_v, noise = 0.6, norm = "batch", verbose = True):
    self.num_rows, self.num_cols = target_size
    self.coef_v = coef_v
    self.noise = noise
    self.norm = norm

    self.net_init() # assumes img is a PIL image
    if verbose:
      print ("Loaded NST transform net")

  @staticmethod
  def conv_norm_block(filters, filter_size, strides = (1, 1), norm = "batch", include_relu = True,
                      transpose = False, padding = "same"):
    def _conv_norm_block(x):
      if transpose:
        a = keras.layers.Conv2DTranspose(filters, filter_size, strides = strides, padding = padding)(x)
      else:
        a = keras.layers.Conv2D(filters, filter_size, strides = strides, padding = padding)(x)
      if norm == "batch":
        a = keras.layers.BatchNormalization()(a)
      elif norm == "instance":
        a = Instance_Norm()(a)
      if include_relu:
        a = keras.layers.Activation("relu")(a)
      return a
    return _conv_norm_block

  @staticmethod
  def conv_res_block(filters, filter_size, norm = "batch"):
    def _conv_res_block(x):
      identity = keras.layers.convolutional.Cropping2D(cropping = ((2, 2), (2, 2)))(x) # 2 is the number of conv layers
      a = NST_Transform.conv_norm_block(filters, filter_size, norm = norm, include_relu = True, padding = "valid")(x)
      a = NST_Transform.conv_norm_block(filters, filter_size, norm = norm, include_relu = False,padding = "valid")(a)
      return keras.layers.merge.add([identity, a])
    return _conv_res_block

  def net_init(self):
    x = keras.layers.Input(shape = (self.num_rows, self.num_cols, 3))
    a = Noisy_Normalize(noise = self.noise)(x)

    a = Reflection_Padding2D((40, 40))(a)
    a = NST_Transform.conv_norm_block(32, (9, 9), norm = self.norm)(a)
    a = NST_Transform.conv_norm_block(64, (3, 3), strides = (2, 2), norm = self.norm)(a)
    a = NST_Transform.conv_norm_block(128, (3, 3), strides = (2, 2), norm = self.norm)(a)

    for i in range(5):
      a = NST_Transform.conv_res_block(128, (3, 3), norm = self.norm)(a)

    a = NST_Transform.conv_norm_block(64, (3, 3), norm = self.norm, strides = (2, 2), transpose = True)(a)
    a = NST_Transform.conv_norm_block(32, (3, 3), norm = self.norm, strides = (2, 2), transpose = True)(a)
    a = NST_Transform.conv_norm_block(3, (9, 9), norm = self.norm, strides = (1, 1), transpose = True)(a)

    y = Denormalize(name = "generated")(a)

    self.k_model = keras.Model(inputs = x, outputs = y)

    tv_regularizer = TV_Regularizer(self.coef_v)(self.k_model.layers[-1])
    self.k_model.layers[-1].add_loss(tv_regularizer)
    # adding total variation loss

# REGULARIZERS FOR NST
class Style_Regularizer(keras.regularizers.Regularizer):

  def __init__(self, style_img, weight):
    self.style_gram = Style_Regularizer.gram_matrix(style_img)
    self.weight = weight
    self.uses_learning_phase = False
    super(Style_Regularizer, self).__init__()

  def __call__(self, x):
    return self.weight * K.sum(K.square(self.style_gram - Style_Regularizer.gram_matrix(x.output[0])))
    # x.output[0] is generated by network, x.output[1] is the true label

  @staticmethod
  def gram_matrix(a):
    a = K.batch_flatten(K.permute_dimensions(a, (2, 0, 1)))
    return K.dot(a, K.transpose(a))

class Content_Regularizer(keras.regularizers.Regularizer):

  def __init__(self, weight):
    self.weight = weight
    self.uses_learning_phase = False
    super(Content_Regularizer, self).__init__()

  def __call__(self, x):
    return self.weight * K.sum(K.square(x.output[0] - x.output[1]))

class TV_Regularizer(keras.regularizers.Regularizer):

  def __init__(self, weight):
    self.weight = weight
    self.uses_learning_phase = False

  def __call__(self, x):
    shape = K.shape(x.output)
    num_rows, num_cols, channels = shape[0], shape[1], shape[2]
    # tensors are not iterable unless eager execution is enabled
    a = K.square(x.output[:, :num_rows - 1, :num_cols - 1, :] - x.output[:, 1:, :num_cols - 1, :])
    b = K.square(x.output[:, :num_rows - 1, :num_cols - 1, :] - x.output[:, :num_rows - 1, 1:, :])

    return K.sum(K.pow(a + b, 1.25))