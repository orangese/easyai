
"""

"easyai._advanced._nets"

Advanced keras nets. Not for use by easyai users-- does not use easyai API.

"""

from easyai._advanced._layers import *
from easyai._advanced._losses import *

# NST TRANSFORM NET
class NSTTransform(NetworkInterface):
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
      print("Loaded NST transform net")

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
        a = InstanceNorm()(a)
      if include_relu:
        a = keras.layers.Activation("relu")(a)
      return a
    return _conv_norm_block

  @staticmethod
  def conv_res_block(filters, filter_size, norm = "batch"):
    def _conv_res_block(x):
      identity = keras.layers.convolutional.Cropping2D(cropping = ((2, 2), (2, 2)))(x) # 2 is the number of conv layers
      a = NSTTransform.conv_norm_block(filters, filter_size, norm = norm, include_relu = True, padding = "valid")(x)
      a = NSTTransform.conv_norm_block(filters, filter_size, norm = norm, include_relu = False,padding = "valid")(a)
      return keras.layers.merge.add([identity, a])
    return _conv_res_block

  def net_init(self):
    x = keras.layers.Input(shape = (self.num_rows, self.num_cols, 3), name = "img_transform_input")
    a = Normalize(noise = self.noise)(x)

    a = ReflectionPadding2D((40, 40))(a)
    a = NSTTransform.conv_norm_block(32, (9, 9), norm = self.norm)(a)
    a = NSTTransform.conv_norm_block(64, (3, 3), strides = (2, 2), norm = self.norm)(a)
    a = NSTTransform.conv_norm_block(128, (3, 3), strides = (2, 2), norm = self.norm)(a)

    for i in range(5):
      a = NSTTransform.conv_res_block(128, (3, 3), norm = self.norm)(a)

    a = NSTTransform.conv_norm_block(64, (3, 3), norm = self.norm, strides = (2, 2), transpose = True)(a)
    a = NSTTransform.conv_norm_block(32, (3, 3), norm = self.norm, strides = (2, 2), transpose = True)(a)
    a = NSTTransform.conv_norm_block(3, (9, 9), norm = self.norm, strides = (1, 1), transpose = True)(a)

    y = Denormalize(name = "img_transform_output")(a)

    self.k_model = keras.Model(inputs = x, outputs = y)

    tv_regularizer = TVRegularizer(self.coef_v)(self.k_model.layers[-1])
    self.k_model.layers[-1].add_loss(tv_regularizer)
    # adding total variation loss

# VGG NETS
class NSTLoss(StaticInterface):
  """Pre-trained NST loss networks. Copied from keras source code and changed to fit this API's needs."""

  WEIGHTS_PATH_NO_TOP = {
    "vgg16": "https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights"
             "_tf_dim_ordering_tf_kernels_notop.h5",
    "vgg19": "https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights"
             "_tf_dim_ordering_tf_kernels_notop.h5"
  }

  @staticmethod
  def VGG16(input_tensor):
    """VGG16. Input tensor required."""
    img_input = input_tensor

    # block 1
    x = keras.layers.Conv2D(64, (3, 3), activation = "relu", padding = "same", name = "block1_conv1")(img_input)
    x = keras.layers.Conv2D(64, (3, 3), activation = "relu", padding = "same", name = "block1_conv2")(x)
    x = keras.layers.MaxPooling2D((2, 2), strides = (2, 2), name = "block1_pool")(x)

    # block 2
    x = keras.layers.Conv2D(128, (3, 3), activation = "relu", padding = "same", name = "block2_conv1")(x)
    x = keras.layers.Conv2D(128, (3, 3), activation = "relu", padding = "same", name = "block2_conv2")(x)
    x = keras.layers.MaxPooling2D((2, 2), strides = (2, 2), name = "block2_pool")(x)

    # block 3
    x = keras.layers.Conv2D(256, (3, 3), activation = "relu", padding = "same", name = "block3_conv1")(x)
    x = keras.layers.Conv2D(256, (3, 3), activation = "relu", padding = "same", name = "block3_conv2")(x)
    x = keras.layers.Conv2D(256, (3, 3), activation = "relu", padding = "same", name = "block3_conv3")(x)
    x = keras.layers.MaxPooling2D((2, 2), strides = (2, 2), name="block3_pool")(x)

    # block 4
    x = keras.layers.Conv2D(512, (3, 3), activation = "relu", padding = "same", name = "block4_conv1")(x)
    x = keras.layers.Conv2D(512, (3, 3), activation = "relu", padding = "same", name = "block4_conv2")(x)
    x = keras.layers.Conv2D(512, (3, 3), activation = "relu", padding = "same", name = "block4_conv3")(x)
    x = keras.layers.MaxPooling2D((2, 2), strides = (2, 2), name = "block4_pool")(x)

    # block 5
    x = keras.layers.Conv2D(512, (3, 3), activation = "relu", padding = "same", name = "block5_conv1")(x)
    x = keras.layers.Conv2D(512, (3, 3), activation = "relu", padding = "same", name = "block5_conv2")(x)
    x = keras.layers.Conv2D(512, (3, 3), activation = "relu", padding = "same", name = "block5_conv3")(x)
    x = keras.layers.MaxPooling2D((2, 2), strides = (2, 2), name = "block5_pool")(x)

    # create model
    inputs = keras.engine.network.get_source_inputs(input_tensor)
    model = keras.Model(inputs, x, name = "vgg16")

    # load weights
    weights_path = keras.utils.data_utils.get_file("vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5",
                                                   NSTLoss.WEIGHTS_PATH_NO_TOP["vgg16"], cache_subdir = "models")
    model.load_weights(weights_path, by_name = True)

    return model

  @staticmethod
  def VGG19(input_tensor):
    """VGG19. Input tensor required."""
    img_input = input_tensor

    # block 1
    x = keras.layers.Conv2D(64, (3, 3), activation = "relu", padding = "same", name = "block1_conv1")(img_input)
    x = keras.layers.Conv2D(64, (3, 3), activation = "relu", padding = "same", name = "block1_conv2")(x)
    x = keras.layers.MaxPooling2D((2, 2), strides = (2, 2), name = "block1_pool")(x)

    # block 2
    x = keras.layers.Conv2D(128, (3, 3), activation = "relu", padding = "same", name = "block2_conv1")(x)
    x = keras.layers.Conv2D(128, (3, 3), activation = "relu", padding = "same", name = "block2_conv2")(x)
    x = keras.layers.MaxPooling2D((2, 2), strides = (2, 2), name = "block2_pool")(x)

    # block 3
    x = keras.layers.Conv2D(256, (3, 3), activation = "relu", padding = "same", name = "block3_conv1")(x)
    x = keras.layers.Conv2D(256, (3, 3), activation = "relu", padding = "same", name = "block3_conv2")(x)
    x = keras.layers.Conv2D(256, (3, 3), activation = "relu", padding = "same", name = "block3_conv3")(x)
    x = keras.layers.Conv2D(256, (3, 3), activation = "relu", padding = "same", name = "block3_conv4")(x)
    x = keras.layers.MaxPooling2D((2, 2), strides = (2, 2), name = "block3_pool")(x)

    # block 4
    x = keras.layers.Conv2D(512, (3, 3), activation = "relu", padding = "same", name = "block4_conv1")(x)
    x = keras.layers.Conv2D(512, (3, 3), activation = "relu", padding = "same", name = "block4_conv2")(x)
    x = keras.layers.Conv2D(512, (3, 3), activation = "relu", padding = "same", name = "block4_conv3")(x)
    x = keras.layers.Conv2D(512, (3, 3), activation = "relu", padding = "same", name = "block4_conv4")(x)
    x = keras.layers.MaxPooling2D((2, 2), strides = (2, 2), name = "block4_pool")(x)

    # block 5
    x = keras.layers.Conv2D(512, (3, 3), activation = "relu", padding = "same", name = "block5_conv1")(x)
    x = keras.layers.Conv2D(512, (3, 3), activation = "relu", padding = "same", name = "block5_conv2")(x)
    x = keras.layers.Conv2D(512, (3, 3), activation = "relu", padding = "same", name = "block5_conv3")(x)
    x = keras.layers.Conv2D(512, (3, 3), activation = "relu", padding = "same", name = "block5_conv4")(x)
    x = keras.layers.MaxPooling2D((2, 2), strides = (2, 2), name = "block5_pool")(x)

    # create model
    inputs = keras.engine.network.get_source_inputs(input_tensor)
    model = keras.Model(inputs, x, name = "vgg19")

    # load weights
    weights_path = keras.utils.data_utils.get_file("vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5",
                                                   NSTLoss.WEIGHTS_PATH_NO_TOP["vgg19"], cache_subdir = "models")
    model.load_weights(weights_path, by_name = True)

    return model