
"""

"easyai.core.py"

Wrapper program for easy use of keras. Targeted towards those with little to no experience with Python and
machine learning.

"""

import keras as K
import tensorflow as tf
import numpy as np
from time import time
from typing import Union

# SUPPORT
class Static_Interface(object):
  """Static interface for other programs. An object of this class cannot be created."""

  def __init__(self):
    """As Static_Interface objects should not be created, __init__ throws a NotImplementedError.

    :raises NotImplementedError
    """
    raise NotImplementedError("class is static")

class Error_Handling(Static_Interface):
  """Static class for error handling."""

  @staticmethod
  def suppress_tf_warnings():
    """Suppresses tensorflow warnings. Does not work if tensorflow is outdated."""
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf
    try:
      tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    except AttributeError:
      tf.logging.set_verbosity(tf.logging.ERROR)

# CORE LAYERS
class Abstract_Layer(Static_Interface):
  """Abstract class that acts as the base for all layer classes. Should not be implemented"""

  def __init__(self, num_neurons: int , actv: str):
    """As Abstract_Layer objects should not be created, __init__ throws a NotImplementedError.

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
    [advanced] Creates keras mask. This mask will be used for training and all computations.

    :raises AssertionError if self.prev is not initialized.
    """
    assert self.prev, "self.prev must be initialized"
    raise NotImplementedError("cannot be implemented. How did you even call this function?")

class Input(Abstract_Layer):
  """MLP input layer with no activation, bias, or weights of its own. [advanced] No keras mask."""

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
  """MLP layer (aka a dense layer). [advanced] Has a keras mask."""

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
  """Convolutional layer. [advanced] Has a keras mask, stride = 1, padding = "valid"."""

  def __init__(self, filter_size: tuple, num_filters: int, actv: str = "sigmoid"):
    """
    Initializes Conv object. [advanced] No padding and a stride of 1 is assumed.

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

  def __init__(self, pool_size: tuple):
    """
    Initializes Pooling object. [advanced] Only max pooling is implemented, stride = None.

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
  """Uses easyai layer objects to create a functional keras model."""

  def __init__(self, layers: list = None, cost: str = "binary_crossentropy"):
    """
    Initializes NN (neural network) object.

    :param layers: layers of the network. Expects easyai core layer objects.
    :param cost: [advanced feature] cost used by the neural network. Default is categorical_crossentropy.
    """
    if layers is None:
      layers = []
    self.layers = tuple(layers)
    self.link_layers()

    self.cost = cost
    self.is_trained = False

  def link_layers(self):
    """[advanced] Links wrapper objects and creates a wrapper keras Sequential object."""
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
    """Trains and compiles this NN object. [advanced] Only SGD is used.

    :param x: input data. For example, if classifying an image, `x` would the pixel vectors.
    :param y: labels. For example, if classifying an image, `y` would be the image labels.
    [advanced] The `y` data should be comprised of one-hot encodings.
    :param lr: [advanced] learning rate used in SGD. Default is 3.0.
    :param epochs: number of epochs. Default is 1.
    :param batch_size: [advanced] minibatch size. Default is 10.
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

# APPLICATIONS OF NEURAL NETWORKS
class Neural_Style_Transfer(object):
  
  CONTENT_LAYER = "block5_conv2"
  STYLE_LAYERS = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"]

  def __init__(self, net: Union[str, NN] = None, num_rows = 400):
    self.net = net if not (net is None) else "vgg19"
    if isinstance(self.net, str):
      assert self.net == "vgg19", "only the vgg19 pre-trained model is supported"
    self.generated = None
    self.img_tensor = None

    self.num_rows = num_rows
    self.num_cols = None

  def net_init(self):
    if self.net == "vgg19":
      self.k_model = K.applications.vgg19.VGG19(input_tensor = self.img_tensor, weights = "imagenet",
                                                include_top = False)
    elif isinstance(self.net, NN):
      self.k_model = self.net.k_model

    self.k_model.trainable = False
    self.outputs = dict([(layer.name, layer.output) for layer in self.k_model.layers])

  def train(self, content_path: str, style_path: str, lr: float = 0.1, epochs: int = 1, verbose: bool = True):
    content, style, generated = self.image_init(content_path, style_path)

    self.img_tensor = K.backend.concatenate([content, style, generated], axis = 0)
    self.img_order = ["content", "style", "generated"]
    
    self.net_init()
    
    if verbose:
      print ("Loaded {0} model".format(self.net if isinstance(self.net, str) else type(self.net)))

    for epoch in range(epochs):
      start = time()

      cost = self.get_cost(Neural_Style_Transfer.CONTENT_LAYER, Neural_Style_Transfer.STYLE_LAYERS)
      grads = K.backend.gradients(cost, generated)

      generated -= tf.scalar_mul(lr, K.backend.variable(grads))
      K.preprocessing.image.save_img("home/ryan/PycharmProjects/easyai/result.png", self.deprocess(generated))

      if verbose:
        print ("Epoch {0}/{1}".format(epoch + 1, epochs))
        print (" - {0}s - cost: {1}".format(round(time() - start), cost))

  # COST CALCULATIONS
  def get_cost(self, content_layer, style_layers, coef_C = 1.0, coef_S = 2.0):

    def content_cost(layer):
      layer_features = self.outputs[layer]

      content_actvs = layer_features[self.img_order.index("content"), :, :, :]
      generated_actvs = layer_features[self.img_order.index("generated"), :, :, :]

      return K.backend.sum(K.backend.square(content_actvs - generated_actvs))

    def layer_style_cost(a_G, a_S):

      def gram_matrix(a):
        a = K.backend.batch_flatten(K.backend.permute_dimensions(a, (2, 0, 1)))
        return K.backend.dot(a, K.backend.transpose(a))

      gram_s = gram_matrix(a_S)
      gram_g = gram_matrix(a_G)

      return K.backend.sum(K.backend.square(gram_s - gram_g)) * (1. / (2 * int(np.prod(a_S.shape))) ** 2)

    cost = K.backend.variable(0.0)

    cost += coef_C * content_cost(content_layer)

    for layer in style_layers:
      layer_features = self.outputs[layer]

      generated_actvs = layer_features[self.img_order.index("generated"), :, :, :]
      style_actvs = layer_features[self.img_order.index("style"), :, :, :]

      cost += layer_style_cost(generated_actvs, style_actvs) * (coef_S / len(style_layers))

    return cost

  # IMAGE PROCESSING
  def image_init(self, content_path: str, style_path: str) -> tuple:
    content = self.preprocess(content_path)
    style = self.preprocess(style_path)

    self.gen_shape = (self.num_rows, self.num_cols, 3)
    print ("Generated image shape: {0}".format(self.gen_shape))

    if self.net == "vgg19":
      content = K.applications.vgg19.preprocess_input(content)
      style = K.applications.vgg19.preprocess_input(style)
    else:
      raise NotImplementedError("data normalization for other nets is not supported yet")

    generated = K.backend.placeholder(shape = (1, *self.gen_shape))

    return content, style, generated

  def preprocess(self, img, target_size=None):
    if target_size is None:
      if self.num_cols is None:
        width, height = K.preprocessing.image.load_img(img).size
        self.num_cols = int(width * self.num_rows / height)
      target_size = (self.num_rows, self.num_cols)
    if isinstance(img, str):
      img = K.preprocessing.image.load_img(img, target_size = target_size)
    return K.backend.variable(np.expand_dims(K.preprocessing.image.img_to_array(img), axis = 0))

  def deprocess(self, generated):
    generated = generated.reshape((self.num_rows, self.num_cols, 3))

    generated[:, :, 0] += 103.939
    generated[:, :, 1] += 116.779
    generated[:, :, 2] += 123.68

    generated = generated[:, :, ::-1]
    generated = np.clip(generated, 0, 255).astype('uint8')
    return generated

Error_Handling.suppress_tf_warnings()
net = Neural_Style_Transfer("vgg19")
net.train("/home/ryan/PycharmProjects/easyai/dog.jpg", "/home/ryan/PycharmProjects/easyai/picasso.jpg")