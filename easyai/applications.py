
"""

"easyai.applications.py"

Applications of core layers and networks in larger, more real-world-based algorithms. Example: neural style transfer.

"""

import importlib

import matplotlib.pyplot as plt
from PIL import Image
from scipy.optimize import fmin_l_bfgs_b
import tensorflow as tf

from easyai.core import *
from easyai.advanced import *

# SUPPORT
class Evaluator(object):
  """
  Class used for custom loss and gradient functions. Should be used in conjunction with scipy.optimize.[whatever].
  """

  def __init__(self, obj: object):
    """
    Initializes Evaluator object.

    :param obj: obj that has some function used to evaluate loss and gradients, called "loss_and_grads"
    :raises AssertionError: obj must have loss_and_grads function
    """
    self.obj = obj
    assert hasattr(obj, "loss_and_grads"), "obj must have loss_and_grads function"
    self.reset()

  def f_loss(self, img: np.ndarray):
    """
    Calculates loss.

    :param img: image (array) used to calculate loss.
    :return: loss.
    """
    loss, grads = self.obj.loss_and_grads(img)
    self.loss = loss
    self.grads = grads
    return self.loss

  def f_grads(self, img):
    """
    Calculates gradients.

    :param img: image (array) used to calculate gradients.
    :return: gradients.
    """
    grads = np.copy(self.grads)
    self.reset()
    return grads

  def reset(self):
    self.loss = None
    self.grads = None

# NEURAL NETWORK APPLICATION
class Slow_NST(Network_Interface):
  """
  Class implementation of neural style transfer learning. As of August 2019, only VGG19 and VGG16 is supported.
  Borrowed heavily from the keras implementation of neural style transfer.
  """

  HYPERPARAMS = {"CONTENT_LAYER": "block5_conv2",
                 "STYLE_LAYERS": ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"],
                 "COEF_C": 1e0,
                 "COEF_S": 1e3,
                 "COEF_V": 1e1,
                 "MEANS": [103.939, 116.779, 123.68], # not a hp-- don't edit
                 "IMG_ORDER": ["content", "style", "generated"]
                 }

  MODELS = {"vgg16": "VGG16",
            "vgg19": "VGG19"
            }

  # INITS
  def __init__(self, net: str = None, num_rows: int = 512):
    """
    Initializes Slow_NST object.

    :param net: pre-trained model. Should be a string represeting the name of model, e.g., "vgg19".
    :param num_rows: number of rows that the image has. Is a pre-defined but editable hyperparameter.
    """
    self.net = net if net is not None else "vgg19"

    self.num_rows = num_rows
    self.num_cols = None

  def train_init(self, content: Image.Image, style: Image.Image, noise: float = 0.6, verbose: bool = True,
                 gen_tensor: tf.Tensor = None):
    """
    Initializes processed images (content, style, generated) as keras tensors in preparation for training.
    Also initializes the created image as a copy. All variables are object attributes.

    :param content: content image as PIL Image.
    :param style: style image as PIL Image.
    :param noise: amount of noise in initially generated image. Range is [0., 1.].
    :param verbose: if true, prints additional information.
    :param gen_tensor: support for fast NST.
    """
    self.k_net_module = importlib.import_module("keras.applications.{0}".format(self.net))

    self.image_init(content, style, gen_tensor = gen_tensor)

    self.img_tensor = K.backend.concatenate([getattr(self, img) for img in Slow_NST.get_hps("img_order")], axis = 0)

    self.model_init()

    if verbose:
      print("Loaded {0} model".format(self.net))

    if gen_tensor is None:
      noise_image = np.random.uniform(-20.0, 20.0, size = K.backend.int_shape(self.generated)[1])
      self.img = noise_image * noise + self.preprocess(content) * (1.0 - noise)

  def model_init(self):
    """
    Initializes model based on net type provided in __init__.
    """
    try:
      net_name = getattr(self.k_net_module, Slow_NST.MODELS[self.net])
    except KeyError:
      raise ModuleNotFoundError("{0} is not currently supported for neural style transfer".format(self.net))

    self.k_model = net_name(input_tensor = self.img_tensor, weights = "imagenet",
                            include_top = False) # no need for FC layers since no predictions occur

    for layer in self.k_model.layers:
      layer.trainable = False
    self.outputs = dict([(layer.name, layer.output) for layer in self.k_model.layers])

  def image_init(self, content: Image.Image, style: Image.Image, gen_tensor: tf.Tensor = None):
    """
    Initializes preprocessed images (content, style, generated) as object attributes.

    :param content: content image as PIL Image.
    :param style: style image as PIL Image.
    :param gen_tensor: support for fast NST.
    """
    self.content = K.backend.variable(self.preprocess(content))
    self.style = K.backend.variable(self.preprocess(style))

    self.gen_shape = (self.num_rows, self.num_cols, 3)
    print ("Generated image shape: {0}".format(self.gen_shape))

    self.generated = gen_tensor if gen_tensor is not None else K.backend.placeholder(shape = (1, *self.gen_shape))
    # 1 is number of images in batch

  def tensor_init(self, content_layer: str, style_layers: list):
    """
    Initializes the keras function that will be used to calculate gradients and loss. Also initializes Evaluator object.

    :param content_layer: layer at which content loss will be evaluated. Is a pre-defined hyperparameter.
    :param style_layers: layer(s) at which style loss will be evaluated. Is a pre-defined hyperparameter.
    """
    coef_C, coef_S, coef_V = self.get_hyperparams("coef_C", "coef_S", "coef_V")
    
    loss = self.loss_tensor(content_layer, style_layers, coef_c, coef_s, coef_v)
    grads = K.backend.gradients(loss, self.generated)
    outputs = [loss, *grads] if isinstance(grads, (list, tuple)) else [loss, grads]

    self.model_func = K.backend.function([self.generated], outputs)
    self.evaluator = Evaluator(self)

  # TRAINING
  def train(self, content: Image.Image, style: Image.Image, epochs: int = 1, init_noise: float = 0.6,
            verbose: bool = True, save_path: str = None) -> np.ndarray:
    """
    Trains a Slow_NST object. More precisely, the pixel values of the created image are optimized using
    scipy's implementation of L-BFGS-B.

    :param content: content image as PIL Image.
    :param style: style image as PIL Image.
    :param epochs: number of iterations or epochs.
    :param init_noise: amount of noise in initially generated image. Range is [0., 1.].
    :param verbose: if true, prints information about each epoch.
    :param save_path: (optional) path at which to save the created image at each iteration.
    :return: final created image.
    """
    self.train_init(content, style, verbose = verbose, noise = init_noise)

    content_layer, style_layers = self.get_hyperparams("content_layer", "style_layers")

    self.tensor_init(content_layer, style_layers)

    if verbose:
      Slow_NST.display_original(content, style)

    for epoch in range(epochs):
      start = time()

      # updating pixel values using L-BFGS-B
      self.img, loss, info = fmin_l_bfgs_b(func = self.evaluator.f_loss, x0 = self.img.flatten(),
                                           fprime = self.evaluator.f_grads, maxfun = 20) # 20 iterations per epoch
      plt.close()

      if verbose:
        print ("Epoch {0}/{1}".format(epoch + 1, epochs))
        print (" - {0}s - loss: {1} [broken]".format(round(time() - start), loss)) # loss is broken-- it's way too high
        self.display_img(self.img, "Epoch {0}/{1}".format(epoch + 1, epochs))

      if save_path is not None:
        full_save_path = save_path + "/epoch{0}.png".format(epoch + 1)
        K.preprocessing.image.save_img(full_save_path, self.deprocess(self.img))

    return self.deprocess(self.img)

  # loss CALCULATIONS
  def loss_and_grads(self, img: np.ndarray) -> tuple:
    """
    Computes loss and gradients using img. Utilizes the keras function created in tensor_init().

    :param img: image used to calculate loss and gradients w.r.t. the loss.
    :return: loss, gradients.
    """
    img = img.reshape(self.gen_shape)
    outputs = self.model_func([img]) # gets output of self.model_func evaluated at img
    loss = outputs[0]
    if len(outputs[1:]) == 1:
      grads = outputs[1].flatten().astype(np.float64)
      # scipy's L-BFGS-B function requires that the dtype of these variables be float64
    else:
      grads = np.array(outputs[1:]).flatten().astype(np.float64)
    return loss, grads

  def loss_tensor(self, content_layer: str, style_layers: list, coef_c: float, coef_s: float, coef_v: float):
    """
    Gets the symbolic loss tensor as a keras tensor. This tensor will be used to calculate loss and
    the gradients of loss. Is broken but training still works.

    :param content_layer: layer at which content loss will be evaluated. Is a pre-defined hyperparameter.
    :param style_layers: layer(s) at which style loss will be evaluated. Is a pre-defined hyperparameter.
    :param coef_c: content weight. Is a pre-defined hyperparameter.
    :param coef_s: stye weight. Is a pre-defined hyperparameter.
    :param coef_v: total variation weight.Is a pre-defined hyperparameter.
    :return: loss tensor.
    """

    def content_loss(layer, outputs):
      """Computes the content loss at layer "layer"."""
      layer_features = outputs[layer]

      content_actvs = layer_features[Slow_NST.get_hps("img_order").index("content"), :, :, :]
      generated_actvs = layer_features[Slow_NST.get_hps("img_order").index("generated"), :, :, :]

      return K.backend.sum(K.backend.square(generated_actvs - content_actvs)) # i.e., squared norm

    def layer_style_loss(a_G, a_S):
      """Computes the style loss at layer "layer". Is broken but training still works."""

      def gram_matrix(a):
        """Computes the gram matrix of a."""
        a = K.backend.batch_flatten(K.backend.permute_dimensions(a, (2, 0, 1)))
        return K.backend.dot(a, K.backend.transpose(a))

      gram_s = gram_matrix(a_S)
      gram_g = gram_matrix(a_G)

      return K.backend.sum(K.backend.square(gram_s - gram_g))

    def total_variation_loss(a_G, num_rows, num_cols):
      """Computes the total variation loss of the generated image. According to keras, it is designed to keep
      the generated image locally coherent."""
      a = K.backend.square(a_G[:, :num_rows - 1, :num_cols - 1, :] - a_G[:, 1:, :num_cols - 1, :])
      b = K.backend.square(a_G[:, :num_rows - 1, :num_cols - 1, :] - a_G[:, :num_rows - 1, 1:, :])

      return K.backend.sum(K.backend.pow(a + b, 1.25))

    loss = K.backend.variable(0.0) # it is necessary to initialize loss like this

    loss += coef_c * content_loss(content_layer, self.outputs) # content loss

    # this loop is probably broken (and/or the computation of layer_style_loss)
    for layer in style_layers: # style loss
      layer_features = self.outputs[layer]

      generated_actvs = layer_features[Slow_NST.get_hps("img_order").index("generated"), :, :, :]
      style_actvs = layer_features[Slow_NST.get_hps("img_order").index("style"), :, :, :]

      loss += (coef_s / len(style_layers)) * layer_style_loss(generated_actvs, style_actvs) / \
              (4.0 * (int(self.gen_shape[-1]) ** 2) * ((self.num_rows * self.num_cols) ** 2)) # normalization

    loss += coef_v * total_variation_loss(self.generated, self.num_rows, self.num_cols)

    return loss

  # IMAGE PROCESSING
  def preprocess(self, img: Image.Image, target_size = None) -> np.ndarray:
    """
    Preprocesses an image.

    :param img: image to preprocess.
    :param target_size: target size of the image. If is none, defaults to object attributes.
    :return: processed image.
    """
    # because of pass-by-assignment properties, a copy must be made to prevent tampering with original img
    if target_size is None:
      if self.num_cols is None:
        width, height = img.size
        self.num_cols = int(width * self.num_rows / height)
      target_size = (self.num_rows, self.num_cols)
    img = img.resize(reversed(target_size), Image.NEAREST) # resizing image with interpolation = NEAREST
    img = np.expand_dims(K.preprocessing.image.img_to_array(img), axis = 0)

    return self.k_net_module.preprocess_input(img)

  def deprocess(self, img_: np.ndarray) -> np.ndarray:
    """
    Reverses effect of preprocess

    :param img_: image to deprocess.
    :return: deprocessed image.
    """
    img = np.copy(img_).reshape(*self.generated.shape[1:])
    means = self.get_hyperparams("means")
    for i in range(len(means)):
      img[:, :, i] += means[i] # adding mean pixel values
    img = img[:, :, ::-1] #BGR -> RBG
    img = np.clip(img, 0, 255).astype('uint8')
    return img

  def display_img(self, img: np.ndarray, title: str):
    """
    Displays image.

    :param img: image to be displayed.
    :param title: title of maptlotlib plot.
    """
    try:
      img = self.deprocess(img)
    except np.core._exceptions.UFuncTypeError:
      pass

    fig = plt.gcf()

    fig.canvas.set_window_title("Training...")
    fig.suptitle(title)
    plt.axis("off")

    plt.imshow(img.reshape(*self.gen_shape[1:]))
    plt.pause(0.1)

    plt.show(block = False)

  @staticmethod
  def display_original(content: Image.Image, style: Image.Image):
    """
    Displays original images.

    :param content: path to content image.
    :param style: path to style image.
    """
    fig, (content_ax, style_ax) = plt.subplots(1, 2)
    fig.suptitle("Original images")
    fig.canvas.set_window_title("Pre-training")

    content_ax.imshow(content)
    content_ax.axis("off")

    style_ax.imshow(style)
    style_ax.axis("off")
    plt.pause(0.1)

    plt.show(block = False)

  # MISCELLANEOUS
  def get_hyperparams(self, *hyperparams: str) -> Union[str, tuple, float]:
    """
    Fetches hyperparameters. Merely a syntax simplification tool.

    :param hyperparams: any number of hyperparameters to fetch.
    :return: feched hyperparameters.
    """
    try:
      fetched = tuple(Neural_Style_Transfer.HYPERPARAMS[hp.upper()][self.net] for hp in hyperparams)
    except KeyError:
      fetched = tuple(Neural_Style_Transfer.HYPERPARAMS[hp.upper()]["other"] for hp in hyperparams)
    return fetched[0] if len(fetched) == 1 else fetched

class Fast_NST(Slow_NST):

  def __init__(self, img_transform_net: Network_Interface = NST_Transform_Net(),
               loss_net: Network_Interface = Slow_NST(), num_rows: int = 512):
    self.img_transform_net = img_transform_net
    self.loss_net = loss_net
    self.num_rows = num_rows

  def train_init(self, content: Image.Image, style: Image.Image, noise = 0.6, verbose: bool = True):
    """
    Initializes processed images (content, style, generated) from paths as keras tensors in preparation for training.
    Also initializes the created image as a copy. All variables are object attributes.

    :param content: content image as PIL Image.
    :param style: style image as PIL Image.
    :param noise: amount of noise in initially generated image. 0. ≤ noise ≤ 1.
    :param verbose: if true, prints additional information.
    """
    # target shape init
    width, height = content.size
    self.num_cols = int(width * self.num_rows / height)

    # partial train init
    self.img_transform_net.train_init(content, noise = noise)
    self.loss_net.train_init(content, style, noise = noise, verbose = verbose,
                             gen_tensor = self.img_transform_net.k_model.output)
    self.generated = self.img_transform_net.k_model.layers[-1].output # aka, output of the image transform net
    self.gen_shape = self.generated.shape[1:]

    # concatenating models
    # outs = self.loss_net.k_model(self.img_transform_net.k_model.layers[-1].output)
    # self.k_model = K.Model(self.img_transform_net.k_model.input, outs)

    # getting dict for loss net loss calculation
    self.outputs = dict([(layer.name, layer.output) for layer in self.loss_net.k_model.layers])
    # wanted to name self.outputs as self.loss_outputs but it is inconvenient due to implementation of Slow_NST methods

    # loss net as a loss function
    self.bleeb = K.backend.placeholder(shape = self.loss_net.k_model.layers[0].output_shape)
    outs = [self.loss_tensor(*Slow_NST.get_hps("content_layer", "style_layers", "coef_c", "coef_s", "coef_v"))]
    self.loss_func = K.backend.function([self.bleeb], outs)

    # compiling model
    self.img_transform_net.k_model.compile(optimizer = K.optimizers.Adam(), loss = self.loss)

  def loss(self, y_true, y_pred):
    return self.loss_func([y_true])[0]

  def train(self, content_imgs: Image.Image, style_imgs: Image.Image, epochs: int = 1,
              init_noise: float = 0.6, verbose: bool = True):
    self.train_init(content_imgs, style_imgs, noise = init_noise, verbose = verbose)

from easyai.support.load import load_nst_imgs
content, style = load_nst_imgs("/home/ryan/Pictures/nst_generated/dog_blue_green_stained_glass.jpg",
                               "/home/ryan/Pictures/nst_generated/dog_blue_green_stained_glass.jpg")

test = Fast_NST()
test.train(content, style)