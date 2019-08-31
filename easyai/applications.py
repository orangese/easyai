
"""

"easyai.applications.py"

Applications of neural networks in larger, more real-world-based algorithms. Does not use easyai API; instead, relies
on keras since easyai is not flexible enough to meet the demands of these algorithms.

"""

import importlib
import os
import random

import matplotlib.pyplot as plt
from PIL import Image
from scipy.optimize import fmin_l_bfgs_b
import tensorflow as tf

from easyai._advanced import HidePrints
from easyai._advanced._nets import *
from easyai._advanced._losses import *

# NEURAL NETWORK APPLICATION
class SlowNST(NetworkInterface):
  """
  Class implementation of neural style transfer learning. As of August 2019, only VGG19 and VGG16 is supported.
  Borrowed heavily from the keras implementation of neural style transfer.
  """

  # CONSTANTS
  HYPERPARAMS = {"CONTENT_LAYER": "block5_conv2",
                 "STYLE_LAYERS": ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"],
                 "COEF_C": 1e0,
                 "COEF_S": 1e3,
                 "COEF_V": 1e-4,
                 "MEANS": [103.939, 116.779, 123.68], # not a hp-- don"t edit
                 "IMG_ORDER": ["content", "style", "generated"]
                 }

  MODELS = {"vgg16": "VGG16",
            "vgg19": "VGG19"
            }

  # INITS
  def __init__(self, net: str = "vgg19", num_rows: int = 400):
    """
    Initializes SlowNST object.

    :param net: pre-trained model. Either "vgg16" or "vgg19"
    :param num_rows: number of rows that the image has. Is a pre-defined but editable hyperparameter.
    """
    self.net = net

    self.num_rows = num_rows
    self.num_cols = None

  def train_init(self, content: Image.Image, style: Image.Image, noise: float = 0.6, verbose: bool = True):
    """
    Initializes processed images (content, style, generated) as keras tensors in preparation for training.
    Also initializes the created image as a copy. All variables are object attributes.

    :param content: content image as PIL Image.
    :param style: style image as PIL Image.
    :param noise: amount of noise in initially generated image. Range is [0., 1.].
    :param verbose: if true, prints additional information.
    :raises ModuleNotFoundError: if provided net is not valid.
    """
    self.k_net_module = importlib.import_module("keras.applications.{0}".format(self.net))

    self.image_init(content, style)

    self.img_tensor = K.concatenate([getattr(self, img) for img in SlowNST.get_hps("img_order")], axis = 0)

    self.model_init()

    noise_image = np.random.uniform(-20.0, 20.0, size = K.int_shape(self.generated))
    self.img = noise_image * noise + self.preprocess(content) * (1.0 - noise)

    if verbose:
      print("Loaded {0} model".format(self.net))

  def model_init(self):
    """
    Initializes model based on net type provided in __init__.

    :raises ModuleNotFoundError: if provided net is not valid.
    """
    try:
      net_name = getattr(self.k_net_module, SlowNST.MODELS[self.net])
    except KeyError:
      raise ModuleNotFoundError("{0} is not supported for slow neural style transfer".format(self.net))

    self.k_model = net_name(input_tensor = self.img_tensor, weights = "imagenet", include_top = False)
    # no need for FC layers since no predictions occur

    self.k_model.trainable = False
    self.outputs = dict([(layer.name, layer.output) for layer in self.k_model.layers])

  def image_init(self, content: Image.Image, style: Image.Image):
    """
    Initializes preprocessed images (content, style, generated) as object attributes.

    :param content: content image as PIL Image.
    :param style: style image as PIL Image.
    """
    self.content = K.variable(self.preprocess(content))
    self.style = K.variable(self.preprocess(style))

    self.generated = K.placeholder(shape = (1, self.num_rows, self.num_cols, 3)) # 1 is number of images in batch

    print("Generated image shape: {0}".format(self.generated.shape[1:]))

  def tensor_init(self, content_layer: str, style_layers: list):
    """
    Initializes the keras function that will be used to calculate gradients and loss. Also initializes Evaluator object.

    :param content_layer: layer at which content loss will be evaluated. Is a pre-defined hyperparameter.
    :param style_layers: layer(s) at which style loss will be evaluated. Is a pre-defined hyperparameter.
    """
    coef_c, coef_s, coef_v = SlowNST.get_hps("coef_c", "coef_s", "coef_v")

    loss = self.loss_tensor(content_layer, style_layers, coef_c, coef_s, coef_v)
    grads = K.gradients(loss, self.generated)
    outputs = [loss, *grads] if isinstance(grads, (list, tuple)) else [loss, grads]

    self.model_func = K.function([self.generated], outputs)
    self.evaluator = Evaluator(self)

  # TRAINING
  def train(self, content: Image.Image, style: Image.Image, epochs: int = 1, init_noise: float = 0.6,
            verbose: bool = True, save_path: str = None) -> np.ndarray:
    """
    Trains a SlowNST object. More precisely, the pixel values of the created image are optimized using
    scipy's implementation of L-BFGS-B.

    :param content: content image as PIL Image.
    :param style: style image as PIL Image.
    :param epochs: number of iterations or epochs.
    :param init_noise: amount of noise in initially generated image. Range is [0., 1.].
    :param verbose: if true, prints information about each epoch.
    :param save_path: (optional) path at which to save the created image at each iteration.
    :return: final created image.
    """
    num_iters = 20 # number of L-BFGS-B iterations per epoch

    self.train_init(content, style, verbose = verbose, noise = init_noise)

    content_layer, style_layers = SlowNST.get_hps("content_layer", "style_layers")

    self.tensor_init(content_layer, style_layers)

    print("Training with L-BFGS-B (another gradient-based optimization algorithm) in a {0}-D space. During "
           "each epoch, the pixels of the generated image will be changed {1} times in an attempt to minimize loss."
           .format(np.prod(self.generated.shape[1:]), num_iters))

    if verbose:
      SlowNST.display_original(content, style)

    for epoch in range(epochs):
      if verbose:
        print("Epoch {0}/{1}".format(epoch + 1, epochs))

      start = time()

      # updating pixel values using L-BFGS-B
      self.img, loss, info = fmin_l_bfgs_b(func = self.evaluator.f_loss, x0 = self.img.flatten(),
                                           fprime = self.evaluator.f_grads, maxfun = num_iters)
      plt.close()

      if verbose:
        print(" - {}s - loss: {:.4e}".format(round(time() - start), loss))
        SlowNST.display_img(self.img, "Epoch {0}/{1}".format(epoch + 1, epochs), self.generated.shape[1:])

      if save_path is not None:
        full_save_path = save_path + "/epoch{0}.png".format(epoch + 1)
        keras.preprocessing.image.save_img(full_save_path, SlowNST.deprocess(self.img, self.generated.shape[1:]))

    return SlowNST.deprocess(self.img, self.generated.shape[1:])

  # LOSS CALCULATIONS
  def loss_and_grads(self, img: np.ndarray) -> tuple:
    """
    Computes loss and gradients using img. Utilizes the keras function created in tensor_init().

    :param img: image used to calculate loss and gradients w.r.t. the loss.
    :return: loss, gradients.
    """
    img = img.reshape(self.generated.shape)
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

      content_actvs = layer_features[SlowNST.get_hps("img_order").index("content"), :, :, :]
      generated_actvs = layer_features[SlowNST.get_hps("img_order").index("generated"), :, :, :]

      return K.sum(K.square(generated_actvs - content_actvs)) # i.e., squared norm

    def layer_style_loss(a_g, a_s):
      """Computes the style loss at layer "layer". Is broken but training still works."""

      def gram_matrix(a):
        """Computes the gram matrix of a."""
        a = K.batch_flatten(K.permute_dimensions(a, (2, 0, 1)))
        return K.dot(a, K.transpose(a))

      gram_s = gram_matrix(a_s)
      gram_g = gram_matrix(a_g)

      return K.sum(K.square(gram_s - gram_g))

    def total_variation_loss(a_g, num_rows, num_cols):
      """Computes the total variation loss of the generated image. According to keras, it is designed to keep
      the generated image locally coherent."""
      a = K.square(a_g[:, :num_rows - 1, :num_cols - 1, :] - a_g[:, 1:, :num_cols - 1, :])
      b = K.square(a_g[:, :num_rows - 1, :num_cols - 1, :] - a_g[:, :num_rows - 1, 1:, :])

      return K.sum(K.pow(a + b, 1.25))

    loss = K.variable(0.0) # it is necessary to initialize loss like this

    loss += coef_c * content_loss(content_layer, self.outputs) # content loss

    for layer in style_layers: # style loss
      layer_features = self.outputs[layer]

      generated_actvs = layer_features[SlowNST.get_hps("img_order").index("generated"), :, :, :]
      style_actvs = layer_features[SlowNST.get_hps("img_order").index("style"), :, :, :]

      layer_shape = layer_features.get_shape().as_list()
      norm_term = 4.0 * (layer_shape[-1] ** 2) * (np.prod(layer_shape[1:-1]) ** 2)
      # norm_term = 4.0 * (K.int_shape(self.generated)[-1] ** 2) * ((self.num_rows * self.num_cols) ** 2)
      loss += (coef_s / len(style_layers)) * layer_style_loss(generated_actvs, style_actvs) / norm_term

    loss += coef_v * total_variation_loss(self.generated, self.num_rows, self.num_cols)

    return loss

  # IMAGE PROCESSING
  def preprocess(self, img: Image.Image, target_size = None) -> np.ndarray:
    """
    Preprocesses an image.

    :param img: image to preprocess.
    :param target_size: target size of the image. If is None, defaults to object attributes.
    :return: processed image.
    """
    if target_size is None:
      if self.num_cols is None:
        width, height = img.size
        self.num_cols = int(width * self.num_rows / height)
      target_size = (self.num_rows, self.num_cols)
    img = img.resize(reversed(target_size)) # resizing image with interpolation = NEAREST
    img = np.expand_dims(keras.preprocessing.image.img_to_array(img), axis = 0)
    img = self.k_net_module.preprocess_input(img)
    return img

  @staticmethod
  def deprocess(img_: np.ndarray, target_shape: tuple = None) -> np.ndarray:
    """
    Reverses effect of preprocess.

    :param img_: image to deprocess.
    :param target_shape: target shape.
    :return: deprocessed image.
    """
    img = np.copy(img_)
    if target_shape is not None:
      img = img.reshape(target_shape)
    means = SlowNST.get_hps("means")
    for i in range(len(means)):
      img[:, :, i] += means[i] # adding mean pixel values
    img = img[:, :, ::-1] #BGR -> RBG
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

  @staticmethod
  def display_img(img: np.ndarray, title: str, target_shape: tuple = None, deprocess: bool = True):
    """
    Displays image.

    :param img: image to be displayed.
    :param title: title of maptlotlib plot.
    :param target_shape: target shape.
    :param deprocess: whether or not to deprocess the image before displaying it.
    """
    if deprocess:
      img = SlowNST.deprocess(img, target_shape = target_shape)

    fig = plt.gcf()

    fig.canvas.set_window_title("Training...")
    fig.suptitle(title)
    plt.axis("off")

    plt.imshow(img.reshape(target_shape))
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
  @staticmethod
  def get_hps(*hyperparams: str) -> Union[str, tuple, float]:
    """
    Fetches hyperparameters. Merely a syntax simplification tool.

    :param hyperparams: any number of hyperparameters to fetch.
    :return: feched hyperparameters.
    """
    fetched = tuple(SlowNST.HYPERPARAMS[hp.upper()] for hp in hyperparams)
    return fetched[0] if len(fetched) == 1 else fetched

class FastNST(NetworkInterface):
  """
  Fast neural style transfer, uses implementation of slow neural style transfer.

  Paper: https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16.pdf
  Supplementary material: https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16Supplementary.pdf

  Fast NST trains on the COCO dataset (~4 hours) and then can stylize images 1e3 times faster than slow NST. However,
  each unique style requires another round of training on the COCO dataset.
  """

  # INITS
  def __init__(self, img_transform_net: NetworkInterface = NSTTransform(), loss_net: str = "vgg16"):
    """
    Initializes fast NST object. This network has two parts: a trainable network (img_transform_net) and a fixed
    network (loss_net).

    :param img_transform_net: image transform networkeras. Will be trained.
    :param loss_net: fixed network, either "vgg16" or "vgg19". Is pre-trained.
    """
    self.img_transform_net = img_transform_net
    self.loss_net = loss_net

    self.vgg_num = int(self.loss_net.replace("vgg", ""))

  def train_init(self, style: Image.Image, target_size, noise = 0.6, norm = "batch", verbose: bool = True):
    """
    Initializes processed images (style and generated) as keras tensors in preparation for training.
    Also initializes the function and models that will be used for training.

    :param style: style image as PIL Image.
    :param target_size: all training images will be reshaped to target_size.
    :param noise: amount of noise in initially generated image. Range is [0., 1.].
    :param norm: type of normalization to apply. Either "batch" for batch norm or "instance" for instance norm.
    :param verbose: if true, prints additional information.
    :raises ModuleNotFoundError: if provided loss net is not valid.
    """
    # IMAGE INIT
    self.num_rows, self.num_cols = target_size
    self.style = style.resize(target_size)

    # NET INIT
    self.img_transform_net.train_init(target_size, coef_v = SlowNST.get_hps("coef_v"), noise = noise, norm = norm)
    self.loss_net_init()

    self.k_model.compile(optimizer = keras.optimizers.Adam(), loss = FastNST.dummy_loss)

    # CALLBACKS INIT
    self.losses = []

    if verbose:
      print("Loaded NST loss net")

  def loss_net_init(self):

    vgg_num = self.vgg_num

    def add_regularizers(model, style_img, target_size):

      def add_style_loss(style_img, layers, outputs, target_size):
        # retrieving hyperparameters
        style_layers = SlowNST.get_hps("style_layers")

        # preprocessing
        style_img = style_img.resize(reversed(target_size))
        style_img = np.expand_dims(keras.preprocessing.image.img_to_array(style_img), axis = 0)

        style_func = K.function([model.layers[-(vgg_num + 3)].input], [outputs[layer] for layer in style_layers])
        # +3 to account for Normalize, Denormalize, and Concatenate layers
        style_features = style_func([style_img])

        weight = SlowNST.get_hps("coef_s") / len(style_layers)
        for layer_num, layer_name in enumerate(style_layers): # adding style loss
          layer = layers[layer_name]
          style_regularizer = StyleRegularizer(K.variable(style_features[layer_num][0]), weight)(layer)
          layer.add_loss(style_regularizer)

      def add_content_loss(layers):
        content_layer = layers[SlowNST.get_hps("content_layer")]
        content_regularizer = ContentRegularizer(SlowNST.get_hps("coef_c"))(content_layer)
        content_layer.add_loss(content_regularizer)

      def add_tv_loss(tv_layer):
        tv_layer.add_loss(TVRegularizer(SlowNST.get_hps("coef_v"))(tv_layer))

      outputs = dict([(layer.name, layer.output) for layer in model.layers[-(vgg_num + 2):]])
      layers = dict([(layer.name, layer) for layer in model.layers[-(vgg_num + 2):]])
      # +2 to account for the addition of the input tensor (Concatenate) and the VGGNormalize layer

      add_style_loss(style_img, layers, outputs, target_size)
      add_content_loss(layers)
      add_tv_loss(model.get_layer("img_transform_output")) # tv loss is evaluated on the output of the transform net

    generated, content = self.img_transform_net.k_model.output, self.img_transform_net.k_model.input

    # no concatenation of style (that occurs in the regularizers)
    img_tensor = keras.layers.merge.concatenate([generated, content], axis = 0)
    img_tensor = VGGNormalize(name = "vgg_normalize")(img_tensor)

    try:
      loss_net = getattr(NSTLoss, SlowNST.MODELS[self.loss_net])
    except AttributeError:
      raise ModuleNotFoundError("{0} is not supported for fast neural style transfer".format(self.loss_net))

    self.k_model = loss_net(input_tensor = img_tensor) # automatically excludes top

    # only freezing VGG layers-- image transform net is trainable
    for layer in self.k_model.layers[-(self.vgg_num + 2):]:
      layer.trainable = False

    add_regularizers(self.k_model, self.style, (self.num_rows, self.num_cols))
    # adding loss and regularizers

  # TRAINING
  def train(self, style: Image.Image, epochs: int = 1, batch_size = 2, init_noise: float = 0.6,
            target_size: tuple = (256, 256), verbose: bool = True, save_path: str = None) -> np.ndarray:
    """
    Trains the image transform network on the MS COCO dataset (https://cocodataset.org/#download) to match a certain
    style. This dataset does not come preinstalled with easyai and takes a while to download (~4 hours).
    If you start training without COCO installed, you will be prompted to run easyai"s COCO installer script.

    :param style: style image as PIL Image.
    :param epochs: number of iterations or epochs.
    :param batch_size: batch size.
    :param init_noise: amount of noise in initially generated image. Range is [0., 1.].
    :param target_size: all training images will be reshaped to target_size.
    :param verbose: if true, prints additional training information.
    :param save_path: path to save training results
    :return: fast NST run on content example (numpy array).
    :raises ModuleNotFoundError: if provided loss net is not valid.
    """
    assert batch_size == 2, "only batch size of 2 is supported"

    path_to_coco = os.getenv("HOME") + "/coco"
    coco_dataset = "unlabeled2017"

    self.train_init(style, target_size = target_size, noise = init_noise, norm = "instance", verbose = verbose)

    with HidePrints(): # hiding print messages called when using ImageDataGenerator
      datagen = keras.preprocessing.image.ImageDataGenerator()
      generator = datagen.flow_from_directory(path_to_coco, target_size = target_size, batch_size = batch_size,
                                              classes = [coco_dataset], class_mode = "input")

    print("Training with Adam (another gradient-based optimization algorithm) in a {0}-D space. During each epoch, "
          "network parameters will be changed approximately {1} times in an attempt to minimize loss."
          .format(self.img_transform_net.k_model.count_params(), int(generator.samples / batch_size)))

    num_batches = round(generator.samples / batch_size)

    # example for verbose output
    path_to_coco_imgs = path_to_coco + "/" + coco_dataset
    content_example = Image.open(path_to_coco_imgs + "/" + random.choice(os.listdir(path_to_coco + "/" + coco_dataset)))
    content_example = np.array(content_example.resize(target_size)).astype(np.uint8)
    SlowNST.display_img(content_example, "Content example image", deprocess = False)

    for epoch in range(epochs):
      if verbose:
        print("Epoch {0}/{1}".format(epoch + 1, epochs))

      start = time()

      batch_nums = [1, 1] # batch_nums[0] is the current batch, and batch_nums[1] is the current batch of batches
      batch_start = time()

      eta = None # estimated time until to the next epoch

      for batch, labels in generator: # batch == labels == content image

        # TRAINING
        loss = self.k_model.train_on_batch(batch, labels) # train single batch

        self.losses.append(loss)

        # VERBOSE OUTPUT
        if verbose and batch_nums[0] % int(num_batches / 20) == 0: # if verbose, display information ~20 times per epoch
          elapsed = round(time() - batch_start)

          if eta is None:
            eta = round(elapsed * num_batches - elapsed)
          else:
            eta = round((0.95 * eta + 0.05 * (elapsed * (num_batches - batch_nums[0]))) - elapsed)
            # exponentially weighted average of previous ETAs and current elapsed time

          print(" - {}s - batches {}-{} of {} completed ({}%) - time until next epoch - {}s - loss - {:.4e}".format(
            elapsed, *reversed(batch_nums), num_batches, round(batch_nums[0] / num_batches * 100, 1), eta,
            self.losses[-1])) # {:.4e} represents the loss in scientific notation, rounded to 4 significant figures
          SlowNST.display_img(self.run_nst(content_example), "Epoch {0}: batch {1}".format(epoch, batch_nums[0]),
                              deprocess = False)

          batch_nums[1] = batch_nums[0] + 1
          batch_start = time()

        # BATCH COUNT
        batch_nums[0] += 1

        if batch_nums[0] >= generator.samples or (generator.samples - batch_nums[0]) < batch_size:
          break

      if verbose:
        print("Epoch {0} completed in {1}s with average loss of {2}".format(epoch, round(time() - start),
                                                                            np.average(np.array(self.losses))))
        self.losses = []

      if save_path is not None:
        full_save_path = save_path + "/epoch{0}.png".format(epoch + 1)
        keras.preprocessing.image.save_img(full_save_path, self.run_nst(content_example))

    return self.run_nst(content_example)

  # TESTING
  def run_nst(self, content: Union[Image.Image, np.ndarray]) -> np.ndarray:
    """
    Run fast NST on content image.

    :param content: image (as PIL image or numpy array) on which to run NST.
    :return: final result as a numpy array.
    """
    if isinstance(content, Image.Image):
      content = np.array(content)
    content = content.astype(np.float32) # converting image to proper dtype
    try:
      return np.squeeze(self.img_transform_net.k_model.predict(content)).astype(np.uint8)
    except ValueError:
      return np.squeeze(self.img_transform_net.k_model.predict(np.expand_dims(content, axis = 0))).astype(np.uint8)

  # CUSTOM LOSS
  @staticmethod
  def dummy_loss(y_true, y_pred) -> tf.Variable:
    """
    Dummy loss function. Loss is not optimized; instead, regularizers are used.

    :param y_true: true label.
    :param y_pred: network prediction.
    :return: keras variable with value = 0.0
    """
    return K.variable(0.0)

if __name__ == "__main__":
  # SlowNST.HYPERPARAMS["COEF_S"] = 0
  # SlowNST.HYPERPARAMS["COEF_V"] = 0

  from easyai.support.load import load_imgs
  style = load_imgs("https://drive.google.com/uc?export=download&id=18MpTOAt40ngCRpX1xcckwxUXNhOiBemJ")

  test = FastNST()
  test.train(style, batch_size = 2)