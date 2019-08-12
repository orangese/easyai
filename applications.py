
"""

"easyai.applications.py"

Applications of core layers and networks in larger, more real-world-based algorithms. Example: neural style transfer.

"""

from scipy.optimize import fmin_l_bfgs_b
from core import *

# SUPPORT
class Evaluator(object):

  def __init__(self, obj):
    self.obj = obj
    assert hasattr(obj, "loss_and_grads"), "obj must have loss_and_grads function"
    self.reset()

  def f_loss(self, img):
    loss, grads = self.obj.loss_and_grads(img)
    self.loss = loss
    self.grads = grads
    return self.loss

  def f_grads(self, img):
    grads = np.copy(self.grads)
    self.reset()
    return grads

  def reset(self):
    self.loss = None
    self.grads = None

# NEURAL NETWORK APPLICATION
class Neural_Style_Transfer(object):

  HYPERPARAMS = {"CONTENT_LAYER":
                   {
                     "vgg19": "block5_conv2",
                     "NN": ""},
                 "STYLE_LAYERS":
                   {
                     "vgg19": ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"],
                     "NN": ""
                   }
                 }

  def __init__(self, net: Union[str, NN] = None, num_rows: int = 400):
    self.net = net if not (net is None) else "vgg19"
    if isinstance(self.net, str):
      assert self.net == "vgg19", "only the vgg19 pre-trained model is supported"
    self.generated = None
    self.img_tensor = None

    self.num_rows = num_rows
    self.num_cols = None

  def train_init(self, content_path: str, style_path: str, verbose: bool):
    self.image_init(content_path, style_path)

    self.img_tensor = K.backend.concatenate([self.content, self.style, self.generated], axis = 0)
    self.img_order = ["content", "style", "generated"]

    self.model_init()

    if verbose:
      print("Loaded {0} model".format(self.net))

    self.img = self.preprocess(content_path)

  def model_init(self):
    if self.net == "vgg19":
      self.k_model = K.applications.vgg19.VGG19(input_tensor = self.img_tensor, weights = "imagenet",
                                                include_top = False)
    elif isinstance(self.net, NN):
      self.k_model = self.net.k_model
      self.net = "NN"

    self.k_model.trainable = False
    self.outputs = dict([(layer.name, layer.output) for layer in self.k_model.layers])

  def image_init(self, content_path: str, style_path: str):
    self.content = K.backend.variable(self.preprocess(content_path))
    self.style = K.backend.variable(self.preprocess(style_path))

    self.gen_shape = (self.num_rows, self.num_cols, 3)
    print ("Generated image shape: {0}".format(self.gen_shape))

    self.generated = K.backend.placeholder(shape = (1, *self.gen_shape))

  def func_init(self, content_layer: str, style_layers: str):
    cost = self.cost_tensor(content_layer, style_layers)
    grads = K.backend.gradients(cost, self.generated)
    outputs = [cost, *grads] if isinstance(grads, (list, tuple)) else [cost, grads]

    self.model_func = K.backend.function([self.generated], outputs)
    self.evaluator = Evaluator(self)

  def train(self, content_path: str, style_path: str, epochs: int = 1, verbose: bool = True):
    content_layer, style_layers = self.get_hyperparams("content_layer", "style_layers")

    self.train_init(content_path, style_path, verbose = verbose)
    self.func_init(content_layer, style_layers)

    for epoch in range(epochs):
      start = time()
      self.img, cost, throwaway = fmin_l_bfgs_b(func = self.evaluator.f_loss, x0 = self.img.flatten(),
                                                fprime = self.evaluator.f_grads, maxfun = 20)
      if verbose:
        print("Epoch {0}/{1}".format(epoch + 1, epochs))
        print(" - {0}s - cost: {1}".format(round(time() - start), cost))

      K.preprocessing.image.save_img("/home/ryan/PycharmProjects/easyai/result.png", self.deprocess(self.img))

  # COST CALCULATIONS
  def loss_and_grads(self, img):
    img = img.reshape(self.generated.shape)
    outputs = self.model_func([img])
    cost = outputs[0]
    if len(outputs[1:]) == 1:
      grads = outputs[1].flatten().astype(np.float64)
    else:
      grads = np.array(outputs[1:]).flatten().astype(np.float64)
    return cost, grads

  def cost_tensor(self, content_layer, style_layers, coef_C = 1e-2, coef_S = 1e3):

    def content_cost(layer):
      layer_features = self.outputs[layer]

      content_actvs = layer_features[self.img_order.index("content"), :, :, :]
      generated_actvs = layer_features[self.img_order.index("generated"), :, :, :]

      return K.backend.sum(K.backend.square(generated_actvs - content_actvs))

    def layer_style_cost(a_G, a_S):

      def gram_matrix(a):
        a = K.backend.batch_flatten(K.backend.permute_dimensions(a, (2, 0, 1)))
        return K.backend.dot(a, K.backend.transpose(a))

      gram_s = gram_matrix(a_S)
      gram_g = gram_matrix(a_G)

      return K.backend.sum(K.backend.square(gram_s - gram_g))

    cost = K.backend.variable(0.0)

    cost += coef_C * content_cost(content_layer)

    for layer in style_layers:
      layer_features = self.outputs[layer]

      generated_actvs = layer_features[self.img_order.index("generated"), :, :, :]
      style_actvs = layer_features[self.img_order.index("style"), :, :, :]

      cost += layer_style_cost(generated_actvs, style_actvs) * (coef_S / len(style_layers)) / \
              (4.0 * (int(self.generated.shape[-1]) ** 2) * (self.num_rows * self.num_cols ** 2))

    return cost

  # IMAGE PROCESSING
  def preprocess(self, img, target_size = None):
    if target_size is None:
      if self.num_cols is None:
        width, height = K.preprocessing.image.load_img(img).size
        self.num_cols = int(width * self.num_rows / height)
      target_size = (self.num_rows, self.num_cols)
    if isinstance(img, str):
      img = K.preprocessing.image.load_img(img, target_size = target_size)
    img = np.expand_dims(K.preprocessing.image.img_to_array(img), axis = 0)
    if self.net == "vgg19":
      return K.applications.vgg19.preprocess_input(img)
    else:
      raise NotImplementedError("data normalization for other nets is not supported yet")

  def deprocess(self, img_):
    img = np.copy(img_).reshape(*self.generated.shape[1:])

    if self.net == "vgg19":
      img[:, :, 0] += 103.939
      img[:, :, 1] += 116.779
      img[:, :, 2] += 123.68

      img = img[:, :, ::-1]
      img = np.clip(img, 0, 255).astype('uint8')
    else:
      raise NotImplementedError("data normalization for other nets is not supported yet")

    return img

  # MISCELLANEOUS
  def get_hyperparams(self, *hyperparams):
    fetched = tuple(Neural_Style_Transfer.HYPERPARAMS[hp.upper()][self.net] for hp in hyperparams)
    return fetched[0] if len(fetched) == 1 else fetched

Error_Handling.suppress_tf_warnings()
net = Neural_Style_Transfer("vgg19")
net.train("/home/ryan/PycharmProjects/easyai/dog.jpg", "/home/ryan/PycharmProjects/easyai/picasso.jpg", epochs = 1000)