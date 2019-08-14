
"""

"easyai.support.examples.py"

Program that implements easyai.support.datasets and easyai.core in examples like MNIST.

"""

import random

from easyai.applications import *
from easyai.support.datasets import *

# CLASSES
class MNIST(Static_Interface):
  """
  Contains examples using MNIST and Fashion-MNIST datasets.
  """

  @staticmethod
  def mlp(version: str = "digits") -> NN:
    """
    MNIST multi-layer perceptron network.

    :param version: "digits" for MNIST dataset or "fashion" for Fashion-MNIST dataset.
    :return: trained NN model.
    """
    suppress_tf_warnings()

    (x_train, y_train), (x_test, y_test) = Builtins.load_mnist(version = version, mode = "mlp")
    print ("Loaded MNIST data\n")

    mlp = NN([Input(784), Dense(100), Dense(10, actv = "softmax")], cost = "categorical_crossentropy")
    print (mlp.summary())

    mlp.train(x_train, y_train, lr = 3.0, epochs = 1)
    mlp.evaluate(x_test, y_test)

    return mlp

  @staticmethod
  def cnn(version: str = "digits") -> NN:
    """
    MNIST convolutional network.

    :param version: "digits" for MNIST dataset or "fashion" for Fashion-MNIST dataset.
    :return: trained NN model.
    """
    suppress_tf_warnings()

    (x_train, y_train), (x_test, y_test) = Builtins.load_mnist(version = version, mode = "conv")
    print ("Loaded MNIST data")

    conv_nn = NN([Input((28, 28)), Conv((5, 5), 20), Pooling((2, 2)), Dense(100), Dense(10, actv = "softmax")],
                 cost = "categorical_crossentropy")
    print (conv_nn.summary())

    conv_nn.train(x_train, y_train, lr = 0.1, epochs = 60)
    conv_nn.evaluate(x_test, y_test)

    return conv_nn

class Lending_Club(Static_Interface):
  """
  Contains examples using LendingClub credit rating dataset.
  """

  @staticmethod
  def mlp() -> NN:
    """
    LendingClub MLP.

    :return: trained NN model.
    """
    suppress_tf_warnings()

    (x_train, y_train), (x_test, y_test) = Extras.load_lending_club()
    print("Loaded LendingClub data")

    mlp = NN([Input(9), Dense(200, actv = "relu"), Dense(200, actv = "relu"), Dense(7, actv = "softmax")],
             cost = "categorical_crossentropy")
    print (mlp.summary())

    mlp.train(x_train, y_train, lr = 0.01, epochs = 50)
    mlp.evaluate(x_test, y_test)

    return mlp

class Art(Static_Interface):
  """
  Art generated with AI.
  """

  @staticmethod
  def neural_style_transfer(content = None, style = None, save_path = None):
    """
    Neural style transfer with art and photographs.

    :param content: name of content image from dataset. Default is a random image from built-in datasets.
    :param style: name of style image from dataset. Default is a random image from built-in datasets.
    :param save_path: path to which to save final result. Default is None.
    :return: trained Neural_Style_Transfer object.
    """
    def get_img(img_name, type_, images):
      if img_name is None:
        return random.choice(list(images[type_].items()))
      else:
        try:
          return img_name, images[type_][img_name]
        except KeyError:
          raise ValueError("supported {0} images are {1}".format(type_, list(images[type_].keys())))

    suppress_tf_warnings()

    images = Extras.load_nst_dataset()
    print ("Loaded NST images")

    content_name, content_img = get_img(content, "content", images)
    style_name, style_img = get_img(style, "style", images)

    print ("Using content image \"{0}\" and style image \"{1}\"".format(content_name, style_name))

    model = Neural_Style_Transfer("vgg19")

    final_img = model.train(content_img, style_img, epochs = 25, init_noise = 0.6)

    model.display_img(final_img, "Final result")

    if save_path is not None:
      full_save_path = save_path + "/{0}_{1}.jpg".format(content_name, style_name)
      K.preprocessing.image.save_img(full_save_path, final_img)
      print ("Saved image at \"{0}\"".format(full_save_path))

#--BELOW NOT SUPPORTED--
from tkinter import *
from PIL import Image, ImageDraw
import PIL
import numpy as np

import keras as K

class Unsupported(Static_Interface):

  @staticmethod
  def draw(fileName):

    width = 200
    height = 200
    white = (255, 255, 255)

    def save():
      image1.save(fileName)

    def drawIm(event):
      x1, y1 = (event.x - 5), (event.y - 5)
      x2, y2 = (event.x + 5), (event.y + 5)
      cv.create_oval(x1, y1, x2, y2, width=5, fill="black")
      draw.line(((x2, y2), (x1, y1)), fill="black", width=10)

    root = Tk()
    cv = Canvas(root, width=width, height=height, bg="white")
    cv.pack()
    image1 = PIL.Image.new("RGB", (width, height), white)
    draw = ImageDraw.Draw(image1)
    cv.bind("<B1-Motion>", drawIm)
    button = Button(text="save", command=save)
    button.pack()
    root.mainloop()

  @staticmethod
  def getActivation(fileName):
    img = Image.open(fileName)
    img = img.resize((28, 28))
    img = np.take(np.asarray(img), [0], axis=2).reshape(28, 28)
    return np.abs(img - 255)

  @staticmethod
  def mnist_demo(h5_model: str = None):

    def get_user_draw():
      raise NotImplementedError()

    def write(text: any):
      raise NotImplementedError()

    if h5_model is None:
      nn = MNIST.mlp()
      h5_model = "h5_model.h5"
      nn.save(h5_model)

    model = K.models.load_model("h5_model")
    write("Test evalaution: {0}%".format(model.evaluate()[-1] * 100))

    digit = get_user_draw()

    digits = np.arange(model.layers[-1].units + 1)
    pred = digits[np.argmax(model.predict(digit))]

    write("Network prediction: \"{0}\"".format(pred))

  @staticmethod
  def style_transfer(img_path):
    """
    Neural style transfer to make a content image (C) look a certain style (S). Uses VGG as base network.

    :return: trained NN model.
    """
    suppress_tf_warnings()

  @staticmethod
  def display_image(pixels, label=None):
    # function that displays an image using matplotlib-- not really necessary for the digit classifier
    import matplotlib.pyplot as plt
    figure = plt.gcf()
    figure.canvas.set_window_title("Number display")

    if label: plt.title("Label: \"{label}\"".format(label=label))
    else: plt.title("No label")

    plt.imshow(pixels, cmap="gray")
    plt.show()

if __name__ == "__main__":
  # filename = "test.png"
  # Unsupported.draw("test.png")
  # activation = Unsupported.getActivation(filename)
  # Unsupported.display_image(activation)

  styles = list(Links.NST.style.keys())
  contents = list(Links.NST.content.keys())

  for style in styles:
    for content in contents:
      Art.neural_style_transfer(content, style, save_path = "/home/ryan/Documents")