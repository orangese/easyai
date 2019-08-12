
"""

"easyai.support.examples.py"

Program that implements easyai.support.datasets and easyai.core in examples like MNIST.

"""

import os
import random
from support.datasets import *
from applications import *

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
    Error_Handling.suppress_tf_warnings()

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
    Error_Handling.suppress_tf_warnings()

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
    Error_Handling.suppress_tf_warnings()

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
  def neural_style_transfer(path, content_image = None, style_image = None):
    """
    Neural style transfer with art and photographs.

    :param path: path to easyai project.
    :param content_image: name of content image. Default is a random image from built-in datasets.
    :param style_image: name of style image. Default is a random image from built-in datasets.
    :return: trained Neural_Style_Transfer object.
    """
    Error_Handling.suppress_tf_warnings()

    if content_image is None or style_image is None:
      content_path = path + "/support/raw_datasets/neural_style_transfer/content/"
      content_path += random.choice(os.listdir(content_path))

      style_path = path + "/support/raw_datasets/neural_style_transfer/style/"
      style_path += random.choice(os.listdir(style_path))
    else:
      content_path = path + "/support/raw_datasets/neural_style_transfer/content/" + content_image
      style_path = path + "/support/raw_datasets/neural_style_transfer/style/" + style_image

    content_name = [char for char in content_path.split("/")][-1]
    style_name = [char for char in style_path.split("/")][-1]
    print ("Using content image \"{0}\" and style image \"{1}\"".format(content_name, style_name))

    model = Neural_Style_Transfer()

    final_img = model.train(content_path, style_path, epochs = 100, init_noise = 0.8)

    model.display_img(final_img, "Final result")

Art.neural_style_transfer("/home/ryan/PycharmProjects/easyai", "flower.jpg", "the_scream_munch.jpg")

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
    Error_Handling.suppress_tf_warnings()

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

# filename = "test.png"
# Unsupported.draw("test.png")
# activation = Unsupported.getActivation(filename)
# Unsupported.display_image(activation)