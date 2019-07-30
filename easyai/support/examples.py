
"""

"easyai.support.examples.py"

Program that implements easyai.support.datasets and easyai.core in examples like MNIST.

"""

from easyai.core import *
from easyai.support.datasets import *

# CLASSES
class MNIST(Static_Interface):
  """Contains examples using MNIST and Fashion-MNIST datasets."""

  @staticmethod
  def mlp(version = "digits"):
    """MNIST multi-layer perceptron network.

    :param version: "digits" for MNIST dataset or "fashion" for Fashion-MNIST dataset.
    """
    Error_Handling.suppress_tf_warnings()

    (x_train, y_train), (x_test, y_test) = Builtins.load_mnist(version = version, mode = "mlp")

    mlp = NN([Input(784), Dense(100), Dense(10, actv = "softmax")], cost = "categorical_crossentropy")
    print (mlp.summary())

    mlp.train(x_train, y_train, lr = 3.0, epochs = 60)
    mlp.evaluate(x_test, y_test)

  @staticmethod
  def cnn(version = "digits"):
    """MNIST convolutional network.

    :param version: "digits" for MNIST dataset or "fashion" for Fashion-MNIST dataset.
    """
    Error_Handling.suppress_tf_warnings()

    (x_train, y_train), (x_test, y_test) = Builtins.load_mnist(version = version, mode = "conv")

    conv_nn = NN([Input((28, 28)), Conv((5, 5), 20), Pooling((2, 2)), Dense(100), Dense(10, actv = "softmax")],
                 cost = "categorical_crossentropy")
    print (conv_nn.summary())

    conv_nn.train(x_train, y_train, lr = 0.1, epochs = 60)
    conv_nn.evaluate(x_test, y_test)

class Lending_Club(Static_Interface):
  """Contains examples using LendingClub credit rating dataset."""

  @staticmethod
  def mlp():
    """LendingClub MLP."""
    Error_Handling.suppress_tf_warnings()

    (x_train, y_train), (x_test, y_test) = Extras.load_lending_club()

    mlp = NN([Input(9), Dense(200, actv = "relu"), Dense(200, actv = "relu"), Dense(7, actv = "softmax")],
             cost = "categorical_crossentropy")
    print (mlp.summary())

    mlp.train(x_train, y_train, lr = 0.01, epochs = 50)
    mlp.evaluate(x_test, y_test)

#--BELOW NOT SUPPORTED--
from tkinter import *
from PIL import Image, ImageDraw
import PIL
import numpy as np

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

MNIST.cnn()