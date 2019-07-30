
"""

"easyai.support.examples.py"

Program that implements easyai.support.datasets and easyai.core in examples like MNIST.

"""

from tkinter import *
from PIL import Image, ImageDraw
import numpy as np
from easyai.core import *
from easyai.support.datasets import Builtins

class MNIST(Static_Interface):

  @staticmethod
  def mlp():
    """MNIST multi-layer perceptron network."""
    Error_Handling.suppress_tf_warnings()

    (x_train, y_train), (x_test, y_test) = Builtins.load_mnist(mode = "mlp")

    mlp = NN([Input(784), Dense(100), Dense(10)])
    print (mlp.summary())

    mlp.train(x_train, y_train, lr = 3.0)
    mlp.evaluate(x_test, y_test)

  @staticmethod
  def cnn():
    """MNIST convolutional network."""
    Error_Handling.suppress_tf_warnings()
    (x_train, y_train), (x_test, y_test) = Builtins.load_mnist(mode = "conv")

    mlp = NN([Input((28, 28)), Conv((5, 5), 20), Pooling((2, 2)), Dense(100), Dense(10)])
    print (mlp.summary())

    mlp.train(x_train, y_train, lr = 0.1)
    mlp.evaluate(x_test, y_test)

  """--BELOW NOT SUPPORTED--"""
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