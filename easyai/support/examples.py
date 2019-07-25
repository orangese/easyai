
"""

"easyai.support.examples.py"

Program that implements easyai.support.datasets and easyai.core in examples like MNIST.

"""

from easyai.core import *
from easyai.support.datasets import *

class MNIST(Static_Interface):

  @staticmethod
  def mlp():
    """MNIST multi-layer perceptron network. One sigmoidal hidden layer of 100 neurons and MSE cost"""
    mlp = NN([Input(784), Dense(100), Dense(10)])
    (x_train, y_train), (x_test, y_test) = load_mnist()
    print (mlp.summary())
    mlp.train(x_train, y_train)
    mlp.evaluate(x_test, y_test)

  @staticmethod
  def cnn():
    """MNIST convolutional network."""
