
"""

"easyai.support.datasets.py"

Program that implements easyai.core and provides wrapper for keras data loaders.

"""

import keras as K
from easyai.core import Static_Interface

class Builtins(Static_Interface):

  @staticmethod
  def load_mnist(mode = "mlp"):
    """loads MNIST data for use by a keras Sequential object"""
    (x_train, y_train), (x_test, y_test) = K.datasets.mnist.load_data()

    x_train = (x_train / 255).astype("float32")
    x_test = (x_test / 255).astype("float32")

    y_train = y_train.astype("float32")
    y_test = y_test.astype("float32")

    if mode == "mlp":
      x_train = x_train.reshape(x_train.shape[0], -1)
      x_test = x_test.reshape(x_test.shape[0], -1)
    elif mode == "conv":
      if K.backend.image_data_format() == "channels_first":
        x_train.resize(x_train.shape[0], 1, *x_train.shape[1:])
        x_test.resize(x_test.shape[0], 1, *x_test.shape[1:])
      else:
        x_train.resize(*x_train.shape, 1)
        x_test.resize(*x_test.shape, 1)

    y_train = K.utils.to_categorical(y_train, 10)
    y_test = K.utils.to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)
