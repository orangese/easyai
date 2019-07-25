
"""

"easyai.support.datasets.py"

Program that implements easyai.core and provides wrapper for keras data loaders.

"""

import keras as K

def load_mnist():
  """loads MNIST data for use by a keras Sequential object"""
  (x_train, y_train), (x_test, y_test) = K.datasets.mnist.load_data()

  x_train = (x_train.reshape(x_train.shape[0], -1) / 255).astype("float32")
  y_train = K.utils.to_categorical(y_train, 10).astype("float32")

  x_test = (x_test.reshape(x_test.shape[0], -1) / 255).astype("float32")
  y_test = K.utils.to_categorical(y_test, 10).astype("float32")

  return (x_train, y_train), (x_test, y_test)