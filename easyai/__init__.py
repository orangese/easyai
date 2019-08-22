# ERROR HANDLING
def suppress_tf_warnings():
  """
  Suppresses tensorflow warnings. Does not work if tensorflow is outdated.
  """
  import os
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

  import warnings
  warnings.simplefilter(action = "ignore", category = FutureWarning)

  import tensorflow as tf
  try:
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
  except AttributeError:
    tf.logging.set_verbosity(tf.logging.ERROR)
  # compatible with tensorflow == 1.14.0 and tensorflow-gpu == 1.8.0

suppress_tf_warnings()

from . import applications
from . import core
from . import support

# Importable from root
from .core import StaticInterface
from .core import NN
