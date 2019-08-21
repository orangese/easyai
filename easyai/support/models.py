
"""

"easyai.support.models.py"

Holds links to pre-trained easyai models stored on Google Drive.

"""

import random

from easyai.core import *

class Trained_Model_Interface(Static_Interface):
  """
  Model for all model classes.
  """

  models = {} # dictionary containing links to pretrained models

  @staticmethod
  def load_net(net_name: str) -> keras.Model:
    raise NotImplementedError("abstract class methods should not be implemented")

  @staticmethod
  def random_net() -> keras.Model:
    raise NotImplementedError("abstract class methods should not be implemented")

class Fast_NST_Models(Trained_Model_Interface):
  """
  Contains links to pretrained fast NST models as well as model loaders.
  """

  models = {} # TODO: created pretrained fast NST net and write load_net function

  @staticmethod
  def load_net(net_name: str) -> Network_Interface:
    """
    Loads pretrained fast NST model.

    :param net_name: name of model to be loaded.
    :return: pretrained model.
    """
    raise NotImplementedError()

  @staticmethod
  def random_net() -> Network_Interface:
    """
    Loads random pretrained fast NST model.

    :return: random pretrained model.
    """
    return Fast_NST_Models.load_net(random.choice(list(Fast_NST_Models.models.keys())))