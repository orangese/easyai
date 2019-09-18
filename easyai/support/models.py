
"""

"easyai.support.models.py"

Holds links to pre-trained easyai models stored on Google Drive.

"""

# TODO: implement firebase (use existing code structure)

import random

from easyai.core import *

class TrainedModelInterface(StaticInterface):
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

class FastNSTModels(TrainedModelInterface):
  """
  Contains links to pretrained fast NST models as well as model loaders.
  """

  models = {} # models will be a dictionary mapping style name to h5 file or link to h5 file
  # TODO: created pretrained fast NST net and write load_net function

  @staticmethod
  def load_net(net_name: str) -> AbstractNetwork:
    """
    Loads pretrained fast NST model.

    :param net_name: name of model to be loaded.
    :return: pretrained model as a AbstractNetwork.
    """
    raise NotImplementedError()

  @staticmethod
  def random_net() -> AbstractNetwork:
    """
    Loads random pretrained fast NST model.

    :return: random pretrained model.
    """
    return FastNSTModels.load_net(random.choice(list(FastNSTModels.models.keys()))) # should not be changed