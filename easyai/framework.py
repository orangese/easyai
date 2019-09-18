
"""

"easyai.framework.py"

Framework for other files (abstract classes, interfaces, etc.).

"""

from typing import Union

# FRAMEWORK
class StaticInterface(object):
  """
  Static interface for other programs. An object of this class cannot be created.
  """

  def __init__(self, *args, **kwargs):
    """
    As StaticInterface objects should not be created, __init__ throws a NotImplementedError.

    :raises NotImplementedError
    """
    raise NotImplementedError("class is static")


class NetworkInterface(object):
  """Interface for all network-like classes."""

  HYPERPARAMS = {} # to be filled by child classes

  def __init__(self, *args, **kwargs):
    """
    Initialization for interfaces should not be used.

    :raises NotImplementedError: class is an interface
    """
    self.k_model = None  # to be explicit, every NetworkInterface should have a k_model attribute
    raise NotImplementedError("class is an interface")

  def train_init(self, *args, **kwargs):
    """
    Creates keras mask.

    :raises NotImplementedError: class is an interface
    """
    raise NotImplementedError("class is an interface")

  def train(self, *args, **kwargs):
    """
    Trains network.

    :raises NotImplementedError: class is an interface
    """
    raise NotImplementedError("class is an interface")

  # PRE-IMPLEMENTED FUNCTIONS
  def get_hps(self, *hyperparams: str) -> Union[str, tuple, float]:
    """
    Fetches hyperparameters from a NetworkInterface class. Merely a syntax simplification tool.

    :param hyperparams: any number of hyperparameters to fetch.
    :return: feched hyperparameters.
    """
    fetched = tuple(self.HYPERPARAMS[hp.upper()] for hp in hyperparams)
    return fetched[0] if len(fetched) == 1 else fetched

class AbstractLayer(object):
  """
  Abstract class that acts as the base for all layer classes. Should not be implemented.
  """

  def __init__(self, num_neurons: int, actv: str, *args, **kwargs):
    """
    As AbstractLayer objects should not be created, __init__ throws a NotImplementedError.

    :raises NotImplementedError
    """
    self.num_neurons = num_neurons
    self.actv = actv
    self.prev = None
    self.k_model = None
    raise NotImplementedError("class is abstract")

  def __str__(self):
    try:
      return "{0} {1} layer: {2} neurons".format(self.__class__.__name__, self.actv, self.num_neurons)
    except AttributeError:
      return "{0} layer: {1} neurons".format(self.__class__.__name__, self.num_neurons)

  def __repr__(self):
    return self.__str__()

  def train_init(self, *args, **kwargs):
    """
     Creates keras mask. This mask will be used for training and all computations.

    :raises AssertionError if self.prev is not initialized.
    """
    assert self.prev, "self.prev must be initialized"
    raise NotImplementedError("cannot be implemented. How did you even call this function?")
