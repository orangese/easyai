
"""

"easyai.core.py"

Wrapper program for easy use of keras. Targeted towards those with little to no experience with Python and
machine learning.

"""

import keras as K
from time import time

#Support for other programs
class Static_Interface(object):

  def __init__(self):
    raise ValueError("class is static")

#NN classes
class Abstract_Layer(object):

  def __init__(self, num_neurons, actv):
    self.num_neurons = num_neurons
    self.actv = actv
    raise ValueError("abstract class should not be implemented")

  def __str__(self):
    try: return "{0} {1} layer: {2} neurons".format(self.__class__.__name__, self.actv, self.num_neurons)
    except AttributeError: return "{0} layer: {1} neurons".format(self.__class__.__name__, self.num_neurons)

  def __repr__(self):
    return self.__str__()

class Input(Abstract_Layer):
  """input layer, no keras mask"""

  def __init__(self, num_neurons):
    self.num_neurons = num_neurons

class Dense(Abstract_Layer):
  """wrapper for keras' Dense class"""

  def __init__(self, num_neurons, actv = "sigmoid", prev = None):
    self.num_neurons = num_neurons
    self.actv = actv
    self.prev = prev
    self.k_mask = None

  def update_mask(self):
    self.k_mask = K.layers.Dense(units = self.num_neurons, input_shape = (self.prev.num_neurons, ),
                                 activation = self.actv)

class NN(object):
  """uses wrapper classes to create a functional network-- wrapper for Sequential"""

  def __init__(self, layers = [], cost = "mse"):
    self.layers = tuple(layers)
    self.cost = cost
    self.link_layers()
    self.is_trained = False

  def __setattr__(self, name, value):
    """tweaked so that only the layer/model attributes can be changed"""
    if hasattr(self, name) and not (name in "k_model, k_layers, layer, is_trained"):
      raise ValueError("cannot modify attribute {0}".format(name))
    super().__setattr__(name, value)
    if name == "layers": self.link_layers()

  def link_layers(self):
    """links wrapper objects and creates a wrapper keras Sequential object"""
    self.k_layers = []
    for prev, layer in zip(self.layers, self.layers[1:]):
      layer.prev = prev
      layer.update_mask()
      self.k_layers.append(layer.k_mask)
    self.k_model = K.Sequential(self.k_layers)

  def add_layer(self, layer, position = None):
    """adds a layer"""
    new_layers = list(self.layers)
    if not position: position = len(self.layers)
    new_layers.insert(position, layer)
    self.layers = new_layers

  def rm_layer(self, position = None, layer = None):
    """removes a layer"""
    assert position or layer, "position or layer arguments must be provided"
    new_layers = list(self.layers)
    if layer: new_layers.remove(layer)
    elif position: del new_layers[position]
    self.layers = new_layers

  def train(self, x, y, lr = 3.0, epochs = 1, batch_size = 10):
    """trains and compiles k_model-- simplified version of keras' compile and fit"""
    optimizer = K.optimizers.SGD(lr = lr)
    metrics = ["categorical_accuracy"] if self.layers[-1].num_neurons != 1 else ["binary_accuracy"]
    self.k_model.compile(optimizer = optimizer, loss = self.cost, metrics = metrics)
    self.k_model.fit(x, y, epochs = epochs, batch_size = batch_size, validation_split = 0.2, verbose = 2)
    self.is_trained = True

  def evaluate(self, x, y, verbose = True):
    """wrapper for Sequential evaluate"""
    start = time()
    evaluation = self.k_model.evaluate(x, y, verbose = 2)
    result = "Test evaluation\n - {0}s".format(round(time() - start))
    for name, evaluation in zip(self.k_model.metrics_names, evaluation):
      result += (" - test_{0}: {1}".format(name, round(evaluation, 4)))
    if verbose: print (result)
    return evaluation

  def predict(self, x):
    """wrapper for Sequential predict"""
    return self.k_model.predict(x)

  def summary(self, advanced = False):
    """beginner-oriented version of keras' model.summary()"""
    result = "Layers: \n"
    for layer in self.layers:
      result += "  {0}. {1}\n".format(self.layers.index(layer) + 1, layer)
    result += "Trained: {0}\n".format(self.is_trained)
    if advanced and self.is_trained:
      result += "Advanced: \n"
      result += "  1. Cost function: {0}\n".format(self.cost)
    return result

if __name__ == "__main__":
  test = NN([Input(784), Dense(100), Dense(10)])
  print (test.summary(True))