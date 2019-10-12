"""

"easyai.core.py"

Contains core functionality for easyai: the NN class.

"""

# IMPORTS
from time import time

import keras
import numpy as np

from easyai.framework import *


# NEURAL NETWORK IMPLEMENTATION
class NN(ABNetwork):
    """
    Uses easyai layer objects to create a functional keras model.
    """

    def __init__(self, *layers, loss: str = "binary_crossentropy"):
        """
        Initializes NN (neural network) object.

        :param layers: layers of the network. Expects easyai core layer objects.
        :param loss: loss used by the neural network. Default is categorical_crossentropy.
        """
        if layers is None:
            layers = []
        self.layers = layers
        self.link_layers()

        self.loss = loss
        self.is_trained = False

    def link_layers(self):
        """ Links wrapper objects and creates a wrapper keras Sequential object."""
        self.k_layers = []
        for prev, layer in zip(self.layers, self.layers[1:]):
            layer.prev = prev
            layer.train_init()
            if isinstance(layer.k_model, list):
                for mask in layer.k_model:
                    self.k_layers.append(mask)
            else:
                self.k_layers.append(layer.k_model)
        self.k_model = keras.Sequential(layers=[keras.layers.InputLayer(self.layers[0].output_shape), *self.k_layers],
                                        name="k_model")

    def add_layer(self, layer: ABLayer, position: int = None):
        """Adds a layer and creates a new keras object.

        :param layer: layer to be added. Should be instance of easyai core layer classes.
        :param position: position at which to insert new layer. Uses Python's list "insert".
        """
        new_layers = list(self.layers)
        if not position:
            position = len(self.layers)
        new_layers.insert(position, layer)
        self.__init__(new_layers)

    def rm_layer(self, position: int = None, layer: ABLayer = None):
        """Removes a layer and creates a new keras object.

        :param position: position at which layer should be removed. Recommended instead of `layer`.
        :param layer: layer that should be removed. Not recommended due to possible duplicate layers.
        """
        assert position or layer, "position or layer arguments must be provided"
        new_layers = list(self.layers)
        if layer:
            new_layers.remove(layer)
        elif position:
            del new_layers[position]
        self.__init__(new_layers)

    def train(self, x: np.ndarray, y: np.ndarray, lr: float = 0.1, epochs: int = 1, batch_size: int = 10):
        """Trains and compiles this NN object. Only SGD is used.

        :param x: input data. For example, if classifying an image, `x` would the pixel vectors.
        :param y: labels. For example, if classifying an image, `y` would be the image labels.
               The `y` data should be comprised of one-hot encodings.
        :param lr:  learning rate used in SGD. Default is 3.0.
        :param epochs: number of epochs. Default is 1.
        :param batch_size:  minibatch size. Default is 10.
        """
        optimizer = keras.optimizers.SGD(lr=lr)
        metrics = ["categorical_accuracy"] if self.layers[-1].num_neurons > 2 else ["binary_accuracy"]
        # fixes weird keras feature in which "accuracy" metric causes unexpected results if using "binary_crossentropy"
        self.k_model.compile(optimizer=optimizer, loss=self.loss, metrics=metrics)
        print("Training with stochastic gradient descent in a {0}-D space. During epoch (training step), {1} "
              "training examples and their corresponding labels will be used to minimize loss.".format(
            self.k_model.count_params(), len(x)))
        self.k_model.fit(x, y, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=2)
        self.is_trained = True

    def evaluate(self, x: np.ndarray, y: np.ndarray, verbose: bool = True) -> list:
        """Evaluates this NN object using test data.

        :param x: inputs. See train documentation for more information.
        :param y: labels. See train documentation for more information.
        :param verbose: if true, this function gives more information about the evaluation process.
        :return: evaluation.
        """
        start = time()
        evaluation = self.k_model.evaluate(x, y, verbose=2)
        result = "Test evaluation\n - {0}s".format(round(time() - start))
        for name, evaluation in zip(self.k_model.metrics_names, evaluation):
            result += (" - test_{0}: {1}".format(name, round(evaluation, 4)))
        if verbose:
            print(result)
        return evaluation

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predicts the labels given input data.

        :param x: input data.
        :return: prediction.
        """
        print("Getting output encoding of x")
        return self.k_model.predict(x)

    def summary(self, advanced: bool = False) -> None:
        """
        Summary of model.

        :param advanced: if true, print advanced information.
        """
        alphabet = list(map(chr, range(97, 123)))
        result = "Network summary: \n"
        result += "  1. Layers: \n"
        for layer in self.layers:
            result += "    {0}. {1}\n".format(alphabet[self.layers.index(layer)], layer)
        result += "  2. Trained: {0}\n".format(self.is_trained)
        if advanced and self.is_trained:
            result += "Advanced: \n"
            result += "    1. Loss function: {0}\n".format(self.loss)
            result += "    2. Training algorithm: {0}\n".format("stochastic gradient descent")
        print(result)

    def save(self, filename: str):
        """
        Saves self as hdf5 file.

        :param filename: hdf5 file to save to.
        """
        self.k_model.save(filename)
