"""

"easyai.framework.py"

Framework for other files (abstract classes, interfaces, etc.).

"""

import keras
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# FRAMEWORK
class Static(object):
    """
    Static interface for other programs. An object of this class cannot be created. It's for organizational purposes.
    """

    def __init__(self, *args, **kwargs):
        """
        As Static objects should not be created, __init__ throws a NotImplementedError.

        :raises NotImplementedError
        """
        raise NotImplementedError("class is static")


class ABNetwork(object):
    """Abstract parent for all network-like classes."""

    HYPERPARAMS = {}  # to be filled by child classes

    def __init__(self, *args, **kwargs):
        """
        Initialization for ABCs should not be used.

        :raises NotImplementedError: class is abstract
        """
        self.k_model = None  # to be explicit, every ABNetwork should have a k_model attribute
        raise NotImplementedError("class is abstract")

    def train_init(self, *args, **kwargs):
        """
        Creates keras mask.

        :raises NotImplementedError: class is abstract
        """
        raise NotImplementedError("class is abstract")

    def train(self, *args, **kwargs):
        """
        Trains network.

        :raises NotImplementedError: class is abstract
        """
        raise NotImplementedError("class is abstract")

    # PRE-IMPLEMENTED FUNCTIONS
    def get_hps(self, *hyperparams: str) -> any:
        """
        Fetches hyperparameters from a ABNetwork class. Merely a syntax simplification tool.

        :param hyperparams: any number of hyperparameters to fetch.
        :return: feched hyperparameters.
        """
        fetched = tuple(self.HYPERPARAMS[hp.upper()] for hp in hyperparams)
        return fetched[0] if len(fetched) == 1 else fetched


class ABImageNetwork(ABNetwork):

    def __init__(self):
        self.k_net_module = None
        self.num_cols, self.num_rows = None, None

    # IMAGE PROCESSING
    def preprocess(self, img: Image.Image, target_size=None) -> np.ndarray:
        """
        Preprocesses an image.

        :param img: image to preprocess.
        :param target_size: target size of the image. If is None, defaults to object attributes.
        :return: processed image.
        """
        if target_size is None:
            try:
                if self.num_cols is None:
                    width, height = img.size
                    self.num_cols = int(width * self.num_rows / height)
                target_size = reversed((self.num_rows, self.num_cols))
            except AttributeError:
                target_size = img.size
        img = img.resize(target_size)
        img = np.expand_dims(keras.preprocessing.image.img_to_array(img), axis=0)
        img = self.k_net_module.preprocess_input(img)
        return img

    @staticmethod
    def deprocess(img: np.ndarray, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError("deprocess is abstract")

    # DISPLAY
    @staticmethod
    def display_img(img: np.ndarray, title: str, target_shape: tuple = None, deprocess=None):
        """
        Displays image.

        :param img: image to be displayed.
        :param title: title of maptlotlib plot.
        :param target_shape: target shape.
        :param deprocess: whether or not to deprocess the image before displaying it.
        """
        if deprocess:
            img = deprocess(img, target_shape=target_shape)

        fig = plt.gcf()

        fig.canvas.set_window_title("Training...")
        fig.suptitle(title)
        plt.axis("off")

        plt.imshow(img.reshape(target_shape))
        plt.pause(0.1)

        plt.show(block=False)


class ABLayer(object):
    """
    Abstract base class that acts as the base for all layer classes. Should not be implemented.
    """

    def __init__(self, num_neurons: int, actv: str, *args, **kwargs):
        """
        As ABLayer objects should not be created, __init__ throws a NotImplementedError.

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

    def train_init(self):
        """
         Creates keras mask. This mask will be used for training and all computations.

        :raises AssertionError if self.prev is not initialized.
        """
        assert self.prev, "self.prev must be initialized"
        raise NotImplementedError("cannot be implemented. How did you even call this function?")
