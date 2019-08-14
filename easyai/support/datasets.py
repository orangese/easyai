
"""

"easyai.support.datasets.py"

Program that implements easyai.core and provides wrapper for keras data loaders.

"""

from io import BytesIO

import pandas as pd
import requests
from PIL import Image

from easyai.core import *

# DATASETS
class Links(Static_Interface):

  class NST(Static_Interface):

    content = {"wwi_portrait": "https://drive.google.com/uc?export=download&id=1e9RdRiLOvMn9ZzqtRVdAX59Kf_NJaVAt",
               "girl_with_balloon_banksy":
                 "https://drive.google.com/uc?export=download&id=1FnhuYsx2sjSqI5qwZQxd7cmh8A5z74tj",
               "mountain_painting": "https://drive.google.com/uc?export=download&id=1_bmWZxTdE4cXvSiZZ_HQpzFsiytkZV5z",
               "mona_lisa_da_vinci": "https://drive.google.com/uc?export=download&id=1nfJ0-200MkjbuQEWsmKeO0C-byrZSQRF",
               "nyc": "https://drive.google.com/uc?export=download&id=1EZiItOnGA8LuetejrW5HYMzk6sGYTlqN",
               "golden_gate_bridge": "https://drive.google.com/uc?export=download&id=1XH5vHXQfdKs4XOhWj4DN31Npwg6PVykb",
               "flower": "https://drive.google.com/uc?export=download&id=1baUPRVEh82szkeQ0601O4gpj_mNNbpbe",
               "einstein": "https://drive.google.com/uc?export=download&id=1FKNW8QYoAQ6y7V6sPss-YsmmOMAUUWZS",
               "dog": "https://drive.google.com/uc?export=download&id=1f8aNYIJqBrMqYEYYSMPOmzgVmS_9C969",
               "arno_river_buildings":
                 "https://drive.google.com/uc?export=download&id=1ynBPvxuUOUo4upgTNE4KCab4F26leSLi"
               }

    style = {"wheatfields_with_crows_van_gogh":
               "https://drive.google.com/uc?export=download&id=1YH2IE42KcwxOzu3C_xs6F_cLT4VJqpAW",
             "water_lilies_monet": "https://drive.google.com/uc?export=download&id=1I8hkXBpwEq_tuOsnDcEWt8naMCLnXiDZ",
             "the_scream_munch": "https://drive.google.com/uc?export=download&id=1kUybhXBJF8NfWm8A1-Ry7KksbG3jx8hb",
             "starry_night_van_gogh":
               "https://drive.google.com/uc?export=download&id=18MpTOAt40ngCRpX1xcckwxUXNhOiBemJ",
             "blue_green_stained_glass":
               "https://drive.google.com/uc?export=download&id=1LxjKiXAiPM1OFg2s9c-cbFEqIwHp2ffN",
             "little_village_duncanson":
               "https://drive.google.com/uc?export=download&id=1ys1wrH6Glhfvr28kvgvLLwP5NoSL_0JS",
             "udnie_picabia": "https://drive.google.com/uc?export=download&id=1sFVDVyWgNle7La5sgz9vks42uNAUYIVv",
             "femme_nue_assise_picasso":
               "https://drive.google.com/uc?export=download&id=1IACcGYMY0kYS9K5Ng-mTPQc3USCEu7Or",
             "cubist_karthik": "https://drive.google.com/uc?export=download&id=19AKRyeJxNHWg8w50xE0VWyI33r-VMZTe",
             "bottle_and_fishes_braque":
               "https://drive.google.com/uc?export=download&id=1JfBSd7CHIUIEKI7Zyl4XKX4Wy8W6X4e-",
             "great_wave_off_kanagawa_hokusai":
               "https://drive.google.com/uc?export=download&id=104dqfFU4z3ul-hGgbNYAvLUP4OuD2GBw"
             }

# DATA LOADERS
class Builtins(Static_Interface):
  """
  Data loaders (including preprocessing) for built-in keras datasets. These include: MNIST, Fashion-MNIST,
  CIFAR-10, CIFAR-100, Boston housing, IMDB, Reuters.
  """

  @staticmethod
  def load_mnist(version: str = "digits", mode: str = "mlp") -> tuple:
    """
    Loads MNIST or Fashion-MNIST data. These two datasets are combined into one method because they are so similar.

    :param version: either "digits" for regular MNIST or "fashion" for Fashion-MNIST.
    :param mode: either "mlp" or "conv".
    :return: two tuples: (x_train, y_train) and (x_test, y_test).
    """
    assert version == "digits" or version == "fashion", "only MNIST or Fashion-MNIST are available"

    if version == "digits":
      (x_train, y_train), (x_test, y_test) = K.datasets.mnist.load_data()
    else:
      (x_train, y_train), (x_test, y_test) = K.datasets.fashion_mnist.load_data()

    x_train = (x_train / 255).astype("float32")
    x_test = (x_test / 255).astype("float32")

    y_train = y_train.astype("float32")
    y_test = y_test.astype("float32")

    if mode == "mlp":
      x_train = x_train.reshape(x_train.shape[0], -1)
      x_test = x_test.reshape(x_test.shape[0], -1)
    elif mode == "conv":
      x_train.resize(*x_train.shape, 1)
      x_test.resize(*x_test.shape, 1)

    num_classes = 10
    y_train = K.utils.to_categorical(y_train, num_classes)
    y_test = K.utils.to_categorical(y_test, num_classes)

    return (x_train, y_train), (x_test, y_test)

  @staticmethod
  def load_cifar(version: int) -> tuple:
    """
    Loads CIFAR-10 or CIFAR-100 data.

    :param version: either 10 for CIFAR-10 or 100 for CIFAR-100
    :return: two tuples: (x_train, y_train) and (x_test, y_test).
    """
    assert version == 10 or version == 100, "only CIFAR-10 and CIFAR-100 are available"

    (x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()

    num_classes = version
    y_train = K.utils.to_categorical(y_train, num_classes)
    y_test = K.utils.to_categorical(y_test, num_classes)

    return (x_train, y_train), (x_test, y_test)

  @staticmethod
  def load_boston_housing():
    raise NotImplementedError()

  @staticmethod
  def load_imdb():
    raise NotImplementedError()

  @staticmethod
  def load_reuters():
    raise NotImplementedError()

class Extras(Static_Interface):
  """
  Data loaders (including pre-processing) for other datasets, including user-importd ones and native easyai
  datasets. These include: LendingClub (credit rating) dataset and neural style transfer images.
  """

  @staticmethod
  def load_nst_dataset() -> dict:
    """
    Method to load images used as examples for neural style transfer.

    :return: a dictionary of two dictionaries. The first dictionary contains the names of the content images mapped to
             the content images (Image objects), and the second contains the names of the style images mapped to the
              style images.
    """
    # helper variables and lambdas
    urls_to_imgs = lambda urls: [Image.open(BytesIO(requests.get(url).content)) for url in urls]
    dict_from_lists = lambda keys, values: dict(zip(keys, values))
    content_dict = Links.NST.content
    style_dict = Links.NST.style

    images = {"content": None, "styles": None}

    #retrieving names of images and values and mapping them
    content_keys = list(content_dict.keys())
    content_imgs = urls_to_imgs(list(content_dict.values()))
    images["content"] = dict_from_lists(content_keys, content_imgs)

    style_keys = list(style_dict.keys())
    style_imgs = urls_to_imgs(list(style_dict.values()))
    images["style"] = dict_from_lists(style_keys, style_imgs)

    return images

  # BROKEN
  @staticmethod
  def load_lending_club() -> tuple:
    """
    Loads LendingClub credit rating dataset.

    :return: two tuples: (x_train, y_train) and (x_test, y_test).
    """
    globals_ = {}

    def sigmoid_normalize(raw_array, range_ = None):
      """Function that converts a list of values between any range to [0, 1]."""
      array = np.copy(raw_array).astype(np.float32)
      if range_ is None:
        range_ = (min(array), max(array))
      if range_ == (0, 1):
        return array
      #  Step 1: subtract minimum from everything
      array -= range_[0]
      #  Step 2: divide by range
      dist = abs(range_[0]) + abs(range_[1])
      array /= dist
      return np.nan_to_num(array)

    def convert_categorical(categoricals, range_):
      """Converts a list of categorical variables to an integer list, range = [0, 1]."""
      to_int = len(range_)
      fractions = np.array([i / (to_int - 1) for i in range(to_int)], dtype = np.float32)
      if isinstance(categoricals, str):
        return fractions[range_.index(categoricals)]
      else:
        return np.nan_to_num(np.array([fractions[range_.index(categorical)]
                                       for categorical in categoricals], dtype = np.float32))

    def to_int(n):
      """Turns every element in a list into an int."""
      if isinstance(n, list):
        fin = []
        for element in n:
          try:
            fin.append(int(element))
          except ValueError:
            fin.append(0)
        return np.nan_to_num(np.array(fin, dtype = np.float32))
      else:
        try:
          return int(n)
        except ValueError:
          return 0

    def strip(n):
      """Strips a list of strings of everything but digits and decimals."""
      if isinstance(n, str) or isinstance(n, float) or isinstance(n, int):
        return "".join(ch for ch in str(n) if str(ch).isdigit() or str(ch) == ".")
      else:
        return ["".join(ch for ch in str(s) if str(ch).isdigit() or str(ch) == ".") for s in n]

    def vectorize(value, range_):
      """Takes a value and vectorizes it (one-hot encoder)."""
      result = np.zeros((len(range_), ))
      result[range_.index(value)] = 1.0
      return result

    def get_range(data):
      """Gets the ranges for a list."""
      ranges = []
      for element in data:
        if element in ranges:
          continue
        else:
          ranges.append(element)
      return ranges

    def unison_shuffle(a, b):
      """Returns unison shuffled copies of two np.arrays."""
      p = np.random.permutation(len(a))
      return a[p], b[p]

    def load_file(filestream):
      """Reads a specific excel file and prepares it for data processing."""
      data = pd.ExcelFile(filestream, engine = "xlrd")
      del data["loan_status"]
      del data["funded_amnt"]
      del data["sub_grade"]
      del data["funded_amnt_inv"]
      del data["inq_last_6mths"]
      del data["open_acc"]
      del data["revol_bal"]
      del data["revol_util"]
      del data["total_acc"]
      del data["total_pymnt"]
      del data["total_pymnt_inv"]
      del data["total_rec_prncp"]
      del data["total_rec_int"]
      del data["total_rec_late_fee"]
      del data["recoveries"]
      del data["collection_recovery_fee"]
      del data["last_pymnt_amnt"]

      labels = []
      range_ = get_range(data["grade"])
      for label in np.asarray(data["grade"]):
        labels.append(vectorize(label, range_))
      del data["grade"]

      for feature in data.columns:
        if feature == "term" or feature == "emp_length":
          globals_[feature] = range_
          data[feature] = to_int(strip(data[feature]))
        try:
          globals_[feature] = (min(data[feature]), max(data[feature]))
          data[feature] = sigmoid_normalize(data[feature])
        except ValueError:
          range_ = get_range(data[feature])
          globals_[feature] = [r.lower() for r in range_]
          data[feature] = convert_categorical(data[feature], range_)

      return data.values.reshape(len(data.index), len(data.columns), 1), np.array(labels)

    def load_data(file, ratio = 0.8):
      """Data processer (essentially a wrapper for load_file). Ratio is the fraction of data that is training data."""
      inputs, labels = load_file(file)

      big_data = unison_shuffle(inputs.reshape(len(inputs), len(inputs[0])),
                                labels.reshape(len(labels), len(labels[0])))
      num_train = int(ratio * len(big_data[0]))

      return (big_data[0][:num_train], big_data[1][:num_train]), (big_data[0][num_train:], big_data[1][num_train:])

    return load_data(
      "https://drive.google.com/uc?export=download&id=1HYE7qIgAjdYve_sS1ixZH8TJTE5wB-sv")