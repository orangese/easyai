
"""

"easyai.support.load.py"

Loads new, user-supplied data.

"""

from PIL import Image
from glob import glob
from typing import Tuple
from easyai.core import *

def load_nst_images(content_path: str, style_path: str) -> Tuple[Image.Image, Image.Image]:
  """
  Loads new images for neural style transfer.

  :param content_path: path to content image.
  :param style_path: path to style image.
  :return: tuple of images as numpy arrays.
  """
  get_img = lambda path: Image.open(glob(path))
  return get_img(content_path), get_img(style_path)