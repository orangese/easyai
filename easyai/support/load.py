
"""

"easyai.support.load.py"

Loads new, user-supplied data.

"""

from typing import Tuple

from PIL import Image

def load_nst_imgs(content_path: str, style_path: str) -> Tuple[Image.Image, Image.Image]:
  """
  Loads new images for neural style transfer.

  :param content_path: path to content image.
  :param style_path: path to style image.
  :return: tuple of images as numpy arrays.
  """
  return Image.open(content_path), Image.open(style_path)