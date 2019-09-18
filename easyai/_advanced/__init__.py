import os
import sys

# PRINT HANDLING
class HidePrints(object):
  """Temporarily hides prints."""

  def __enter__(self):
    self.to_show = sys.stdout
    sys.stdout = open(os.devnull, "w")

  def __exit__(self, exc_type, exc_val, exc_tb):
    sys.stdout.close()
    sys.stdout = self.to_show

from . import HidePrints

from . import _layers
from . import _losses
from . import _nets