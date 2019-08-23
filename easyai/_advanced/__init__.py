import os
import sys

# PRINT HANDLING
class HidePrints:
  """Temporarily hides prints. Copied from StackOverflow."""

  def __enter__(self):
    self._original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')

  def __exit__(self, exc_type, exc_val, exc_tb):
    sys.stdout.close()
    sys.stdout = self._original_stdout

from . import HidePrints

from . import _layers
from . import _losses
from . import _nets