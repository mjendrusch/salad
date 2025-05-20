"""This module contains a utility type `dotdict`, which is a dictionary with dot access."""

from copy import deepcopy

class dotdict(dict):
  """Dictionary with a dot-access operator."""
  __getattr__ = dict.get
  __setattr__ = dict.__setitem__
  __delattr__ = dict.__delitem__

  def __deepcopy__(self, memo=None):
    return dotdict(deepcopy(dict(self), memo=memo))
