"""
Define all npdimension operators.
"""

__all__ = ['take', 'compress']

def take(a, indices, axis=None):
    """
    Take elements from a Block or Scaffold along a named axis. See numpy.take.
    """
    return a.take(indices, axis=axis)

def compress(a, condition, axis=None):
    """
    Return selected slices of a Block or Scaffold along a named axis. See numpy.compress.
    """
    return a.compress(condition, axis=axis)
