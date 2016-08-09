"""
The operators of the linalg module of npdimension.
"""

__all__ = ['norm']

def norm(a, axis=None):
    """
    Vector norm (does not support matrix norm).
    """
    if axis is None:
        raise Exception("The axis argument is required for norm")
    return a._norm(axis=axis)
