"""
Define all npdimension operators.
"""
# pylint: disable=W0108

from collections import namedtuple

__all__ = ['NDARRAY_MEMBERS']

def axes_reversed(original):
    return reversed(original)

# All numpy ndarray member functions and attributes
Member = namedtuple('Member', ['name', 'binary', 'transform_axes'])
NDARRAY_MEMBERS = [
    Member(name='T', binary=False, transform_axes=axes_reversed),
    Member(name='__abs__', binary=False, transform_axes=None),
    Member(name='__add__', binary=True, transform_axes=None),
    Member(name='__and__', binary=True, transform_axes=None),
    Member(name='__array__', binary=False, transform_axes=None),
    # Member(name='__array_finalize__', binary=False, transform_axes=None),
    # Member(name='__array_interface__', binary=False, transform_axes=None),
    # Member(name='__array_prepare__', binary=False, transform_axes=None),
    # Member(name='__array_priority__', binary=False, transform_axes=None),
    # Member(name='__array_struct__', binary=False, transform_axes=None),
    # Member(name='__array_wrap__', binary=False, transform_axes=None),
    # Member(name='__class__', binary=False, transform_axes=None),
    Member(name='__contains__', binary=False, transform_axes=None),
    Member(name='__copy__', binary=False, transform_axes=None),
    Member(name='__deepcopy__', binary=False, transform_axes=None),
    # Member(name='__delattr__', binary=False, transform_axes=None),
    Member(name='__delitem__', binary=False, transform_axes=None),
    Member(name='__delslice__', binary=False, transform_axes=None),
    Member(name='__div__', binary=True, transform_axes=None),
    Member(name='__divmod__', binary=True, transform_axes=None),
    # Member(name='__doc__', binary=False, transform_axes=None),
    Member(name='__eq__', binary=True, transform_axes=None),
    Member(name='__float__', binary=False, transform_axes=None),
    Member(name='__floordiv__', binary=True, transform_axes=None),
    # Member(name='__format__', binary=False, transform_axes=None),
    Member(name='__ge__', binary=False, transform_axes=None),
    # Member(name='__getattribute__', binary=False, transform_axes=None),
    Member(name='__getitem__', binary=False, transform_axes=None),
    Member(name='__getslice__', binary=False, transform_axes=None),
    Member(name='__gt__', binary=True, transform_axes=None),
    Member(name='__hash__', binary=False, transform_axes=None),
    Member(name='__hex__', binary=False, transform_axes=None),
    Member(name='__iadd__', binary=True, transform_axes=None),
    Member(name='__iand__', binary=True, transform_axes=None),
    Member(name='__idiv__', binary=True, transform_axes=None),
    Member(name='__ifloordiv__', binary=True, transform_axes=None),
    Member(name='__ilshift__', binary=True, transform_axes=None),
    Member(name='__imod__', binary=True, transform_axes=None),
    Member(name='__imul__', binary=True, transform_axes=None),
    Member(name='__index__', binary=False, transform_axes=None),
    Member(name='__init__', binary=False, transform_axes=None),
    Member(name='__int__', binary=False, transform_axes=None),
    Member(name='__invert__', binary=False, transform_axes=None),
    Member(name='__ior__', binary=True, transform_axes=None),
    Member(name='__ipow__', binary=True, transform_axes=None),
    Member(name='__irshift__', binary=True, transform_axes=None),
    Member(name='__isub__', binary=True, transform_axes=None),
    Member(name='__iter__', binary=False, transform_axes=None),
    Member(name='__itruediv__', binary=True, transform_axes=None),
    Member(name='__ixor__', binary=True, transform_axes=None),
    Member(name='__le__', binary=True, transform_axes=None),
    Member(name='__len__', binary=False, transform_axes=None),
    Member(name='__long__', binary=False, transform_axes=None),
    Member(name='__lshift__', binary=True, transform_axes=None),
    Member(name='__lt__', binary=True, transform_axes=None),
    Member(name='__mod__', binary=True, transform_axes=None),
    Member(name='__mul__', binary=True, transform_axes=None),
    Member(name='__ne__', binary=True, transform_axes=None),
    Member(name='__neg__', binary=False, transform_axes=None),
    Member(name='__new__', binary=False, transform_axes=None),
    Member(name='__nonzero__', binary=False, transform_axes=None),
    Member(name='__oct__', binary=False, transform_axes=None),
    Member(name='__or__', binary=True, transform_axes=None),
    Member(name='__pos__', binary=False, transform_axes=None),
    Member(name='__pow__', binary=False, transform_axes=None), # TODO: Binary...?
    Member(name='__radd__', binary=True, transform_axes=None),
    Member(name='__rand__', binary=True, transform_axes=None),
    Member(name='__rdiv__', binary=True, transform_axes=None),
    Member(name='__rdivmod__', binary=True, transform_axes=None),
    # Member(name='__reduce__', binary=False, transform_axes=None),
    # Member(name='__reduce_ex__', binary=False, transform_axes=None),
    # Member(name='__repr__', binary=False, transform_axes=None),
    Member(name='__rfloordiv__', binary=True, transform_axes=None),
    Member(name='__rlshift__', binary=True, transform_axes=None),
    Member(name='__rmod__', binary=True, transform_axes=None),
    Member(name='__rmul__', binary=True, transform_axes=None),
    Member(name='__ror__', binary=True, transform_axes=None),
    Member(name='__rpow__', binary=True, transform_axes=None),
    Member(name='__rrshift__', binary=True, transform_axes=None),
    Member(name='__rshift__', binary=True, transform_axes=None),
    Member(name='__rsub__', binary=True, transform_axes=None),
    Member(name='__rtruediv__', binary=True, transform_axes=None),
    Member(name='__rxor__', binary=True, transform_axes=None),
    # Member(name='__setattr__', binary=False, transform_axes=None),
    Member(name='__setitem__', binary=False, transform_axes=None),
    Member(name='__setslice__', binary=False, transform_axes=None),
    # Member(name='__setstate__', binary=False, transform_axes=None),
    # Member(name='__sizeof__', binary=False, transform_axes=None),
    Member(name='__str__', binary=False, transform_axes=None),
    Member(name='__sub__', binary=True, transform_axes=None),
    # Member(name='__subclasshook__', binary=False, transform_axes=None),
    Member(name='__truediv__', binary=True, transform_axes=None),
    Member(name='__xor__', binary=True, transform_axes=None),
    Member(name='all', binary=False, transform_axes=None), # TODO: transform_axes
    Member(name='any', binary=False, transform_axes=None), # TODO: transform_axes
    Member(name='argmax', binary=False, transform_axes=None), # TODO: transform_axes
    Member(name='argmin', binary=False, transform_axes=None), # TODO: transform_axes
    Member(name='argpartition', binary=False, transform_axes=None), # TODO: transform_axes
    Member(name='argsort', binary=False, transform_axes=None), # TODO: transform_axes
    Member(name='astype', binary=False, transform_axes=None),
    # Member(name='base', binary=False, transform_axes=None),
    Member(name='byteswap', binary=False, transform_axes=None),
    Member(name='choose', binary=True, transform_axes=None), # TODO: transform_axes
    Member(name='clip', binary=False, transform_axes=None),
    Member(name='compress', binary=True, transform_axes=None), # TODO: transform_axes
    Member(name='conj', binary=False, transform_axes=None),
    Member(name='conjugate', binary=False, transform_axes=None),
    Member(name='copy', binary=False, transform_axes=None),
    # Member(name='ctypes', binary=False, transform_axes=None),
    Member(name='cumprod', binary=False, transform_axes=None), # TODO: transform_axes
    Member(name='cumsum', binary=False, transform_axes=None), # TODO: transform_axes
    # Member(name='data', binary=False, transform_axes=None),
    Member(name='diagonal', binary=False, transform_axes=None),
    Member(name='dot', binary=True, transform_axes=None),
    Member(name='dtype', binary=False, transform_axes=None),
    # Member(name='dump', binary=False, transform_axes=None),
    # Member(name='dumps', binary=False, transform_axes=None),
    Member(name='fill', binary=False, transform_axes=None),
    # Member(name='flags', binary=False, transform_axes=None),
    # Member(name='flat', binary=False, transform_axes=None),
    Member(name='flatten', binary=False, transform_axes=None), # TODO: transform_axes
    Member(name='getfield', binary=False, transform_axes=None),
    # Member(name='imag', binary=False, transform_axes=None),
    Member(name='item', binary=False, transform_axes=None),
    Member(name='itemset', binary=False, transform_axes=None),
    # Member(name='itemsize', binary=False, transform_axes=None),
    Member(name='max', binary=False, transform_axes=None), # TODO: transform_axes
    Member(name='mean', binary=False, transform_axes=None), # TODO: transform_axes
    Member(name='min', binary=False, transform_axes=None), # TODO: transform_axes
    # Member(name='nbytes', binary=False, transform_axes=None),
    Member(name='ndim', binary=False, transform_axes=None),
    Member(name='newbyteorder', binary=False, transform_axes=None),
    Member(name='nonzero', binary=False, transform_axes=None),
    Member(name='partition', binary=False, transform_axes=None),
    Member(name='prod', binary=False, transform_axes=None), # TODO: transform_axes
    Member(name='ptp', binary=False, transform_axes=None), # TODO: transform_axes
    Member(name='put', binary=True, transform_axes=None),
    Member(name='ravel', binary=False, transform_axes=None), # TODO: transform_axes
    # Member(name='real', binary=False, transform_axes=None),
    Member(name='repeat', binary=False, transform_axes=None), # TODO: transform_axes
    Member(name='reshape', binary=False, transform_axes=None), # TODO: transform_axes
    Member(name='resize', binary=False, transform_axes=None), # TODO: transform_axes
    Member(name='round', binary=False, transform_axes=None),
    Member(name='searchsorted', binary=False, transform_axes=None), # TODO
    Member(name='setfield', binary=False, transform_axes=None),
    Member(name='setflags', binary=False, transform_axes=None),
    # Member(name='shape', binary=False, transform_axes=None),
    Member(name='size', binary=False, transform_axes=None), # TODO
    Member(name='sort', binary=False, transform_axes=None),
    Member(name='squeeze', binary=False, transform_axes=None), # TODO: transform_axes
    Member(name='std', binary=False, transform_axes=None), # TODO: transform_axes
    # Member(name='strides', binary=False, transform_axes=None),
    Member(name='sum', binary=False, transform_axes=None), # TODO: transform_axes
    Member(name='swapaxes', binary=False, transform_axes=None), # TODO: transform_axes
    Member(name='take', binary=True, transform_axes=None), # TODO: transform_axes
    Member(name='tobytes', binary=False, transform_axes=None),
    # Member(name='tofile', binary=False, transform_axes=None),
    # Member(name='tolist', binary=False, transform_axes=None),
    # Member(name='tostring', binary=False, transform_axes=None),
    Member(name='trace', binary=False, transform_axes=None),
    Member(name='transpose', binary=False, transform_axes=None), # TODO: transform_axes
    Member(name='var', binary=False, transform_axes=None), # TODO: transform_axes
    Member(name='view', binary=False, transform_axes=None)
]
