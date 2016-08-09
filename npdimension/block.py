"""
An ndarray-like class with labelled dimensions.
"""

import numpy as np
import six


class Block(object):
    __slots__ = ['_data', '_axes']

    def __init__(self, data, axes=None):
        """
        Create a new Block with `data` and `axes`.

        `data` must be an array-like object and `axes` must be a list-like object.
        """
        self._data = np.asarray(data)
        self._axes = list(axes)

    def __repr__(self):
        return repr({
            'data': self._data,
            'axes': self.axes
        })

    def __call__(self, **kwargs):
        """
        Perform indexing on `data` according to dimension names. Returns a new Block.

        If possible, a view is returned, rather than a copy.
        """
        # Determine how the indexing applies to the Block
        indexing = []
        for axis in self._axes:
            if axis in kwargs:
                # Apply dimension=slice from arguments to value
                indexing.append(kwargs[axis])
            else:
                # If no argument is given, keep everything in the dimension
                indexing.append(slice(None, None, None))

        # Index the data
        indexed = self._data[tuple(indexing)]
        return Block(indexed, axes=self._axes)

    def __lt__(self, other):
        return Block(self.data.__lt__(other), axes=self._axes)

    def __le__(self, other):
        return Block(self.data.__le__(other), axes=self._axes)

    def __gt__(self, other):
        return Block(self.data.__gt__(other), axes=self._axes)

    def __ge__(self, other):
        return Block(self.data.__ge__(other), axes=self._axes)

    @property
    def data(self):
        return self._data

    @property
    def axes(self):
        return self._axes

    @property
    def shape(self):
        return {axis: dimension for axis, dimension in zip(self._axes, self._data.shape)}

    def get_axis_index(self, axis):
        return self._axes.index(axis)

    def _map_data(self, function, *args, **kwargs):
        """
        Returns a Block where `function` has been applied to `data` with `args` and `kwargs`.

        If the keyword argument `axis` is provided as a string, it is converted to an index before
        calling the given function. If the keyword argument `new_axes` is provided, its value is
        used as the new axes of the returned Block. Otherwise, the original axes of the called Block
        are used.
        """
        # Prepare the kwargs
        if 'axis' in kwargs:
            # Rewrite string axes as integer indices
            if isinstance(kwargs['axis'], six.string_types):
                kwargs['axis'] = self._axes.index(kwargs['axis'])

        # Determine new axes
        if 'new_axes' in kwargs:
            new_axes = kwargs['new_axes']
            del kwargs['new_axes']
        else:
            new_axes = self._axes

        return Block(function(self._data, *args, **kwargs), axes=new_axes)

    def take(self, indices, axis=None):
        """
        Take elements from a Block or Scaffold along a named axis. See numpy.take.
        """
        if axis is None:
            raise Exception("The axis argument is required for take.")
        return self._map_data(np.take, indices, axis=axis)

    def compress(self, condition, axis=None):
        """
        Return selected slices of a Block or Scaffold along a named axis. See numpy.compress.
        """
        if axis is None:
            raise Exception("The axis argument is required for compress.")
        return self._map_data(np.compress, condition, axis=axis)

    def mean(self, axis=None):
        """
        Compute the arithmetic mean along the specified axis.
        """
        # Determine the new axes
        if axis is not None:
            new_axes = [candidate for candidate in self._axes if candidate != axis]
        else:
            new_axes = self.axes

        return self._map_data(np.mean, axis=axis, new_axes=new_axes)

    def _norm(self, axis=None):
        """
        Vector norm (matrix norm is not yet supported).
        """
        # Determine the new axes
        new_axes = [candidate for candidate in self._axes if candidate != axis]

        return self._map_data(np.linalg.norm, axis=axis, new_axes=new_axes)
