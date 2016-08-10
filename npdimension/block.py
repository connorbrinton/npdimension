"""
An ndarray-like class with labelled dimensions.
"""

import numpy as np

# pylint: disable=W0212

class Block(object):
    __slots__ = ['_data', '_axes']

    def __init__(self, data, axes=None):
        """
        Create a new Block with `data` and `axes`.

        `data` must be an array-like object and `axes` must be a list-like object.
        """
        self._data = np.asarray(data)
        self._axes = list(axes)
        # TODO: Check to make sure _data shape and _axes length are consistent

    def __repr__(self):
        parts = []
        # Print the class name
        prefix = "%s(" % type(self).__name__
        parts.append(prefix)

        # Print the axes
        parts.append("axes=%s" % self.axes)
        parts.append("\n")

        # Print the (indented) numpy array
        array = repr(self._data)
        indent = " " * len(prefix)
        indented = "\n".join(indent + line for line in array.split('\n'))
        parts.append(indented)

        # Add the end parentheses
        parts.append(")")

        return "".join(parts)

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

    @property
    def data(self):
        return self._data

    @property
    def axes(self):
        return self._axes

    @property
    def shape(self):
        return {axis: dimension for axis, dimension in zip(self._axes, self._data.shape)}
