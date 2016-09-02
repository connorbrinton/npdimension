"""
NPDIndexer slices either `Block`s or `Scaffold`s according to given indexes.
"""

# pylint: disable=E1101

import six

import npd


class NPDIndexer(object):
    """
    The indexing object for `Block`s and `Scaffold`s.
    """

    def __init__(self, data, index):
        # Make sure the index is the right shape
        if len(index.axes) != 1:
            raise Exception("The index must be one-dimensional.")

        # Store all values
        self._data = data
        self._index = index
        self._axis = index.axes[0]
        self.__sorter = None

    @property
    def _sorter(self):
        """
        Lazily load and cache the sorter when requested.
        """
        if self.__sorter is None:
            self.__sorter = npd.argsort(self._index)
        return self.__sorter

    def __getitem__(self, selection):
        # Selection is either a scalar, a single string, or a slice or array of either
        if isinstance(selection, slice):
            # TODO: Turn slice into array
            selection = None

        if isinstance(selection, six.string_types):
            # Find a single string in the index
            indices = npd.flatnonzero(self._index == selection)
        else:
            # Must be either a scalar or an array/iterator
            indices = npd.searchsorted(self._index, selection, sorter=self._sorter)

        return npd.take(self._data, indices, axis=self._axis)
