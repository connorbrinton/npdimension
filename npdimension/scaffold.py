"""
A dictionary-like container for aligned npdimension Blocks.
"""

# TODO: Verify that shapes match according to axes on insertion

from __future__ import print_function
import numpy as np

class Scaffold(dict):

    def apply(self, function, *args, **kwargs):
        """
        Return a new dict where `function` has been applied to all members of the dictionary with
        additional arguments `args` and `kwargs`.

        If the keyword argument `axis` is provided, the function will only be applied to dictionary
        members containing the given axis.
        """
        applied = Scaffold()
        for key, value in self.iteritems():
            if 'axis' not in kwargs:
                applied[key] = function(value, *args, **kwargs)
            else:
                if kwargs['axis'] in value.axes:
                    applied[key] = function(value, *args, **kwargs)
                else:
                    applied[key] = value

        return applied

    def __call__(self, **kwargs):
        """
        Perform indexing on the elements in the Scaffold according to dimension names.
        """
        def index(value):
            return value(**kwargs)

        return self.apply(index)

    def __repr__(self):
        normally = super(Scaffold, self).__repr__()
        return "%s(%s)" % (Scaffold.__name__, normally)

    def loc(self, index):
        """
        Returns a NPDIndexer indexing on the member named `index`.

        The member named `index` must be a 1D array. Indexing and slicing will take place along the
        axis of this member. The returned NPDIndexer reacts to indexing by:
        (1) Looking up the indices of the indexed values
        (2) Returning this Scaffold sliced according to the indices of the indexed values.
        """
        return NPDIndexer(self, self[index])

    @property
    def shape(self):
        """
        Return the shape of the Scaffold as a dictionary of axis-dimension pairs.
        """
        shape = {}
        for value in self.itervalues():
            # Get the shape of each element
            vshape = value.shape
            for axis, size in vshape.iteritems():
                # Update the Scaffold shape with new axis dimensions
                if axis not in shape:
                    shape[axis] = size
                else:
                    # Raise an error if inconsistent dimensions are detected
                    if shape[axis] != size:
                        # TODO: Create or use a specific exception type
                        raise Exception("Inconsistent dimension for Scaffold along %s" % axis)

        return shape
