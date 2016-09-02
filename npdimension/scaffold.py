"""
A dictionary-like container for aligned npdimension Blocks.
"""

# TODO: Verify that shapes match according to axes on insertion

from __future__ import print_function

import six

from npdindexer import NPDIndexer


class Scaffold(dict):
    def __getitem__(self, selection):
        # If a string is requested, return it
        if isinstance(selection, six.string_types):
            return super(Scaffold, self).__getitem__(selection)

        # Otherwise, slice everything with applicable axes according to selection
        if isinstance(selection, dict):
            applied = Scaffold()
            axes = set(selection.keys())
            for key, value in self.iteritems():
                # Determine if the selection affects the Scaffold member
                intersection = axes.intersection(value.axes)
                if len(intersection) == 0:
                    applied[key] = value
                else:
                    subselection = {k: v for k, v in selection.iteritems() if k in intersection}
                    applied[key] = value[subselection]
            return applied

        raise Exception("Scaffolds can only be subscripted with a string or dictionary.")


    def __setitem__(self, selection, value):
        if isinstance(selection, six.string_types):
            return super(Scaffold, self).__setitem__(selection, value)

        raise Exception("Only string types can be used to set Scaffold members.")


    def __repr__(self):
        normally = super(Scaffold, self).__repr__()
        return "%s(%s)" % (Scaffold.__name__, normally)

    @property
    def axes(self):
        """
        Return an unordered set representing the axes present in this Scaffold.
        """
        axes = set()
        for value in self.itervalues():
            axes.update(value.axes)

        return axes

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
