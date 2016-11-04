"""
Define all npdimension operators.
"""
# pylint: disable=C0111
# pylint: disable=W0108
# pylint: disable=W0613

from __future__ import print_function
from collections import namedtuple
import itertools
from types import ModuleType

import numpy as np
from block import Block
from scaffold import Scaffold


__all__ = ['NP_MEMBERS', 'NDARRAY_MEMBERS', 'load_proxies']


# Data structure for proxied object members
Parameters = namedtuple('Parameters', [
    'determine_axes', # Generate the new axes for the returned Block
    'transform_args', # Prepare the arguments for the numpy function
    'override', # Inject this member even if it already exists
    'passthrough', # Directly return the result of the numpy function
    'transform_result' # Modify or wrap the result before returning it
])

# Provide Parameters defaults
Parameters.__new__.__defaults__ = (None,) * len(Parameters._fields)


def get_next_block(iterator):
    for value in iterator:
        if isinstance(value, Block):
            return value

    return None


def manual_axes(*args, **kwargs):
    if 'axes' not in kwargs:
        raise Exception("Function requires 'axes' keyword argument.")

    return kwargs['axes']


def remove_axes_kwarg(*args, **kwargs):
    if 'axes' in kwargs:
        del kwargs['axes']

    return args, kwargs


def reverse_axes(*args, **kwargs):
    return reversed(get_next_block(args).axes)


def only_axis(*args, **kwargs):
    return [kwargs.get('axis')]


def insert_axes(*args, **kwargs):
    # Get the block, indexer, and axis arguments
    iterator = iter(args)
    block = get_next_block(iterator)
    indexer = next(iterator) # We require that the indexer come immediately after the Block
    axis = kwargs.get('axis')

    if indexer is None:
        raise Exception("An indexing argument is required.")

    # If the indexer is a Block, merge its axes with `block`'s
    if isinstance(indexer, Block):
        # Make sure there's an axis argument
        if axis is None:
            raise Exception("The axis keyword argument is required when indexing with a Block.")

        # Find the index of the axis argument in the first axes
        index = block.axes.index(axis)

        # The axis along which the original block is being sliced takes precedence over the slicer's
        # axes, so we increase the index by one to keep the original slicing axis.
        index = index + 1

        # Substitute the second Block's axes where the specified axis exists
        return block.axes[:index] + indexer.axes[1:] + block.axes[index:]

    # If it's a scalar, just return the normal axes or nothing, depending on the axis kwarg
    if np.isscalar(indexer):
        # With no axis, these functions use the flattened array
        if axis is None:
            return only_singular_axes(block)

        # Otherwise, remove the requested axis
        return remove_axis(block, axis=axis)

    # If it's not a Block or a scalar, make sure it's 1D
    if np.ndim(indexer) != 1:
        raise Exception("A Block, 1D array-like object, or scalar is required for this function.")

    # Return the normal axes
    return block.axes


def remove_axis(*args, **kwargs):
    # If an axis wasn't specified, don't worry about it
    if 'axis' not in kwargs or kwargs['axis'] is None:
        return None

    # Otherwise, remove the specified axis
    return [original for original in get_next_block(args).axes if original != kwargs['axis']]


def only_singular_axes(*args, **kwargs):
    # If there was only one dimension, keep using the old axes
    block = get_next_block(args)
    if len(block.axes) == 1:
        return block.axes

    # Otherwise, give up
    return None


def secondary_axes(*args, **kwargs):
    """
    Use the axes of the secondary block, if present (otherwise return an ndarray).

    The secondary block is that which comes immediately after the subject block, which is either the
    first argument or self, depending on whether the method is bound or unbound.
    """
    iterator = iter(args)
    block = get_next_block(iterator)
    secondary = get_next_block(iterator)

    if secondary is not None:
        return secondary.axes

    return None


def swap_axes(*args, **kwargs):
    # Load the block
    block = get_next_block(args)

    # Find the indices of the axes to swap
    one = block.axes.index(args[1])
    two = block.axes.index(args[2])

    # Make a copy of the axes and swap
    axes = list(block.axes)
    axes[one], axes[two] = axes[two], axes[one]

    return axes


def transform_swap_axes_args(*args, **kwargs):
    # Make args mutable
    args = list(args)

    # Load the block axes
    axes = get_next_block(args).axes

    # Replace axis arguments with numbers
    args[1] = axes.index(args[1])
    args[2] = axes.index(args[2])

    return args, kwargs


def transform_apply_along_axis_args(*args, **kwargs):
    """
    Transform apply_along_axis(func, block, *args, axis='?', **kwargs) into
    apply_along_axis(func, axis_index, block, *args, **kwargs)
    """
    # Look up the function, block, and axis
    func = args[0]
    block = args[1]
    axis = kwargs.pop('axis')
    remaining = args[2:]

    # Look up the axis index
    index = block.axes.index(axis)

    # Create new arguments
    args = [func, index, block] + list(remaining)

    return args, kwargs


def expand_dims(*args, **kwargs):
    """
    Determine the new axes for an invocation of npd.expand_dims(block, after, name).
    """
    # Look up arguments
    block = args[0]
    after = args[1]
    name = args[2]

    # Find index of after in axes
    axes = block.axes
    index = axes.index(after)

    # Return axes with name inserted after index
    return axes[:(index + 1)] + [name] + axes[(index + 1):]


def transform_expand_dims(*args, **kwargs):
    """
    Transform an invocation of npd.expand_dims(block, after, name) into
    np.expand_dims(block, after_index).
    """
    # Look up arguments
    block = args[0]
    after = args[1]

    # Find index of after in axes
    axes = block.axes
    index = axes.index(after)

    # Generate transformed arguments
    args = [block, index]

    return args, kwargs


def transform_indexing_axes(*args, **kwargs):
    """
    Remove axes indexed by scalar values.
    """
    # Look up the block
    block = get_next_block(args)

    # Look up the indexer
    indexer = args[1]

    # Scalars and non-tuple array-like objects are treated like single-element tuples
    # Basically anything except for a tuple or dictionary
    if not isinstance(indexer, (tuple, dict)):
        indexer = (indexer,)

    # Tuples are filled to the full dimensionality of the Block
    if isinstance(indexer, tuple):
        # Ellipses in tuples are replaced to fill the tuple with full slices
        if Ellipsis in indexer:
            index = indexer.index(Ellipsis)
            left, right = indexer[:(index - 1)], indexer[index:]
        else:
            left, right = indexer, []
        filler = [slice(None)] * (len(block.axes) - len(left) - len(right))
        indexer = tuple(itertools.chain(left, filler, right))

    # Check its type (similar to xarray semantics)
    if isinstance(indexer, tuple):
        return [axis for axis, index in zip(block.axes, indexer) if not np.isscalar(index)]
    elif isinstance(indexer, dict):
        axes = []
        for axis in block.axes:
            try:
                # Look up the slice value
                value = indexer[axis]

                # If not a scalar, the axis will be present
                if not np.isscalar(value):
                    axes.append(axis)
            except KeyError:
                # Include the unsliced axis
                axes.append(axis)

        return axes
    else:
        # We don't know this type of selection :(
        raise Exception("Unknown selection type %s provided. A scalar, tuple, or dict is expected."
                        % repr(type(indexer).__name__))


def transform_indexing_args(*args, **kwargs):
    """
    Prepare indexing arguments for ndarray.__getitem__ by replacing a dict indexer with an
    appropriate indexing tuple.
    """
    # Look up the indexer
    indexer = args[1]

    # Don't do anything if it's a not a dictionary
    if not isinstance(indexer, dict):
        return args, kwargs

    # Make the arguments mutable
    args = list(args)

    # If it's a dictionary, we'll need block information
    block = get_next_block(args)

    # Build a new slice object
    slicer = []
    for axis in block.axes:
        try:
            slicer.append(indexer[axis])
        except KeyError:
            slicer.append(slice(None))

    # Convert to a tuple
    slicer = tuple(slicer)

    # Replace the old indexer
    args[1] = slicer

    # Return the new arguments
    return args, kwargs


# Functions shared between the numpy module and the ndarray object
NP_COMMON = {
    'all': Parameters(determine_axes=remove_axis),
    'any': Parameters(determine_axes=remove_axis),
    'argmax': Parameters(determine_axes=only_axis),
    'argmin': Parameters(determine_axes=only_axis),
    'argpartition': Parameters(determine_axes=only_axis),
    'argsort': Parameters(determine_axes=only_axis),
    'choose': Parameters(),
    'clip': Parameters(),
    'compress': Parameters(),
    'copy': Parameters(),
    'cumprod': Parameters(),
    'cumsum': Parameters(),
    'diagonal': Parameters(),
    'imag': Parameters(),
    'max': Parameters(determine_axes=remove_axis),
    'mean': Parameters(determine_axes=remove_axis),
    'nonzero': Parameters(),
    'partition': Parameters(),
    'prod': Parameters(determine_axes=remove_axis),
    'ptp': Parameters(determine_axes=remove_axis),
    'put': Parameters(),
    'ravel': Parameters(determine_axes=only_singular_axes),
    'real': Parameters(),
    # 'repeat': Parameters(),  # TODO: determine_axes
    # 'reshape': Parameters(),  # TODO: determine_axes
    # 'resize': Parameters(),  # TODO: determine_axes
    'round': Parameters(),
    'searchsorted': Parameters(determine_axes=secondary_axes),
    # 'size': Parameters(),  # TODO
    'sort': Parameters(),
    'squeeze': Parameters(),  # TODO: determine_axes
    'std': Parameters(determine_axes=remove_axis),
    'sum': Parameters(determine_axes=remove_axis),
    'swapaxes': Parameters(determine_axes=swap_axes, transform_args=transform_swap_axes_args),
    'take': Parameters(determine_axes=insert_axes),
    'trace': Parameters(),
    'transpose': Parameters(determine_axes=reverse_axes),
    'var': Parameters(determine_axes=remove_axis),
}

# All numpy module functions
NP_MEMBERS = {
    'alen': Parameters(),
    # 'allclose': Parameters(), # TODO: Implement
    # 'alltrue': Parameters(), # TODO: Implement
    # 'alterdot': Parameters(), # TODO: Implement
    # 'amax': Parameters(), # TODO: Implement
    # 'amin': Parameters(), # TODO: Implement
    # 'angle': Parameters(), # TODO: Implement
    # 'append': Parameters(), # TODO: Implement
    'apply_along_axis': Parameters(determine_axes=remove_axis, # TODO: Only works for 1d returns
                                   transform_args=transform_apply_along_axis_args),
    # 'apply_over_axes': Parameters(), # TODO: Implement
    'arange': Parameters(determine_axes=manual_axes, transform_args=remove_axes_kwarg),
    # 'argwhere': Parameters(), # TODO: Implement
    # 'around': Parameters(), # TODO: Implement
    # 'array2string': Parameters(), # TODO: Implement
    # 'array_equal': Parameters(), # TODO: Implement
    # 'array_equiv': Parameters(), # TODO: Implement
    # 'array_repr': Parameters(), # TODO: Implement
    # 'array_split': Parameters(), # TODO: Implement
    # 'array_str': Parameters(), # TODO: Implement
    # 'asanyarray': Parameters(), # TODO: Implement
    # 'asarray': Parameters(), # TODO: Implement
    # 'asarray_chkfinite': Parameters(), # TODO: Implement
    # 'ascontiguousarray': Parameters(), # TODO: Implement
    # 'asfarray': Parameters(), # TODO: Implement
    # 'asfortranarray': Parameters(), # TODO: Implement
    # 'asmatrix': Parameters(), # TODO: Implement
    # 'asscalar': Parameters(), # TODO: Implement
    # 'atleast_1d': Parameters(), # TODO: Implement
    # 'atleast_2d': Parameters(), # TODO: Implement
    # 'atleast_3d': Parameters(), # TODO: Implement
    # 'average': Parameters(), # TODO: Implement
    # 'bartlett': Parameters(), # TODO: Implement
    # 'base_repr': Parameters(), # TODO: Implement
    # 'binary_repr': Parameters(), # TODO: Implement
    # 'blackman': Parameters(), # TODO: Implement
    # 'bmat': Parameters(), # TODO: Implement
    # 'broadcast_arrays': Parameters(), # TODO: Implement
    # 'broadcast_to': Parameters(), # TODO: Implement
    # 'byte_bounds': Parameters(), # TODO: Implement
    # 'char': {}, # TODO: Add char functions
    # 'column_stack': Parameters(), # TODO: Implement
    # 'common_type': Parameters(), # TODO: Implement
    # 'convolve': Parameters(), # TODO: Implement
    # 'corrcoef': Parameters(), # TODO: Implement
    # 'correlate': Parameters(), # TODO: Implement
    # 'cov': Parameters(), # TODO: Implement
    # 'cross': Parameters(), # TODO: Implement
    # 'cumproduct': Parameters(), # TODO: Implement
    # 'delete': Parameters(), # TODO: Implement
    # 'deprecate': Parameters(), # TODO: Implement
    # 'deprecate_with_doc': Parameters(), # TODO: Implement
    # 'diag': Parameters(), # TODO: Implement
    # 'diag_indices': Parameters(), # TODO: Implement
    # 'diag_indices_from': Parameters(), # TODO: Implement
    # 'diagflat': Parameters(), # TODO: Implement
    # 'diff': Parameters(), # TODO: Implement
    # 'disp': Parameters(), # TODO: Implement
    # 'dsplit': Parameters(), # TODO: Implement
    # 'dstack': Parameters(), # TODO: Implement
    # 'ediff1d': Parameters(), # TODO: Implement
    # 'emath': {} # TODO: Add emath functions
    'empty': Parameters(determine_axes=manual_axes, transform_args=remove_axes_kwarg),
    'expand_dims': Parameters(determine_axes=expand_dims, transform_args=transform_expand_dims),
    # 'extract': Parameters(), # TODO: Implement
    # 'eye': Parameters(), # TODO: Implement
    # 'fft': {}, # TODO: Add fft functions
    # 'fill_diagonal': Parameters(), # TODO: Implement
    # 'find_common_type': Parameters(), # TODO: Implement
    # 'fix': Parameters(), # TODO: Implement
    'flatnonzero': Parameters(determine_axes=only_singular_axes), # TODO: Implement
    # 'fliplr': Parameters(), # TODO: Implement
    # 'flipud': Parameters(), # TODO: Implement
    # 'fromfunction': Parameters(), # TODO: Implement
    # 'fromregex': Parameters(), # TODO: Implement
    'full': Parameters(determine_axes=manual_axes, transform_args=remove_axes_kwarg),
    # 'full_like': Parameters(), # TODO: Implement
    # 'fv': Parameters(), # TODO: Implement
    # 'genfromtxt': Parameters(), # TODO: Implement
    # 'get_array_wrap': Parameters(), # TODO: Implement
    # 'get_include': Parameters(), # TODO: Implement
    # 'get_printoptions': Parameters(), # TODO: Implement
    # 'getbufsize': Parameters(), # TODO: Implement
    # 'geterr': Parameters(), # TODO: Implement
    # 'geterrcall': Parameters(), # TODO: Implement
    # 'gradient': Parameters(), # TODO: Implement
    # 'hamming': Parameters(), # TODO: Implement
    # 'hanning': Parameters(), # TODO: Implement
    # 'histogram': Parameters(), # TODO: Implement
    # 'histogram2d': Parameters(), # TODO: Implement
    # 'histogramdd': Parameters(), # TODO: Implement
    # 'hsplit': Parameters(), # TODO: Implement
    # 'hstack': Parameters(), # TODO: Implement
    # 'i0': Parameters(), # TODO: Implement
    # 'identity': Parameters(), # TODO: Implement
    'in1d': Parameters(),
    # 'indices': Parameters(), # TODO: Implement
    # 'info': Parameters(), # TODO: Implement
    # 'insert': Parameters(), # TODO: Implement
    # 'interp': Parameters(), # TODO: Implement
    # 'intersect1d': Parameters(), # TODO: Implement
    # 'ipmt': Parameters(), # TODO: Implement
    # 'irr': Parameters(), # TODO: Implement
    # 'isclose': Parameters(), # TODO: Implement
    # 'iscomplex': Parameters(), # TODO: Implement
    # 'iscomplexobj': Parameters(), # TODO: Implement
    # 'isfortran': Parameters(), # TODO: Implement
    # 'isneginf': Parameters(), # TODO: Implement
    # 'isposinf': Parameters(), # TODO: Implement
    # 'isreal': Parameters(), # TODO: Implement
    # 'isrealobj': Parameters(), # TODO: Implement
    # 'isscalar': Parameters(), # TODO: Implement
    # 'issctype': Parameters(), # TODO: Implement
    # 'issubclass_': Parameters(), # TODO: Implement
    # 'issubdtype': Parameters(), # TODO: Implement
    # 'issubsctype': Parameters(), # TODO: Implement
    # 'iterable': Parameters(), # TODO: Implement
    # 'ix_': Parameters(), # TODO: Implement
    # 'kaiser': Parameters(), # TODO: Implement
    # 'kron': Parameters(), # TODO: Implement
    'lib': {}, # TODO: add lib functions
    'linalg': {
        'cholesky': Parameters(), # TODO: Implement
        'cond': Parameters(), # TODO: Implement
        'det': Parameters(), # TODO: Implement
        'eig': Parameters(), # TODO: Implement
        'eigh': Parameters(), # TODO: Implement
        'eigvals': Parameters(), # TODO: Implement
        'eigvalsh': Parameters(), # TODO: Implement
        'info': Parameters(), # TODO: Implement
        'inv': Parameters(), # TODO: Implement
        'lstsq': Parameters(), # TODO: Implement
        'matrix_power': Parameters(), # TODO: Implement
        'matrix_rank': Parameters(), # TODO: Implement
        'norm': Parameters(determine_axes=remove_axis),
        'pinv': Parameters(), # TODO: Implement
        'qr': Parameters(), # TODO: Implement
        'slogdet': Parameters(), # TODO: Implement
        'solve': Parameters(), # TODO: Implement
        'svd': Parameters(), # TODO: Implement
        'tensorinv': Parameters(), # TODO: Implement
        'tensorsolve': Parameters() # TODO: Implement
    },
    # 'linspace': Parameters(), # TODO: Implement
    # 'load': Parameters(), # TODO: Implement
    # 'loadtxt': Parameters(), # TODO: Implement
    'logical_and': Parameters(),
    'logical_not': Parameters(),
    'logical_or': Parameters(),
    # 'logspace': Parameters(), # TODO: Implement
    # 'lookfor': Parameters(), # TODO: Implement
    # 'ma': {}, # TODO: Add ma functions
    # 'mafromtxt': Parameters(), # TODO: Implement
    # 'mask_indices': Parameters(), # TODO: Implement
    # 'mat': Parameters(), # TODO: Implement
    # 'math': {}, # TODO: Add math functions
    # 'maximum_sctype': Parameters(), # TODO: Implement
    # 'median': Parameters(), # TODO: Implement
    # 'meshgrid': Parameters(), # TODO: Implement
    # 'mintypecode': Parameters(), # TODO: Implement
    # 'mirr': Parameters(), # TODO: Implement
    # 'moveaxis': Parameters(), # TODO: Implement
    # 'msort': Parameters(), # TODO: Implement
    # 'nan_to_num': Parameters(), # TODO: Implement
    # 'nanargmax': Parameters(), # TODO: Implement
    # 'nanargmin': Parameters(), # TODO: Implement
    # 'nanmax': Parameters(), # TODO: Implement
    # 'nanmean': Parameters(), # TODO: Implement
    # 'nanmedian': Parameters(), # TODO: Implement
    # 'nanmin': Parameters(), # TODO: Implement
    # 'nanpercentile': Parameters(), # TODO: Implement
    # 'nanprod': Parameters(), # TODO: Implement
    # 'nanstd': Parameters(), # TODO: Implement
    # 'nansum': Parameters(), # TODO: Implement
    # 'nanvar': Parameters(), # TODO: Implement
    # 'ndfromtxt': Parameters(), # TODO: Implement
    # 'ndim': Parameters(), # TODO: Implement
    # 'nper': Parameters(), # TODO: Implement
    # 'npv': Parameters(), # TODO: Implement
    # 'obj2sctype': Parameters(), # TODO: Implement
    # 'ones': Parameters(), # TODO: Implement
    # 'ones_like': Parameters(), # TODO: Implement
    # 'outer': Parameters(), # TODO: Implement
    # 'pad': Parameters(), # TODO: Implement
    # 'percentile': Parameters(), # TODO: Implement
    # 'piecewise': Parameters(), # TODO: Implement
    # 'pkgload': Parameters(), # TODO: Implement
    # 'place': Parameters(), # TODO: Implement
    # 'pmt': Parameters(), # TODO: Implement
    # 'poly': Parameters(), # TODO: Implement
    # 'polyadd': Parameters(), # TODO: Implement
    # 'polyder': Parameters(), # TODO: Implement
    # 'polydiv': Parameters(), # TODO: Implement
    # 'polyfit': Parameters(), # TODO: Implement
    # 'polyint': Parameters(), # TODO: Implement
    # 'polymul': Parameters(), # TODO: Implement
    # 'polysub': Parameters(), # TODO: Implement
    # 'polyval': Parameters(), # TODO: Implement
    # 'ppmt': Parameters(), # TODO: Implement
    # 'product': Parameters(), # TODO: Implement
    # 'pv': Parameters(), # TODO: Implement
    # 'random': {}, # TODO: Add random functions
    # 'rank': Parameters(), # TODO: Implement
    # 'rate': Parameters(), # TODO: Implement
    # 'real_if_close': Parameters(), # TODO: Implement
    # 'rec': {} # TODO: Add rec functions
    # 'recfromcsv': Parameters(), # TODO: Implement
    # 'recfromtxt': Parameters(), # TODO: Implement
    # 'require': Parameters(), # TODO: Implement
    # 'restoredot': Parameters(), # TODO: Implement
    # 'roll': Parameters(), # TODO: Implement
    # 'rollaxis': Parameters(), # TODO: Implement
    # 'roots': Parameters(), # TODO: Implement
    # 'rot90': Parameters(), # TODO: Implement
    # 'row_stack': Parameters(), # TODO: Implement
    # 'safe_eval': Parameters(), # TODO: Implement
    # 'save': Parameters(), # TODO: Implement
    # 'savetxt': Parameters(), # TODO: Implement
    # 'savez': Parameters(), # TODO: Implement
    # 'savez_compressed': Parameters(), # TODO: Implement
    # 'sctype2char': Parameters(), # TODO: Implement
    # 'select': Parameters(), # TODO: Implement
    # 'set_printoptions': Parameters(), # TODO: Implement
    # 'set_string_function': Parameters(), # TODO: Implement
    # 'setbufsize': Parameters(), # TODO: Implement
    # 'setdiff1d': Parameters(), # TODO: Implement
    # 'seterr': Parameters(), # TODO: Implement
    # 'seterrcall': Parameters(), # TODO: Implement
    # 'setxor1d': Parameters(), # TODO: Implement
    # 'shape': Parameters(), # TODO: Implement
    # 'show_config': Parameters(), # TODO: Implement
    # 'sinc': Parameters(), # TODO: Implement
    # 'sometrue': Parameters(), # TODO: Implement
    # 'sort_complex': Parameters(), # TODO: Implement
    # 'source': Parameters(), # TODO: Implement
    # 'split': Parameters(), # TODO: Implement
    # 'stack': Parameters(), # TODO: Implement
    # 'tensordot': Parameters(), # TODO: Implement
    # 'tile': Parameters(), # TODO: Implement
    # 'trapz': Parameters(), # TODO: Implement
    # 'tri': Parameters(), # TODO: Implement
    # 'tril': Parameters(), # TODO: Implement
    # 'tril_indices': Parameters(), # TODO: Implement
    # 'tril_indices_from': Parameters(), # TODO: Implement
    # 'trim_zeros': Parameters(), # TODO: Implement
    # 'triu': Parameters(), # TODO: Implement
    # 'triu_indices': Parameters(), # TODO: Implement
    # 'triu_indices_from': Parameters(), # TODO: Implement
    # 'typename': Parameters(), # TODO: Implement
    'union1d': Parameters(), # TODO: Implement
    'unique': Parameters(determine_axes=only_singular_axes), # TODO: Implement
    # 'unwrap': Parameters(), # TODO: Implement
    # 'vander': Parameters(), # TODO: Implement
    # 'vsplit': Parameters(), # TODO: Implement
    # 'vstack': Parameters(), # TODO: Implement
    # 'who': Parameters(), # TODO: Implement
    # 'zeros_like': Parameters() # TODO: Implement
}
NP_MEMBERS.update(NP_COMMON)

# All numpy ndarray member functions and attributes
NDARRAY_MEMBERS = {
    'T': Parameters(determine_axes=reverse_axes),
    '__abs__': Parameters(),
    '__add__': Parameters(),
    '__and__': Parameters(),
    # '__array__': Parameters(passthrough=False),
    # '__array_finalize__': Parameters(),
    # '__array_interface__': Parameters(),
    # '__array_prepare__': Parameters(),
    # '__array_priority__': Parameters(),
    # '__array_struct__': Parameters(),
    # '__array_wrap__': Parameters(),
    # '__class__': Parameters(),
    '__contains__': Parameters(),
    '__copy__': Parameters(),
    '__deepcopy__': Parameters(),
    # '__delattr__': Parameters(),
    '__delitem__': Parameters(),
    '__delslice__': Parameters(),
    '__div__': Parameters(),
    '__divmod__': Parameters(),
    # '__doc__': Parameters(),
    '__eq__': Parameters(),
    '__float__': Parameters(),
    '__floordiv__': Parameters(),
    # '__format__': Parameters(),
    '__ge__': Parameters(override=True),
    # '__getattribute__': Parameters(),
    '__getitem__': Parameters(determine_axes=transform_indexing_axes,
                              transform_args=transform_indexing_args),
    '__getslice__': Parameters(),
    '__gt__': Parameters(override=True),
    # '__hash__': Parameters(), # This is just None in ndarray, strange...
    '__hex__': Parameters(),
    '__iadd__': Parameters(),
    '__iand__': Parameters(),
    '__idiv__': Parameters(),
    '__ifloordiv__': Parameters(),
    '__ilshift__': Parameters(),
    '__imod__': Parameters(),
    '__imul__': Parameters(),
    '__index__': Parameters(),
    # '__init__': Parameters(),
    '__int__': Parameters(),
    '__invert__': Parameters(),
    '__ior__': Parameters(),
    '__ipow__': Parameters(),
    '__irshift__': Parameters(),
    '__isub__': Parameters(),
    '__iter__': Parameters(),
    '__itruediv__': Parameters(),
    '__ixor__': Parameters(),
    '__le__': Parameters(override=True),
    '__len__': Parameters(),
    '__long__': Parameters(),
    '__lshift__': Parameters(),
    '__lt__': Parameters(override=True),
    '__mod__': Parameters(),
    '__mul__': Parameters(),
    '__ne__': Parameters(override=True),
    '__neg__': Parameters(),
    # '__new__': Parameters(),
    '__nonzero__': Parameters(),
    '__oct__': Parameters(),
    '__or__': Parameters(),
    '__pos__': Parameters(),
    '__pow__': Parameters(),
    '__radd__': Parameters(),
    '__rand__': Parameters(),
    '__rdiv__': Parameters(),
    '__rdivmod__': Parameters(),
    # '__reduce__': Parameters(),
    # '__reduce_ex__': Parameters(),
    # '__repr__': Parameters(),
    '__rfloordiv__': Parameters(),
    '__rlshift__': Parameters(),
    '__rmod__': Parameters(),
    '__rmul__': Parameters(),
    '__ror__': Parameters(),
    '__rpow__': Parameters(),
    '__rrshift__': Parameters(),
    '__rshift__': Parameters(),
    '__rsub__': Parameters(),
    '__rtruediv__': Parameters(),
    '__rxor__': Parameters(),
    # '__setattr__': Parameters(),
    '__setitem__': Parameters(transform_args=transform_indexing_args),
    '__setslice__': Parameters(),
    # '__setstate__': Parameters(),
    # '__sizeof__': Parameters(),
    # '__str__': Parameters(),
    '__sub__': Parameters(),
    # '__subclasshook__': Parameters(),
    '__truediv__': Parameters(),
    '__xor__': Parameters(),
    'astype': Parameters(),
    'base': Parameters(),
    'byteswap': Parameters(),
    'conj': Parameters(),
    'conjugate': Parameters(),
    'ctypes': Parameters(),
    # 'data': Parameters(),
    'dot': Parameters(),
    'dtype': Parameters(),
    # 'dump': Parameters(),
    # 'dumps': Parameters(),
    'fill': Parameters(),
    'flags': Parameters(),
    'flat': Parameters(),
    'flatten': Parameters(determine_axes=only_singular_axes),
    'getfield': Parameters(),
    'item': Parameters(),
    'itemset': Parameters(),
    'itemsize': Parameters(),
    'min': Parameters(determine_axes=remove_axis),
    'nbytes': Parameters(),
    'ndim': Parameters(),
    'newbyteorder': Parameters(),
    'setfield': Parameters(),
    'setflags': Parameters(),
    # 'shape': Parameters(),
    'strides': Parameters(),
    'tobytes': Parameters(),
    # 'tofile': Parameters(),
    # 'tolist': Parameters(),
    # 'tostring': Parameters(),
    'view': Parameters()
}
NDARRAY_MEMBERS.update(NP_COMMON)


def closure(func, params):
    # Create a function that can be bound to Block
    def proxy(*args, **kwargs):
        """
        Invoke the numpy function specified by the closure variable `what` on behalf of the caller.

        The arguments and kwargs are rewritten before invoking the numpy function. The following
        checks and transformations take place:
        - If a Scaffold object is present in the arguments, this function is reinvoked with each
          member of the Scaffold in turn. If the keyword argument `axis` is specified, Scaffold
          members that do not have the requested `axis` are simply copied to the returned data
          structure (and thus do not have the function applied to them). If the results are scalars
          or ndarrays, a dictionary with the same keys as the Scaffold and the result for each key
          is returned. If the results are all Blocks, a new Scaffold with the same keys and the
          result for each key is returned.
        - If `what` specifies a `determine_axes` function, invoke it with the arguments to the
          function and save its return value as `transformed_axes`.
        - If `what` specifies a `transform_args` function, use it to transform the function
          arguments (both args and kwargs).
        - Arguments of type `Block` have their axes saved and are replaced by simple ndarrays.
        - The keyword argument `axis` is saved to a local variable called `axis` and the kwarg is
          replaced `axis`'s index in the invoking `Block`'s axes. It follows that `axis`, if
          specified, must be a string, rather than an integer.
        - The numpy function is invoked with the current state of the arguments.
        - If the return value of the numpy function is not an `ndarray`, return it immediately.
        - Return a new Block where `data` is the returned `ndarray` and the `axes` are the
          `transformed_axes`.
        """
        # Allow assignment to args
        args = list(args)

        # Deal with Scaffolds as arguments
        if any(isinstance(arg, Scaffold) for arg in args):
            return peel_scaffolds(proxy, *args, **kwargs)

        # We figure out which Block is first in the args, which will either be `self` (for a bound
        # method) or simply the first Block for an unbound method.
        first = get_next_block(args)

        # determine_axes needs to use original arguments, so we invoke it first
        transformed_axes = determine_axes(params, first, *args, **kwargs)

        # Prepare arguments for numpy function
        args, kwargs = transform_args(params, first, *args, **kwargs)

        # Obtain a result
        result = func(*args, **kwargs)

        # If we're in passthrough mode, just return it
        if params.passthrough:
            return result

        # If it's not an ndarray, just return it
        if not isinstance(result, np.ndarray):
            return result

        # If the new axes couldn't be determined, just return the ndarray
        if transformed_axes is None:
            return result

        # If the new axes could be determined, use them to create a new Block with the result
        return Block(result, axes=transformed_axes)

    # Preserve name
    if hasattr(func, '__name__'):
        proxy.__name__ = func.__name__

    # Preserve documentation
    if hasattr(func, '__doc__'):
        proxy.__doc__ = func.__doc__

    # Return the closured function
    return proxy


def peel_scaffolds(func, *args, **kwargs):
    """
    Call `func` repeatedly with a Scaffold in `args` replaced with member Blocks, and return a
    Scaffold or dict representing the result.

    If the `axis` keyword argument is present, Scaffold members without that `axis` will not be
    transformed according to func, but simply placed in the output data structure.

    Only one Scaffold argument may be present. I can't think of any applications for more than one
    Scaffold argument anyways. If you really want it, handle it yourself.
    """
    # Allow assignment to args
    args = list(args)

    # Make sure there's only one Scaffold argument
    scaffolds = [(index, arg) for index, arg in enumerate(args) if isinstance(arg, Scaffold)]
    if len(scaffolds) != 1:
        raise Exception("Only one Scaffold argument may be provided to a function.")

    # Get the single scaffold
    index, scaffold = scaffolds[0]

    # Look up the axis, for deciding which members to manipulate
    axis = kwargs.get('axis')

    results = {}
    for key, value in scaffold.iteritems():
        if axis is not None and axis not in value.axes:
            # Pass the value through if the axis is specified but not present in the member
            results[key] = scaffold[key]
        else:
            # Otherwise, replace the argument with the Block and save the result
            args[index] = value
            results[key] = func(*args, **kwargs)

    # If all values are Blocks, create a new Scaffold object
    if all(isinstance(result, Block) for result in results.values()):
        return Scaffold(results)

    # Otherwise just return the dictionary
    return results


def determine_axes(params, first, *args, **kwargs):
    if params.determine_axes is not None:
        return params.determine_axes(*args, **kwargs)
    else:
        return first.axes


def transform_args(params, first, *args, **kwargs):
    # Perform any function-specific argument transformation
    if params.transform_args is not None:
        args, kwargs = params.transform_args(*args, **kwargs)

    # Allow argument assignment
    args = list(args)

    # Replace `Block` arguments with ndarrays
    for index, arg in enumerate(args):
        if isinstance(arg, Block):
            args[index] = arg.data

    # Rewrite the axis keyword argument
    if 'axis' in kwargs:
        axis = kwargs['axis']
        if axis is None:
            del kwargs['axis']
        if axis is not None:
            kwargs['axis'] = first.axes.index(axis)

    return args, kwargs


def load_proxies(source, which, destination):
    """
    Load proxies to functions from `source` according to `which` into `destination`.
    """
    # Generate a filter for injectable attributes
    def injectable(name, value, destination):
        return not hasattr(destination, name) or isinstance(value, Parameters) and value.override

    # Only inject attributes that can be injected
    which = {k: v for k, v in which.iteritems() if injectable(k, v, destination)}

    # For each member listed in which
    for name, value in which.iteritems():
        # Load the member
        member = getattr(source, name, None)
        if member is None:
            raise Exception("Member %s not found in %s" % (name, source))

        # Base case, it's a function
        if isinstance(value, Parameters):
            proxy = closure(member, value)
        else:
            # Otherwise, it's a module, so load its members recursively
            proxy = ModuleType(name)
            load_proxies(member, value, proxy)

        # Store in destination
        setattr(destination, name, proxy)
