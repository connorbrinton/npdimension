"""
Define all npdimension operators.
"""
# pylint: disable=C0111
# pylint: disable=W0108
# pylint: disable=W0613

from collections import namedtuple
from types import ModuleType
import numpy as np
from block import Block
from scaffold import Scaffold

__all__ = ['NP_MEMBERS', 'NDARRAY_MEMBERS', 'load_proxies']


def get_next_block(iterator):
    for value in iterator:
        if isinstance(value, Block):
            return value

    return None


def reverse_axes(*args, **kwargs):
    return reversed(get_next_block(args).axes)


def only_axis(*args, **kwargs):
    return [kwargs.get('axis')]


def insert_axes(*args, **kwargs):
    # Get the first and second arguments
    iterator = iter(args)
    first = get_next_block(iterator)
    second = get_next_block(iterator)

    # TODO: If no Block was found, look for something simply list-like
    if second is None:
        raise Exception("A Block is required for this function.")

    # Find the index of the axis argument in the first axes
    index = first.axes.index(kwargs['axis'])

    # Substitute the second Block's axes where the specified axis exists
    return first.axes[:index] + second.axes + first.axes[(index + 1):]


def remove_axis(*args, **kwargs):
    # If an axis wasn't specified, don't worry about it
    if kwargs['axis'] is None:
        return None

    # Otherwise, remove the specified axis
    return [original for original in get_next_block(args).axes if original != kwargs['axis']]


def only_singular_axes(original, *args, **kwargs):
    # If there was only one dimension, keep using the old axes
    first = get_next_block(args)
    if len(first.axes) == 1:
        return first.axes

    # Otherwise, give up
    return None


def swap_axes(*args, **kwargs):
    pass # TODO: Implement


def transform_swap_axes_args(*args, **kwargs):
    pass # TODO: Implement

# Data structure for proxied object members
Parameters = namedtuple('Parameters', ['transform_axes', 'transform_args'])

# Provide Parameters defaults
Parameters.__new__.__defaults__ = (None,) * len(Parameters._fields)


# All numpy module functions
NP_MEMBERS = {
    'alen': Parameters(),
    # 'all': Parameters(), # TODO: Implement
    # 'allclose': Parameters(), # TODO: Implement
    # 'alltrue': Parameters(), # TODO: Implement
    # 'alterdot': Parameters(), # TODO: Implement
    # 'amax': Parameters(), # TODO: Implement
    # 'amin': Parameters(), # TODO: Implement
    # 'angle': Parameters(), # TODO: Implement
    # 'any': Parameters(), # TODO: Implement
    # 'append': Parameters(), # TODO: Implement
    # 'apply_along_axis': Parameters(), # TODO: Implement
    # 'apply_over_axes': Parameters(), # TODO: Implement
    # 'argmax': Parameters(), # TODO: Implement
    # 'argmin': Parameters(), # TODO: Implement
    # 'argpartition': Parameters(), # TODO: Implement
    # 'argsort': Parameters(), # TODO: Implement
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
    # 'choose': Parameters(), # TODO: Implement
    # 'clip': Parameters(), # TODO: Implement
    # 'column_stack': Parameters(), # TODO: Implement
    # 'common_type': Parameters(), # TODO: Implement
    # 'compress': Parameters(), # TODO: Implement
    # 'convolve': Parameters(), # TODO: Implement
    # 'copy': Parameters(), # TODO: Implement
    # 'corrcoef': Parameters(), # TODO: Implement
    # 'correlate': Parameters(), # TODO: Implement
    # 'cov': Parameters(), # TODO: Implement
    # 'cross': Parameters(), # TODO: Implement
    # 'cumprod': Parameters(), # TODO: Implement
    # 'cumproduct': Parameters(), # TODO: Implement
    # 'cumsum': Parameters(), # TODO: Implement
    # 'delete': Parameters(), # TODO: Implement
    # 'deprecate': Parameters(), # TODO: Implement
    # 'deprecate_with_doc': Parameters(), # TODO: Implement
    # 'diag': Parameters(), # TODO: Implement
    # 'diag_indices': Parameters(), # TODO: Implement
    # 'diag_indices_from': Parameters(), # TODO: Implement
    # 'diagflat': Parameters(), # TODO: Implement
    # 'diagonal': Parameters(), # TODO: Implement
    # 'diff': Parameters(), # TODO: Implement
    # 'disp': Parameters(), # TODO: Implement
    # 'dsplit': Parameters(), # TODO: Implement
    # 'dstack': Parameters(), # TODO: Implement
    # 'ediff1d': Parameters(), # TODO: Implement
    # 'emath': {} # TODO: Add emath functions
    # 'expand_dims': Parameters(), # TODO: Implement
    # 'extract': Parameters(), # TODO: Implement
    # 'eye': Parameters(), # TODO: Implement
    # 'fft': {}, # TODO: Add fft functions
    # 'fill_diagonal': Parameters(), # TODO: Implement
    # 'find_common_type': Parameters(), # TODO: Implement
    # 'fix': Parameters(), # TODO: Implement
    # 'flatnonzero': Parameters(), # TODO: Implement
    # 'fliplr': Parameters(), # TODO: Implement
    # 'flipud': Parameters(), # TODO: Implement
    # 'fromfunction': Parameters(), # TODO: Implement
    # 'fromregex': Parameters(), # TODO: Implement
    # 'full': Parameters(), # TODO: Implement
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
    # 'imag': Parameters(), # TODO: Implement
    # 'in1d': Parameters(), # TODO: Implement
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
        'norm': Parameters(transform_axes=remove_axis),
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
    # 'logspace': Parameters(), # TODO: Implement
    # 'lookfor': Parameters(), # TODO: Implement
    # 'ma': {}, # TODO: Add ma functions
    # 'mafromtxt': Parameters(), # TODO: Implement
    # 'mask_indices': Parameters(), # TODO: Implement
    # 'mat': Parameters(), # TODO: Implement
    # 'math': {}, # TODO: Add math functions
    # 'maximum_sctype': Parameters(), # TODO: Implement
    # 'mean': Parameters(), # TODO: Implement
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
    # 'nonzero': Parameters(), # TODO: Implement
    # 'nper': Parameters(), # TODO: Implement
    # 'npv': Parameters(), # TODO: Implement
    # 'obj2sctype': Parameters(), # TODO: Implement
    # 'ones': Parameters(), # TODO: Implement
    # 'ones_like': Parameters(), # TODO: Implement
    # 'outer': Parameters(), # TODO: Implement
    # 'pad': Parameters(), # TODO: Implement
    # 'partition': Parameters(), # TODO: Implement
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
    # 'prod': Parameters(), # TODO: Implement
    # 'product': Parameters(), # TODO: Implement
    # 'ptp': Parameters(), # TODO: Implement
    # 'put': Parameters(), # TODO: Implement
    # 'pv': Parameters(), # TODO: Implement
    # 'random': {}, # TODO: Add random functions
    # 'rank': Parameters(), # TODO: Implement
    # 'rate': Parameters(), # TODO: Implement
    # 'ravel': Parameters(), # TODO: Implement
    # 'real': Parameters(), # TODO: Implement
    # 'real_if_close': Parameters(), # TODO: Implement
    # 'rec': {} # TODO: Add rec functions
    # 'recfromcsv': Parameters(), # TODO: Implement
    # 'recfromtxt': Parameters(), # TODO: Implement
    # 'repeat': Parameters(), # TODO: Implement
    # 'require': Parameters(), # TODO: Implement
    # 'reshape': Parameters(), # TODO: Implement
    # 'resize': Parameters(), # TODO: Implement
    # 'restoredot': Parameters(), # TODO: Implement
    # 'roll': Parameters(), # TODO: Implement
    # 'rollaxis': Parameters(), # TODO: Implement
    # 'roots': Parameters(), # TODO: Implement
    # 'rot90': Parameters(), # TODO: Implement
    # 'round_': Parameters(), # TODO: Implement
    # 'row_stack': Parameters(), # TODO: Implement
    # 'safe_eval': Parameters(), # TODO: Implement
    # 'save': Parameters(), # TODO: Implement
    # 'savetxt': Parameters(), # TODO: Implement
    # 'savez': Parameters(), # TODO: Implement
    # 'savez_compressed': Parameters(), # TODO: Implement
    # 'sctype2char': Parameters(), # TODO: Implement
    # 'searchsorted': Parameters(), # TODO: Implement
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
    # 'size': Parameters(), # TODO: Implement
    # 'sometrue': Parameters(), # TODO: Implement
    # 'sort': Parameters(), # TODO: Implement
    # 'sort_complex': Parameters(), # TODO: Implement
    # 'source': Parameters(), # TODO: Implement
    # 'split': Parameters(), # TODO: Implement
    # 'squeeze': Parameters(), # TODO: Implement
    # 'stack': Parameters(), # TODO: Implement
    # 'std': Parameters(), # TODO: Implement
    # 'sum': Parameters(), # TODO: Implement
    # 'swapaxes': Parameters(), # TODO: Implement
    # 'take': Parameters(), # TODO: Implement
    # 'tensordot': Parameters(), # TODO: Implement
    # 'tile': Parameters(), # TODO: Implement
    # 'trace': Parameters(), # TODO: Implement
    # 'transpose': Parameters(), # TODO: Implement
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
    # 'union1d': Parameters(), # TODO: Implement
    # 'unique': Parameters(), # TODO: Implement
    # 'unwrap': Parameters(), # TODO: Implement
    # 'vander': Parameters(), # TODO: Implement
    # 'var': Parameters(), # TODO: Implement
    # 'vsplit': Parameters(), # TODO: Implement
    # 'vstack': Parameters(), # TODO: Implement
    # 'who': Parameters(), # TODO: Implement
    # 'zeros_like': Parameters() # TODO: Implement
}


# All numpy ndarray member functions and attributes
NDARRAY_MEMBERS = {
    'T': Parameters(transform_axes=reverse_axes),
    '__abs__': Parameters(),
    '__add__': Parameters(),
    '__and__': Parameters(),
    '__array__': Parameters(),
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
    '__ge__': Parameters(),
    # '__getattribute__': Parameters(),
    '__getitem__': Parameters(),
    '__getslice__': Parameters(),
    '__gt__': Parameters(),
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
    '__init__': Parameters(),
    '__int__': Parameters(),
    '__invert__': Parameters(),
    '__ior__': Parameters(),
    '__ipow__': Parameters(),
    '__irshift__': Parameters(),
    '__isub__': Parameters(),
    '__iter__': Parameters(),
    '__itruediv__': Parameters(),
    '__ixor__': Parameters(),
    '__le__': Parameters(),
    '__len__': Parameters(),
    '__long__': Parameters(),
    '__lshift__': Parameters(),
    '__lt__': Parameters(),
    '__mod__': Parameters(),
    '__mul__': Parameters(),
    '__ne__': Parameters(),
    '__neg__': Parameters(),
    '__new__': Parameters(),
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
    '__setitem__': Parameters(),
    '__setslice__': Parameters(),
    # '__setstate__': Parameters(),
    # '__sizeof__': Parameters(),
    '__str__': Parameters(),
    '__sub__': Parameters(),
    # '__subclasshook__': Parameters(),
    '__truediv__': Parameters(),
    '__xor__': Parameters(),
    'all': Parameters(transform_axes=remove_axis),
    'any': Parameters(transform_axes=remove_axis),
    'argmax': Parameters(transform_axes=only_axis),
    'argmin': Parameters(transform_axes=only_axis),
    'argpartition': Parameters(transform_axes=only_axis),
    'argsort': Parameters(transform_axes=only_axis),
    'astype': Parameters(),
    'base': Parameters(),
    'byteswap': Parameters(),
    'choose': Parameters(),  # TODO: transform_axes
    'clip': Parameters(),
    'compress': Parameters(),
    'conj': Parameters(),
    'conjugate': Parameters(),
    'copy': Parameters(),
    'ctypes': Parameters(),
    'cumprod': Parameters(),  # TODO: transform_axes
    'cumsum': Parameters(),  # TODO: transform_axes
    # 'data': Parameters(),
    'diagonal': Parameters(),
    'dot': Parameters(),
    'dtype': Parameters(),
    # 'dump': Parameters(),
    # 'dumps': Parameters(),
    'fill': Parameters(),
    'flags': Parameters(),
    'flat': Parameters(),
    'flatten': Parameters(transform_axes=only_singular_axes),
    'getfield': Parameters(),
    'imag': Parameters(),
    'item': Parameters(),
    'itemset': Parameters(),
    'itemsize': Parameters(),
    'max': Parameters(transform_axes=remove_axis),
    'mean': Parameters(transform_axes=remove_axis),
    'min': Parameters(transform_axes=remove_axis),
    'nbytes': Parameters(),
    'ndim': Parameters(),
    'newbyteorder': Parameters(),
    'nonzero': Parameters(),
    'partition': Parameters(),
    'prod': Parameters(transform_axes=remove_axis),
    'ptp': Parameters(transform_axes=remove_axis),
    'put': Parameters(),
    'ravel': Parameters(transform_axes=only_singular_axes),
    'real': Parameters(),
    'repeat': Parameters(),  # TODO: transform_axes
    'reshape': Parameters(),  # TODO: transform_axes
    'resize': Parameters(),  # TODO: transform_axes
    'round': Parameters(),
    'searchsorted': Parameters(),  # TODO
    'setfield': Parameters(),
    'setflags': Parameters(),
    # 'shape': Parameters(),
    'size': Parameters(),  # TODO
    'sort': Parameters(),
    'squeeze': Parameters(),  # TODO: transform_axes
    'std': Parameters(transform_axes=remove_axis),
    'strides': Parameters(),
    'sum': Parameters(transform_axes=remove_axis),
    'swapaxes': Parameters(transform_axes=swap_axes, transform_args=transform_swap_axes_args),  # TODO: transform_axes, transform_args
    'take': Parameters(transform_axes=insert_axes),
    'tobytes': Parameters(),
    # 'tofile': Parameters(),
    # 'tolist': Parameters(),
    # 'tostring': Parameters(),
    'trace': Parameters(),
    'transpose': Parameters(transform_axes=reverse_axes),
    'var': Parameters(transform_axes=remove_axis),
    'view': Parameters()
}


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
        - If `what` specifies a `transform_axes` function, invoke it with the arguments to the
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

        # transform_axes needs to use original arguments, so we invoke it first
        transformed_axes = transform_axes(params, first, *args, **kwargs)

        # Prepare arguments for numpy function
        args, kwargs = transform_args(params, first, *args, **kwargs)

        # Obtain a result
        result = func(*args, **kwargs)

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


def transform_axes(params, first, *args, **kwargs):
    if params.transform_axes is not None:
        return params.transform_axes(*args, **kwargs)
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
        if not hasattr(destination, name):
            setattr(destination, name, proxy)
