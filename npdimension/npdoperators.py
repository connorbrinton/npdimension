"""
Define all npdimension operators.
"""
# pylint: disable=C0111
# pylint: disable=W0108
# pylint: disable=W0613

from collections import namedtuple
import numpy as np
from block import Block
from scaffold import Scaffold

__all__ = ['load_ndarray_members']


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

    # Find the index of the axis argument in the first axes
    index = first.axes.index(kwargs['axis'])

    # Substitute the second Block's axes where the specified axis exists
    return first[:index] + second.axes + first[(index + 1):]


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
Member = namedtuple('Member', ['name', 'transform_axes', 'transform_args'])

# Provide Member defaults
Member.__new__.__defaults__ = (None,) * (len(Member._fields) - 1)

# All numpy ndarray member functions and attributes
NDARRAY_MEMBERS = [
    Member(name='T', transform_axes=reverse_axes),
    Member(name='__abs__'),
    Member(name='__add__'),
    Member(name='__and__'),
    Member(name='__array__'),
    # Member(name='__array_finalize__'),
    # Member(name='__array_interface__'),
    # Member(name='__array_prepare__'),
    # Member(name='__array_priority__'),
    # Member(name='__array_struct__'),
    # Member(name='__array_wrap__'),
    # Member(name='__class__'),
    Member(name='__contains__'),
    Member(name='__copy__'),
    Member(name='__deepcopy__'),
    # Member(name='__delattr__'),
    Member(name='__delitem__'),
    Member(name='__delslice__'),
    Member(name='__div__'),
    Member(name='__divmod__'),
    # Member(name='__doc__'),
    Member(name='__eq__'),
    Member(name='__float__'),
    Member(name='__floordiv__'),
    # Member(name='__format__'),
    Member(name='__ge__'),
    # Member(name='__getattribute__'),
    Member(name='__getitem__'),
    Member(name='__getslice__'),
    Member(name='__gt__'),
    Member(name='__hash__'),
    Member(name='__hex__'),
    Member(name='__iadd__'),
    Member(name='__iand__'),
    Member(name='__idiv__'),
    Member(name='__ifloordiv__'),
    Member(name='__ilshift__'),
    Member(name='__imod__'),
    Member(name='__imul__'),
    Member(name='__index__'),
    Member(name='__init__'),
    Member(name='__int__'),
    Member(name='__invert__'),
    Member(name='__ior__'),
    Member(name='__ipow__'),
    Member(name='__irshift__'),
    Member(name='__isub__'),
    Member(name='__iter__'),
    Member(name='__itruediv__'),
    Member(name='__ixor__'),
    Member(name='__le__'),
    Member(name='__len__'),
    Member(name='__long__'),
    Member(name='__lshift__'),
    Member(name='__lt__'),
    Member(name='__mod__'),
    Member(name='__mul__'),
    Member(name='__ne__'),
    Member(name='__neg__'),
    Member(name='__new__'),
    Member(name='__nonzero__'),
    Member(name='__oct__'),
    Member(name='__or__'),
    Member(name='__pos__'),
    Member(name='__pow__'),
    Member(name='__radd__'),
    Member(name='__rand__'),
    Member(name='__rdiv__'),
    Member(name='__rdivmod__'),
    # Member(name='__reduce__'),
    # Member(name='__reduce_ex__'),
    # Member(name='__repr__'),
    Member(name='__rfloordiv__'),
    Member(name='__rlshift__'),
    Member(name='__rmod__'),
    Member(name='__rmul__'),
    Member(name='__ror__'),
    Member(name='__rpow__'),
    Member(name='__rrshift__'),
    Member(name='__rshift__'),
    Member(name='__rsub__'),
    Member(name='__rtruediv__'),
    Member(name='__rxor__'),
    # Member(name='__setattr__'),
    Member(name='__setitem__'),
    Member(name='__setslice__'),
    # Member(name='__setstate__'),
    # Member(name='__sizeof__'),
    Member(name='__str__'),
    Member(name='__sub__'),
    # Member(name='__subclasshook__'),
    Member(name='__truediv__'),
    Member(name='__xor__'),
    Member(name='all', transform_axes=remove_axis),
    Member(name='any', transform_axes=remove_axis),
    Member(name='argmax', transform_axes=only_axis),
    Member(name='argmin', transform_axes=only_axis),
    Member(name='argpartition', transform_axes=only_axis),
    Member(name='argsort', transform_axes=only_axis),
    Member(name='astype'),
    Member(name='base'),
    Member(name='byteswap'),
    Member(name='choose'),  # TODO: transform_axes
    Member(name='clip'),
    Member(name='compress'),
    Member(name='conj'),
    Member(name='conjugate'),
    Member(name='copy'),
    Member(name='ctypes'),
    Member(name='cumprod'),  # TODO: transform_axes
    Member(name='cumsum'),  # TODO: transform_axes
    # Member(name='data'),
    Member(name='diagonal'),
    Member(name='dot'),
    Member(name='dtype'),
    # Member(name='dump'),
    # Member(name='dumps'),
    Member(name='fill'),
    Member(name='flags'),
    Member(name='flat'),
    Member(name='flatten', transform_axes=only_singular_axes),
    Member(name='getfield'),
    Member(name='imag'),
    Member(name='item'),
    Member(name='itemset'),
    Member(name='itemsize'),
    Member(name='max', transform_axes=remove_axis),
    Member(name='mean', transform_axes=remove_axis),
    Member(name='min', transform_axes=remove_axis),
    Member(name='nbytes'),
    Member(name='ndim'),
    Member(name='newbyteorder'),
    Member(name='nonzero'),
    Member(name='partition'),
    Member(name='prod', transform_axes=remove_axis),
    Member(name='ptp', transform_axes=remove_axis),
    Member(name='put'),
    Member(name='ravel', transform_axes=only_singular_axes),
    Member(name='real'),
    Member(name='repeat'),  # TODO: transform_axes
    Member(name='reshape'),  # TODO: transform_axes
    Member(name='resize'),  # TODO: transform_axes
    Member(name='round'),
    Member(name='searchsorted'),  # TODO
    Member(name='setfield'),
    Member(name='setflags'),
    # Member(name='shape'),
    Member(name='size'),  # TODO
    Member(name='sort'),
    Member(name='squeeze'),  # TODO: transform_axes
    Member(name='std', transform_axes=remove_axis),
    Member(name='strides'),
    Member(name='sum', transform_axes=remove_axis),
    Member(name='swapaxes', transform_axes=swap_axes, transform_args=transform_swap_axes_args),  # TODO: transform_axes, transform_args
    Member(name='take', transform_axes=insert_axes),
    Member(name='tobytes'),
    # Member(name='tofile'),
    # Member(name='tolist'),
    # Member(name='tostring'),
    Member(name='trace'),
    Member(name='transpose', transform_axes=reverse_axes),
    Member(name='var', transform_axes=remove_axis),
    Member(name='view')
]


# NP_FUNCTIONS = {
#     'alen': Member(name='alen'),
#     'all': Member(name='all'),
#     'allclose': Member(name='allclose'),
#     'alltrue': Member(name='alltrue'),
#     'alterdot': Member(name='alterdot'),
#     'amax': Member(name='amax'),
#     'amin': Member(name='amin'),
#     'angle': Member(name='angle'),
#     'any': Member(name='any'),
#     'append': Member(name='append'),
#     'apply_along_axis': Member(name='apply_along_axis'),
#     'apply_over_axes': Member(name='apply_over_axes'),
#     'argmax': Member(name='argmax'),
#     'argmin': Member(name='argmin'),
#     'argpartition': Member(name='argpartition'),
#     'argsort': Member(name='argsort'),
#     'argwhere': Member(name='argwhere'),
#     'around': Member(name='around'),
#     'array2string': Member(name='array2string'),
#     'array_equal': Member(name='array_equal'),
#     'array_equiv': Member(name='array_equiv'),
#     'array_repr': Member(name='array_repr'),
#     'array_split': Member(name='array_split'),
#     'array_str': Member(name='array_str'),
#     'asanyarray': Member(name='asanyarray'),
#     'asarray': Member(name='asarray'),
#     'asarray_chkfinite': Member(name='asarray_chkfinite'),
#     'ascontiguousarray': Member(name='ascontiguousarray'),
#     'asfarray': Member(name='asfarray'),
#     'asfortranarray': Member(name='asfortranarray'),
#     'asmatrix': Member(name='asmatrix'),
#     'asscalar': Member(name='asscalar'),
#     'atleast_1d': Member(name='atleast_1d'),
#     'atleast_2d': Member(name='atleast_2d'),
#     'atleast_3d': Member(name='atleast_3d'),
#     'average': Member(name='average'),
#     'bartlett': Member(name='bartlett'),
#     'base_repr': Member(name='base_repr'),
#     'binary_repr': Member(name='binary_repr'),
#     'blackman': Member(name='blackman'),
#     'bmat': Member(name='bmat'),
#     'broadcast_arrays': Member(name='broadcast_arrays'),
#     'broadcast_to': Member(name='broadcast_to'),
#     'byte_bounds': Member(name='byte_bounds'),
#     'char': {}, # TODO: Add char functions
#     'choose': Member(name='choose'),
#     'clip': Member(name='clip'),
#     'column_stack': Member(name='column_stack'),
#     'common_type': Member(name='common_type'),
#     'compress': Member(name='compress'),
#     'convolve': Member(name='convolve'),
#     'copy': Member(name='copy'),
#     'corrcoef': Member(name='corrcoef'),
#     'correlate': Member(name='correlate'),
#     'cov': Member(name='cov'),
#     'cross': Member(name='cross'),
#     'cumprod': Member(name='cumprod'),
#     'cumproduct': Member(name='cumproduct'),
#     'cumsum': Member(name='cumsum'),
#     'delete': Member(name='delete'),
#     'deprecate': Member(name='deprecate'),
#     'deprecate_with_doc': Member(name='deprecate_with_doc'),
#     'diag': Member(name='diag'),
#     'diag_indices': Member(name='diag_indices'),
#     'diag_indices_from': Member(name='diag_indices_from'),
#     'diagflat': Member(name='diagflat'),
#     'diagonal': Member(name='diagonal'),
#     'diff': Member(name='diff'),
#     'disp': Member(name='disp'),
#     'dsplit': Member(name='dsplit'),
#     'dstack': Member(name='dstack'),
#     'ediff1d': Member(name='ediff1d'),
#     'emath': {} # TODO: Add emath functions
#     'expand_dims': Member(name='expand_dims'),
#     'extract': Member(name='extract'),
#     'eye': Member(name='eye'),
#     'fft': {}, # TODO: Add fft functions
#     'fill_diagonal': Member(name='fill_diagonal'),
#     'find_common_type': Member(name='find_common_type'),
#     'fix': Member(name='fix'),
#     'flatnonzero': Member(name='flatnonzero'),
#     'fliplr': Member(name='fliplr'),
#     'flipud': Member(name='flipud'),
#     'fromfunction': Member(name='fromfunction'),
#     'fromregex': Member(name='fromregex'),
#     'full': Member(name='full'),
#     'full_like': Member(name='full_like'),
#     'fv': Member(name='fv'),
#     'genfromtxt': Member(name='genfromtxt'),
#     'get_array_wrap': Member(name='get_array_wrap'),
#     'get_include': Member(name='get_include'),
#     'get_printoptions': Member(name='get_printoptions'),
#     'getbufsize': Member(name='getbufsize'),
#     'geterr': Member(name='geterr'),
#     'geterrcall': Member(name='geterrcall'),
#     'gradient': Member(name='gradient'),
#     'hamming': Member(name='hamming'),
#     'hanning': Member(name='hanning'),
#     'histogram': Member(name='histogram'),
#     'histogram2d': Member(name='histogram2d'),
#     'histogramdd': Member(name='histogramdd'),
#     'hsplit': Member(name='hsplit'),
#     'hstack': Member(name='hstack'),
#     'i0': Member(name='i0'),
#     'identity': Member(name='identity'),
#     'imag': Member(name='imag'),
#     'in1d': Member(name='in1d'),
#     'indices': Member(name='indices'),
#     'info': Member(name='info'),
#     'insert': Member(name='insert'),
#     'interp': Member(name='interp'),
#     'intersect1d': Member(name='intersect1d'),
#     'ipmt': Member(name='ipmt'),
#     'irr': Member(name='irr'),
#     'isclose': Member(name='isclose'),
#     'iscomplex': Member(name='iscomplex'),
#     'iscomplexobj': Member(name='iscomplexobj'),
#     'isfortran': Member(name='isfortran'),
#     'isneginf': Member(name='isneginf'),
#     'isposinf': Member(name='isposinf'),
#     'isreal': Member(name='isreal'),
#     'isrealobj': Member(name='isrealobj'),
#     'isscalar': Member(name='isscalar'),
#     'issctype': Member(name='issctype'),
#     'issubclass_': Member(name='issubclass_'),
#     'issubdtype': Member(name='issubdtype'),
#     'issubsctype': Member(name='issubsctype'),
#     'iterable': Member(name='iterable'),
#     'ix_': Member(name='ix_'),
#     'kaiser': Member(name='kaiser'),
#     'kron': Member(name='kron'),
#     'linalg': {}, # TODO: Add linalg functions
#     'linspace': Member(name='linspace'),
#     'load': Member(name='load'),
#     'loadtxt': Member(name='loadtxt'),
#     'logspace': Member(name='logspace'),
#     'lookfor': Member(name='lookfor'),
#     'ma': {}, # TODO: Add ma functions
#     'mafromtxt': Member(name='mafromtxt'),
#     'mask_indices': Member(name='mask_indices'),
#     'mat': Member(name='mat'),
#     'math': {}, # TODO: Add math functions
#     'maximum_sctype': Member(name='maximum_sctype'),
#     'mean': Member(name='mean'),
#     'median': Member(name='median'),
#     'meshgrid': Member(name='meshgrid'),
#     'mintypecode': Member(name='mintypecode'),
#     'mirr': Member(name='mirr'),
#     'moveaxis': Member(name='moveaxis'),
#     'msort': Member(name='msort'),
#     'nan_to_num': Member(name='nan_to_num'),
#     'nanargmax': Member(name='nanargmax'),
#     'nanargmin': Member(name='nanargmin'),
#     'nanmax': Member(name='nanmax'),
#     'nanmean': Member(name='nanmean'),
#     'nanmedian': Member(name='nanmedian'),
#     'nanmin': Member(name='nanmin'),
#     'nanpercentile': Member(name='nanpercentile'),
#     'nanprod': Member(name='nanprod'),
#     'nanstd': Member(name='nanstd'),
#     'nansum': Member(name='nansum'),
#     'nanvar': Member(name='nanvar'),
#     'ndfromtxt': Member(name='ndfromtxt'),
#     'ndim': Member(name='ndim'),
#     'nonzero': Member(name='nonzero'),
#     'nper': Member(name='nper'),
#     'npv': Member(name='npv'),
#     'obj2sctype': Member(name='obj2sctype'),
#     'ones': Member(name='ones'),
#     'ones_like': Member(name='ones_like'),
#     'outer': Member(name='outer'),
#     'pad': Member(name='pad'),
#     'partition': Member(name='partition'),
#     'percentile': Member(name='percentile'),
#     'piecewise': Member(name='piecewise'),
#     'pkgload': Member(name='pkgload'),
#     'place': Member(name='place'),
#     'pmt': Member(name='pmt'),
#     'poly': Member(name='poly'),
#     'polyadd': Member(name='polyadd'),
#     'polyder': Member(name='polyder'),
#     'polydiv': Member(name='polydiv'),
#     'polyfit': Member(name='polyfit'),
#     'polyint': Member(name='polyint'),
#     'polymul': Member(name='polymul'),
#     'polysub': Member(name='polysub'),
#     'polyval': Member(name='polyval'),
#     'ppmt': Member(name='ppmt'),
#     'prod': Member(name='prod'),
#     'product': Member(name='product'),
#     'ptp': Member(name='ptp'),
#     'put': Member(name='put'),
#     'pv': Member(name='pv'),
#     'random': {}, # TODO: Add random functions
#     'rank': Member(name='rank'),
#     'rate': Member(name='rate'),
#     'ravel': Member(name='ravel'),
#     'real': Member(name='real'),
#     'real_if_close': Member(name='real_if_close'),
#     'rec': {} # TODO: Add rec functions
#     'recfromcsv': Member(name='recfromcsv'),
#     'recfromtxt': Member(name='recfromtxt'),
#     'repeat': Member(name='repeat'),
#     'require': Member(name='require'),
#     'reshape': Member(name='reshape'),
#     'resize': Member(name='resize'),
#     'restoredot': Member(name='restoredot'),
#     'roll': Member(name='roll'),
#     'rollaxis': Member(name='rollaxis'),
#     'roots': Member(name='roots'),
#     'rot90': Member(name='rot90'),
#     'round_': Member(name='round_'),
#     'row_stack': Member(name='row_stack'),
#     'safe_eval': Member(name='safe_eval'),
#     'save': Member(name='save'),
#     'savetxt': Member(name='savetxt'),
#     'savez': Member(name='savez'),
#     'savez_compressed': Member(name='savez_compressed'),
#     'sctype2char': Member(name='sctype2char'),
#     'searchsorted': Member(name='searchsorted'),
#     'select': Member(name='select'),
#     'set_printoptions': Member(name='set_printoptions'),
#     'set_string_function': Member(name='set_string_function'),
#     'setbufsize': Member(name='setbufsize'),
#     'setdiff1d': Member(name='setdiff1d'),
#     'seterr': Member(name='seterr'),
#     'seterrcall': Member(name='seterrcall'),
#     'setxor1d': Member(name='setxor1d'),
#     'shape': Member(name='shape'),
#     'show_config': Member(name='show_config'),
#     'sinc': Member(name='sinc'),
#     'size': Member(name='size'),
#     'sometrue': Member(name='sometrue'),
#     'sort': Member(name='sort'),
#     'sort_complex': Member(name='sort_complex'),
#     'source': Member(name='source'),
#     'split': Member(name='split'),
#     'squeeze': Member(name='squeeze'),
#     'stack': Member(name='stack'),
#     'std': Member(name='std'),
#     'sum': Member(name='sum'),
#     'swapaxes': Member(name='swapaxes'),
#     'take': Member(name='take'),
#     'tensordot': Member(name='tensordot'),
#     'tile': Member(name='tile'),
#     'trace': Member(name='trace'),
#     'transpose': Member(name='transpose'),
#     'trapz': Member(name='trapz'),
#     'tri': Member(name='tri'),
#     'tril': Member(name='tril'),
#     'tril_indices': Member(name='tril_indices'),
#     'tril_indices_from': Member(name='tril_indices_from'),
#     'trim_zeros': Member(name='trim_zeros'),
#     'triu': Member(name='triu'),
#     'triu_indices': Member(name='triu_indices'),
#     'triu_indices_from': Member(name='triu_indices_from'),
#     'typename': Member(name='typename'),
#     'union1d': Member(name='union1d'),
#     'unique': Member(name='unique'),
#     'unwrap': Member(name='unwrap'),
#     'vander': Member(name='vander'),
#     'var': Member(name='var'),
#     'vsplit': Member(name='vsplit'),
#     'vstack': Member(name='vstack'),
#     'who': Member(name='member'),
#     'zeros_like': Member(name='zeros_like')
# }



def closure(what):
    # Look up the ndarray function
    func = getattr(np.ndarray, what.name, None)
    if func is None:
        raise Exception("Function %s not found in ndarray" % what.name)

    # Create a function that can be bound to Block
    def proxy(*args, **kwargs):
        """
        Invoke the numpy function specified by the closure variable `what` on behalf of the caller.

        The arguments and kwargs are rewritten before invoking the numpy function. The following
        checks and transformations take place:
        - If a Scaffold object is present in the arguments, this function is reinvoked with each
          member of the Scaffold in turn. If the results are scalars or ndarrays, a dictionary with
          the same keys as the Scaffold and the result for each key is returned. If the results are
          all Blocks, a new Scaffold with the same keys and the result for each key is returned.
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
        # TODO: Only call functions on Scaffold members that have the axis denoted by the kwarg
        # `axis`, if specified.
        for index, arg in enumerate(args):
            if isinstance(arg, Scaffold):
                # If we find any Scaffold object, deal with each member independently
                results = {}
                for key, value in arg.iteritems():
                    # Replace the Scaffold with a member
                    args[index] = value

                    # Determine the result
                    results[key] = proxy(*args, **kwargs)

                # If all values are Blocks, create a new Scaffold object
                if all(isinstance(result, Block) for result in results.values()):
                    return Scaffold(results)

                # Otherwise, just return the dictionary
                return results

        # We figure out which Block is first in the args, which will either be `self` (for a bound
        # method) or simply the first Block for an unbound method.
        first = get_next_block(args)

        # transform_axes needs to use original arguments, so we invoke it first
        transformed_axes = transform_axes(what, first, *args, **kwargs)

        # Prepare arguments for numpy function
        args, kwargs = transform_args(what, first, *args, **kwargs)

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


def transform_axes(what, first, *args, **kwargs):
    if what.transform_axes is not None:
        return what.transform_axes(*args, **kwargs)
    else:
        return first.axes


def transform_args(what, first, *args, **kwargs):
    # Perform any function-specific argument transformation
    if what.transform_args is not None:
        args, kwargs = what.transform_args(*args, **kwargs)

    # Allow argument assignment
    args = list(args)

    # Replace `Block` arguments with ndarrays
    for index, arg in enumerate(args):
        if isinstance(arg, Block):
            args[index] = arg.data

    # Rewrite the axis keyword argument
    if 'axis' in kwargs:
        kwargs['axis'] = first.axes.index(kwargs['index'])

    return args, kwargs


def load_ndarray_members(destination):
    # Inject all ndarray functions
    for member in NDARRAY_MEMBERS:
        # Don't overwrite pre-existing members
        if not hasattr(destination, member.name):
            setattr(destination, member.name, closure(member))
