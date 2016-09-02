"""
Label dimensions of numpy ndarrays and perform bulk operations on aligned ndarrays in a dict-like
container.
"""

# Scientific imports
import numpy

# Local imports
import npd
import npproxy

# Inject all members
npproxy.load_proxies(numpy, npproxy.NP_MEMBERS, npd)
npproxy.load_proxies(numpy.ndarray, npproxy.NDARRAY_MEMBERS, npd.Block)
npproxy.load_proxies(numpy.ndarray, npproxy.NDARRAY_MEMBERS, npd.Scaffold)

# Import results and other members
from npd import *

__all__ = ['Block', 'NPDIndexer', 'Scaffold'] + list(npproxy.NP_MEMBERS.keys())
