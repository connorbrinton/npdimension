"""
Label dimensions of numpy ndarrays and perform bulk operations on aligned ndarrays in a dict-like
container.
"""

# Imports
import sys
import numpy
from block import Block
from scaffold import Scaffold
import npproxy

# Load numpy members from npproxy
npproxy.load_proxies(numpy, npproxy.NP_MEMBERS, sys.modules[__name__])
npproxy.load_proxies(numpy.ndarray, npproxy.NDARRAY_MEMBERS, Block)
npproxy.load_proxies(numpy.ndarray, npproxy.NDARRAY_MEMBERS, Scaffold)

__all__ = ['Block', 'Scaffold'] + list(npproxy.NP_MEMBERS.keys())
