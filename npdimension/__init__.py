"""
Label dimensions of numpy ndarrays and perform bulk operations on aligned ndarrays in a dict-like
container.
"""

# Imports
import sys
import numpy

import npd
import block
import scaffold
import npproxy

# Load numpy members from npproxy
npproxy.load_proxies(numpy, npproxy.NP_MEMBERS, npd)
npproxy.load_proxies(numpy.ndarray, npproxy.NDARRAY_MEMBERS, block.Block)
npproxy.load_proxies(numpy.ndarray, npproxy.NDARRAY_MEMBERS, scaffold.Scaffold)

# Load all top-level members
from npd import *
from block import Block
from scaffold import Scaffold
from npdindexer import NPDIndexer

__all__ = ['Block', 'NPDIndexer', 'Scaffold'] + list(npproxy.NP_MEMBERS.keys())
