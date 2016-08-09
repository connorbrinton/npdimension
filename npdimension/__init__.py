"""
Label dimensions of numpy ndarrays and perform bulk operations on aligned ndarrays in a dict-like
container.
"""

from block import Block
from scaffold import Scaffold
import operators
from operators import *

import linalg

__all__ = ['Block', 'Scaffold', 'linalg'] + operators.__all__
