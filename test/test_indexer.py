import numpy as np
import npdimension as npd

def test_indexer():
    indexer = npd.NPDIndexer(npd.Block(np.arange(9) + 100, axes=['one']),
                             npd.Block(np.arange(9), axes=['one']))
    assert indexer[0] == 100
    assert indexer[8] == 108
    assert np.all(indexer[[0, 1]] == [100, 101])
