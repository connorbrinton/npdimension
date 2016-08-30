import nose.tools
import numpy as np
import npdimension as npd

def test_indexer_values():
    # Test correct values
    indexer = npd.NPDIndexer(npd.Block(np.arange(9) + 100, axes=['one']),
                             npd.Block(np.arange(9), axes=['one']))
    assert indexer[0] == 100
    assert indexer[8] == 108
    assert np.all(indexer[[0, 1]] == [100, 101])

def test_indexer_axes():
    # Test axes produced by indexing
    indexer = npd.NPDIndexer(npd.Block(np.arange(9) + 100, axes=['one']),
                             npd.Block(np.arange(9), axes=['one']))

    assert indexer[[0]].axes == ['one']
    assert indexer[npd.Block([[0, 1]], axes=['one', 'two'])].axes == ['one', 'two']
    nose.tools.assert_raises(Exception, '__getitem__', indexer, [[0, 1]])
