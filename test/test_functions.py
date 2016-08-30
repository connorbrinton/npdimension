import numpy as np
import npdimension as npd

def test_functions():
    block = npd.Block(np.random.randint(0, 3, size=(4, 5, 2)), axes=['latitude', 'longitude', 'time'])
    #block
    scaffold = npd.Scaffold()
    scaffold['elevation'] = block
    frames = npd.Block(np.arange(2), ['time'])
    scaffold['time'] = frames
    #scaffold
    scaffold.take(npd.Block([0, 1, 2], axes=['latitude']), axis='latitude')

    assert isinstance(block < 2, npd.Block)

def test_swapaxes():
    npd.Block(np.arange(9).reshape(-1, 3), axes=['one', 'two']).swapaxes('one', 'two')

def test_getitem():
    block = npd.Block(np.arange(9).reshape(-1, 3), axes=['one', 'two'])
    assert block[0, 0] == 0
    assert block[:, 0].axes == ['one']
    assert block[0, :].axes == ['two']
    assert block[:, :].axes == ['one', 'two']
    assert np.all(block[dict(one=0)] == npd.Block(np.arange(3), axes=['two']))
    assert np.all(block[dict()] == block)

    single = npd.Block(np.arange(9) + 10, axes=['one'])
    assert single[0] == 10
    assert single[8] == 18
    assert np.all(single[[0, 1]] == [10, 11])
