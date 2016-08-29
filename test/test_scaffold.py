"""
Test the API of the Block and Scaffold object from npdimension.
"""

from __future__ import print_function
import numpy as np
import npdimension as npd

# pylint: disable=E1101

def test_initialization():
    # Measure the elevation of a mountain range in tibet over time
    data = np.random.randint(0, 5, size=(4, 5, 3))
    elevation = npd.Block(data, axes=['latitude', 'longitude', 'time'])
    latitude = npd.Block(np.linspace(30.29103, 31.29103, num=4), axes=['latitude'])
    longitude = npd.Block(np.linspace(92.41153, 93.41153, 5), axes=['longitude'])

    # Build the scaffold
    scaffold = npd.Scaffold()
    scaffold['elevation'] = elevation
    scaffold['latitude'] = latitude
    scaffold['longitude'] = longitude

    # Check to make sure the sizes are correct
    # print(scaffold.shape)

    # Slice a smaller portion of the map
    band = scaffold(latitude=[0, 1])
#    print(band)
