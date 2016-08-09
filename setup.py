"""
Setup script for npdimension.
"""

from setuptools import setup, Command
import os
import sys

packages = ['npdimension']
scripts = []
package_data = { }

# TODO: Add dependency on numpy
setup(name='npdimension',
      version='0.0.1',
      description="Label dimensions of numpy ndarrays and perform bulk operations on aligned"
                  "ndarrays in a dict-like container.",
      author='Connor Brinton',
      author_email='connor.brinton+NPD@gmail.com',
      url='https://github.com/connorbrinton/npdimension',
      license='Proprietary',
      package_data=package_data,
      packages=packages,
      scripts=scripts,
      test_suite='nose2.collector.collector')
