#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
from setuptools import setup
import sys


# Python version
if sys.version_info[:2] < (3, 3):
    print('DGFS requires Python 3.3 or newer')
    sys.exit(-1)

# DGFS version
vfile = open('dgfs1D/_version.py').read()
vsrch = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", vfile, re.M)

if vsrch:
    version = vsrch.group(1)
else:
    print('Unable to find a version string in dgfs1D/_version.py')

# Modules
modules = [
    'dgfs1D.sphericaldesign',

    'dgfs1D.std',
    'dgfs1D.std.kernels',
    'dgfs1D.std.kernels.bcs',
    'dgfs1D.std.kernels.scattering',

    'dgfs1D.bi',
    'dgfs1D.bi.kernels',
    'dgfs1D.bi.kernels.bcs',
    'dgfs1D.bi.kernels.scattering'
]

# Data
package_data = {
    'dgfs1D.sphericaldesign': [
        'symmetric/*.txt'
    ],

    'dgfs1D.std.kernels': ['*.mako'],
    'dgfs1D.std.kernels.bcs': ['*.mako'],
    'dgfs1D.std.kernels.scattering': ['*.mako'],

    'dgfs1D.bi.kernels': ['*.mako'],
    'dgfs1D.bi.kernels.bcs': ['*.mako'],
    'dgfs1D.bi.kernels.scattering': ['*.mako']
}

# Hard dependencies
install_requires = [
    'appdirs >= 1.4.0',
    'gimmik >= 2.0',
    'h5py >= 2.6',
    'mako >= 1.0.0',
    'numpy >= 1.8',
    'pytools >= 2016.2.1',
    'pycuda >= 2015.1'
]

# Soft dependencies
extras_require = {
    
}

# Scripts
console_scripts = [
    'dgfsStd1D = dgfs1D.std.std:__main__',
    'dgfsAsymStd1D = dgfs1D.std.astd:__main__',
    'dgfsBi1D = dgfs1D.bi.bi:__main__'
]

# Info
classifiers = [
    'License :: GNU GPL v2',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3.3',
    'Topic :: Scientific/Engineering'
]

long_description = '''dgfs1D is an open-source minimalistic implementation of 
Discontinuous Galerkin Fast Spectral method in one dimension'''

setup(name='dgfs1D',
      version=version,
      description='Discontinuous Galerkin Fast Spectral in One Dimension',
      long_description=long_description,
      author='Purdue University, West Lafayette',
      author_email='jaisw7@gmail.com',
      url='http://www.github.com/jaisw7',
      license='GNU GPL v2',
      keywords='Applied Mathematics',
      packages=['dgfs1D'] + modules,
      package_data=package_data,
      entry_points={'console_scripts': console_scripts},
      install_requires=install_requires,
      extras_require=extras_require,
      classifiers=classifiers
)
