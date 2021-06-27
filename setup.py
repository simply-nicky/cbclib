import os
import sys
from distutils.core import Extension
from setuptools import setup, find_packages
import numpy

try:
    from Cython.Build import cythonize
except ImportError:
    USE_CYTHON = False
else:
    USE_CYTHON = True

ext = '.pyx' if USE_CYTHON else '.c'
extension_args = {'language': 'c',
                  'extra_compile_args': ['-fopenmp', '-std=c99'],
                  'extra_link_args': ['-lgomp', '-Wl,-rpath,/usr/local/lib'],
                  'libraries': ['fftw3', 'fftw3_omp'],
                  'library_dirs': ['/usr/local/lib',
                                   os.path.join(sys.prefix, 'lib')],
                  'include_dirs': [numpy.get_include(),
                                   os.path.join(sys.prefix, 'include'),
                                   os.path.join(os.path.dirname(__file__), 'cbclib/include')]}

src_files = ['cbclib/include/lsd.c',]
extensions = [Extension(name='cbclib.bin.line_detector',
                        sources=['cbclib/bin/line_detecotr' + ext,] + src_files, **extension_args),]

if USE_CYTHON:
    extensions = cythonize(extensions, annotate=True, language_level="3",
                           compiler_directives={'cdivision': True,
                                                'boundscheck': False,
                                                'wraparound': False,
                                                'binding': True,
                                                'embedsignature': True})

with open('README.md', 'r') as readme:
    long_description = readme.read()

setup(name='cbclib',
      version='0.1.0',
      long_description=long_description,
      long_description_content_type='text/markdown',
      packages=find_packages(),
      install_requires=['h5py', 'numpy', 'scipy'],
      ext_modules=extensions,
      classifiers=[
          "Programming Language :: Python",
          "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
          "Operating System :: OS Independent"
      ],
      python_requires='>=3.6')