import os
import sys
from setuptools import setup, find_namespace_packages, Extension
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
                  'libraries': ['fftw3', 'fftw3f', 'fftw3_omp', 'fftw3f_omp'],
                  'library_dirs': ['/usr/local/lib',
                                   os.path.join(sys.prefix, 'lib')],
                  'include_dirs': [numpy.get_include(),
                                   os.path.join(sys.prefix, 'include'),
                                   os.path.join(os.path.dirname(__file__), 'cbclib/include')]}

extensions = [Extension(name='cbclib.bin.cbc_indexing',
                        sources=['cbclib/bin/cbc_indexing' + ext, 'cbclib/include/array.c',
                                 'cbclib/include/geometry.c'],
                        **extension_args),
              Extension(name='cbclib.bin.image_proc',
                        sources=['cbclib/bin/image_proc' + ext, 'cbclib/include/pocket_fft.c',
                                 'cbclib/include/img_proc.c', 'cbclib/include/fft_functions.c',
                                 'cbclib/include/median.c', 'cbclib/include/array.c',
                                 'cbclib/include/cephes.c', 'cbclib/include/kd_tree.c'],
                        **extension_args),
              Extension(name='cbclib.bin.line_detector',
                        sources=['cbclib/bin/line_detector' + ext, 'cbclib/include/lsd.c',
                                 'cbclib/include/img_proc.c', 'cbclib/include/array.c',
                                 'cbclib/include/cephes.c', 'cbclib/include/kd_tree.c'],
                        **extension_args),
              Extension(name='cbclib.bin.signal_proc',
                        sources=['cbclib/bin/signal_proc' + ext, 'cbclib/include/sgn_proc.c',
                                 'cbclib/include/array.c'],
                        **extension_args),
              Extension(name='cbclib.bin.pyfftw',
                        sources=['cbclib/bin/pyfftw' + ext], **extension_args)]

if USE_CYTHON:
    extensions = cythonize(extensions, annotate=True, language_level="3",
                           include_path=['cbclib/bin',],
                           compiler_directives={'cdivision': True,
                                                'boundscheck': False,
                                                'wraparound': False,
                                                'binding': True,
                                                'embedsignature': True})

with open('README.md', 'r') as readme:
    long_description = readme.read()

setup(name='cbclib',
      version='0.7.2',
      author='Nikolay Ivanov',
      author_email="nikolay.ivanov@desy.de",
      long_description=long_description,
      long_description_content_type='text/markdown',
      url="https://github.com/simply-nicky/cbclib",
      packages=find_namespace_packages(),
      include_package_data=True,
      install_requires=['h5py', 'numpy', 'scipy'],
      ext_modules=extensions,
      classifiers=[
          "Programming Language :: Python",
          "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
          "Operating System :: OS Independent"
      ],
      python_requires='>=3.6')
