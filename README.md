# cbclib
Convergent beam crystallography project (**cbclib**) is a library for
data processing of convergent beam crystallography datasets.

## Dependencies

- [Python](https://www.python.org/) 3.6 or later (Python 2.x is **not** supported).
- [h5py](https://www.h5py.org) 2.10.0 or later.
- [NumPy](https://numpy.org) 1.19.0 or later.
- [SciPy](https://scipy.org) 1.5.2 or later.

## Installation from source
In order to build the package from source simply execute the following command:

    python setup.py install

or:

    pip install -r requirements.txt -e . -v

That cythonizes the Cython extensions and builds them into ``/cbclib/bin``.