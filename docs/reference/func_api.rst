Low-level functions
===================

Most of the heavy computations are performed by functions written in C. The corresponding function are wrapped
with `Cython <https://cython.org>`_. Most of the funtions perform the computations in concurrent manner and use
the `OpenMPI <https://www.open-mpi.org>`_ library.

Image data processing
---------------------

Here is a list of image processing routines used in this software. This routines are employed to process the experimentally
measured CBC patterns. Some of these routines rely on the 2-dimensional Fourier transform. All of the pertained functions
use `FFTW <https://www.fftw.org>`_ library to perform the fast Fourier transform (FFT).

Also, the `pyFFTW wrapper <https://github.com/pyFFTW/pyFFTW>`_ of the `FFTW <https://www.fftw.org>`_ library is incorporated
into the library for computing dicrete Fourier transforms of multidimensional arrays.

.. toctree::
    :maxdepth: 1

    classes/fftw
    funcs/next_fast_len
    funcs/fft_convolve
    funcs/gaussian_filter
    funcs/gaussian_gradient_magnitude
    funcs/median
    funcs/median_filter
    funcs/maximum_filter

Thick line rasterisation
------------------------

Routines to draw thick lines with variable thickness and the antialiasing applied on a single frame by using the
Bresenham's algorithm [BSH]_.

.. toctree::
    :maxdepth: 1

    funcs/draw_line_image
    funcs/draw_line_mask
    funcs/draw_line_table

CBC patterns processing
-----------------------

Routines used to prepare and create sparse CBC patterns out of exerimentally measured patterns.

.. toctree::
    :maxdepth: 1

    funcs/normalise_pattern
    funcs/refine_pattern
    funcs/subtract_background
    funcs/project_effs

CBD forward model
-----------------

Some of the routines required to implement the CBD geometry and simulate a CBD pattern are implemented in C to boost the
performance and enable the concurrent computations.

.. toctree::
    :maxdepth: 1

    funcs/euler_angles
    funcs/euler_matrix
    funcs/tilt_angles
    funcs/tilt_matrix
    funcs/spherical_to_cartesian
    funcs/cartesian_to_spherical
    funcs/calc_source_lines
    funcs/filter_hkl
    funcs/ce_criterion

Intensity scaling
-----------------

A set of functions used in the CBC intensity scaling and the estimation of a crystal diffraction efficiency map.

.. toctree::
    :maxdepth: 1

    funcs/binterpolate
    funcs/poisson_criterion
    funcs/ls_criterion
    funcs/kr_predict
    funcs/kr_grid
    funcs/unmerge_signal
