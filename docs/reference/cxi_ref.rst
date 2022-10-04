Working with HDF5 files
=======================

:class:`CXIProtocol <cbclib.CXIProtocol>`
-----------------------------------------

HDF5 protocol (:class:`cbclib.CXIProtocol`) is a helper class for a :class:`cbclib.CXIStore`
HDF5 file handler, which tells it where to look for the necessary data fields in a HDF5
file. The class is fully customizable so you can tailor it to your particular data
structure of HDF5 file. The protocol consists of the following parts for each data
attribute (`data`, `whitefield`, etc.):

* **datatypes** : Data type (`float`, `int`, `uint`, or `bool`) of the given attribute.
* **load_paths** : List of paths inside a HDF5 file, where the given data attribute may be
  saved.
* **kinds** : The attribute's kind, that specifies data dimensionality. This information
  is required to know how load, save and process the data. The attribute may be one of
  the four following kinds:

  * *scalar* : Data is either 0D, 1D, or 2D. The data is saved and loaded plainly
    without any transforms or indexing.
  * *sequence* : A time sequence array. Data is either 1D, 2D, or 3D. The data is
    indexed, so the first dimension of the data array must be a time dimension. The
    data points for the given index are not transformed.
  * *frame* : Frame array. Data must be 2D, it may be transformed with any of
    :class:`cbclib.Transform` objects. The data shape is identical to the detector
    pixel grid.
  * *stack* : A time sequnce of frame arrays. The data must be 3D. It's indexed in the
    same way as `sequence` attributes. Each frame array may be transformed with any of
    :class:`cbclib.Transform` objects.

.. note::

    You can save protocol to an INI file with :func:`cbclib.CXIProtocol.to_ini`
    and import protocol from INI file with :func:`cbclib.CXIProtocol.import_ini`.

The default protocol can be accessed with :func:`cbclib.CXIProtocol.import_default`. The protocol
is given by:

.. code-block:: ini

    [datatypes]
    background = float
    cor_data = float
    data = uint
    frames = uint
    mask = bool
    streak_data = float
    tilts = float
    translations = float
    wavelength = float
    whitefield = float
    x_pixel_size = float
    y_pixel_size = float

    [load_paths]
    background = [/entry/crystallography/background,]
    cor_data = [/entry/data/cor_data,]
    data = [/entry/data/data, /entry_1/data/data, /entry_1/data_1/data, /data/data, /entry_1/instrument_1/detector_1/data]
    frames = [/entry/crystallography/frames, /frame_selector/frames, /process_3/frames]
    mask = [/entry/instrument/detector/mask, /entry_1/instrument_1/detector_1/mask, /entry_1/instrument/detector/mask]
    tilts = [/entry/crystallography/tilts,]
    translations = [/entry/crystallography/translations,]
    streak_data = [/entry/crystallography/streak_data,]
    wavelength = [/entry/instrument/beam/wavelength, /entry/instrument/beam/incident_wavelength, /entry_1/instrument_1/beam/wavelength]
    whitefield = [/entry/crystallography/whitefield,]
    x_pixel_size = [/entry/instrument/detector/x_pixel_size,]
    y_pixel_size = [/entry/instrument/detector/y_pixel_size,]

    [kinds]
    background = stack
    cor_data = stack
    data = stack
    frames = sequence
    mask = stack
    streak_data = stack
    tilts = sequence
    translations = sequence
    wavelength = scalar
    whitefield = frame
    x_pixel_size = scalar
    y_pixel_size = scalar

:class:`CXIStore <cbclib.CXIStore>`
-----------------------------------

HDF5 file handler class (:class:`cbclib.CXIStore`) accepts a set of paths to the files together with
a protocol object. :class:`cbclib.CXIStore` searches the files for any data attributes defined by
the protocol. It provides an interface to load the data of the given attribute from the files
(see :func:`cbclib.CXIStore.load_attribute`) and save the data of the attribute to the first file in
the set (see :func:`cbclib.CXIStore.save_attribute`). The files may be multiple or a single one.

.. automodule:: cbclib.cxi_protocol

Contents
--------

.. toctree::
    :maxdepth: 1

    classes/cxi_protocol
    classes/cxi_store