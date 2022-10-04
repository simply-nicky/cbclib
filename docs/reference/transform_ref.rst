Image transforms
================

Transforms are common image transformations. They can be chained together using :class:`cbclib.ComposeTransforms`.
You pass a :class:`cbclib.Transform` instance to a data container :class:`cbclib.CrystData`. All transform classes
are inherited from the abstract :class:`cbclib.Transform` class. Use :func:`cbclib.Transform.forward` to apply
transform to an image.

:class:`Transform <cbclib.Transform>`
-----------------------------------------------------

.. autoclass:: cbclib.Transform
    :members:
    :inherited-members:

:class:`ComposeTransforms <cbclib.ComposeTransforms>`
-----------------------------------------------------

.. autoclass:: cbclib.ComposeTransforms
    :members:
    :inherited-members:


Transforms on images
--------------------

.. toctree::
    :maxdepth: 1

    classes/crop
    classes/mirror
    classes/downscale