Image transforms
================

Transforms are common image transformations. They can be chained together using :class:`ComposeTransforms <cbclib.ComposeTransforms>`. You pass a
:class:`Transform <cbclib.Transform>` instance to a data container :class:`CrystData <cbclib.CrystData>`. All transform classes are inherited from the
abstract :class:`Transform <cbclib.Transform>` class. Use :func:`forward <cbclib.Transform.forward>` to apply transform to an image.

:class:`Transform <cbclib.Transform>`
-------------------------------------

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