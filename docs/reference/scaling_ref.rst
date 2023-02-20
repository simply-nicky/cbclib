Intensity scaling
=================

The intensity scaling, integration and merging for CBD data are performed in an iterative manner to achieve convergence to the optimal structure factor
estimates and diffraction power of the crystal in one process. The intensity profile of a particular Bragg reflection captured on the detector can be
broken down to a product of three terms:

.. math::

    I_{hkl}(\mathbf{x}) = |q_{hkl}|^2 \chi(\mathbf{u}(\mathbf{x})) f^2_{hkl}(\mathbf{x}),

where :math:`\chi(\mathbf{k}_{in})` is the projection map of the sample, :math:`q_{hkl}` is the structure factor, :math:`f_{hkl}` is the standard profile,
:math:`\mathbf{u}(\mathbf{x}) = \mathbf{k}_{out}(\mathbf{x}) - \mathbf{g}_{hkl}` is the geometrical mapping from the detector plane to the aperture function.

:class:`IntensityScaler <cbclib.IntensityScaler>` recovers the estimates of projection maps :math:`\chi` and :math:`q_{hkl}` iteratively with :func:`train <cbclib.IntensityScaler.train>`
by minimising one of two criteria (see :func:`fitness <cbclib.IntensityScaler.fitness>`):

1. The former one is **the Poisson negative log-likelihood**:

   .. math:: \epsilon^{NLL} = \sum_{ni} \log \mathrm{P}(I_n(\mathbf{x}_i), I_{hkl}(\mathbf{x}_i) + I_{bgd}(\mathbf{x}_i)),

   where the likelihood :math:`\mathrm{P}` follows the Poisson distribution :math:`\log \mathrm{P}(I, \lambda) = I \log \lambda - I`.

2. The latter one is **the least-squares criterion**:
   
   .. math:: \epsilon^{LS} = \sum_{ni} f\left( \frac{I_n(\mathbf{x}_i) - I_{hkl}(\mathbf{x}_i) - I_{bgd}}{\sigma_I^2} \right),

   where :math:`f(x)` is either l2, l1, or Huber loss function, and :math:`\sigma_I` is the standard deviation of measured photon counts for a given diffraction
   streak.

.. note::

   :class:`IntensityScaler <cbclib.IntensityScaler>` incorporates intensity merging into iterative update with :func:`merge_hkl <cbclib.IntensityScaler.merge_hkl>`
   method that takes a symmetry type of the crystal lattice. Given the supplied symmetry, the structure factors for the symmetric reflections are assumed to be identical
   so that the optimisation in :func:`train <cbclib.IntensityScaler.train>` is carried out for the whole subset of symmetric reflections in one fell swoop.

:class:`IntensityScaler <cbclib.IntensityScaler>`
-------------------------------------------------

.. autoclass:: cbclib.IntensityScaler
    :members:
    :inherited-members: