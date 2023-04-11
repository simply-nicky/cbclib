Intensity scaling
=================

The intensity scaling, integration and merging for CBD data are performed in an iterative manner to achieve convergence to the optimal structure factor
estimates and diffraction power of the crystal in one process. The intensity profile of a particular Bragg reflection captured on the detector can be
broken down to a product of three terms:

.. math::

   I_{hkl}(\mathbf{x}) = |q_{hkl}|^2 \chi(\mathbf{u}(\mathbf{x})) f^2_{hkl}(\mathbf{x}),

where :math:`\chi(\mathbf{k}_{in})` is the projection map of the sample, :math:`q_{hkl}` is the structure factor, :math:`f_{hkl}` is the standard profile,
:math:`\mathbf{u}(\mathbf{x}) = \mathbf{k}_{out}(\mathbf{x}) - \mathbf{g}_{hkl}` is the geometrical mapping from the detector plane to the aperture function.

To accurately handle the intensity measurements in the overlapping streak regions, we assume that the diffracted signal from different diffraction orders is
summed up incoherently. Thus, the modelled diffraction pattern is given by:

.. math::

   \hat{I}_n(\mathbf{x}_i) = I_{bgd}(\mathbf{x}_i) + \sum_{hkl} |q_{hkl}|^2 \chi_n(\mathbf{u}(\mathbf{x}_i)) f^2_{hkl}(\mathbf{x}_i),

where and :math:`I_{bgd}(\mathbf{x})` is the white-field and :math:`\chi_n(\mathbf{k}_{in})` is the projection mapping of the crystal at the n-th frame.

:class:`IntensityScaler <cbclib.IntensityScaler>` recovers the estimates of projection maps :math:`\chi` and :math:`q_{hkl}` iteratively with :func:`train <cbclib.IntensityScaler.train>`
by minimising one of two criteria (see :func:`fitness <cbclib.IntensityScaler.fitness>`):

1. The former one is **the Poisson negative log-likelihood**:

   .. math:: \varepsilon^{NLL} = \sum_{ni} \varepsilon_n^{NLL}(\mathbf{x}_i)  = \sum_{ni} \log \mathrm{P}(I_n(\mathbf{x}_i), \hat{I}_n(\mathbf{x}_i)),

   where the likelihood :math:`\mathrm{P}` follows the Poisson distribution :math:`\log \mathrm{P}(I, \lambda) = I \log \lambda - I`.

2. The latter one is **the least-squares criterion**:
   
   .. math:: \varepsilon^{LS} = \sum_{ni} \varepsilon_n^{LS}(\mathbf{x}_i) = \sum_{ni} f\left( \frac{I_n(\mathbf{x}_i) - \hat{I}_n(\mathbf{x}_i)}{\sigma_I} \right),

   where :math:`f(x)` is either l2, l1, or Huber loss function, and :math:`\sigma_I` is the standard deviation of measured photon counts for a given diffraction
   streak.

.. note::

   :class:`IntensityScaler <cbclib.IntensityScaler>` incorporates intensity merging into iterative update with :func:`merge_hkl <cbclib.IntensityScaler.merge_hkl>`
   method that takes a symmetry type of the crystal lattice. Given the supplied symmetry, the structure factors for the symmetric reflections are assumed to be identical
   so that the optimisation in :func:`train <cbclib.IntensityScaler.train>` is carried out for the whole subset of symmetric reflections in one fell swoop.

At each iteration, the update for the structure factors is performed by minimising one of the criteria above, where :math:`\chi_n` is held constant. At the
next step, the projection mapping are calculated based on the new estimates of crystal factors as follows:

.. math::

   \hat{\chi}_n(\mathbf{k}_{in}) = \frac{\sum_i \hat{I}^{hkl}_n(\mathbf{x}_i) f^2_{hkl}(\mathbf{x}_i) K((\mathbf{k}_{in} - \mathbf{u}(\mathbf{x}_i)) / h)}
   {\sum_i |q_{hkl}|^2 f^4_{hkl}(\mathbf{x}_i) K((\mathbf{k}_{in} - \mathbf{u}(\mathbf{x}_i)) / h)},

where :math:`K(\mathbf{x})` is the kernel function, and :math:`h` is the kernel bandwidth, and :math:`\hat{I}^{hkl}_n` is the estimate of the intensity measurements
pertaining to a certain Bragg reflection:

.. math:: 

   \hat{I}^{hkl}_n(\mathbf{x}_i) = \frac{(I_n(\mathbf{x}_i) - I_{bgd}(\mathbf{x}_i)) I_{hkl}(\mathbf{x}_i)}{\sum_{hkl} I_{hkl}(\mathbf{x}_i)}.

.. note::

   :math:`\hat{I}^{hkl}_n(\mathbf{x}_i)` differs from the measured intensities only in the regions where the diffraction streaks overlap.

:class:`IntensityScaler <cbclib.IntensityScaler>`
-------------------------------------------------

.. autoclass:: cbclib.IntensityScaler
   :members:
   :inherited-members: