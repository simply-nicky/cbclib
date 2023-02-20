cbclib API
==========

The data processing of Convergent Beam Crystallography (CBC) rotational datasets can be broken down to
three stages:

:doc:`data_proc_ref`
--------------------

Raw CBC patterns measured by a detector are preprocessed to get rid of the background and hot pixels (see :class:`CrystData <cbclib.CrystData>`).
Background subtracted CBC patterns are used to detect the diffraction streaks (:class:`LSDetector <cbclib.LSDetector>`). A set of detected
streaks yields a normalised sparse CBC pattern. The dataset of sparse CBC patterns is then saved to a tabular format (see
:class:`CBCTable <cbclib.CBCTable>`).

:doc:`refinement_ref`
---------------------

CBD scheme has more complicated scattering geometry as compared to the traditional X-ray crystallography experiments using
monochromatic collimated beam. Even little uncertainties of a cristalline sample position and orientation impose substantial
alterations to the structure of a CBD pattern. Therefore, sample positions and sample orientations ought to be reconstructed
with high precision at every step of a scan.

The forward modelling of a CBC pattern (see :class:`CBDModel <cbclib.CBDModel>`) is used to refine the experimental setup and the crystal
lattice. The forward model project the reciprocal lattice points to the detector plane and yields a predicted pattern of
standard reflections. Thus, starting with the preliminary estimations the experimental geometry and the sample lattice
can be refined by minimising the discrepancy between the predicted and experimentally measured patterns (see :class:`SampleRefiner <cbclib.SampleRefiner>`
and :class:`SetupRefiner <cbclib.SetupRefiner>`).

Before performing the minimisation, one needs a way to estimate the unit cell of the crystalline sample directly from the
measured CBC pattern. The Fourier auto-indexing algorithm specifically tailored to CBC datasets was implemented in cbclib
software suite (see :class:`FourierIndexer <cbclib.FourierIndexer>`).

Indexing
--------

The problem of finding the incoming wave-vector, the scattered wave-vector and the recipocal lattice point that correpond to
each Bragg reflection measured on the detector is called **indexing**. After the corect experimental geometry and sample lattice
are obtained, one can detect the diffraction streaks and assign to them the correct Miller indices with the forward model of
CBC patterns (see :class:`ModelDetector <cbclib.ModelDetector>`). The predicted reflection is presumed to be present in the measured patterns if the
signal-to-noise ratio of measured intensities is above the certain threshold.

:doc:`scaling_ref`
------------------

In contrast to crystallographic data from the monochromatic collimated beam, different pixels of each Bragg reflection streak of
CBD data arise from different incident wave-vectors of the source beam as well as different regions of the crystal which has
varying diffracting power. This characteristic of CBD data requires scaling of individual pixel intensities of each streak before
integration and merging for structure determination.

cbclib uses an iterative algorithm to simultaneously reconstruct the projection maps and structure factors of the crystalline
sample (see :class:`IntensityScaler <cbclib.IntensityScaler>`). The sample projection maps and structure factors are updated by fitting the
theoretical intensity profiles of Bragg reflections to the experimentally measured intensities. Either negative log likelihood or
least squares error is calculated to estimate how well the modelled intensity profiles fit to the experimental intensities (see
:func:`IntensityScaler.fitness <cbclib.IntensityScaler.fitness>`).

Contents
--------

.. toctree::
    :maxdepth: 1

    log_ref
    cxi_ref
    transform_ref
    data_proc_ref
    cbd_model_ref
    refinement_ref
    scaling_ref
    func_api
    misc_types