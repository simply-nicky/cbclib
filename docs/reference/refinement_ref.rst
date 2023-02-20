Geometry refinement
===================

.. image:: ../_static/cbclib_indexing.png
    :width: 100 %
    :alt: A flow diagram of the main classes.

The problem of finding the incoming wave-vector, the scattered wave-vector and the recipocal lattice point that
correpond to each Bragg reflection measured on the detector is called **indexing**. Before indexing the CBC patterns,
one must retrieve the correct experimental geometry and crystalline lattice. The software suite provides a set of tools
to retrieve them from the experimentally measured patterns.

Before repforming the geometry and lattice refinement, one needs a CBC table generated from the detected diffraction
streaks (:func:`cbclib.LSDetector.export_table`) and an initial estimate of the experimental geometry (see :class:`ScanSetup <cbclib.ScanSetup>`).
The following classes are central to the setup and lattice refinement of CBC patterns:

* **CBC Table** : :class:`cbclib.CBCTable` is the main data container for CBC tables. It provides an interface to output any
  specific CBC patterns in the scan in different formats. Besides, it contains a set of methods to perform
  the indexing and sample refinement procedure.
* **Fourier Indexer** : :class:`cbclib.FourierIndexer` is the Fourier auto-indexing tool specifically tailored to CBC
  pattern. It contains a rasterised 3D image of normalised intensity profiles mapped to the space of scattering momentum.
  It provides methods to perform the Fourier transform (:func:`cbclib.FourierIndexer.fft`) and find the highest
  peaks that correspond to the lattice unit vectors of the sample (:func:`cbclib.FourierIndexer.find_peaks`).
* **Sample Refiner** : :class:`cbclib.SampleRefiner` performs the sample refinement. It employs :class:`cbclib.CBDModel`
  CBD pattern prediction to find the sample positions and alignments, that yield the best fit with between the simulated
  and experimentally measured patterns. The refinement is performed by minimizing the cross-entropy criterion between the
  predicted and experimental patterns. The sample positions and orientations are refined for each frame separately.
* **Setup Refiner** : :class:`cbclib.SetupRefiner` performs the experimental setup refinement. It employs :class:`cbclib.CBDModel`
  CBD pattern prediction to find the experimental setup parameters, that yield the best fit with between the simulated
  and experimentally measured patterns. The refinement is performed by minimizing the cross-entropy criterion between
  the predicted and experimental patterns of the whole dataset.
* **Scan Samples** : :class:`cbclib.ScanSamples` is a data container that stores the results of the sample refinement.
  It provides an interface to save and load the results to a :class:`pandas.DataFrame`. Also, one can regularise the
  sample positions through the scan by using `the gaussian processes <https://scikit-learn.org/stable/modules/gaussian_process.html>`_.

Contents
--------

.. toctree::
    :maxdepth: 1

    classes/cbc_table
    classes/fourier_indexer
    classes/sample_refiner
    classes/setup_refiner
    classes/scan_samples
