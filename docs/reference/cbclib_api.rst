cbclib API
==========

The data processing of Convergent Beam Crystallography (CBC) rotational datasets can be broken down to
three stages:

Preprocessing and streak detection
----------------------------------

Raw CBC patterns measured by a detector are preprocessed to get rid of the background and hot pixels. Background subtracted
CBC patterns undergo the streak line detection (:class:`cbclib.LSDetector`) to yield a sparse representation of CBC pattern.
A sparse CBC pattern can be saved as a :class:`pandas.DataFrame` table.

Indexing and sample refinement
------------------------------

CBD scheme has more complicated scattering geometry as compared to the traditional X-ray crystallography experiments using
monochromatic collimated beam. Even little uncertainties of a cristalline sample position and orientation impose substantial
alterations to the structure of a CBD pattern. Therefore, sample positions and sample orientations are ought to be reconstructed
with high precision at every step of a scan.

Intensity scaling and merging
-----------------------------

Content
^^^^^^^

.. toctree::
    :maxdepth: 1

    cxi_ref
    transform_ref
    data_proc_ref
    exp_geom
    indexing_ref
    func_api
    misc_types