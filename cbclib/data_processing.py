from multiprocessing import cpu_count
import numpy as np
from .data_container import DataContainer, dict_to_object
from .bin import subtract_background, median, median_filter, LSD, maximum_filter

class CrystData(DataContainer):
    attr_set = {'data', 'tilts', 'translations'}
    init_set = {'cor_data', 'flatfields', 'good_frames', 'mask', 'num_threads',
                'pupil', 'roi', 'streak_data', 'streaks', 'whitefield'}

    def __init__(self, protocol, **kwargs):
        # Initialize protocol for the proper data type conversion in __setattr__
        self.__dict__['protocol'] = protocol

        # Initialize attr_dict
        super(CrystData, self).__init__(**kwargs)

        # Add protocol to the attr_dict in order to get the dict_update decorator working
        self.__dict__['attr_dict']['protocol'] = protocol

        # Initialize init_set attributes
        self._init_dict()

    def _init_dict(self):
        # Set number of threads, num_threads is not a part of the protocol
        if self.num_threads is None:
            self.num_threads = np.clip(1, 64, cpu_count())
        # Set ROI, good frames array, mask, and whitefield
        if self.roi is None:
            self.roi = np.array([0, self.data.shape[1], 0, self.data.shape[2]])
        if self.good_frames is None:
            self.good_frames = np.arange(self.data.shape[0])
        if self.mask is None:
            self.mask = np.ones(self.data.shape, dtype=self.protocol.get_dtype('mask'))
        if self.mask.shape == self.data.shape[1:]:
            self.mask = np.tile(self.mask[None, :], (self.data.shape[0], 1, 1))

        if self._iswhitefield and self.cor_data is None:
            self.cor_data = subtract_background(self.data, self.mask, self.whitefield,
                                                self.good_frames, self.num_threads)

        if not self.pupil is None:
            self.mask_region.inplace_update(self.pupil)

        # Initialize a list of StreakDetector objects
        self._det_objects = []

    @property
    def _iswhitefield(self):
        return not self.whitefield is None

    def _check_dtype(self, attr, value):
        dtype = self.protocol.get_dtype(attr)
        if dtype is None:
            return value
        if isinstance(value, np.ndarray):
            return np.asarray(value, dtype=dtype)
        return dtype(value)

    def __setattr__(self, attr, value):
        if attr in self.attr_set | self.init_set and not value is None:
            if isinstance(value, dict):
                for key in value:
                    value[key] = self._check_dtype(attr, value[key])
            else:
                value = self._check_dtype(attr, value)
            super(CrystData, self).__setattr__(attr, value)
        else:
            super(CrystData, self).__setattr__(attr, value)

    @dict_to_object
    def bin_data(self, bin_ratio=2):
        """Return a new :class:`CrystData` object with the data binned by
        a factor `bin_ratio`.

        Parameters
        ----------
        bin_ratio : int, optional
            Binning ratio. The frame size will decrease by the factor of
            `bin_ratio`.

        Returns
        -------
        CrystData
            New :class:`CrystData` object with binned `data`.
        """
        data = self.data[:, ::bin_ratio, ::bin_ratio]
        mask = self.mask[:, ::bin_ratio, ::bin_ratio]
        data_dict = {'data': data, 'mask': mask, 'roi': self.roi // bin_ratio}

        if self._iswhitefield:
            data_dict['whitefield'] = self.whitefield[::bin_ratio, ::bin_ratio]
            data_dict['cor_data'] = self.cor_data[:, ::bin_ratio, ::bin_ratio]

        return data_dict

    @dict_to_object
    def crop_data(self, roi):
        """Return a new :class:`CrystData` object with the updated `roi`.

        Parameters
        ----------
        roi : iterable
            Region of interest in the detector plane.

        Returns
        -------
        CrystData
            New :class:`CrystData` object with the updated `roi`.
        """
        return {'roi': np.asarray(roi, dtype=int)}

    def get(self, attr, value=None):
        """Return a dataset with `mask` and `roi` applied.
        Return `value` if the attribute is not found.

        Parameters
        ----------
        attr : str
            Attribute to fetch.
        value : object, optional
            Return if `attr` is not found.

        Returns
        -------
        numpy.ndarray or object
            `attr` dataset with `mask` and `roi` applied.
            `value` if `attr` is not found.
        """
        if attr in self:
            val = super(CrystData, self).get(attr)
            if not val is None:
                if attr in ['cor_data', 'data', 'flatfields', 'mask', 'whitefield']:
                    val = np.ascontiguousarray(val[..., self.roi[0]:self.roi[1],
                                                        self.roi[2]:self.roi[3]])
                if attr in ['cor_data', 'data', 'flatfields', 'mask']:
                    val = np.ascontiguousarray(val[self.good_frames])
            return val
        return value

    def get_pca(self):
        """Perform the Principal Component Analysis [PCA]_ of the measured data and
        return a set of eigen flat fields (EFF).

        Returns
        -------
        effs_var : numpy.ndarray
            Variance ratio for each EFF, that it describes.
        effs : numpy.ndarray
            Set of eigen flat fields.

        References
        ----------
        .. [PCA] Vincent Van Nieuwenhove, Jan De Beenhouwer, Francesco De Carlo,
                 Lucia Mancini, Federica Marone, and Jan Sijbers, "Dynamic intensity
                 normalization using eigen flat fields in X-ray imaging," Opt. Express
                 23, 27975-27989 (2015).
        """
        if not self._iswhitefield:
            raise AttributeError("No whitefield in the container")

        mat_svd = np.tensordot(self.cor_data, self.cor_data, axes=((1, 2), (1, 2)))
        eig_vals, eig_vecs = np.linalg.eig(mat_svd)
        effs = np.tensordot(eig_vecs, self.cor_data, axes=((0,), (0,)))
        return eig_vals / eig_vals.sum(), effs

    @dict_to_object
    def mask_frames(self, good_frames):
        """Return a new :class:`CrystData` object with the updated
        good frames mask.

        Parameters
        ----------
        good_frames : iterable
            List of good frames' indices.

        Returns
        -------
        CrystData
            New :class:`CrystData` object with the updated `good_frames`.
        """
        return {'good_frames': np.asarray(good_frames, dtype=np.int)}

    @dict_to_object
    def mask_region(self, roi):
        """Return a new :class:`CrystData` object with the updated
        mask. The region defined by the `roi` will be masked out.

        Parameters
        ----------
        roi : iterable
            Bad region of interest in the detector plane.

        Returns
        -------
        CrystData
            New :class:`CrystData` object with the updated `mask`.
        """
        mask = self.mask.copy()
        mask[:, roi[0]:roi[1], roi[2]:roi[3]] = False

        if self._iswhitefield:
            cor_data = self.cor_data.copy()
            cor_data[:, roi[0]:roi[1], roi[2]:roi[3]] = 0
            return {'cor_data': cor_data, 'mask': mask}

        return {'mask': mask}

    @dict_to_object
    def import_mask(self, mask, update='reset'):
        """Return a new :class:`CrystData` object with the new
        mask.

        Parameters
        ----------
        mask : np.ndarray
            New mask array.
        update : {'reset', 'multiply'}, optional
            Multiply the new mask and the old one if 'multiply',
            use the new one if 'reset'.

        Raises
        ------
        ValueError
            If the mask shape is incompatible with the data.

        Returns
        -------
        CrystData
            New :class:`CrystData` object with the updated `mask`.
        """
        if mask.shape != self.data.shape[1:]:
            raise ValueError('mask and data have incompatible shapes: '\
                             f'{mask.shape:s} != {self.data.shape[1:]:s}')
        if update == 'reset':
            return {'mask': mask, 'cor_data': None}
        elif update == 'multiply':
            return {'mask': mask * self.mask, 'cor_data': None}
        else:
            raise ValueError(f'Invalid update keyword: {update:s}')

    @dict_to_object
    def import_whitefield(self, whitefield):
        """Return a new :class:`CrystData` object with the new
        whitefield.

        Parameters
        ----------
        whitefield : np.ndarray
            New whitefield array.

        Raises
        ------
        ValueError
            If the whitefield shape is incompatible with the data.

        Returns
        -------
        CrystData
            New :class:`CrystData` object with the updated `whitefield`.
        """
        if whitefield.shape != self.data.shape[1:]:
            raise ValueError('whitefield and data have incompatible shapes: '\
                             f'{whitefield.shape:s} != {self.data.shape[1:]:s}')
        return {'whitefield': whitefield, 'cor_data': None}

    @dict_to_object
    def update_flatfields(self, method='median', size=11, effs=None):
        """Return a new :class:`CrystData` object with new
        flatfields. The flatfields are generated by the dint of
        median filtering or Principal Component Analysis [PCA]_.

        Parameters
        ----------
        method : {'median', 'pca'}, optional
            Method to generate the flatfields. The following keyword
            values are allowed:

            * 'median' : Median `data` along the first axis.
            * 'pca' : Generate a set of flatfields based on Eigen Flatfields
              `effs`. `effs` can be obtained with :func:`CrystData.get_pca`
              method.
        size : int, optional
            Size of the filter window in pixels used for the 'median' generation
            method.
        effs : np.ndarray, optional
            Set of Eigen flatfields used for the 'pca' generation method.

        Raises
        ------
        ValueError
            If the `method` keyword is invalid.
        AttributeError
            If the `whitefield` is absent in the :class:`CrystData` container
            when using the 'pca' generation method.
        ValuerError
            If `effs` were not provided when using the 'pca' generation method.

        Returns
        -------
        CrystData
            New :class:`CrystData` object with the updated `flatfields`.

        References
        ----------
        .. [PCA] Vincent Van Nieuwenhove, Jan De Beenhouwer, Francesco De Carlo,
                 Lucia Mancini, Federica Marone, and Jan Sijbers, "Dynamic
                 intensity normalization using eigen flat fields in X-ray
                 imaging," Opt. Express 23, 27975-27989 (2015).

        See Also
        --------
        CrystData.get_pca : Method to generate Eigen Flatfields.
        """
        if method not in ['median', 'pca']:
            raise ValueError('invalid method argument')

        if method == 'median':
            good_flats = median_filter(self.get('data'), size, num_threads=self.num_threads)
        elif method == 'pca':
            if not self._iswhitefield:
                raise AttributeError('No whitefield in the data container')
            if effs is None:
                raise ValueError('No eigen flat fields were provided')

            weights = np.tensordot(self.get('cor_data'), effs, axes=((1, 2), (1, 2))) / np.sum(effs * effs, axis=(1, 2))
            good_flats = np.tensordot(weights, effs, axes=((1,), (0,))) + self.get('whitefield')

        flatfields = np.zeros(self.data.shape, dtype=self.protocol.get_dtype('flatfields'))
        flatfields[self.good_frames, self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]] = good_flats
        return {'flatfields': flatfields}

    @dict_to_object
    def update_mask(self, method='perc-bad', pmin=0.01, pmax=99.99, vmin=0, vmax=65535,
                    update='reset'):
        """Return a new :class:`CrystData` object with the updated
        bad pixels mask.

        Parameters
        ----------
        method : {'no-bad', 'range-bad', 'perc-bad'}, optional
            Bad pixels masking methods:

            * 'no-bad' (default) : No bad pixels.
            * 'range-bad' : Mask the pixels which values lie outside
              of (`vmin`, `vmax`) range.
            * 'perc-bad' : Mask the pixels which values lie outside
              of the (`pmin`, `pmax`) percentiles.
        vmin, vmax : float, optional
            Lower and upper intensity values of 'range-bad' masking
            method.
        pmin, pmax : float, optional
            Lower and upper percentage values of 'perc-bad' masking
            method.
        update : {'reset', 'multiply'}, optional
            Multiply the new mask and the old one if 'multiply',
            use the new one if 'reset'.

        Returns
        -------
        CrystData
            New :class:`CrystData` object with the updated `mask`.
        """
        if update == 'reset':
            data = self.get('data')
        elif update == 'multiply':
            data = self.get('data') * self.get('mask')
        else:
            raise ValueError(f'Invalid update keyword: {update:s}')

        if method == 'no-bad':
            mask = np.ones((self.good_frames.size, self.roi[1] - self.roi[0],
                            self.roi[3] - self.roi[2]), dtype=bool)
        elif method == 'range-bad':
            mask = (data >= vmin) & (data < vmax)
        elif method == 'perc-bad':
            offsets = (data.astype(np.int32) -
                       median_filter(data, (1, 3, 3),
                                     num_threads=self.num_threads).astype(np.int32))
            mask = (offsets >= np.percentile(offsets, pmin)) & \
                   (offsets <= np.percentile(offsets, pmax))
        else:
            ValueError('invalid method argument')
        mask_full = self.mask.copy()

        if update == 'reset':
            mask_full[self.good_frames, self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]] = mask
        elif update == 'multiply':
            mask_full[self.good_frames, self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]] *= mask
        else:
            raise ValueError(f'Invalid update keyword: {update:s}')

        if self._iswhitefield:
            cor_data = self.cor_data.copy()
            cor_data[mask_full] = 0
            return {'cor_data': cor_data, 'mask': mask_full}

        return {'mask': mask_full}

    @dict_to_object
    def update_whitefield(self):
        """Return a new :class:`CrystData` object with new
        whitefield as the median taken through the stack of
        measured frames.

        Returns
        -------
        CrystData
            New :class:`CrystData` object with the updated `whitefield`.
        """
        good_wf = median(data=self.get('data'), mask=self.get('mask'), axis=0,
                         num_threads=self.num_threads)
        whitefield = np.zeros(self.data.shape[1:], dtype=self.protocol.get_dtype('whitefield'))
        whitefield[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]] = good_wf
        return {'whitefield': whitefield, 'cor_data': None}

    def detect_streaks(self, vmin, vmax, size=(1, 3, 3)):
        if self.pupil is None:
            return None
        return StreakDetector.import_data(self, vmin, vmax, size)

    def update_streaks(self, det_obj):
        if not det_obj in self._det_objects:
            raise ValueError("The StreakDetector object doesn't belong to the data container")

        self.streak_data = np.zeros(self.data.shape, dtype=np.float64)
        self.streak_data[self.good_frames, self.roi[0]:self.roi[1],
                         self.roi[2]:self.roi[3]] = det_obj.streak_data

        self.streaks = {}
        for frame_idx in det_obj.streaks:
            self.streaks[frame_idx] = det_obj.streaks[frame_idx].copy()
            self.streaks[frame_idx][:, :4:2] += self.roi[2]
            self.streaks[frame_idx][:, 1:4:2] += self.roi[0]

    def write_cxi(self, cxi_file, overwrite=True):
        """Write all the `attr` to a CXI file `cxi_file`.

        Parameters
        ----------
        cxi_file : h5py.File
            :class:`h5py.File` object of the CXI file.
        overwrite : bool, optional
            Overwrite the content of `cxi_file` file if it's True.

        Raises
        ------
        ValueError
            If `overwrite` is False and the data is already present
            in `cxi_file`.
        """
        for attr, data in self.items():
            if attr in self.protocol:
                self.protocol.write_cxi(attr, data, cxi_file, overwrite=overwrite)

class StreakDetector(DataContainer):
    attr_set = {'center', 'data', 'num_threads', 'streak_data'}
    init_set = {'lsd_obj', 'streak_mask', 'streaks'}
    footprint = np.array([[[False, False,  True, False, False],
                           [False,  True,  True,  True, False],
                           [ True,  True,  True,  True,  True],
                           [False,  True,  True,  True, False],
                           [False, False,  True, False, False]]])

    def __init__(self, cryst_data, **kwargs):
        self.__dict__['_reference'] = cryst_data
        self._reference._det_objects.append(self)

        super(StreakDetector, self).__init__(**kwargs)

        if self.lsd_obj is None:
            self.update_lsd.inplace_update()

    @classmethod
    def import_data(cls, cryst_data, vmin, vmax, size=(1, 3, 3)):
        data = cryst_data.get('cor_data')
        if not size is None:
            data = median_filter(data, size=size, num_threads=cryst_data.num_threads)
        streak_data = (np.clip(data, vmin, vmax) - vmin) / (vmax - vmin)
        center = np.array([cryst_data.pupil[:2].mean() - cryst_data.roi[0],
                           cryst_data.pupil[2:].mean() - cryst_data.roi[2]])
        return cls(cryst_data=cryst_data, center=center, data=data,
                   streak_data=streak_data, num_threads=cryst_data.num_threads)

    @dict_to_object
    def update_lsd(self, scale=0.9, sigma_scale=0.9, log_eps=0.,
                   ang_th=60.0, density_th=0.5, quant=2.0e-2):
        return {'cryst_data': self._reference, 'lsd_obj': LSD(scale=scale,
                sigma_scale=sigma_scale, log_eps=log_eps, ang_th=ang_th,
                density_th=density_th, quant=quant, y_c=self.center[0],
                x_c=self.center[1])}

    @dict_to_object
    def update_mask(self, dilation=15, radius=1.0, filter_lines=True):
        out_dict = self.lsd_obj.mask(self.streak_data, max_val=1, dilation=dilation,
                                     radius=radius, filter_lines=filter_lines,
                                     return_lines=True, num_threads=self.num_threads)
        return {'cryst_data': self._reference, 'streak_mask': out_dict['mask'],
                'streaks': out_dict['lines']}

    @dict_to_object
    def update_streak_data(self, iterations=10):
        if self.streak_mask is None:
            raise AttributeError("'streak_mask' must be generated before.")

        divisor = self.data.copy()
        for _ in range(iterations):
            divisor = maximum_filter(divisor, mask=self.streak_mask,
                                     footprint=self.footprint,
                                     num_threads=self.num_threads)
        streak_data = np.where(divisor, self.data / divisor, 0)
        return {'cryst_data': self._reference, 'streak_data': streak_data}
