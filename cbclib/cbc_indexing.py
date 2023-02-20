from __future__ import annotations
<<<<<<< HEAD
from typing import Any, Dict, Iterable, Iterator
import numpy as np
from .ini_parser import INIParser
from .data_container import DataContainer
from .bin import tilt_matrix
=======
from multiprocessing import cpu_count
from typing import Any, Dict, Optional, Sequence, Tuple, TypeVar, Union
from dataclasses import dataclass, field, fields
import numpy as np
from scipy.ndimage import label, center_of_mass, mean
>>>>>>> dev-dataclass

from .bin import (FFTW, empty_aligned, median_filter, gaussian_filter, gaussian_grid,
                  binterpolate, filter_direction)
from .cbc_setup import Basis
from .cxi_protocol import Indices

M = TypeVar('M', bound='Map3D')

@dataclass
class Map3D():
    val         : np.ndarray
    x           : np.ndarray
    y           : np.ndarray
    z           : np.ndarray
    num_threads : int = cpu_count()

    def __post_init__(self):
        if (self.x.size != self.val.shape[2] or self.y.size != self.val.shape[1] or
            self.z.size != self.val.shape[0]):
            raise ValueError('values have incompatible shape with the coordinates')

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.val.shape

    @property
    def grid(self) -> np.ndarray:
        return np.stack(np.meshgrid(self.z, self.y, self.x, indexing='ij')[::-1], axis=-1)

    def __getitem__(self: M, indices: Tuple[Indices, Indices, Indices]) -> M:
        idxs = []
        for index, size in zip(indices, self.shape):
            if isinstance(index, (int, slice)):
                idxs.append(np.atleast_1d(np.arange(size)[index]))
            elif isinstance(index, np.ndarray):
                idxs.append(index.ravel())
            else:
                idxs.append(index)
        ii, jj, kk = np.meshgrid(*idxs, indexing='ij')
        return self.replace(val=self.val[ii, jj, kk], x=self.x[idxs[2]], y=self.y[idxs[1]],
                            z=self.z[idxs[0]])

    def __add__(self: M, obj: Any) -> M:
        if np.isscalar(obj):
            return self.replace(val=self.val + obj)
        if isinstance(obj, Map3D):
            if not self.is_compatible(obj):
                raise TypeError("Can't sum two incompatible Map3D objects")
            return self.replace(val=self.val + obj.val)
        return NotImplemented

    def __sub__(self: M, obj: Any) -> M:
        if np.isscalar(obj):
            return self.replace(val=self.val - obj)
        if isinstance(obj, Map3D):
            if not self.is_compatible(obj):
                raise TypeError("Can't subtract two incompatible Map3D objects")
            return self.replace(val=self.val - obj.val)
        return NotImplemented

    def __rmul__(self: M, obj: Any) -> M:
        if np.isscalar(obj):
            return self.replace(val=obj * self.val)
        return NotImplemented

    def __mul__(self: M, obj: Any) -> M:
        if np.isscalar(obj):
            return self.replace(val=obj * self.val)
        if isinstance(obj, Map3D):
            if not self.is_compatible(obj):
                raise TypeError("Can't multiply two incompatible Map3D objects")
            return self.replace(val=self.val * obj.val)
        return NotImplemented

    def __truediv__(self: M, obj: Any) -> M:
        if np.isscalar(obj):
            return self.replace(val=self.val / obj)
        if isinstance(obj, Map3D):
            if self.is_compatible(obj):
                raise TypeError("Can't divide two incompatible Map3D objects")
            return self.replace(val=self.val / obj.val)
        return NotImplemented

    def clip(self: M, vmin: float, vmax: float) -> M:
        """Clip the 3D data in a range of values :code:`[vmin, vmax]`.

        Args:
            vmin : Lower bound.
            vmax : Upper bound.

        Returns:
            A new 3D data object.
        """
        return self.replace(val=np.clip(self.val, vmin, vmax))

    def fft(self: M) -> M:
        """Perform 3D Fourier transform. `FFTW <https://www.fftw.org>`_ C library is used to
        compute the transform.

        Returns:
            A 3Ddata object with the Fourier image data.

        See Also:
            cbclib.bin.FFTW : Python wrapper of FFTW library.
        """
        val = empty_aligned(self.shape, dtype='complex64')
        fft_obj = FFTW(self.val.astype(np.complex64), val, threads=self.num_threads,
                       axes=(0, 1, 2), flags=('FFTW_ESTIMATE',))

        kx = np.fft.fftshift(np.fft.fftfreq(self.shape[2], d=self.x[1] - self.x[0]))
        ky = np.fft.fftshift(np.fft.fftfreq(self.shape[1], d=self.y[1] - self.y[0]))
        kz = np.fft.fftshift(np.fft.fftfreq(self.shape[0], d=self.z[1] - self.z[0]))
        val = np.fft.fftshift(np.abs(fft_obj())) / np.sqrt(np.prod(self.shape))
        return self.replace(val=val, x=kx, y=ky, z=kz)

    def gaussian_blur(self: M, sigma: Union[float, Tuple[float, float, float]]) -> M:
        """Apply Gaussian blur to the 3D data.

        Args:
            sigma : width of the gaussian blur.

        Returns:
            A new 3D data object with blurred out data.
        """
        val = gaussian_filter(self.val, sigma, num_threads=self.num_threads)
        return self.replace(val=val)

    def get_coordinates(self, index: np.ndarray) -> np.ndarray:
        """Transform a set of data indices to a set of coordinates.

        Args:
            index : An array of indices.

        Returns:
            An array of coordinates.
        """
        index = index.reshape(-1, 3)[:, ::-1]
        idx0 = np.rint(index).astype(int)
        crd0 = np.stack((np.take(self.x, idx0[:, 0]), np.take(self.y, idx0[:, 1]),
                         np.take(self.z, idx0[:, 2])), axis=-1)
        idx1 = idx0 + np.array(~np.isclose(index, idx0), dtype=int)
        crd1 = np.stack((np.take(self.x, idx1[:, 0]), np.take(self.y, idx1[:, 1]),
                         np.take(self.z, idx1[:, 2])), axis=-1)
        return crd0 + (index - idx0) * (crd1 - crd0)

    def interpolate(self, coordinates: np.ndarray) -> np.ndarray:
        """Interpolate the 3D grid at a given array of coordiantes ``coordinates``.

        Args:
            coordinates : An array of coordinates.

        Returns:
            Array of interpolated values.
        """
        return binterpolate(data=self.val, grid=(self.x, self.y, self.z), coords=coordinates,
                            num_threads=self.num_threads)

    def is_compatible(self: M, map_3d: M) -> bool:
        """Check if 3D data object has a compatible set of coordinates.

        Args:
            map_3d : 3D data object.

        Returns:
            True if the 3D data object ``map_3d`` has a compatible set of coordinates.
        """
        return ((self.x == map_3d.x).all() and (self.y == map_3d.y).all() and
                (self.z == map_3d.z).all())

    def replace(self: M, **kwargs: Any) -> M:
        """Return a new :class:`Map3D` object with a set of attributes replaced.

        Args:
            kwargs : A set of attributes and the values to to replace.

        Returns:
            A new :class:`Map3D` object with the updated attributes.
        """
        return type(self)(**dict(self.to_dict(), **kwargs))

    def to_dict(self) -> Dict[str, Any]:
        """Export the :class:`Map3D` object to a :class:`dict`.

        Returns:
            A dictionary of :class:`Map3D` object's attributes.
        """
        return {field.name: getattr(self, field.name) for field in fields(self)}

@dataclass
class FourierIndexer(Map3D):
    """3D data object designed to perform Fourier auto-indexing. Projects measured intensities to
    the reciprocal space and provides several tools to works with a 3D data in the reciprocal
    space. The container uses the `FFTW <https://www.fftw.org>`_ C library to perform the
    3-dimensional Fourier transform.

    Args:
        val : 3D data array.
        x : x coordinates.
        y : y coordinates.
        z : z coordinates.
        num_threads : Number of threads used in the calculations.
    """
    val         : np.ndarray
    x           : np.ndarray
    y           : np.ndarray
    z           : np.ndarray
    num_threads : int = field(default=1)

<<<<<<< HEAD
    foc_pos - focus position relative to the detector [m]
    pix_size - detector pixel size [m]
    rot_axis - axis of rotation
    smp_pos - sample position relative to the detector [m]
    """
    attr_dict = {'exp_geom': ('foc_pos', 'rot_axis', 'smp_pos', 'wavelength',
                              'x_pixel_size', 'y_pixel_size', 'kin_min', 'kin_max')}
    fmt_dict = {'exp_geom': 'float'}

    foc_pos : np.ndarray
    rot_axis : np.ndarray
    smp_pos : np.ndarray
    kin_min : np.ndarray
    kin_max : np.ndarray
    wavelength : float
    x_pixel_size : float
    y_pixel_size : float

    def __init__(self, foc_pos: np.ndarray, rot_axis: np.ndarray, smp_pos: np.ndarray,
                 kin_min: np.ndarray, kin_max: np.ndarray, wavelength: float,
                 x_pixel_size: float, y_pixel_size: float) -> None:
        exp_geom={'foc_pos': foc_pos, 'rot_axis': rot_axis, 'smp_pos': smp_pos,
                  'kin_min': kin_min, 'kin_max': kin_max, 'wavelength': wavelength,
                  'x_pixel_size': x_pixel_size, 'y_pixel_size': y_pixel_size}
        super(ScanSetup, self).__init__(exp_geom=exp_geom)

    @classmethod
    def _lookup_dict(cls) -> Dict[str, str]:
        lookup = {}
        for section in cls.attr_dict:
            for option in cls.attr_dict[section]:
                lookup[option] = section
        return lookup

    def __iter__(self) -> Iterator[str]:
        return self._lookup.__ifter__()

    def __contains__(self, attr: str) -> bool:
        return attr in self._lookup

    def __repr__(self) -> str:
        return self._format(self.export_dict()).__repr__()

    def __str__(self) -> str:
        return self._format(self.export_dict()).__str__()

    def keys(self) -> Iterable[str]:
        return list(self)

    @classmethod
    def import_ini(cls, ini_file: str, **kwargs: Any) -> ScanSetup:
        """Initialize a :class:`ScanSetup` object with an
        ini file.

        Parameters
        ----------
        ini_file : str, optional
            Path to the ini file. Load the default parameters
            if None.
        **kwargs : dict
            Experimental geometry parameters.
            Initialized with `ini_file` if not provided.

        Returns
        -------
        scan_setup : ScanSetup
            A :class:`ScanSetup` object with all the attributes
            imported from the ini file.
        """
        attr_dict = cls._import_ini(ini_file)
        for option, section in cls._lookup_dict().items():
            if option in kwargs:
                attr_dict[section][option] = kwargs[option]
        return cls(**attr_dict['exp_geom'])

    def _det_to_k(self, y_coords: np.ndarray, x_coords: np.ndarray, source: np.ndarray) -> np.ndarray:
        delta_y = y_coords * self.y_pixel_size - source[1]
        delta_x = x_coords * self.x_pixel_size - source[0]
        phis = np.arctan2(delta_y, delta_x)
        thetas = np.arctan(np.sqrt(delta_x**2 + delta_y**2) / source[2])
        return np.stack((np.sin(thetas) * np.cos(phis),
                         np.sin(thetas) * np.sin(phis),
                         np.cos(thetas)), axis=-1)

    def _k_to_det(self, karr: np.ndarray, source: np.ndarray) -> np.ndarray:
        theta, phi = np.arccos(karr[..., 2]), np.arctan2(karr[..., 1], karr[..., 0])
        det_x = source[2] * np.tan(theta) * np.cos(phi) + source[0]
        det_y = source[2] * np.tan(theta) * np.sin(phi) + source[1]
        return np.stack((det_y / self.y_pixel_size, det_x / self.x_pixel_size), axis=-1)

    def detector_to_kout(self, y_coords: np.ndarray, x_coords: np.ndarray) -> np.ndarray:
        return self._det_to_k(y_coords, x_coords, self.smp_pos)

    def kout_to_detector(self, kout: np.ndarray) -> np.ndarray:
        return self._k_to_det(kout, self.smp_pos)

    def detector_to_kin(self, y_coords: np.ndarray, x_coords: np.ndarray) -> np.ndarray:
        return self._det_to_k(y_coords, x_coords, self.foc_pos)

    def kin_to_detector(self, kin: np.ndarray) -> np.ndarray:
        return self._k_to_det(kin, self.foc_pos)

    def index_pts(self, streaks: np.ndarray) -> np.ndarray:
        delta_x = streaks[:, :4:2] * self.x_pixel_size - self.smp_pos[0]
        delta_y = streaks[:, 1:4:2] * self.y_pixel_size - self.smp_pos[1]
        taus_x = delta_x[:, 1] - delta_x[:, 0]
        taus_y = delta_y[:, 1] - delta_y[:, 0]
        taus_abs = taus_x**2 + taus_y**2
        products = (delta_y[:, 0] * taus_x - delta_x[:, 0] * taus_y) / taus_abs
        return np.stack((-taus_y * products, taus_x * products), axis=1)

    def tilt_matrices(self, tilts: np.ndarray) -> np.ndarray:
        return tilt_matrix(tilts, self.rot_axis)

class ScanStreaks():
    columns = {'frames', 'streaks', 'x', 'y', 'I', 'bgd'}
    init_set = {}

    def __init__(self, indices: np.ndarray) -> None:
        super(ScanStreaks, self).__init__(indices=indices)
=======
    @staticmethod
    def _find_reduced(vectors: np.ndarray, basis: np.ndarray) -> np.ndarray:
        """Return vector indices, that satisfy the first condition of
        Lenstra-Lenstra-Lovász lattice basis reduction [LLL]_.

        Args:
            vectors : array of vectors of shape (N, 3).
            basis : basis set of shape (M, 3).

        References:
            .. [LLL]: "Lenstra-Lenstra-Lovász lattice basis reduction algorithm."
                      Wikipedia, Wikimedia Foundation, 4 Jul. 2022.
        """
        prod = vectors.dot(basis.T)
        mask = 2.0 * np.abs(prod) < (basis * basis).sum(axis=1)
        return np.where(mask.all(axis=1))[0]

    def filter_direction(self, axis: Sequence[float], rng: float, sigma: float) -> FourierIndexer:
        """Mask out a specific direction in 3D data. Useful for correcting artifacts in a
        Fourier image caused by the detector gaps.

        Args:
            axis : Direction of the masking line.
            rng : Width of the masking line.
            sigma : Smoothness of the masking line.

        Returns:
            New :class:`FourierIndexer` object with the masked out 3D data.
        """
        val = self.val * filter_direction(grid=self.grid, axis=axis, rng=rng, sigma=sigma,
                                          num_threads=self.num_threads)
        return self.replace(val=val)

    def find_peaks(self, val: float, dmin: float=0.0, dmax: float=np.inf) -> np.ndarray:
        """Find a set of basis vectors, that correspond to the peaks in the 3D data, that lie
        above the threshold ``val``.

        Args:
            val : Threshold value.
            dmin : Minimum peak distance. All the peaks below the bound are discarded.
            dmin : Maximum peak distance. All the peaks above the bound are discarded.

        Returns:
            A set of peaks in the 3D data in order of distance.
        """
        mask = self.val > val
        peak_lbls, peak_num = label(mask)
        index = np.arange(1, peak_num + 1)
        peaks = np.array(center_of_mass(self.val, labels=peak_lbls, index=index))
        dists = np.array(mean(np.sum(self.grid**2, axis=-1), labels=peak_lbls, index=index))
        mask = (dists > dmin**2) & (dists < dmax**2)
        idxs = np.argsort(dists[mask])
        peaks = np.concatenate((peaks[[np.argmin(dists)]], peaks[mask][idxs]), axis=0)
        return self.get_coordinates(peaks)

    def fitness(self, x: np.ndarray, center: np.ndarray, sigma: float,
                cutoff: float) -> Tuple[float, np.ndarray]:
        """Criterion function for Fourier autoindexing based on maximising the intersection
        between the experimental mapping and a grid of guassian peaks defined by a
        set of basis vectors ``x`` and lying in the sphere of radius ``cutoff``.

        Args:
            x : Flattened matrix of basis vectors.
            center : Center of the modelled grid.
            sigma : A width of diffraction orders.
            cutoff : Distance cutoff for a modelled grid.

        Returns:
            The intersection criterion and the gradient.
        """
        return gaussian_grid(p_arr=self.val, x_arr=self.x, y_arr=self.y, z_arr=self.z,
                             basis=x[:9].reshape((3, 3)), center=center, sigma=sigma,
                             cutoff=cutoff, num_threads=self.num_threads)

    def reduce_peaks(self, center: np.ndarray, peaks: np.ndarray, sigma: float,
                     cutoff: Optional[float]=None) -> Basis:
        """Reduce a set of peaks ``peaks`` to three basis vectors, that maximise the intersection
        between the experimental mapping and a grid of peaks formed by the basis. The grid of peaks
        is confined in a sphere of radius ``cutoff``.

        Args:
            center : Center of the grid.
            peaks : Set of peaks.
            sigma : Width of a peak in the grid.
            cutoff : A distance to the furthest peak in the grid. Defined by the distance to the
                furthest peak in ``peaks`` if None.

        Returns:

        """
        if cutoff is None:
            cutoff = np.max(np.sum(peaks**2, axis=1))**0.5

        basis = np.zeros((3, 3))
        idxs = np.arange(peaks.shape[0])
        for i in range(3):
            criteria = []
            for peak in peaks[idxs]:
                basis[i] = peak
                criteria.append(self.fitness(x=basis.ravel(), center=center, sigma=sigma,
                                             cutoff=cutoff)[0])
            basis[i] = peaks[idxs][np.argmin(criteria)]
            idxs = self._find_reduced(peaks, basis[:i + 1])
        return Basis.import_matrix(basis)

    def white_tophat(self, structure: np.ndarray) -> Map3D:
        """Perform 3-dimensional white tophat filtering.

        Args:
            structure : Structuring element used for the filter.

        Returns:
            New 3D data container with the filtered data.
        """
        val = median_filter(self.val, footprint=structure, num_threads=self.num_threads)
        return self.replace(val=self.val - val)
>>>>>>> dev-dataclass
