from dataclasses import dataclass, field
from multiprocessing import cpu_count
import os
from typing import Callable, List, Optional, Tuple
from tqdm.auto import tqdm
import hdf5plugin
import numpy as np
import pandas as pd
import pygmo
from scipy.optimize import differential_evolution
from ..cbc_setup import  Basis, ScanSetup, ScanSamples
from ..cbc_scaling import CBCTable, Refiner, SetupRefiner, SampleRefiner
from ..cxi_protocol import CXIStore
from ..data_container import INIContainer, Crop
from ..data_processing import CrystData
from ..log_protocol import LogContainer

@dataclass
class Executor():
    out_path            : str = 'None'
    basis_path          : str = 'None'
    hkl_path            : str = 'None'
    samples_path        : str = 'None'
    setup_path          : str = 'None'
    table_path          : str = 'None'
    num_threads         : int = cpu_count()
    verbose             : bool = True

    basis               : Optional[Basis] = None
    samples             : Optional[ScanSamples] = None
    setup               : Optional[ScanSetup] = None
    table               : Optional[CBCTable] = None
    q_abs               : float = 0.0

    def __post_init__(self):
        if os.path.isfile(self.basis_path):
            self.basis = Basis.import_ini(self.basis_path)
        if os.path.isfile(self.samples_path):
            self.samples = ScanSamples.import_dataframe(pd.read_hdf(self.samples_path, 'data'))
        if os.path.isfile(self.setup_path):
            self.setup = ScanSetup.import_ini(self.setup_path)
        if os.path.isfile(self.table_path) and self.setup:
            self.table = CBCTable.import_hdf(self.table_path, 'data', self.setup)
        if os.path.isfile(self.hkl_path):
            self.hkl = np.load(self.hkl_path)
        elif self.basis is not None:
            self.hkl = self.basis.generate_hkl(self.q_abs)
        else:
            self.hkl = None

@dataclass
class DetectExecutor(Executor, INIContainer):
    __ini_fields__ = {'system': ('basis_path', 'h5_files', 'hkl_path', 'out_path', 'samples_path',
                                 'setup_path', 'table_path', 'wf_path', 'detector', 'num_chunks',
                                 'num_threads', 'verbose'),
                      'detection': ('cor_rng', 'cutoff', 'dilations', 'filter_thr', 'frames',
                                    'group_thr', 'vrange', 'quant', 'q_abs', 'roi', 'snr_thr', 'width')}

    h5_files            : List[str] = field(default_factory=list)
    wf_path             : str = 'None'
    detector            : str = 'lsd'
    num_chunks          : int = 100

    roi                 : Tuple[int, int, int, int] = (0, 0, 0, 0)
    frames              : Tuple[int, int] = (0, 0)
    vrange              : Tuple[int, int] = (0, 0)
    cor_rng             : Tuple[float, float] = (0.0, 0.0)
    quant               : float = 0.02
    cutoff              : float = 0.0
    filter_thr          : float = 0.0
    group_thr           : float = 1.0
    dilations           : Tuple[float, float, float] = (0.0, 1.0, 6.0)
    width               : float = 0.0
    snr_thr             : float = 1.0

    def __post_init__(self):
        super(DetectExecutor, self).__post_init__()
        if os.path.isfile(self.wf_path):
            self.whitefield = np.load(self.wf_path)
        else:
            self.whitefield = None

    def detect_lsd(self, data: CrystData) -> pd.DataFrame:
        lsd_det = data.lsd_detector()
        lsd_det = lsd_det.generate_patterns(vmin=self.cor_rng[0], vmax=self.cor_rng[1],
                                            size=(1, 3, 3))
        lsd_det = lsd_det.update_lsd(quant=self.quant)
        lsd_det = lsd_det.detect(cutoff=self.cutoff, filter_threshold=self.filter_thr,
                                 group_threshold=self.group_thr, dilation=self.dilations[0])
        lsd_det = lsd_det.refine_streaks(self.dilations[1])
        lsd_det = lsd_det.update_patterns(dilations=self.dilations)
        return lsd_det.export_table(concatenate=True)

    def detect_model(self, data: CrystData) -> pd.DataFrame:
        if self.table is not None:
            data = data.import_patterns(self.table.table).update_background()
        mdl_det = data.model_detector(self.basis, self.samples, self.setup)
        mdl_det = mdl_det.detect(hkl=self.hkl, width=self.width, threshold=self.snr_thr)
        mdl_det = mdl_det.refine_streaks(self.dilations[1])
        mdl_det = mdl_det.update_patterns(dilations=self.dilations)
        return mdl_det.export_table(concatenate=True)

    def get_detector(self) -> Callable[[CrystData], pd.DataFrame]:
        if self.detector == 'lsd':
            return self.detect_lsd
        if self.detector == 'model':
            return self.detect_model
        raise ValueError(f'Invalid detector: {self.detector}')

    def run(self):
        data = CrystData(CXIStore(self.h5_files, 'r'), Crop(self.roi),
                         whitefield=self.whitefield)
        atts = ['data', 'good_frames', 'mask', 'frames', 'cor_data', 'background']

        tables = []
        indices = np.array_split(data.input_file.indices()[self.frames[0]:self.frames[1]],
                                 self.num_chunks)
        detector = self.get_detector()
        for idxs in tqdm(indices, total=self.num_chunks, disable=not self.verbose,
                         desc='Processing scan'):
            data = data.clear(atts).load(idxs=idxs, processes=self.num_threads, verbose=False)
            if 'mask' not in data.input_file:
                data = data.update_mask(vmin=self.vrange[0], vmax=self.vrange[1])
            data = data.mask_pupil(self.setup, padding=40)
            data = data.update_background()
            data = data.blur_pupil(self.setup, padding=60, blur=20)
            tables.append(detector(data))

        table = pd.concat(tables, ignore_index=True)
        table.to_hdf(self.out_path, 'data')

def criterion(x: np.ndarray, refiner: Refiner) -> float:
    return refiner.fitness(x)[0]

@dataclass
class RefineExecutor(Executor, INIContainer):
    __ini_fields__ = {'system': ('out_path', 'basis_path', 'hkl_path', 'log_path', 'samples_path',
                                 'setup_path', 'table_path', 'x0_path', 'backend', 'num_threads',
                                 'verbose'),
                      'refinement': ('refine', 'tols', 'q_abs', 'width', 'alpha', 'num_gen',
                                     'pop_size')}

    log_path            : str = 'None'
    x0_path             : str = 'None'
    backend             : str = 'pygmo'

    refine              : str = 'scan'
    tols                : List[float] = field(default_factory=list)
    width               : float = 0.0
    alpha               : float = 0.0
    num_gen             : int = 0
    pop_size            : int = 0

    def __post_init__(self):
        super(RefineExecutor, self).__post_init__()
        if os.path.isfile(self.log_path):
            if self.table.frames[0]:
                idxs = np.insert(self.table.frames, 0, 0)
            else:
                idxs = self.table.frames
            ldata = LogContainer().read_logs(self.log_path, idxs=idxs).read_translations()
            self.tilts = ldata.translations[-self.table.frames.size:, 3] - ldata.translations[0, 3]
        if os.path.isfile(self.x0_path):
            self.x0 = np.load(self.x0_path)
        else:
            self.x0 = None

    def fit_pygmo(self, refiner: Refiner) -> np.ndarray:
        uda = pygmo.de(gen=self.num_gen)
        algo = pygmo.algorithm(uda)
        prob = pygmo.problem(refiner)
        pops = [pygmo.population(size=self.pop_size, prob=prob, b=pygmo.bfe())
                for _ in range(self.num_threads)]
        archi = pygmo.archipelago()
        for pop in pops:
            archi.push_back(algo=algo, pop=pop)

        archi.evolve()
        archi.wait()

        return archi.get_champions_x()[np.argmin(archi.get_champions_f())]

    def fit_scipy(self, refiner: Refiner) -> np.ndarray:
        res = differential_evolution(criterion, bounds=np.stack(refiner.get_bounds()).T,
                                     maxiter=self.num_gen, popsize=self.pop_size,
                                     workers=self.num_threads, updating='deferred',
                                     args=(refiner,))

        return res.x

    def get_fitter(self) -> Callable[[Refiner], np.ndarray]:
        if self.backend == 'pygmo':
            return self.fit_pygmo
        if self.backend == 'scipy':
            return self.fit_scipy
        raise ValueError(f'Invalid backend: {self.backend}')

    def refine_samples(self):
        bnds = SampleRefiner.generate_bounds(z_tol=self.tols[0], tilt_tol=self.tols[1],
                                             frames=np.arange(1), x0=self.x0)
        fitter = self.get_fitter()
        size = np.asarray(bnds[1] - bnds[0], dtype=bool).sum()
        np.save(self.out_path, np.zeros((self.table.frames.size, size)))
        champions = np.load(self.out_path, mmap_mode='r+')

        for idx, frame in tqdm(enumerate(self.table.frames), total=self.table.frames.size,
                               disable=not self.verbose, desc='Run CBC sample refinement'):
            part = self.table.get_frames(frame)
            refiner = part.refine_samples(bounds=bnds, basis=self.basis,
                                          samples=self.samples, hkl=self.hkl,
                                          width=self.width, alpha=self.alpha)
            refiner.filter_hkl(refiner.x0)
            champions[idx] = fitter(refiner)
            champions.flush()

    def refine_setup(self):
        bnds = SetupRefiner.generate_bounds(lat_tol=self.tols[:2], foc_tol=self.tols[2],
                                            rot_tol=self.tols[3], z_tol=self.tols[4],
                                            tilt_tol=self.tols[5], frames=self.table.frames,
                                            x0=self.x0)
        refiner = self.table.refine_setup(bounds=bnds, basis=self.basis, tilts=self.tilts,
                                          hkl=self.hkl, width=self.width, alpha=self.alpha)
        refiner.filter_hkl(refiner.x0)

        fitter = self.get_fitter()
        champion = fitter(refiner)

        np.save(self.out_path, champion)

    def run(self):
        if self.refine == 'setup':
            self.refine_setup()
        elif self.refine == 'samples':
            self.refine_samples()
        else:
            raise ValueError(f'Invalid refine keyword: {self.refine}')
