from dataclasses import dataclass
from multiprocessing import cpu_count
import os
from typing import Callable, Optional, Tuple
from tqdm.auto import tqdm
import hdf5plugin
import numpy as np
import pandas as pd
import pygmo
from scipy.optimize import differential_evolution
from .. import Basis, CBCTable, Crop, CrystData, CXIStore, SampleProblem, ScanSamples, ScanSetup
from ..data_container import INIContainer

@dataclass
class Executor():
    out_path            : str = 'None'
    basis_path          : str = 'None'
    samples_path        : str = 'None'
    setup_path          : str = 'None'
    table_path          : str = 'None'
    num_threads         : int = cpu_count()
    verbose             : bool = True

    basis               : Optional[Basis] = None
    samples             : Optional[ScanSamples] = None
    setup               : Optional[ScanSetup] = None
    table               : Optional[CBCTable] = None

    def __post_init__(self):
        if os.path.isfile(self.basis_path):
            self.basis = Basis.import_ini(self.basis_path)
        if os.path.isfile(self.samples_path):
            self.samples = ScanSamples.import_dataframe(pd.read_hdf(self.samples_path, 'data'))
        if os.path.isfile(self.setup_path):
            self.setup = ScanSetup.import_ini(self.setup_path)
        if os.path.isfile(self.table_path) and self.setup:
            self.table = CBCTable.import_hdf(self.table_path, 'data', self.setup)

@dataclass
class DetectExecutor(Executor, INIContainer):
    __ini_fields__ = {'system': ('out_path', 'basis_path', 'dir_path', 'samples_path', 'setup_path',
                                 'table_path', 'wf_path', 'detector', 'num_chunks', 'num_threads',
                                 'verbose'),
                      'detection': ('scan_num', 'roi', 'frames', 'imax', 'cor_rng', 'quant',
                                    'cutoff', 'filter_thr', 'group_thr', 'dilation', 'q_abs', 
                                    'width', 'snr_thr')}

    dir_path            : str = 'None'
    wf_path             : str = 'None'
    detector            : str = 'lsd'
    num_chunks          : int = 100

    scan_num            : int = 0
    roi                 : Tuple[int, int, int, int] = (0, 0, 0, 0)
    frames              : Tuple[int, int] = (0, 0)
    imax                : int = 10000000
    cor_rng             : Tuple[float, float] = (0.0, 0.0)
    quant               : float = 0.02
    cutoff              : float = 0.0
    filter_thr          : float = 0.0
    group_thr           : float = 0.85
    dilation            : float = 0.0
    q_abs               : float = 0.3
    width               : float = 4.0
    snr_thr             : float = 0.95

    def detect_lsd(self, data: CrystData) -> pd.DataFrame:
        lsd_det = data.lsd_detector()
        lsd_det = lsd_det.generate_pattern(vmin=self.cor_rng[0], vmax=self.cor_rng[1],
                                            size=(1, 3, 3))
        lsd_det = lsd_det.update_lsd(quant=self.quant)
        lsd_det = lsd_det.detect(cutoff=self.cutoff, filter_threshold=self.filter_thr,
                                    group_threshold=self.group_thr, dilation=self.dilation)
        lsd_det = lsd_det.draw_streaks()
        lsd_det = lsd_det.draw_background()
        lsd_det = lsd_det.update_pattern()
        return lsd_det.export_table(concatenate=True)

    def detect_model(self, data: CrystData) -> pd.DataFrame:
        data = data.import_patterns(self.table.table).update_background()
        mdl_det = data.model_detector(self.basis, self.samples, self.setup)
        mdl_det = mdl_det.detect(q_abs=self.q_abs, width=self.width, threshold=self.snr_thr)
        mdl_det = mdl_det.draw_streaks()
        mdl_det = mdl_det.draw_background()
        mdl_det = mdl_det.update_pattern()
        return mdl_det.export_table(concatenate=True)

    def get_detector(self) -> Callable[[CrystData], pd.DataFrame]:
        if self.detector == 'lsd':
            return self.detect_lsd
        if self.detector == 'model':
            return self.detect_model
        raise ValueError(f'Invalid detector: {self.detector}')

    def run(self):
        scan_setup = ScanSetup.import_ini(self.setup_path)

        h5_dir = os.path.join(self.dir_path, f'scan_frames/Scan_{self.scan_num:d}')
        h5_files = sorted([os.path.join(h5_dir, path) for path in os.listdir(h5_dir)
                           if path.endswith(('LambdaFar.nxs', '.h5'))])
        data = CrystData(CXIStore(h5_files, 'r'), Crop(self.roi))

        tables = []
        indices = np.array_split(data.input_file.indices()[self.frames[0]:self.frames[1]],
                                 self.num_chunks)
        detector = self.get_detector()
        for idxs in tqdm(indices, total=self.num_chunks, disable=not self.verbose,
                        desc=f'Processing scan {self.scan_num:d}'):
            data = data.clear().load(idxs=idxs, processes=self.num_threads, verbose=False)
            data = data.update_mask(method='range-bad', vmax=self.imax)
            data = data.mask_pupil(scan_setup, padding=60)
            data = data.import_whitefield(np.load(self.wf_path))
            data = data.blur_pupil(scan_setup, padding=80, blur=20)
            tables.append(detector(data))

        table = pd.concat(tables, ignore_index=True)
        table.to_hdf(self.out_path, 'data')

def criterion(x: np.ndarray, problem: SampleProblem) -> float:
    return problem.fitness(x)[0]

@dataclass
class RefineExecutor(Executor, INIContainer):
    __ini_fields__ = {'system': ('out_path', 'basis_path', 'samples_path', 'setup_path',
                                 'table_path', 'backend', 'num_threads', 'verbose'),
                      'indexing': ('tilt_tol', 'smp_tol', 'q_abs', 'width', 'alpha', 'num_gen',
                                   'pop_size')}

    backend             : str = 'pygmo'

    tilt_tol            : float = 0.0
    smp_tol             : float = 0.0
    q_abs               : float = 0.3
    width               : float = 4.0
    alpha               : float = 0.0
    num_gen             : int = 0
    pop_size            : int = 0

    def fit_pygmo(self, problem: SampleProblem) -> np.ndarray:
        uda = pygmo.de(gen=self.num_gen)
        algo = pygmo.algorithm(uda)
        prob = pygmo.problem(problem)
        pops = [pygmo.population(size=self.pop_size, prob=prob, b=pygmo.bfe())
                for _ in range(self.num_threads)]
        archi = pygmo.archipelago()
        for pop in pops:
            archi.push_back(algo=algo, pop=pop)

        archi.evolve()
        archi.wait()

        return archi.get_champions_x()[np.argmin(archi.get_champions_f())]

    def fit_scipy(self, problem: SampleProblem) -> np.ndarray:
        res = differential_evolution(criterion, bounds=np.stack(problem.get_bounds()).T,
                                     maxiter=self.num_gen, popsize=self.pop_size,
                                     workers=self.num_threads, updating='deferred',
                                     args=(problem,))

        return res.x

    def get_fitter(self) -> Callable[[SampleProblem], np.ndarray]:
        if self.backend == 'pygmo':
            return self.fit_pygmo
        if self.backend == 'scipy':
            return self.fit_scipy
        raise ValueError(f'Invalid backend: {self.backend}')

    def run(self):
        frames = self.table.table['frames'].unique()

        patterns = []
        fitter = self.get_fitter()
        for frame in tqdm(frames, total=frames.size, disable=not self.verbose,
                        desc='Run CBC indexing refinement'):
            problem = self.table.refine(frame=frame, bounds=(self.tilt_tol, self.smp_tol, 0.0),
                                        basis=self.basis, sample=self.samples[frame],
                                        q_abs=self.q_abs, width=self.width, alpha=self.alpha)

            if self.num_gen:
                champion = fitter(problem)
            else:
                champion = problem.x0

            patterns.append(problem.index_frame(champion, frame=frame, 
                                                num_threads=self.num_threads))
            if self.num_gen:
                self.samples[frame] = problem.generate_sample(champion)

        pd.concat(patterns).to_hdf(self.out_path, 'data')
        if self.num_gen:
            self.samples.to_dataframe().to_hdf(self.samples_path, 'data')
