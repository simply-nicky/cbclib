import argparse
from multiprocessing import cpu_count
import numpy as np
import pandas as pd
import pygmo
from scipy.optimize import differential_evolution
from tqdm.auto import tqdm
import cbclib as cbc

def index_pygmo(samples_path: str, data_path: str, setup_path: str, basis_path: str, tilt_tol: float, smp_tol: float,
                q_abs: float, width: float, num_gen: int, pop_size: int, num_threads: int, verbose: bool):
    streaks = cbc.ScanStreaks.import_hdf(data_path, 'data', cbc.ScanSetup.import_ini(setup_path))
    samples = cbc.ScanSamples.import_dataframe(pd.read_hdf(samples_path, 'data'))
    basis = cbc.Basis.read_ini(basis_path)
    frames = streaks.dataframe['frames'].unique()

    for frame in tqdm(frames, total=frames.size, disable=not verbose, desc='Run CBC indexing refinement'):
        problem = streaks.refine_indexing(frame=frame, tol=(tilt_tol, smp_tol, 0.0), basis=basis,
                                          sample=samples[frame], q_abs=q_abs, width=width)

        uda = pygmo.de(gen=num_gen)
        algo = pygmo.algorithm(uda)
        prob = pygmo.problem(problem)
        pops = [pygmo.population(size=pop_size, prob=prob, b=pygmo.bfe()) for _ in range(num_threads)]
        archi = pygmo.archipelago()
        for pop in pops:
            archi.push_back(algo=algo, pop=pop)

        archi.evolve()
        archi.wait()

        champion = archi.get_champions_x()[np.argmin(archi.get_champions_f())]
        problem.index_frame(champion, frame=frame, num_threads=num_threads)
        samples[frame] = problem.generate_sample(champion)

    streaks.dataframe.to_hdf(data_path, 'data')
    samples.to_dataframe().to_hdf(samples_path, 'data')

def criterion(x: np.ndarray, problem: cbc.IndexProblem) -> float:
    return problem.fitness(x)[0]

def index_scipy(samples_path: str, data_path: str, setup_path: str, basis_path: str, tilt_tol: float, smp_tol: float,
                q_abs: float, width: float, num_gen: int, pop_size: int, num_threads: int, verbose: bool):
    streaks = cbc.ScanStreaks.import_hdf(data_path, 'data', cbc.ScanSetup.import_ini(setup_path))
    samples = cbc.ScanSamples.import_dataframe(pd.read_hdf(samples_path, 'data'))
    basis = cbc.Basis.read_ini(basis_path)
    frames = streaks.dataframe['frames'].unique()

    for frame in tqdm(frames, total=frames.size, disable=not verbose, desc='Run CBC indexing refinement'):
        problem = streaks.refine_indexing(frame=frame, tol=(tilt_tol, smp_tol, 0.0), basis=basis,
                                          sample=samples[frame], q_abs=q_abs, width=width)

        res = differential_evolution(criterion, bounds=np.stack(problem.get_bounds()).T, args=(problem,),
                                     maxiter=num_gen, popsize=pop_size, workers=num_threads, updating='deferred')

        champion = res.x
        problem.index_frame(champion, frame=frame, num_threads=num_threads)
        samples[frame] = problem.generate_sample(champion)

    streaks.dataframe.to_hdf(data_path, 'data')
    samples.to_dataframe().to_hdf(samples_path, 'data')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CBC indexing refinement.")
    parser.add_argument('backend', type=str, choices=['pygmo', 'scipy'], help="Choose the back-end for DE refinement")
    parser.add_argument('samples_path', type=str, help="Path to the input Pandas samples table")
    parser.add_argument('data_path', type=str, help="Path to the input Pandas data table")
    parser.add_argument('setup_path', type=str, help="Path to the experimental setup config file")
    parser.add_argument('basis_path', type=str, help="Path to the lattice basis config file")
    parser.add_argument('--tilt_tol', default=0.02, type=float, help="Sample tilting tolerance")
    parser.add_argument('--smp_tol', default=0.02, type=float, help="Sample position tolerance")
    parser.add_argument('--q_abs', default=0.3, type=float, help="Reciprocal space radius")
    parser.add_argument('--width', default=4.0, type=float, help="Streak width in pixels")
    parser.add_argument('--num_gen', default=200, type=int, help="Number of generations")
    parser.add_argument('--pop_size', default=15, type=int, help="Population size in DE refinement")
    parser.add_argument('--num_threads', default=cpu_count(), type=int, help="Number of threads")
    parser.add_argument('-v', '--verbose', action='store_true', help="Set the verbosity")

    args = vars(parser.parse_args())
    backend = args.pop('backend')
    if backend == 'pygmo':
        index_pygmo(**args)
    elif backend == 'scipy':
        index_scipy(**args)
    else:
        raise ValueError('backend option is invalid')
