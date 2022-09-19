import os
import argparse
from multiprocessing import cpu_count
import hdf5plugin
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import cbclib as cbc
from typing import Tuple

def read_whitefield(scan_num: int) -> np.ndarray:
    return np.load(f'results/scan_{scan_num:d}_whitefield.npy')

def main(scan_num: int, dir_path: str, setup_path: str,  data_path: str, table_path: str,
         roi: Tuple[int, int, int, int], frames: Tuple[int, int], mask_max: int,
         cor_range: Tuple[float, float], quant: float, cutoff: float, filter_threshold: float,
         group_threshold: float, num_chunks: int, num_threads: int, save_data: bool, verbose: bool):
    scan_setup = cbc.ScanSetup.import_ini(setup_path)

    h5_dir = os.path.join(dir_path, f'scan_frames/Scan_{scan_num:d}')
    h5_files = sorted([os.path.join(h5_dir, path) for path in os.listdir(h5_dir)
                       if path.endswith(('LambdaFar.nxs', '.h5'))])
    data = cbc.CrystData(cbc.CXIStore(h5_files, 'r'), cbc.Crop(roi))
    if save_data:
        data = data.update_output_file(cbc.CXIStore(data_path, 'a'))

    tables = []
    for idxs in tqdm(np.array_split(data.input_file.indices()[frames[0]:frames[1]], num_chunks), total=num_chunks,
                     desc=f'Processing scan {scan_num:d}', disable=not verbose):
        data = data.clear().load(idxs=idxs, processes=num_threads, verbose=False)
        data = data.update_mask(method='range-bad', vmax=mask_max)
        data = data.mask_pupil(scan_setup, padding=60)
        data = data.import_whitefield(data.transform.forward(read_whitefield(scan_num)))
        data = data.blur_pupil(scan_setup, padding=80, blur=20)
        det_obj = data.get_detector()
        det_obj = det_obj.generate_streak_data(vmin=cor_range[0], vmax=cor_range[1], size=(1, 3, 3))
        det_obj = det_obj.update_lsd(quant=quant)
        det_res = det_obj.detect(cutoff=cutoff, filter_threshold=filter_threshold,
                                 group_threshold=group_threshold)
        det_res = det_res.generate_bgd_mask()
        det_res = det_res.update_streak_data()
        tables.append(det_res.export_table(concatenate=True))
        if save_data:
            data.save(['data', 'good_frames', 'mask', 'frames', 'cor_data', 'background'],
                    mode='insert', idxs=idxs)

    if save_data:
        data.save('whitefield')

    table = pd.concat(tables)
    table.to_hdf(table_path, 'data')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CBC streak detection.")
    parser.add_argument('scan_num', type=int, help="Scan number")
    parser.add_argument('dir_path', type=str, help="Path to the root folder of the experiment")
    parser.add_argument('setup_path', type=str, help="Path to the experimental setup config file")
    parser.add_argument('data_path', type=str, help="Path to the output CXI file")
    parser.add_argument('table_path', type=str, help="Path to the output Pandas table file")
    parser.add_argument('--roi', type=int, nargs=4, help="Region of interest")
    parser.add_argument('--frames', type=int, nargs=2, help="Frame range")
    parser.add_argument('--mask_max', type=int, help="The upper bound of a photon counts range,"\
                                                     "outside of which the data is masked")
    parser.add_argument('--cor_range', type=float, nargs=2, help="Range of background corrected values"\
                                                                 "to take into account for the detection")
    parser.add_argument('--quant', default=2.0e-2, type=float, help="LSD gradient step")
    parser.add_argument('--cutoff', default=70.0, type=float, help="Streaks grouping cut-off")
    parser.add_argument('--filter_threshold', type=float, help="Streaks filtering threshold")
    parser.add_argument('--group_threshold', default=0.7, type=float, help="Streaks grouping threshold")
    parser.add_argument('--num_chunks', default=100, type=int, help="Number of chunks")
    parser.add_argument('--num_threads', default=cpu_count(), type=int, help="Number of threads")
    parser.add_argument('--save_data', action='store_true', help='Save detector data')
    parser.add_argument('-v', '--verbose', action='store_true', help="Set the verbosity")

    args = vars(parser.parse_args())
    main(**args)
