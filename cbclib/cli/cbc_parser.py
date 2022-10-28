import re
import argparse
from typing import Any, List, Optional, Type, Union
from .cbc_executors import Executor, RefineExecutor, DetectExecutor

class CBCParser():
    desc : str = "Command-line Interface for CBC data processing"

    helpers = {'alpha': "Regularisation coefficient",
               'backend': "Choose the back-end for DE refinement between 'pygmo' and 'scipy'",
               'basis_path': "Path to the lattice basis config file",
               'cor_rng': "Range of background corrected intensities used in the detection",
               'cutoff': "Streaks grouping cut-off",
               'detector': "Choose the type of the streak detector between 'lsd' and 'model'",
               'dilation': "Pattern masking dilation in pixels",
               'dir_path': "Folder where the experimental data is saved",
               'filter_thr': "Streaks filtering threshold",
               'frames': "Frame range",
               'group_thr': "Streaks grouping threshold",
               'imax': "The upper bound of good photon counts interval",
               'num_chunks': "Number of dataset chunks",
               'num_gen': "Number of DE generations",
               'num_threads': "Number of threads used in the calculations",
               'out_path': "Path were the output data will be saved",
               'pop_size': "Population size in DE refinement",
               'quant': "LSD gradient step",
               'q_abs': "Reciprocal space radius",
               'roi': "Detector region of interest",
               'samples_path': "Path to the samples table",
               'scan_num': "Scan number",
               'setup_path': "Path to the experimental setup config file",
               'smp_tol': "Sample position tolerance",
               'snr_thr': "Signal-to-noise threshold used in the detection",
               'table_path': "Path to the CBC table file",
               'tilt_tol': "Sample tilting tolerance",
               'verbose': "Set the verbosity",
               'wf_path': "Path to the file containing the white-field profile",
               'width': "Streak width in pixels"}

    @property
    def parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description=self.desc)
        parser.add_argument('command', type=str, metavar='command',
                            choices=['detect', 'refine'],
                            help='Subcommand to run')
        parser.add_argument('-f', '--ini_file', type=str,
                            help="Path to an INI file containing "\
                                 "all the parameters of the subcommand")
        return parser

    @classmethod
    def get_type(self, t: str) -> Any:
        types = {'float': float, 'int': int, 'bool': bool, 'complex': complex, 'str': str}
        is_type = re.search(r'(' + '|'.join(types.keys()) + ')', t)
        if is_type:
            return types[is_type.group(0)]
        return None

    @classmethod
    def get_nargs(self, t: str) -> Union[int, str, None]:
        is_nargs = re.search(r'(list|List|tuple|Tuple)\[([\s\S]*)\]$', t)
        if is_nargs and is_nargs.group(2):
            if '...' in is_nargs.group(2):
                return '+'
            return len(re.findall('(float|int|complex)', is_nargs.group(2)))
        return None

    def generate_parser(self, desc: str, executor: Type[Executor]) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description=desc)
        for attr, field in executor.__dataclass_fields__.items():
            if self.get_type(str(field.type)):
                args = {'type': self.get_type(str(field.type)),
                        'help': self.helpers[attr],
                        'default': field.default}
                if args['type'] in (float, int, complex):
                    args['nargs'] = self.get_nargs(str(field.type))
                if args['type'] == bool:
                    del args['type']
                    args['action'] = 'store_true'
                parser.add_argument('--' + attr, **args)
        return parser

    @property
    def parser_det(self) -> argparse.ArgumentParser:
        return self.generate_parser(desc="Run CBC streak detection",
                                    executor=DetectExecutor)

    @property
    def parser_ref(self) -> argparse.ArgumentParser:
        return self.generate_parser(desc="Run CBC sample refinement",
                                    executor=RefineExecutor)

    def __call__(self, args: List[str]):
        if set(args[1:4]).intersection(('-f', '--ini_file')):
            args, remainder = self.parser.parse_args(args[1:4]), args[4:]
        else:
            args, remainder = self.parser.parse_args(args[1:2]), args[2:]
        getattr(self, args.command)(remainder, args.ini_file)

    def detect(self, args: List[str], ini_file: Optional[str]=None):
        """Command Line Interface for CBC streak detection.
        """
        if ini_file:
            executor = DetectExecutor.import_ini(ini_file)
        else:
            executor = DetectExecutor(**self.parser_det.parse_args(args))
        executor.run()

    def refine(self, args: List[str], ini_file: Optional[str]=None):
        """Command Line Interface for CBC sample refinement.
        """
        if ini_file:
            executor = RefineExecutor.import_ini(ini_file)
        else:
            executor = RefineExecutor(**self.parser_det.parse_args(args))
        executor.run()
