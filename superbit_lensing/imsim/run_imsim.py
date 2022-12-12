# This is to ensure that numpy doesn't have
# OpenMP optimizations clobber our own multiprocessing
import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

from argparse import ArgumentParser

from runner import ImSimRunner

import ipdb

def parse_args():
    parser = ArgumentParser()

    parser.add_argument('config_file', type=str,
                        help='Configuration file for mock sims')
    parser.add_argument('-config_dir', type=str, default=None,
                        help='Directory location of the config file')
    parser.add_argument('-run_name', type=str, default=None,
                        help='Name of mock simulation run')
    parser.add_argument('-outdir', type=str,
                        help='Output directory of simulated files')
    parser.add_argument('-ncores', type=int, default=None,
                        help='Number of cores to use for multiprocessing')
    parser.add_argument('--overwrite', action='store_true', default=None,
                        help='Turn on to overwrite existing files')
    parser.add_argument('--vb', action='store_true', default=None,
                        help='Turn on for verbose prints')

    return parser.parse_args()

def main(args):

    runner = ImSimRunner(args)

    runner.go()

    runner.logprint('Done!')

    return

if __name__ == '__main__':
    args = parse_args()
    main(args)
