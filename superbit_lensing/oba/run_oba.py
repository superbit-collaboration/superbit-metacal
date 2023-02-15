from pathlib import Path
from argparse import ArgumentParser

from oba_io import IOManager
from oba_runner import OBARunner
from superbit_lensing import utils

import ipdb

def parse_args():

    parser = ArgumentParser()

    parser.add_argument('target_name', type=str,
                        help='Name of the target to run on-board analysis for')
    # TODO: We probably want this as a req arg, but we don't know what will
    # be configurable yet! More is hard-coded as well due to QCC requirements
    parser.add_argument('-config_file', type=str, default=None,
                       help='Filename for the on-board analysis configuration')
    parser.add_argument('-config_dir', type=str, default=None,
                        help='Directory of OBA pipeline config file')
    parser.add_argument('-root_dir', type=str, default=None,
                        help='Root directory for OBA run (if testing locally)')
    parser.add_argument('-bands', type=str, default='b,lum',
                        help='List of band names separated by commas (no space)')
    parser.add_argument('-det_bands', type=str, default='b,lum',
                        help='List of band names separated by commas ' +
                        '(no space) to use for the detection image')
    # parser.add_argument('-fname_base', action='store', type=str, default=None,
    #                     help='Basename of image files')
    parser.add_argument('--test', action='store_true', default=False,
                        help='Set to indicate that this is a test run, ' +
                        'which will utilize the TestPrepper')
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='Set to overwrite files')
    parser.add_argument('--vb', action='store_true', default=False,
                        help='Verbosity')

    return parser.parse_args()

def main(args):

    #-----------------------------------------------------------------
    # Initial setup

    config_file = args.config_file
    config_dir = args.config_dir
    root_dir = args.root_dir
    target_name = args.target_name
    bands = args.bands
    det_bands = args.det_bands
    test = args.test
    overwrite = args.overwrite
    vb = args.vb

    # convert bands into a list of strings
    bands = bands.split(',')
    det_bands = det_bands.split(',')

    #-----------------------------------------------------------------
    # Logger setup

    logdir = (Path(utils.get_test_dir()) / 'oba_test').resolve()
    logfile = str(logdir / f'{target_name}_oba.log')

    log = utils.setup_logger(logfile, logdir=logdir)
    logprint = utils.LogPrint(log, vb)

    logprint(f'Log is being saved at {logfile}')

    #-----------------------------------------------------------------
    # config setup

    # TODO: Decide if this is necessary!
    if config_file is not None:
        config_file = Path(config_file)
        if config_dir is not None:
            config_file = Path(config_dir) / config_file
        config_file = config_file.resolve()

    logprint(f'Using config file {config_file}')
    if not config_file.is_file():
        raise ValueError(f'OBA pipeline config file not found: {config_file}')

    #-----------------------------------------------------------------
    # I/O setup (registering filepaths, dirs, etc. for run)

    if root_dir is None:
        logprint('root_dir is None; using QCC paths')
    else:
        logprint(f'root_dir is {root_dir}')

    io_manager = IOManager(root_dir=root_dir, target_name=target_name)
    io_manager.print_dirs(logprint=logprint)

    #-----------------------------------------------------------------
    # Test setup, if needed

    if test is True:
        # handle any needed setup for simulated inputs
        from setup_test import TestPrepper

        logprint('\nTEST == TRUE; Starting test prepper\n')

        prepper = TestPrepper(target_name, bands)
        prepper.go(io_manager, overwrite=overwrite, logprint=logprint)

    #-----------------------------------------------------------------
    # Run pipeline

    runner = OBARunner(
        config_file,
        io_manager,
        target_name,
        bands,
        det_bands,
        logprint,
        test=test
        )

    runner.go(overwrite=overwrite)

    logprint('Done!')

    return 0

if __name__ == '__main__':
    args = parse_args()
    rc = main(args)

    if rc == 0:
        print('\nrun_oba.py completed without error\n')
    else:
        print(f'\nrun_oba.py failed with rc={rc}\n')
