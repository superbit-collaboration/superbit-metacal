import os
from argparse import ArgumentParser

from oba_io import IOManager
from oba_runner import OBARunner

from superbit_lensing.coadd import SWarpRunner
from superbit_lensing.detection import SExtractorRunner
from superbit_lensing import utils

import ipdb

def parse_args():

    parser = ArgumentParser()

    # NOTE: For a normal run, you would pass a top-level config file to specify
    # how to run the on-board analysis pipeline. For testing, you instead pass
    # a path_config file that specifies the minimal needed information to
    # automatically generate test config files to simulate images and run the
    # on-board analysis on
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-config_file', type=str, default=None,
                       help='Filename of the top-level on-board analysis ' +
                       'pipeline config file')
    group.add_argument('-path_config', type=str, default=None,
                       help='A yaml config file that defines the paths ' +
                       'needed to run an on-board analysis test')

    parser.add_argument('-target_name', action='store', type=str, default=None,
                        help='Name of the target to run on-board analysis for')
    # parser.add_argument('basedir', type=str,
    #                     help='Directory containing imaging data')
    parser.add_argument('-root_dir', type=str, default=None,
                        help='Root directory for OBA run (if testing locally)')
    # parser.add_argument('bands', type=str,
    #                     help='List of band names separated by commas (no space)')
    parser.add_argument('-config_dir', type=str, default=None,
                        help='Directory of OBA pipeline config file')
    # parser.add_argument('-det_bands', type=str, default=None,
    #                     help='List of band names separated by commas ' +
    #                     '(no space) to use for the detection image')
    # parser.add_argument('-outfile_base', type=str,
    #                     help='Base name for the output coadd files')
    # parser.add_argument('-outdir', type=str, default=None,
    #                     help='Output directory for MEDS file')
    # parser.add_argument('-fname_base', action='store', type=str, default=None,
    #                     help='Basename of image files')
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='Set to overwrite files')
    parser.add_argument('--vb', action='store_true', default=False,
                        help='Verbosity')

    # NOTE: The following are only intended to be used if testing the
    # on-board analysis locally
    parser.add_argument('--test', action='store_true', default=False,
                        help='Set to run a local test of the OBA')

    return parser.parse_args()

def main(args):

    #-----------------------------------------------------------------
    # Initial setup

    config_file = args.config_file
    config_dir = args.config_dir
    root_dir = args.root_dir
    target_name = args.target_name
    overwrite = args.overwrite
    test = args.test
    vb = args.vb

    # test args
    path_config = args.path_config
    test = args.test

    #-----------------------------------------------------------------
    # Logger setup

    if target_name is None:
        p = f'{target_name}_'
    else:
        p = ''

    logfile = f'{p}oba.log'

    logdir = None
    log = utils.setup_logger(logfile, logdir=logdir)
    logprint = utils.LogPrint(log, vb)

    #-----------------------------------------------------------------
    # Testing & config setup

    if config_dir is not None:
        config_file = os.path.join(config_dir, config_file)

    if test is True:
        if path_config is None:
            raise ValueError('Must pass a path_config if --test is used!')

        logprint(f'Using {path_config} to generate test config')

        if root_dir is None:
            test_dir = utils.get_test_dir()
            root_dir = os.path.join(test_dir, 'oba_test')

    else:
        if config_file is None:
            raise ValueError('A config file must be passed if not in ' +
                             'testing mode!')

        logprint(f'Using config file {config_file}')
        if not os.path.exists(config_file):
            raise ValueError(f'OBA pipeline config file not found!')

    #-----------------------------------------------------------------
    # I/O setup (registering filepaths, dirs, etc. for run)

    io_manager = IOManager(root_dir=root_dir)
    io_manager.print_dirs(logprint=logprint)

    #-----------------------------------------------------------------
    # Run pipeline

    runner = OBARunner(
        config_file, logprint
        )

    runner.go()

    # TODO: ...

    logprint('Done!')

    return 0

if __name__ == '__main__':
    args = parse_args()
    rc = main(args)

    if rc == 0:
        print('\nrun_oba.py completed without error\n')
    else:
        print(f'\nrun_oba.py failed with rc={rc}\n')
