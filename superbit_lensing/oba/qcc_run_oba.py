'''
This is the main script to run the SuperBIT onboard analysis (OBA) pipeline
on raw images from a particular target. The main goal is to run a typical
astronomical image calibration & reduction pipeline through object detection
in order to only send down (raw) image *cutouts* of the sources instead of the
full images in the scenario in which there is insufficient bandwidth

Run usage:

In order to satisfy the QCC commander requirement of only one str argument,
you first specify which "mode" you are activating the OBA in:

- mode 0: Pass only a "target_name"
    Use this mode if you have already run prep_oba.py succesfully and want
    to use the default OBA config file produced for that target at
    {root_dir}/home/bit/oba_temp/{target_name}/{target_name}_oba.yaml

- mode 1: Pass the filepath of a OBA config file
    Use this mode if you want to explicitly pass a specific OBA configuration
    file that may or may not have been produced by prep_oba.py

After specifying the mode, simply pass the corresponding argument for the mode:

# Ex1: mode 0 & target_name
python qcc_run_oba.py 0 Abell2813

# Ex2: mode 1 & specific config file
python qcc_run_oba.py 1 /home/bit/configs/my_oba_config.yaml
'''

import shutil
import os
from pathlib import Path
from argparse import ArgumentParser

from config import OBAConfig
from oba_io import IOManager
from oba_runner import OBARunner
from superbit_lensing import utils

import ipdb

def parse_args():

    parser = ArgumentParser()

    parser.add_argument('mode', type=int, choices=[0, 1],
                        help='Choose which mode to run the OBA; 0 will take ' +
                        'only a target name as the second argument and use ' +
                        'the default config produced by prep_oba.py for the ' +
                        'run configuration, while 1 will take a filepath to ' +
                        'an arbitrary yaml configuration file')
    parser.add_argument('mode_arg', type=str,
                        help='See above - If mode is 0, then pass the target ' +
                        'name and use the default config produced by prep_oba.py' +
                        '; if mode is 1, then pass the path to a yaml config file')

    parser.add_argument('-root_dir', type=str, default=None,
                        help='Root directory for OBA run (if testing locally)')
    parser.add_argument('--test', action='store_true', default=False,
                        help='Set to indicate that this is a test run, ' +
                        'which will utilize the TestPrepper')
    parser.add_argument('--fresh', action='store_true', default=False,
                        help='Set to delete any existing files in the root_dir '
                        'before running')
    # NOTE: --vb and --overwrite have been moved to the OBA config!

    return parser.parse_args()

def main(args):

    #-----------------------------------------------------------------
    # Initial setup

    # NOTE: see script docstring & parse_args() for more details on
    # mode configuration
    mode = args.mode
    mode_arg = args.mode_arg

    root_dir = args.root_dir
    test = args.test
    fresh = args.fresh

    if mode == 0:
        # user passed target_name & wants to use the default OBA config
        # produced by prep_oba.py
        target_name = mode_arg
        config_file = None

    elif mode == 1:
        # user passed a filepath to a specific OBA config file they want
        # to use, may or may not be produced by prep_oba.py
        config_file = Path(mode_arg).resolve()
        target_name = None

    else:
        # shouldn't happen
        print(f'Failed: mode {mode} is not allowed - pass 0 for target_name ' +
              'and 1 for config filepath')
        return 1

    # this is useful for testing the OBA when *not* on the QCC
    if root_dir is None:
        if test is True:
            root_dir = Path(utils.get_test_dir()) / 'oba_test/'

    #-----------------------------------------------------------------
    # I/O & config setup (registering filepaths, dirs, etc. for run)

    if mode == 0:
        # in this case we know the target name, so setup the IO manager first
        # and *then* get the config file

        io_manager = IOManager(root_dir=root_dir, target_name=target_name)

        config_file = io_manager.OBA_TARGET / f'{target_name}_oba.yaml'

        if not config_file.is_file():
            print(f'Failed: OBA pipeline config file not found: {config_file}')
            return 2

        config = OBAConfig(config_file)

        # a target name was passed, so make sure it is consistent with
        # the OBA config
        if target_name != config['run_options']['target_name']:
            print(f'Failed: passed target name {target_name} does not match ' +
                  f'the config value of {config_target_name}')

    elif mode == 1:
        # in this case we know the config filepath but not the target, so load
        # the config first and *then* setup the IO manager

        if not config_file.is_file():
            print('Failed: OBA pipeline config file not found: {config_file}')
            return 2

        config = OBAConfig(config_file)
        target_name = config['run_options']['target_name']

        # now we have what we need to setup the IO manager
        io_manager = IOManager(root_dir=root_dir, target_name=target_name)

    #-----------------------------------------------------------------
    # Logger setup

    # NOTE: log will be moved to target output dir eventually
    logdir = io_manager.OBA_DIR
    logfile = str(logdir / f'{target_name}_oba.log')
    utils.make_dir(logdir)

    log = utils.setup_logger(logfile, logdir=logdir)

    # we won't know whether to run in verbose mode until after config
    # parsing; start verbose and reset when we know
    logprint = utils.LogPrint(log, True)

    logprint(f'Log is being saved temporarily at {logfile}')

    if root_dir is None:
        logprint('root_dir is None; using QCC paths')
    else:
        logprint(f'root_dir is {root_dir}')

    io_manager.print_dirs(logprint=logprint)

    #-----------------------------------------------------------------
    # Config parsing & printing

    # these are guaranteed to be present due to config parsing in OBAConfig
    run_options = config['run_options']
    bands = run_options['bands']
    overwrite = run_options['overwrite']
    vb = run_options['vb']

    # now update LogPrint
    logprint.vb = vb

    logprint(f'Using config file {config_file}')
    logprint()
    for outer_key in config:
        print(f'{outer_key}:')
        if isinstance(config[outer_key], dict):
            for inner_key, val in config[outer_key].items():
                print(f'  {inner_key}: {val}')
        elif isinstance(config[outer_key], list):
            for item in config[outer_key]:
                print(f'  -{item}')
        else:
            print(config[outer_key])
    logprint()

    #-----------------------------------------------------------------
    # Cleanup for fresh runs

    # we want it to match the QCC paths, relative to a local root dir
    if (fresh is True) or (config['run_options']['fresh'] is True):
        target_dir = io_manager.OBA_TARGET
        logprint(f'Deleting old OBA outputs for {target_name} as --fresh is True')
        logprint(f'(copying config first to it isnt deleted)')
        logprint(f'rm {target_dir}')
        try:
            tmp_config_dir = config_file.parents[1]
            tmp_config_file = tmp_config_dir / config_file.name
            shutil.move(str(config_file), str(tmp_config_file))
            shutil.rmtree(str(target_dir))
            utils.make_dir(config_file.parent)
            shutil.move(str(tmp_config_file), str(config_file))
        except FileNotFoundError as e:
            logprint(f'OBA dir for target does not exist. Ignoring --fresh flag')

    #-----------------------------------------------------------------
    # Test setup, if needed

    if test is True:
        # handle any needed setup for simulated inputs
        from setup_test import make_test_prepper

        # guaranteed to be set due to default structure of OBAConfig
        test_type = config['test']['type']

        skip_existing = config['test']['skip_existing']
        logprint(f'Skipping compression of test files: {skip_existing}')

        kwargs = {'skip_existing': skip_existing}
        if test_type == 'hen':
            kwargs['run_name'] = str(config['test']['run_name'])
            kwargs['sim_dir'] = Path(config['test']['sim_dir'])

        logprint(
            f'\nTEST == TRUE; Starting test prepper with type {test_type}\n'
            )

        prepper = make_test_prepper(
            test_type,
            target_name,
            bands,
            **kwargs
        )

        prepper.go(io_manager, overwrite=overwrite, logprint=logprint)

    #-----------------------------------------------------------------
    # Run pipeline

    runner = OBARunner(
        config_file,
        io_manager,
        logprint,
        test=test
        )

    runner.go(overwrite=overwrite)

    logprint(f'\nOBA for {target_name} has finished succesfully!')

    #-----------------------------------------------------------------
    # Prepare log for move to final output location

    log_dest = str(io_manager.OBA_RESULTS / target_name)
    logprint(f'\nCopying (compressed) log file to permanent storage at {log_dest}')

    compressed_logfile = compress_log(Path(logfile))

    log_name = Path(logfile).name
    dest = Path(log_dest) / compressed_logfile.name

    if dest.is_file():
        if overwrite is False:
            logprint(f'Warning: {log_name} already exists at {log_dest} and ' +
                     'overwrite is False! Will not update')
        else:
            dest.unlink()

    shutil.move(compressed_logfile, dest)

    return 0

def compress_log(logfile, cmethod='bzip2', cargs='-z', cext='bz2'):
    '''
    Compress the OBA logfile for final storage on QCC permanent storage

    logfile: pathlib.Path
        The filepath of the log to compress
    cmethod: str
        The compression executable
    cargs: str
        The arguments to pass to the desired compression executable
    cext: str
        The file extension of the compressed file
    '''

    cmd = f'{cmethod} {cargs} {logfile}'

    # we don't use utils.run_command() as it would pipe through a log, which
    # we are currently compressing!
    os.system(cmd)

    logfile_ext = logfile.suffix
    compressed_logfile = logfile.with_suffix(logfile_ext + f'.{cext}')

    return compressed_logfile

if __name__ == '__main__':
    args = parse_args()
    rc = main(args)

    if rc == 0:
        print('\nqcc_run_oba.py completed without error\n')
    else:
        print(f'\nqcc_run_oba.py failed with rc={rc}\n')
