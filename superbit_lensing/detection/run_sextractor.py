import os
from argparse import ArgumentParser

from sextractor_runner import SExtractorRunner
from superbit_lensing import utils

import ipdb

def parse_args():

    parser = ArgumentParser()

    parser.add_argument('config_file', type=str,
                        help='The master SE config file including band: ' +
                        'se_config_file pairs')
    parser.add_argument('run_name', action='store', type=str, default=None,
                        help='Name of mock simulation run')
    parser.add_argument('basedir', type=str,
                        help='Directory containing imaging data')
    parser.add_argument('-bands', type=str, default=None,
                        help='List of band names separated by commas ' +
                        '(no space). Defaults to bands defined in config_file')
    parser.add_argument('-config_dir', type=str, default=None,
                        help='Directory of SExtractor config file')
    parser.add_argument('-outfile_base', type=str,
                        help='Base name for the output coadd files')
    parser.add_argument('-outdir', type=str, default=None,
                        help='Output directory for MEDS file')
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='Set to overwrite files')
    parser.add_argument('--vb', action='store_true', default=False,
                        help='Verbosity')

    return parser.parse_args()

def main(args):
    # TODO: ...
    config_file = args.config_file
    basedir = args.basedir
    run_name = args.run_name
    bands = args.bands
    config_dir = args.config_dir
    outfile_base = args.outfile_base
    outdir = args.outdir
    vb = args.vb

    # convert bands into a list of strings
    if bands is not None:
        bands = bands.split(',')

    #-----------------------------------------------------------------
    # Initial setup

    if outdir is None:
        outdir = os.getcwd()

    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    logdir = outdir
    logfile = 'sextractor_runner.log'
    log = utils.setup_logger(logfile, logdir=logdir)
    logprint = utils.LogPrint(log, vb)

    if config_dir is not None:
        config_file = os.path.join(config_dir, config_file)

    logprint(f'Using master config file {config_file}')
    if not os.path.exists(config_file):
        raise ValueError(f'SExtractor master config file not found!')

    #-----------------------------------------------------------------
    # Create & setup SExtractor runner

    runner = SExtractorRunner(
        config_file,
        run_name,
        basedir,
        bands,
        )

    # Do the source extracting!
    runner.go(
        outfile_base=outfile_base,
        outdir=outdir,
        )

    logprint('Done!')

    return 0

if __name__ == '__main__':
    args = parse_args()
    rc = main(args)

    if rc == 0:
        print('\nrun_sexractor.py completed without error\n')
    else:
        print(f'\nrun_sextractor.py failed with rc={rc}\n')
