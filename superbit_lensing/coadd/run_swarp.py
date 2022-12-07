import os
from argparse import ArgumentParser

from swarp_runner import SWarpRunner
from superbit_lensing import utils

import ipdb

def parse_args():

    parser = ArgumentParser()

    parser.add_argument('config_file', type=str,
                        help='Filename of the SWarp config')
    parser.add_argument('run_name', action='store', type=str, default=None,
                        help='Name of mock simulation run')
    parser.add_argument('basedir', type=str,
                        help='Directory containing imaging data')
    parser.add_argument('bands', type=str,
                        help='List of band names separated by commas (no space)')
    parser.add_argument('-config_dir', type=str, default=None,
                        help='Directory of SWarp config file')
    parser.add_argument('-det_bands', type=str, default=None,
                        help='List of band names separated by commas ' +
                        '(no space) to use for the detection image')
    parser.add_argument('-outfile_base', type=str,
                        help='Base name for the output coadd files')
    parser.add_argument('-outdir', type=str, default=None,
                        help='Output directory for MEDS file')
    parser.add_argument('-fname_base', action='store', type=str, default=None,
                        help='Basename of image files')
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
    det_bands = args.det_bands
    outfile_base = args.outfile_base
    outdir = args.outdir
    fname_base = args.fname_base
    overwrite = args.overwrite
    vb = args.vb

    # default is to use all bands for detection image
    if det_bands is None:
        det_bands = bands

    # convert bands into a list of strings
    bands = bands.split(',')
    det_bands = det_bands.split(',')

    #-----------------------------------------------------------------
    # Initial setup

    if outdir is None:
        outdir = os.getcwd()

    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    logdir = outdir
    logfile = 'swarp_runner.log'
    log = utils.setup_logger(logfile, logdir=logdir)
    logprint = utils.LogPrint(log, vb)

    if config_dir is not None:
        config_file = os.path.join(config_dir, config_file)

    logprint(f'Using config file {config_file}')
    if not os.path.exists(config_file):
        raise ValueError(f'SWarp config file not found!')

    #-----------------------------------------------------------------
    # Create & setup SWarp runner

    runner = SWarpRunner(
        config_file,
        run_name,
        basedir,
        bands,
        det_bands=det_bands,
        fname_base=fname_base,
        )

    # Do the SWarping!
    runner.go(
        outfile_base=outfile_base, outdir=outdir, overwrite=overwrite
        )

    logprint('Done!')

    return 0

if __name__ == '__main__':
    args = parse_args()
    rc = main(args)

    if rc == 0:
        print('\nrun_swarp completed without error\n')
    else:
        print(f'\nrun_swarp failed with rc={rc}\n')
