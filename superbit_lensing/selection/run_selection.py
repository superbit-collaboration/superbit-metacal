import os
from argparse import ArgumentParser

from selector import Selector
from superbit_lensing import utils

def parse_args():
    parser = ArgumentParser()

    parser.add_argument('config_file', type=str,
                        help='Config file that defines sample selection')
    parser.add_argument('mcal_file', type=str,
                        help='Filename of the mcal file to make the selection on')
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='Set to overwrite catalogs')
    parser.add_argument('--vb', action='store_true', default=False,
                        help='Make verbose')

    return parser.parse_args()

def main(args):
    config_file = args.config_file
    mcal_file = args.mcal_file
    overwrite = args.overwrite
    vb = args.vb

    outdir = os.path.dirname(mcal_file)

    # setup logger
    logfile = f'selector.log'
    log = utils.setup_logger(logfile, logdir=outdir)
    logprint = utils.LogPrint(log, vb)

    # NOTE: The Selector is setup to do something more complex in the future,
    # but for now we'll only select on a mcal catalog
    catalogs = {
        'metacal': mcal_file
    }

    selector = Selector(config_file, catalogs)

    selector.run(logprint)
    selector.write(overwrite=overwrite)

    return 0

if __name__ == '__main__':
    args = parse_args()
    rc = main(args)

    if rc == 0:
        print('\nselect_sample.py has completed without errors')
    else:
        print(f'\nselect_sample.py failed with rc={rc}')
