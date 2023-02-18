from argparse import ArgumentParser
from pathlib import Path

from superbit_lensing import utils
from superbit_lensing.cookiecutter import CookieCutter

import ipdb

parser = ArgumentParser()

parser.add_argument('config_file', type=str,
                    help='The CookieCutter yaml config file to use')
parser.add_argument('--vb', action='store_true', default=False,
                    help='Turn on for verbose printing')

def main(args):

    config_file = args.config_file
    vb = args.vb

    config = utils.read_yaml(config_file)

    logdir = (Path(utils.get_test_dir()) / 'cookie_cutter').resolve()
    logfile = str(logdir / f'cookie-cutter.log')

    log = utils.setup_logger(logfile, logdir=logdir)
    logprint = utils.LogPrint(log, vb)

    logprint(f'Log is being saved at {logfile}')

    cookie = CookieCutter(config=config, logprint=logprint)
    cookie.go()

    return 0

if __name__ == '__main__':
    args = parser.parse_args()
    rc = main(args)

    if rc == 0:
        print('test passed succesfully!')
    else:
        print(f'test failed w/ return code of {rc}')
