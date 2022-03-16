import os
import shutil

from argparse import ArgumentParser

import utils
from pipe import SuperBITPipeline, make_test_config

parser = ArgumentParser()

parser.add_argument('--fresh', action='store_true', default=False,
                    help='Clean test directory of old outputs')
parser.add_argument('--clobber', action='store_true', default=False,
                    help='Set to overwrite files')

def main():

    args = parser.parse_args()
    fresh = args.fresh
    clobber = args.clobber

    testdir = utils.get_test_dir()

    if fresh is True:
        outdir = os.path.join(testdir, 'pipe_test')
        print(f'Deleting old test directory {outdir}...')
        shutil.rmtree(outdir)

    logfile = 'pipe_test.log'
    logdir = os.path.join(testdir, 'pipe_test')
    log = utils.setup_logger(logfile, logdir=logdir)

    config_file = make_test_config(clobber=clobber, outdir=logdir)

    config = utils.read_yaml(config_file)
    vb = config['run_options']['vb']

    if vb:
        print(f'config =\n{config}')

    pipe = SuperBITPipeline(config_file, log=log)

    rc = pipe.run()

    return rc

if __name__ == '__main__':
    rc = main()

    if rc == 0:
        print('\nTests have completed without errors')
    else:
        print(f'\nTests failed with rc={rc}')
