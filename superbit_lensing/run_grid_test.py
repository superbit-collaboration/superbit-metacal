from argparse import ArgumentParser

import utils
from pipe import SuperBITPipeline, make_test_config

import ipdb

'''
Run a simple simulate suite with all sources on a hexagonal grid. Cluster
galaxies are not added, but lensing can be done by either a constant shear
field or by the usual NFW halo
'''

parser = ArgumentParser()

parser.add_argument('--fresh', action='store_true', default=False,
                    help='Clean test directory of old outputs')

def main():

    args = parser.parse_args()
    fresh = args.fresh

    testdir = utils.get_test_dir()

    if fresh is True:
        outdir = os.path.join(testdir, 'grid_test')
        print(f'Deleting old test directory {outdir}...')
        shutil.rmtree(outdir)

    logfile = 'grid_test.log'
    logdir = os.path.join(testdir, 'grid_test')
    log = utils.setup_logger(logfile, logdir=logdir)

    config_file = make_test_config(overwrite=True, outdir=logdir)

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
        print('\nScript completed without errors')
    else:
        print(f'\nScript failed with rc={rc}')

