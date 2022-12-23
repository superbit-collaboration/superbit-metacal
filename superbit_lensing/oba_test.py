'''
Classes & functions useful for running a local test of the OBA pipeline
on simulated data

Setup similarly to a "grid test" or "pipe test", but with narrower scope.
Only runs the imsim module to generate a simulated dataset to test if the
on-board analysis (OBA) module works as intended.

See pipe_test.py, grid_test.py, & the README for instructions on how to
run such a test (just change "pipe/grid" -> "oba")
'''

import os
from pathlib import Path
from argparse import ArgumentParser

from superbit_lensing import utils

import ipdb

def parse_args():
    parser = ArgumentParser()

    parser.add_argument('-pipe_config', type=str, default=None,
                        help='A pipeline config to use for the oba test, if ' +
                        'youd rather specify everything yourself')

    # NOTE: Can either pass a path_config file to specify the minimal needed
    # path information to generate an imsim config, or pass a config explicitly
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-path_config', type=str, default=None,
                        help='A yaml config file that defines the paths ' +
                        'needed to run an oba test')
    group.add_argument('-gs_config', type=str, default=None,
                        help='An imsim module config to use for the oba ' +
                        'test, if youd rather specify everything yourself')

    parser.add_argument('--fresh', action='store_true', default=False,
                        help='Clean test directory of old outputs')

    return parser.parse_args()

# The following helper functions create dummy config files for their
# corresponding categories. You can always provide your own if you prefer

def make_test_pipe_config(gs_config, outdir=None, outfile=None,
                          overwrite=False, ncores=8, vb=True):
    '''
    gs_config: str
        The filename of the galsim config file to use
    outfile: str
        The output filename of the generated pipe config
    imsim: str
        The name of the image simulation module to use
    overwrite: bool
        Set to True to overwrite existing config file
    outdir: str, pathlib.Path
        The root directory of the OBA test
    ncores: int
        The number of cores to use for the test run
    vb: bool
        Set for verbose printing
    '''

    if outdir is None:
        # default is to use the repo test dir
        outdir = Path(utils.get_test_dir()) / 'oba_test/'

    utils.make_dir(outdir)

    if outfile is None:
        outfile = outdir / 'oba_test.yaml'

    config = _make_test_pipe_config(
        gs_config, outfile, outdir, overwrite=overwrite, ncores=ncores, vb=vb
        )

    utils.write_yaml(config, outfile)

    return outfile

def _make_test_pipe_config(gs_config, outfile, outdir, overwrite=False,
                           ncores=8, vb=True):

    if (overwrite is True) or (not os.path.exists(outfile)):
        run_name = 'test_target'
        bands = 'b,lum' # test at least 2 bands
        det_bands = 'b,lum'

        config_dir = Path(utils.MODULE_DIR) / 'oba/configs/'
        gs_config = (outdir / gs_config).resolve()
        swarp_config = (config_dir / 'swarp.config').resolve()
        se_config = (config_dir / 'se_configs.yaml').resolve()

        det_cat_file = (outdir / f'{run_name}_coadd_det_cat.fits').resolve()
        meds_file = (outdir / f'{run_name}_meds.fits').resolve()

        oba_config = {
            'run_options': {
                'run_name': run_name,
                'outdir': str(outdir),
                'bands': bands,
                'vb': vb,
                'ncores': ncores,
                'run_diagnostics': True,
                'order': [
                    'imsim',
                    'oba',
                    ]
                },
            f'imsim': {
                'config_file': str(gs_config),
                'run_name': run_name,
                'outdir': str(outdir),
                'overwrite': overwrite
            },
            'oba': {
                # TODO: ...
            }
        }

    return oba_config

def make_test_gs_config(path_config, outdir, outfile='oba_test_gs.yaml',
                        overwrite=False):

    if outdir is not None:
        outfile = str(outdir / outfile)

    if (overwrite is True) or (not os.path.exists(outfile)):

        # use REPO/configs/oba_test_gs.yaml as base

        test_config = utils.read_yaml(
            str(
                (Path(utils.BASE_DIR) /
                 'configs' /
                 'grid_test_gs.yaml').resolve()
                )
            )

        # update with local filepaths
        root_fields = {
            'datadir': 'input',
            'cosmosdir': 'galaxies',
            'cluster_dir': 'cluster_galaxies',
            'cluster_cat_name': 'cluster_galaxies',
            'gaia_dir': 'stars',
        }
        for name, path in path_config.items():
            root = root_fields[name]
            try:
                test_config[root][name] = path
            except KeyError:
                # might not be used
                pass

        test_config['output']['outdir'] = str(outdir)

        utils.write_yaml(test_config, outfile)

    return outfile

def main(args):

    pipe_config_file = args.pipe_config
    path_config_file = args.path_config
    gs_config_file = args.gs_config
    fresh = args.fresh

    argfiles = {
        'pipe_config_file': args.pipe_config,
        'path_config_file': args.path_config,
        'gs_config_file': args.gs_config,
    }
    for name, fname in argfiles.items():
        if fname is not None:
            argfiles[name] = Path(fname).resolve()

    test_dir = Path(utils.get_test_dir()).resolve()
    outdir = test_dir / 'oba_test/'

    if fresh is True:
        print(f'Deleting old test directory {str(outdir)}...')
        try:
            shutil.rmtree(str(outdir))
        except FileNotFoundError as e:
            print('Test directory does not exist. Ignoring --fresh flag')

    # only makes it if it does not currently exist
    utils.make_dir(str(outdir))

    # need to parse local paths before creating galsim config, unless
    # passed explicitly. Argparse makes sure only one is passed
    path_config_file = argfiles['path_config_file']
    if path_config_file is not None:
        print(f'Reading path config file {path_config_file}')
        path_config = utils.read_yaml(path_config_file)
    else:
        path_config = None

    # parse galsim config file first, as it is needed for pipe config
    gs_config_file = argfiles['gs_config_file']
    if gs_config_file is None:
        print('Creating test gs config file...')
        gs_config_file = make_test_gs_config(
            path_config, overwrite=True, outdir=outdir,
            )
        print(f'Generated imsim config file {gs_config_file}')

    # now we have everything we need to create a pipeline config
    pipe_config_file = argfiles['pipe_config_file']
    if pipe_config_file is None:
        # generate a fast config
        print('Creating test pipeline config file...')
        pipe_config_file = make_test_pipe_config(
            gs_config_file, overwrite=True, outdir=outdir
            )
        print(f'Generated pipe config file {pipe_config_file}')

    print('Done!')

    return 0

if __name__ == '__main__':
    args = parse_args()
    rc = main(args)

    if rc == 0:
        print('\nsetup_test.py have completed without errors')
    else:
        print(f'\nsetup_test.py failed with rc={rc}')
