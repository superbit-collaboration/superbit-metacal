import os
import shutil

from argparse import ArgumentParser

import utils
from pipe import SuperBITPipeline

import ipdb

'''
A "grid test" is like a "pipe test", but instead of validating the current
state of the pipeline code, it generates a simplistic set of images to test
the shear calibration in a simple setting. See grid_test.py & the README for
instructions on how to run such a test (just change "pipe" -> "grid")
'''

def parse_args():
    parser = ArgumentParser()

    parser.add_argument('-pipe_config', type=str, default=None,
                        help='A pipeline config to use for the grid test, if ' +
                        'youd rather specify everything yourself')
    parser.add_argument('-selection_config', type=str, default=None,
                        help='A selection config to use for the grid test, if ' +
                        'youd rather specify everything yourself')

    # NOTE: Can either pass a path_config file to specify the minimal needed
    # path information to generate galsim config, or pass a gs config explicitly
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-path_config', type=str, default=None,
                        help='A yaml config file that defines the paths ' +
                        'needed to run a grid test')
    group.add_argument('-gs_config', type=str, default=None,
                        help='A galsim module config to use for the grid' +
                        'test, if youd rather specify everything yourself')

    parser.add_argument('--fresh', action='store_true', default=False,
                        help='Clean test directory of old outputs')

    # NOTE: As ngmix-fit needs some refactoring, this is currently unused
    parser.add_argument('-ngmix_config', type=str, default=None,
                        help='A ngmix-fit module config to use for the grid' +
                        'test, if youd rather specify everything yourself')

    return parser.parse_args()

# The following helper functions create dummy config files for their corresponding
# categories. You can always provide your own if you prefer.

def make_test_pipe_config(gs_config, select_config, outfile='grid_test.yaml',
                          imsim='imsim', outdir=None, overwrite=False):
    '''
    Create a basic yaml config file that tests whether the shear calibration
    succeeds in a simplified setting (low noise, objects on a grid, etc.)

    gs_config: str
        The filename of the galsim config file to use
    select_config: str
        The filename of the selection config file to use
    outfile: str
        The output filename of the generated pipe config
    outdir: str
        The directory to save the config file to
    imsim: str
        The name of the image simulation module to use
    overwrite: bool
        Set to True to overwrite existing config file
    '''

    if outdir is not None:
        filename = os.path.join(outdir, outfile)

    if (overwrite is True) or (not os.path.exists(filename)):
        run_name = 'grid_test'
        outdir = os.path.join(utils.TEST_DIR, run_name)
        se_file = os.path.join(outdir, f'{run_name}_mock_coadd_cat.ldac')
        meds_file = os.path.join(outdir, f'{run_name}_meds.fits')
        mcal_file = os.path.join(outdir, f'{run_name}_mcal.fits')
        ngmix_test_config = make_test_ngmix_config('ngmix_test.yaml',
                                                   outdir=outdir,
                                                   run_name=run_name)

        nfw_file = os.path.join(
            utils.MODULE_DIR, 'shear_profiles/truth/nfw_cl_m7.8e14_z0.25.fits'
            )

        if not os.path.exists(nfw_file):
            raise OSError(f'Warning: grid test nfw file {nfw_file} does not ' +
                          'exist; have you unpacked the sample tar file?')

        test_config = {
            'run_options': {
                'run_name': run_name,
                'outdir': outdir,
                'vb': True,
                'ncores': 8,
                'run_diagnostics': True,
                'order': [
                    # f'{imsim}',
                    # 'medsmaker',
                    # 'metacal',
                    'selection',
                    # 'metacal_v2', # turn on for ngmix v2.X metacal
                    # 'shear_profile',
                    # 'ngmix_fit', # turn on for ngmix photometry (needs updating)
                    ]
                },
            f'{imsim}': {
                'config_file': os.path.join(
                    utils.TEST_DIR, run_name, 'grid_test_gs.yaml'
                    ),
                'run_name': run_name,
                'outdir': outdir,
                'overwrite': overwrite
            },
            'medsmaker': {
                'mock_dir': outdir,
                'outfile': meds_file,
                'fname_base': run_name,
                'run_name': run_name,
                'outdir': outdir,
                'overwrite': overwrite,
                'meds_coadd': True,
                'psf_mode': 'piff'
            },
            'metacal': {
            # 'metacal_v2': {
                'meds_file': meds_file,
                'outfile': mcal_file,
                'outdir': outdir,
                'end': 2000,
                'overwrite': overwrite
            },
            'selection': {
                'config_file': select_config,
                'mcal_file': mcal_file,
                'overwrite': overwrite,
            },
            # 'ngmix_fit': {
            #     'meds_file': meds_file,
            #     'outfile': f'{run_name}_ngmix.fits',
            #     'config': ngmix_test_config,
            #     'outdir': outdir,
            #     'end': 100
            # },
            # 'shear_profile': {
            #     'se_file': se_file,
            #     'mcal_file': mcal_file,
            #     'outfile': f'{run_name}_annular.fits',
            #     'nfw_file': nfw_file,
            #     'outdir': outdir,
            #     'run_name': run_name,
            #     'Nresample': 1, # to run much faster
            #     'overwrite': overwrite,
            # },
        }

        utils.write_yaml(test_config, filename)

    return filename

def make_test_gs_config(path_config, outfile='grid_test_gs.yaml', outdir=None,
                        overwrite=False):

    if outdir is not None:
        filename = os.path.join(outdir, outfile)

    if (overwrite is True) or (not os.path.exists(filename)):

        # use REPO/configs/grid_test_gs.yaml as base

        test_config = utils.read_yaml(
            os.path.join(utils.BASE_DIR, 'configs', 'grid_test_gs.yaml')
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

        test_config['output']['outdir'] = outdir

        utils.write_yaml(test_config, filename)

    return filename

def make_test_ngmix_config(config_file='ngmix_test.yaml', outdir=None,
                           run_name=None, overwrite=False):
    '''
    NOTE: Not currently used, but could be incorporated to run the
    ngmix-fit module
    '''

    if outdir is not None:
        filename = os.path.join(outdir, config_file)

    if run_name is None:
        run_name = 'grid_test'

    if (overwrite is True) or (not os.path.exists(filename)):
        test_config = {
            'gal': {
                'model': 'bdf',
            },
            'psf': {
                'model': 'gauss'
            },
            'priors': {
                'T_range': [-1., 1.e3],
                'F_range': [-100., 1.e9],
                'g_sigma': 0.1,
                'fracdev_mean': 0.5,
                'fracdev_sigma': 0.1
            },
            'fit_pars': {
                'method': 'lm',
                'lm_pars': {
                    'maxfev':2000,
                    'xtol':5.0e-5,
                    'ftol':5.0e-5
                    }
            },
            'pixel_scale': 0.144, # arcsec / pixel
            'nbands': 1,
            'seed': 172396,
            'run_name': run_name
        }

        utils.write_yaml(test_config, filename)

    return filename

def make_test_selection_config(outfile='grid_test_select.yaml',
                               outdir=None, overwrite=False):

    if outdir is not None:
        filename = os.path.join(outdir, outfile)

    if (overwrite is True) or (not os.path.exists(filename)):

        # use REPO/configs/select.yaml as base

        test_config = utils.read_yaml(
            os.path.join(utils.BASE_DIR, 'configs', 'select.yaml')
            )

        # for now, we just use this as-is
        utils.write_yaml(test_config, filename)

    return filename

def main(args):

    pipe_config_file = args.pipe_config
    path_config_file = args.path_config
    selection_config_file = args.selection_config
    gs_config_file = args.gs_config
    fresh = args.fresh

    # NOTE: not currently used
    ngmix_config_file = args.ngmix_config

    testdir = utils.get_test_dir()

    if fresh is True:
        outdir = os.path.join(testdir, 'grid_test')
        print(f'Deleting old test directory {outdir}...')
        try:
            shutil.rmtree(outdir)
        except FileNotFoundError as e:
            print('Test directory does not exist. Ignoring --fresh flag')

    logfile = 'grid_test.log'
    logdir = os.path.join(testdir, 'grid_test')
    log = utils.setup_logger(logfile, logdir=logdir)

    # need to parse local paths before creating galsim config, unless
    # passed explicitly. Argparse makes sure only one is passed
    if path_config_file is not None:
        print(f'Reading path config file {path_config_file}...')
        path_config = utils.read_yaml(path_config_file)
    else:
        path_config = None

    # parse galsim config file first, as it is needed for pipe config
    if gs_config_file is None:
        print('Creating test gs config file...')
        gs_config_file = make_test_gs_config(
            path_config, overwrite=True, outdir=logdir,
            )
        print(f'Using gs config file {gs_config_file}')

    # same thing with a selection config
    if selection_config_file is None:
        selection_config_file = make_test_selection_config(
            overwrite=True, outdir=logdir
            )
        print(f'Using selection config file {selection_config_file}')

    # now we have everything we need to create a pipeline config
    if pipe_config_file is None:
        # generate a fast config
        print('Creating test pipeline config file...')
        pipe_config_file = make_test_pipe_config(
            gs_config_file, selection_config_file, overwrite=True, outdir=logdir
            )
        print(f'Using pipe config file {pipe_config_file}')

    # we saved it to a file instead of returning a dict so that there is
    # a record in the grid_test outdir
    pipe_config = utils.read_yaml(pipe_config_file)

    vb = pipe_config['run_options']['vb']

    if vb:
        print(f'config =\n{pipe_config}')

    pipe = SuperBITPipeline(pipe_config_file, log=log)

    rc = pipe.run()

    return rc

if __name__ == '__main__':
    args = parse_args()
    rc = main(args)

    if rc == 0:
        print('\nTests have completed without errors')
    else:
        print(f'\nTests failed with rc={rc}')
