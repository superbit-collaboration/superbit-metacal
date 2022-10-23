import os
import shutil

from argparse import ArgumentParser

import utils
from pipe import SuperBITPipeline

import ipdb

'''
A "pipe test" is used to validate the current state of the pipeline code,
not correctness. As such it optimizes configuration choices for speed and
not accuracy. Please run a full end-to-end pipe test before commiting code
to the repository.

The simplest way to run a pipe test is to make a copy of

{REPO_DIR}/configs/path_config.yaml

with your local filepaths added where indicated. The patterns suggested
may be different depending on your env & installation setup.

Then simply run the following:

python pipe_test.py -path_config={LOCAL_PATH_CONFIG} --fresh

where --fresh tells the test to reset the pipe_test output directory.

Alternatively, you can specify your own full pipeline configuration & galsim
configuration files in correct format for more control over the pipe test.
You would then run

python pipe_test.py -pipe_config={PIPE_CONFIG} -gs_config={GS_CONFIG} --fresh

For convenience, I add this line to a short shell script and simply call
`source ptest.sh` when I want to run a pipe test.

If the test fails on the n-th module, you may want to edit `run_options['order']`
in the saved pipe config in {TEST_DIR}/pipe_test/ so that you don't unnecessarily
rerun the previous n-1 modules. In this case you would now run

python pipe_test.py -pipe_config={TEST_DIR/pipe_test/pipe_test.yaml}
                    -gs_config={TEST_DIR/pipe_test/pipe_test_gs.yaml}

where it is now important to not use the --fresh flag.

For downloading the required GalSim COSMOS catalogs, see:
https://galsim-developers.github.io/GalSim/_build/html/real_gal.html#downloading-the-cosmos-catalog
'''

def parse_args():
    parser = ArgumentParser()

    parser.add_argument('-pipe_config', type=str, default=None,
                        help='A pipeline config to use for the pipe test, if ' +
                        'youd rather specify everything yourself')

    # NOTE: Can either pass a path_config file to specify the minimal needed
    # path information to generate galsim config, or pass a gs config explicitly
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-path_config', type=str, default=None,
                        help='A yaml config file that defines the paths ' +
                        'needed to run a pipe test')
    group.add_argument('-gs_config', type=str, default=None,
                        help='A galsim module config to use for the pipe ' +
                        'test, if youd rather specify everything yourself')

    parser.add_argument('--fresh', action='store_true', default=False,
                        help='Clean test directory of old outputs')

    # NOTE: As ngmix-fit needs some refactoring, this is currently unused
    parser.add_argument('-ngmix_config', type=str, default=None,
                        help='A ngmix-fit module config to use for the pipe ' +
                        'test, if youd rather specify everything yourself')

    return parser.parse_args()

# The following helper functions create dummy config files for their corresponding
# categories. You can always provide your own if you prefer.

def make_test_pipe_config(gs_config_file, outfile='pipe_test.yaml',
                          imsim='galsim', outdir=None, overwrite=False):
    '''
    Create a basic yaml config file that tests whether the pipeline
    succeeds in running end-to-end. Prioritizes speed over scientific
    value

    gs_config_file: str
        The filename of the galsim config file to use
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
        run_name = 'pipe_test'
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
            raise OSError(f'Warning: pipe test nfw file {nfw_file} does not ' +
                          'exist; have you unpacked the sample tar file?')

        test_config = {
            'run_options': {
                'run_name': run_name,
                'outdir': outdir,
                'vb': True,
                'ncores': 8,
                'run_diagnostics': True,
                'order': [
                    f'{imsim}',
                    'medsmaker',
                    'metacal',
                    # 'metacal_v2', # turn on for ngmix v2.X metacal
                    'shear_profile',
                    # 'ngmix_fit', # turn on for ngmix photometry (needs updating)
                    ]
                },
            f'{imsim}': {
                'config_file': 'pipe_test_gs.yaml',
                'config_dir': os.path.join(utils.TEST_DIR, 'pipe_test'),
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
            'ngmix_fit': {
                'meds_file': meds_file,
                'outfile': f'{run_name}_ngmix.fits',
                'config': ngmix_test_config,
                'outdir': outdir,
                'end': 100
            },
            'shear_profile': {
                'se_file': se_file,
                'mcal_file': mcal_file,
                'outfile': f'{run_name}_annular.fits',
                'nfw_file': nfw_file,
                'outdir': outdir,
                'run_name': run_name,
                'Nresample': 1, # to run much faster
                'overwrite': overwrite,
            },
        }

        utils.write_yaml(test_config, filename)

    return filename

def make_test_gs_config(path_config, outfile='pipe_test_gs.yaml', outdir=None,
                        overwrite=False):

    if outdir is not None:
        filename = os.path.join(outdir, outfile)

    if (overwrite is True) or (not os.path.exists(filename)):

        # use REPO/configs/pipe_test_gs.yaml as base

        test_config = utils.read_yaml(
            os.path.join(utils.BASE_DIR, 'configs', 'pipe_test_gs.yaml')
            )

        # update with local filepaths
        for name, path in path_config.items():
            test_config[name] = path

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
        run_name = 'pipe_test'

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

def main(args):

    pipe_config_file = args.pipe_config
    path_config_file = args.path_config
    gs_config_file = args.gs_config
    fresh = args.fresh

    # NOTE: not currently used
    ngmix_config_file = args.ngmix_config

    testdir = utils.get_test_dir()

    if fresh is True:
        outdir = os.path.join(testdir, 'pipe_test')
        print(f'Deleting old test directory {outdir}...')
        shutil.rmtree(outdir)

    logfile = 'pipe_test.log'
    logdir = os.path.join(testdir, 'pipe_test')
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
            path_config, overwrite=True, outdir=logdir
            )
        print(f'Using gs_config_file {gs_config_file}')

    # now we have everything we need to create a pipeline config
    if pipe_config_file is None:
        # generate a fast config
        print('Creating test pipeline config file...')
        pipe_config_file = make_test_pipe_config(
            gs_config_file, overwrite=True, outdir=logdir
            )
        print(f'Using pipe_config_file {pipe_config_file}')

    # we saved it to a file instead of returning a dict so that there is
    # a record in the pipe_test outdir
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
