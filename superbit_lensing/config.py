import os
import yaml
from argparse import ArgumentParser

import pipe
import utils

parser = ArgumentParser()

parser.add_argument('run_name', type=str,
               help='Name for given pipe run')
parser.add_argument('outfile', type=str,
               help='Output filepath for config file')
parser.add_argument('truth_file', type=str,
               help='Filepath for cluster truth file')
parser.add_argument('gs_config', type=str,
               help='Filepath for galsim mock config file')
parser.add_argument('-outdir', type=str, default=None,
               help='Output directory for outfile')
parser.add_argument('--config_overwrite', action='store_true',
               help='Set to overwrite config file')
parser.add_argument('--run_overwrite', action='store_true',
               help='Set to overwrite run files')

def make_run_config(run_name, outfile, truth_file, gs_config,
                    outdir=None, config_overwrite=False, run_overwrite=False):
    '''
    Makes a standard pipe run config given a few inputs.
    Minor changes can easily be made on the output if desired

    run_name: str
        Name for given pipe run
    outfile: str
        Output filepath for config file
    truth_file: str
        Filepath for cluster truth file
    gs_config: str
        Filepath for galsim mock config file
    outdir: str
        Output directory for outfile
    config_overwrite: bool
        Set to overwrite config file
    run_overwrite: bool
        Set to overwrite run files
    '''

    if outdir is not None:
        utils.make_dir(outdir)
        outfile = os.path.join(outdir, outfile)
    else:
        outdir = ''

    if os.path.exists(outfile):
        if config_overwrite is False:
            raise Exception(f'config_file {outfile} already exists! ' +\
                            'Use config_overwrite if you want to overwrite')

    se_file = os.path.join(outdir, f'{run_name}_mock_coadd_cat.ldac')
    meds_file = os.path.join(outdir, f'{run_name}_meds.fits')
    mcal_file = os.path.join(outdir, f'{run_name}_mcal.fits')
    ngmix_test_config = pipe.make_test_ngmix_config(
        'ngmix_test_config.yaml', outdir=outdir, run_name=run_name
        )

    with open(outfile, 'w') as f:
        config = {
            'run_options': {
                'run_name': run_name,
                'outdir': outdir,
                'vb': True,
                'ncores': 8,
                'run_diagnostics': True,
                'order': [
                    'galsim',
                    'medsmaker',
                    'metacal',
                    'shear_profile',
                    'ngmix_fit'
                    ]
                },
            'galsim': {
                'config_file': gs_config,
                'config_dir': os.path.join(utils.MODULE_DIR,
                                            'galsim',
                                            'config_files'),
                'outdir': outdir,
                'clobber': run_overwrite
            },
            'medsmaker': {
                'mock_dir': outdir,
                'outfile': meds_file,
                'fname_base': run_name,
                'run_name': run_name,
                'outdir': outdir
            },
            'metacal': {
                'meds_file': meds_file,
                'outfile': mcal_file,
                'outdir': outdir,
            },
            'ngmix_fit': {
                'meds_file': meds_file,
                'outfile': f'{run_name}_ngmix.fits',
                'config': ngmix_test_config,
                'outdir': outdir,
            },
            'shear_profile': {
                'se_file': se_file,
                'mcal_file': mcal_file,
                'outfile': f'{run_name}_annular.fits',
                'nfw_file': truth_file,
                'outdir': outdir,
                'run_name': run_name,
                'overwrite': run_overwrite,
            }
        }

        yaml.dump(config, f, default_flow_style=False)

    return outfile

def make_run_config_from_dict(config_dict):
    '''
    Makes a standard pipe run config given an input dictionary.
    Minor changes can easily be made on the output if desired.

    See make_run_config() for details
    '''

    args = {
        'run_name': None,
        'outfile': None,
        'truth_file': None,
        'gs_config': None
    }

    for key in args.keys():
        if key not in config_dict:
            raise KeyError(f'config_dict must include {key}!')
        args[key] = config_dict.pop(key)

    # remaining fields should be optional args
    kwargs = config_dict

    return make_run_config(*args, **kwargs)

def main(args):
    run_name = args.run_name
    outfile = args.outfile
    truth_file = args.truth_file
    gs_config = args.gs_config

    kwargs = {
        'outdir': args.outdir,
        'config_overwrite': args.config_overwrite,
        'run_overwrite': args.run_overwrite
    }

    args = [run_name, outfile, truth_file, gs_config]

    make_run_config(*args, **kwargs)

    return

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
