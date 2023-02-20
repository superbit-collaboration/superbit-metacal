import os
import yaml
from abc import abstractmethod
from pathlib import Path
from glob import glob
from copy import deepcopy
from argparse import ArgumentParser

from superbit_lensing import utils

import ipdb

# TODO: Refactor: library modules should not have script functionality!
parser = ArgumentParser()

parser.add_argument('run_name', type=str,
               help='Name for given pipe run')
parser.add_argument('basedir', type=str,
               help='Base directory for all run outputs')
parser.add_argument('nfw_dir', type=str,
               help='Base directory for NFW cluster truth files')
parser.add_argument('gs_config', type=str,
               help='Filepath for base galsim mock config file')
parser.add_argument('--config_overwrite', action='store_true',
               help='Set to overwrite config files')
parser.add_argument('--run_overwrite', action='store_true',
               help='Set to overwrite run files')
# parser.add_argument('outfile', type=str,
#                help='Output filepath for config file')

class ModuleConfig(object):
    '''
    A base class that adds some logic for required & optional fields
    in a yaml config file for automated parsing & default setting
    (so no more testing if a given field is in the config!)
    '''


    # should be overwritten by any subclasses for more useful error messages
    _name = None

    # during parsing, will error if any field in the list is not present
    _req_params = []

    # during parsing, will *not* error if any field in the dict is not
    # present, and will use the key:val pair to set the default value
    _opt_params = {}

    # set to True if you want the config parsing to error if any fields not
    # registered below in _req_params or _opt_params are present
    _allow_unregistered = False

    # not all fields handle subparsing well, such as single-value or lists
    _skip_subparsing = []

    def __init__(self, config):
        '''
        config: str, pathlib.Path, dict
            An arbitrary config. Either a filename/path for a yaml file
            or a dictionary
        '''

        utils.check_type('config', config, (dict, str, Path))

        if isinstance(config, (str, Path)):
            if isinstance(config, Path):
                config = str(config)
            self.read_config(config)

        elif isinstance(config, dict):
            self.config_file = None
            self.config = config

        self.parse_config()

        return

    def read_config(self, config_file):
        '''
        config: str
            A yaml config filename
        '''
        self.config_file = config_file
        self.config = utils.read_yaml(config_file)

        return

    def parse_config(self):

        # loop over root fields
        for field in self._req_params:
            req = self._req_params[field]
            try:
                opt = self._opt_params[field]

            except KeyError:
                opt = {}

            self.config[field] = utils.parse_config(
                self.config[field],
                req,
                opt,
                self._name,
                allow_unregistered=self._allow_unregistered
                )

        # check for any root fields that only exist in _opt_params
        for field in self._opt_params:
            if field not in self._req_params:
                req = []
            else:
                req = self._req_params[field]
            opt = self._opt_params[field]

            # not all fields handle subparsing well, such as single-value
            # or list entries
            if field in self._skip_subparsing:
                continue

            try:
                if field in self.config:
                    config = self.config[field]
                else:
                    config = {}

                self.config[field] = utils.parse_config(
                    config,
                    req,
                    opt,
                    f'{self.name}[\'{field}\']',
                    allow_unregistered=self._allow_unregistered
                    )
            except KeyError as e:
                raise KeyError(
                    f'Config parsing failed due to missing key {e}!'
                    )

        return

    def __getitem__(self, key):
        '''
        TODO: For backwards compatibility, it may be nice to search
        through all nested fields if a key is not found
        '''
        return self.config[key]

    def __setitem__(self, key, val):
        self.config[key] = val
        return

    def __delitem__(self, key):
        del self.config[key]
        return

    def __iter__(self):
        return iter(self.config)

    def __repr__(self):
        return str(self.config)

    def copy(self):
        return self.__copy__()

    @abstractmethod
    def __copy__(self):
        # should follow the following template:
        # SubClass(deepcopy(self.config))
        pass

    @property
    def name(self):
        return self._name

    def keys(self):
        return self.config.keys()

    def items(self):
        return self.config.items()

    def values(self):
        return self.config.values()

def make_run_config(run_name, outfile, nfw_file, gs_config,
                    outdir=None, config_overwrite=False, seeds=None,
                    run_overwrite=False, ncores=1, run_diagnostics=True,
                    psf_mode='piff', meds_coadd=True, vb=True):
    '''
    Makes a standard pipe run config given a few inputs.
    Minor changes can easily be made on the output if desired

    run_name: str
        Name for given pipe run
    outfile: str
        Output filepath for config file
    nfw_file: str
        Filepath for cluster nfw file
    gs_config: str
        Filepath for galsim mock config file
    outdir: str
        Output directory for outfile
    config_overwrite: bool
        Set to overwrite config file
    seeds: dict
        A dict of 'seed_name': seed pairs for seeds not
        set in the galsim config
    run_overwrite: bool
        Set to overwrite run files
    ncores: int
        Number of processors to use for pipe job
    run_diagnostics: bool
        Set to True to run module diagnostics
    psf_mode: str
        Choose which PSF modeling mode you wish to use
        Options: ['psfex', 'piff', 'true']
    meds_coadd: bool
        Set to True to add coadd stamp to MEDS file
    vb: bool
        Set to True for verbose printing
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
    # ngmix_test_config = make_test_ngmix_config(
    #     'ngmix_test_config.yaml', outdir=outdir, run_name=run_name
    # )

    needed_seeds = 0
    seed_names = ['psf_seed', 'mcal_seed', 'nfw_seed']
    if seeds is not None:
        for name in seed_names:
            if name not in seeds:
                seeds[name] = None
                needed_seeds += 1

    if needed_seeds > 0:
        gen_seeds = utils.generate_seeds(needed_seeds)

        k = 0
        for name in seed_names:
            if seeds[name] is None:
                seeds[name] = gen_seeds[k]
                k += 1
                assert k == needed_seeds

    config = {
        'run_options': {
            'run_name': run_name,
            'outdir': outdir,
            'vb': vb,
            'ncores': ncores,
            'run_diagnostics': run_diagnostics,
            'order': [
                'galsim',
                'medsmaker',
                'metacal',
                'shear_profile',
                # 'ngmix_fit'
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
            'outdir': outdir,
            'psf_seed': seeds['psf_seed'],
            'psf_mode': psf_mode,
            'meds_coadd': meds_coadd
        },
        'metacal': {
            'meds_file': meds_file,
            'outfile': mcal_file,
            'outdir': outdir,
            'seed': seeds['mcal_seed'],
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
            'nfw_file': nfw_file,
            'nfw_seed': seeds['nfw_seed'],
            'outdir': outdir,
            'run_name': run_name,
            'overwrite': run_overwrite,
        }
    }

    utils.write_yaml(config, outfile)

    return outfile

def make_run_config_from_dict(config_dict):
    '''
    Makes a standard pipe run config given an input dictionary.
    Minor changes can easily be made on the output if desired.

    See make_run_config() for details
    '''

    arg_names = ['run_name', 'outfile', 'nfw_file', 'gs_config']
    args = [None, None, None, None]

    # import pdb; pdb.set_trace()

    for i, key in enumerate(arg_names):
        if key not in config_dict:
            raise KeyError(f'config_dict must include {key}!')
        args[i] = config_dict.pop(key)

    # remaining fields should be optional args
    kwargs = config_dict

    return make_run_config(*args, **kwargs)

def update_run_configs(basedir, pipe_update=None, gs_update=None,
                       run_name=None, pipe_regex=None, gs_regex=None):
    '''
    Helper function to update a series of configs in
    standard pipe config dirs

    basedir: str
        The root location of all cluster dirs for a given run
    pipe_update: dict
        A dictionary of key:val updates to the pipe config
    gs_update: dict
        A dictionary of key:val updates to the galsim config
    pipe_regex: str
        a regular expression to find the pipeline config file
    gs_regex: str
        a regular expression to find the galsim config file
    '''

    if run_name is None:
        p = ''
    else:
        p = f'{run_name}_'

    if pipe_regex is None:
        pipe_regex = f'{p}cl*.yaml'

    if gs_regex is None:
        gs_regex = f'{p}gs*.yaml'

    clusters = glob(os.path.join(basedir, 'cl_*'))

    for cluster in clusters:
        if not os.path.isdir(cluster):
            continue
        reals = glob(os.path.join(cluster, 'r*'))
        for real in reals:
            if not os.path.isdir(real):
                continue

            if pipe_update is not None:
                pipe_file = glob(os.path.join(real, pipe_regex))[0]
                pipe = utils.read_yaml(pipe_file)
                pipe.update(pipe_update)
                utils.write_yaml(pipe, pipe_file)

            if gs_update is not None:
                gs_file = glob(os.path.join(real, gs_regex))[0]
                gs = utils.read_yaml(gs_file)
                gs.update(gs_update)
                utils.write_yaml(gs, gs_file)

    return

def main(args):
    run_name = args.run_name
    outfile = args.outfile
    nfw_file = args.nfw_file
    gs_config = args.gs_config

    kwargs = {
        'outdir': args.outdir,
        'config_overwrite': args.config_overwrite,
        'run_overwrite': args.run_overwrite
    }

    args = [run_name, outfile, nfw_file, gs_config]

    make_run_config(*args, **kwargs)

    return

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
