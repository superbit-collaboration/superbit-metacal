import galsim

from superbit_lensing import utils
from copy import deepcopy

import ipdb

class ImSimConfig(object):
    # all parameter names identical to those used in standard
    # SuperBIT image sims
    # TODO: Would be nice to rename a few of these for clarity
    # TODO: Restructure for parent fields

    # these are top-level config fields
    _req_params = {
        'telescope': [
            'diameter',
            'nstruts',
            'strut_thick',
            'strut_theta',
            'obscuration'
            ],

        'bandpasses': [
            # NOTE: each entry needs a name and central wavelength,
            # like the following:
            # - b1_name:
            #   lam: 475
            # - b2_name:
            #   lam: 650
            # ...
            ],

        'detector': [
            'gain'
            ],

        'noise': [
            'read_noise',
            'dark_current',
            'sky_bkg'
            ],

        'psf': [
            'jitter_fwhm'
            ],

        'image': [
            'image_xsize',
            'image_ysize',
            'pixel_scale'
            ],

        'cluster': [
            'type',
            'mass',
            'nfw_conc',
            'nfw_z_halo',
            'center_ra', # NOTE: units set in opt param
            'center_dec', # NOTE: units set in opt param
            ],

        'observation': [
            'nexp',
            'exp_time'
        ],

        'cosmology': [
            'omega_m',
            'omega_lam',
            'h'
        ],

        'input': [
            'cosmosdir',
            'datadir',
            'cat_file_name',
            'cluster_cat_name',
            'cluster_cat_dir'
        ]
    }

    # defaults are assigned
    _opt_params = {
        'run_options': {
            'run_name': None,
            'mpi': False,
            'ncores': 1,
            'overwrite': False,
            'vb': False
            },

        'psf': {
            'use_optics': True
        },

        'output': {
            'outdir': None,
            'format': 'exp_num'
        },

        'stars': {
            # plus any fields required by the galaxy class type
            'type': 'default',
            'Nobjs': None,
            'gaia_dir': None,
            'cat_file': None
            # plus any fields required by the galaxy class type
        },

        'galaxies': {
            'type': 'cosmos',
            'Nobjs': None,
            # plus any fields required by the galaxy class type
        },

        'cluster': {
            # NOTE: We follow the conventions of the old imsim module
            'center_ra_unit': galsim.hours,
            'center_dec_unit': galsim.degrees,
        },

        'cluster_galaxies': {
            'type': 'default',
            'Nobjs': None
            # plus any fields required by the galaxy class type
        },

        'position_sampling': {
            'galaxies': {'type': 'random'},
            'stars': {'type': 'random'},
            'cluster': {'type': 'random'},
        },

        'shear': {
            'type': 'nfw', # default cluster lensing by a NFW halo
            'g1': None,
            'g2': None,
            'e1': None,
            'e2': None
        },

        'seeds': {
            'master': None,
            'noise': None,
            'dithering': None,
            'cluster_galaxies': None,
            'stars': None,
            'galaxies': None
        }
    }

    def __init__(self, config):
        '''
        config: str, dict
            An image simulation config. Either a filename for a yaml file
            or a dictionary
        '''

        if isinstance(config, str):
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
                self.config[field], req, opt, 'ImSim',
                allow_unregistered=True
                )

        # check for any root fields that only exist in _opt_params
        for field in self._opt_params:
            if field not in self._req_params:
                req = []
            else:
                req = self._req_params[field]
            opt = self._opt_params[field]

            try:
                config = self.config[field]
                self.config[field] = utils.parse_config(
                    config, req, opt, f'ImSim[\'{field}\']',
                    allow_unregistered=True
                    )
            except KeyError:
                # don't fill an optional field that wasn't passed
                pass

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

    def __copy__(self):
        return ImSimConfig(deepcopy(self.config))

    def keys(self):
        return self.config.keys()

    def items(self):
        return self.config.items()

    def values(self):
        return self.config.values()
