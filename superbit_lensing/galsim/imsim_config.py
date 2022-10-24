from superbit_lensing import utils
from copy import deepcopy

class ImSimConfig(object):
    # all parameter names identical to those used in standard
    # SuperBIT image sims
    # TODO: Would be nice to rename a few of these for clarity
    # TODO: Restructure for parent fields

    # these are top-level config fields
    _req_params = {
        'telescope': [
            'tel_diam',
            'nstruts',
            'strut_thick',
            'strut_theta',
            'obscuration'
            ],

        'filter': [
            'lam',
            'bandpass'
            ],

        'detector': [
            'gain'
            ],

        'noise': [
            'read_noise',
            'dark_current',
            'dark_current_std'
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
            'mass',
            'nfw_conc',
            'nfw_z_halo',
            'center_ra',
            'center_dec',
            'nclustergal'
            ],

        'observation': [
            'nexp',
            'exp_time'
        ],

        'cosmology': [
            'omega_m',
            'omega_lam'
        ],

        'input': [
            'cosmosdir',
            'datadir',
            'cat_file_name',
            'cluster_cat_name'
        ]
    }

    # defaults are assigned
    _opt_params = {
        'run_options': {
            'run_name': None,
            'mpi': False,
            'ncores': 1,
            'clobber': False,
            'vb': False
            },

        'psf': {
            'use_optics': True
        },

        'output': {
            'outdir': None
        },

        'stars': {
            'nstars': None,
            'sample_gaia_cats': True,
            'gaia_dir': None,
            'star_cat_name': None
        },

        'galaxies': {
            'nobj': None
        },

        'position_sampling': {
            type: 'random'
            # TODO: fill in the rest of the fields!
        }

        'shear': {
            type: 'nfw'
            # TODO: fill in the rest of the fields!
        }

        'seeds': {
            'master_seed': None,
            'noise_seed': None,
            'dithering_seed': None,
            'cluster_seed': None,
            'stars_seed': None,
            'galobj_seed': None
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
        self.config = utils.parse_config(
            self.config, self._req_params, self._opt_params, 'ImSim'
            )

        return

    def __getitem__(self, key):
        '''
        TODO: For backwards compatibility, it may be nice to search
        through all nested fields if a key is not found
        '''
        return self.pars[key]

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
