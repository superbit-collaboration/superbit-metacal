from superbit_lensing import utils
from copy import deepcopy

class ImSimConfig(object):
    # all parameter names identical to those used in standard
    # SuperBIT image sims
    # TODO: Would be nice to rename a few of these for clarity
    _req_params = [
        # telescope params
        'tel_diam', 'nstruts', 'strut_thick', 'strut_theta', 'obscuration'
        # filter params
        'lam', 'bandpass',
        # psf params (optical bits in telescope params)
        'jitter_fwhm'
        # image params
        'image_xsize', 'image_ysize', 'pixel_scale',
        # detector params
        'gain',
        # noise params
        'read_noise', 'dark_current', 'dark_current_std',
        # cluster params
        'mass', 'nfw_conc', 'nfw_z_halo', 'center_ra', 'center_dec',
        # NFW params
        'nfw_conc', 'nfw_z_halo'
        # survey strategy
        'nexp', 'exp_time',
        # sources (nobj is foreground + background)
        'nclustergal', 'nobj', 'nstars',
        # cosmology params
        'omega_m', 'omega_lam',
        # dir params
        'cosmosdir', 'datadir', 'outdir'
        # file params
        'cat_file_name', 'fit_file_name', 'clsuter_cat_name',
        ]
    _opt_params = {
        # general params
        'run_name':None, 'clobber':False, 'use_optics':True,
        # multiprocessing params
        'mpi':False, 'ncores':1,
        # file & dir params
        'star_cat_name':None,
        # star params
        'sample_gaia_cats':True, 'gaia_dir':None,
        # seed params
        'master_seed':None, 'noise_seed':None, 'dithering_seed':None,
        'cluster_seed':None, 'stars_seed':None, 'galobj_seed':None,
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
            self.config, self._req, self._opt, 'ImSim')

        return

        def __getitem__(self, key):
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
