from copy import deepcopy

from superbit_lensing.config import ModuleConfig
from superbit_lensing import utils

import ipdb

class OBAConfig(ModuleConfig):
    '''
    A class that defines the OBA configuration file
    '''

    _name = 'OBA'

    # if not set (or no config is passed), use the (ordered)
    # default modules list
    _default_modules = [
        'preprocessing',
        'cals',
        'masking',
        'background',
        'astrometry',
        'starmask',
        'coadd',
        'detection',
        'output',
        'cleanup'
    ]

    # these are top-level config fields
    _req_params = {
        'run_options': ['target_name', 'bands', 'det_bands']
        }

    # NOTE: a parsed config file will use these defaults if not set by the user
    _opt_params = {
        'run_options': {
            'min_image_quality': 'unverififed',
            'overwrite': False,
            'fresh': False, # delete all previous OBA files on the QCC!
            'vb': False,
        },
        'modules': _default_modules,
        'test': {
            # these are the main suite of sims we are using to test
            # the OBA on the `hen` SuperBIT machine
            'type': 'hen',
            'run_name': None,
            'sim_dir': None,
            # this tells the test prepper to skip compressing test images
            # if they are already present
            'skip_existing': True,
        },
        'preprocessing': {
            # skip the decompressing of raw files if already present in OBA dir
            'skip_decompress': True
            },
        'cals': {
            # dark frame thresholding for hot pixel mask
            'hp_threshold': 1000,
            'ignore_flats': False
        },
        'masking': {
            'cosmic_rays': True,
            'satellites': True,
        },
        'astrometry': {
            # can force the astrometry module to rerun the WCS solution even if
            # there is already a WCS in the image headers from the image checker
            'rerun': False,
            # the search radius about the target position, in deg
            'search_radius': 1, # deg
        },
        'coadd': {
            # the SWarp single-band coadd COMBINE_TYPE
            'combine_type': 'CLIPPED',
            # the SWarp detection coadd COMBINE_TYPE
            'det_combine_type': 'WEIGHTED'
        },
        'output': {
            # Use to make one big central stamp at the target center
            'make_center_stamp': True,
            'center_stamp_size': 512,

            # NOTE: "1d" is the standared CookieCutter output that is
            # *likely* more optimized; it saves *only* the obj cutouts
            # in a long 1D array (along w/ metadata). "2d" is an alt
            # version that replaces the 1D arrays with the reconstructed
            # 2D images, with 0's everywhere outside of the cutouts. This
            # still compresses very nicely but eliminates overlapping pixels
            'make_2d': True
        },
        'cleanup': {
            # NOTE: Will lose intermediate data products if you turn this on!
            'clean_oba_dir': False,
            # This sets the CookieCutter format type that is saved to permanent
            # storage on the QCC. Options are:
            # - "1d": Normal CC def
            # - "2d": 2D IMAGE / 1D MASK version (eliminates overlapping pixels)
            # - "both": copy both 1d & 2d (probably only useful for testing)
            'cc_type': '1d'
        },
        }

    # set to True if you want the config parsing to error if any fields not
    # registered below in _req_params or _opt_params are present
    _allow_unregistered = False

    # the modules field doesn't subparse correctly as it is a list
    _skip_subparsing = ['modules']
    def __copy__(self):
        return ObaConfig(deepcopy(self.config))
