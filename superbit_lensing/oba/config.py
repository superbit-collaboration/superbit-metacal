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
        'run_options': ['target_name', 'bands']
        }

    # NOTE: a parsed config file will use these defaults if not set by the user
    _opt_params = {
        'run_options': {
            'overwrite': False,
            'vb': False,
        },
        'modules': _default_modules,
        'test': {
            # these are the main suite of sims we are using to test
            # the OBA on the `hen` SuperBIT machine
            'type': 'hen',
            'run_name': None,
            'sim_dir': '/home/sweveret/repos/superbit-metacal/tests/ajay/',
            # this tells the test prepper to skip compressing test images
            # if they are already present
            'skip_existing': True,
        },
        'masking': {
            'types': {
                'cosmic_rays': True,
                'satellites': True,
            },
        },
        'astrometry': {
            # can force the astrometry module to rerun the WCS solution even if
            # there is already a WCS in the image headers from the image checker
            'rerun': False,
            # the search radius about the target position, in deg
            'search_radius': 1, # deg
        },
        'coadd': {
            'det_bands': ['b', 'lum'],
        },
        'cleanup': {
            # NOTE: Will lose intermediate data products if you turn this on!
            'clean_oba_dir': False
        },
        }

    # set to True if you want the config parsing to error if any fields not
    # registered below in _req_params or _opt_params are present
    _allow_unregistered = False

    # the modules field doesn't subparse correctly as it is a list
    _skip_subparsing = ['modules']

    # def __init__(self, config):
    #     '''
    #     config: str, dict
    #         An image simulation config. Either a filename for a yaml file
    #         or a dictionary
    #     '''

    #     if isinstance(config, str):
    #         self.read_config(config)
    #     elif isinstance(config, dict):
    #         self.config_file = None
    #         self.config = config

    #     self.parse_config()

    #     return

    # def read_config(self, config_file):
    #     '''
    #     config: str
    #         A yaml config filename
    #     '''
    #     self.config_file = config_file
    #     self.config = utils.read_yaml(config_file)

    #     return

    # def parse_config(self):

    #     # loop over root fields
    #     for field in self._req_params:
    #         req = self._req_params[field]
    #         try:
    #             opt = self._opt_params[field]

    #         except KeyError:
    #             opt = {}

    #         self.config[field] = utils.parse_config(
    #             self.config[field], req, opt, 'Oba',
    #             allow_unregistered=True
    #             )

        # # check for any root fields that only exist in _opt_params
        # for field in self._opt_params:
        #     if field not in self._req_params:
        #         req = []
        #     else:
        #         req = self._req_params[field]
        #     opt = self._opt_params[field]

        #     try:
        #         config = self.config[field]
        #         self.config[field] = utils.parse_config(
        #             config, req, opt, f'Oba[\'{field}\']',
        #             allow_unregistered=True
        #             )
        #     except KeyError:
        #         # don't fill an optional field that wasn't passed
        #         pass

        # return

    # def __getitem__(self, key):
    #     '''
    #     TODO: For backwards compatibility, it may be nice to search
    #     through all nested fields if a key is not found
    #     '''
    #     return self.config[key]

    # def __setitem__(self, key, val):
    #     self.config[key] = val
    #     return

    # def __delitem__(self, key):
    #     del self.config[key]
    #     return

    # def __iter__(self):
    #     return iter(self.config)

    # def __repr__(self):
    #     return str(self.config)

    # def copy(self):
    #     return self.__copy__()

    def __copy__(self):
        return ObaConfig(deepcopy(self.config))

    # def keys(self):
    #     return self.config.keys()

    # def items(self):
    #     return self.config.items()

    # def values(self):
    #     return self.config.values()
