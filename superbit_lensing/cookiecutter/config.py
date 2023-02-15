
from superbit_lensing import utils
from copy import deepcopy

import ipdb

class CookieCutterConfig(object):

    # these are top-level config fields
    _req_params = {
        'images': [],

        'input': [
            'catalog',
        ],

        'output': [
            'filename'
        ],
    }

    # defaults are assigned
    _opt_params = {
        'input': {
            'dir': None,
            'ra tag': 'RA',
            'dec tag': 'DEC',
            'boxsize tag': 'boxsize',
            'catalog_ext': 1, # FITS tables are not stored in primary
        },
        'output': {
            'dir': None,
            'overwrite': False,
        },
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
                self.config[field], req, opt, 'CookieCutter',
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
                    config, req, opt, f'CookieCutter[\'{field}\']',
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
        return CookieCutterConfig(deepcopy(self.config))

    def keys(self):
        return self.config.keys()

    def items(self):
        return self.config.items()

    def values(self):
        return self.config.values()
