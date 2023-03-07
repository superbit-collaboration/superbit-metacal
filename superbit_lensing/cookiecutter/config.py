from superbit_lensing import utils
from superbit_lensing.config import ModuleConfig
from copy import deepcopy

import ipdb

class CookieCutterConfig(ModuleConfig):

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
            'ra_tag': 'RA',
            'dec_tag': 'DEC',
            'ra_unit': 'deg',
            'dec_unit': 'deg',
            'boxsize tag': 'boxsize',
            'catalog_ext': 1, # FITS tables are not stored in primary
        },
        'segmentation': {
            'type': 'minimal',
            'sky': 0,
            'obj': 1,
            'neighbor': 2,
        },
        'output': {
            'dir': None,
            'sci_dtype': None,
            'msk_dtype': None,
            'overwrite': False,
        },
    }

    _image_field_req = ['image_file']
    _image_field_opt = {
        'image_ext': 0,
        'weight_file': None,
        'weight_ext': 0,
        'mask_file': None,
        'mask_ext': 0,
        'skyvar_file': None,
        'skyvar_ext': 0,
        'background_file': None,
        'background_ext': 0,
        'segmentation_file': None,
        'segmentation_ext': 0,
    }

    # to allow for unknown image names
    _allow_unregistered = True

    def parse_config(self):

        super(CookieCutterConfig, self).parse_config()

        # loop over images & set a few defaults
        for image in self.config['images']:
            self.config['images'][image] = utils.parse_config(
                self.config['images'][image],
                self._image_field_req,
                self._image_field_opt,
                'CookieCutter',
                allow_unregistered=False
                )

        return

    def __copy__(self):
        return CookieCutterConfig(deepcopy(self.config))
