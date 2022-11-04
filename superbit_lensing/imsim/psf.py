import galsim

from superbit_lensing import utils

import ipdb

class BasePsf(object):

    _req_params = []
    _opt_params = {}

    _name = None

    # used to determine if the PSF is static across all exposures
    is_static = False

    def __init__(self, config):
        '''
        config: dict
            The `psf` config entries of an ImSim config
        '''

        self.config = utils.parse_config(
            config, self._req_params, self._opt_params, self._name
            )

        # The actual galsim.GSObject
        self._psf = None

        return

    def set_psf(self, psf):
        '''
        psf: galsim.GSObject
            The GalSim GSObject representing the PSF
        '''

        if not isinstance(psf, galsim.GSObject):
            raise TypeError('Must pass a galsim.GSObject for the PSF!')

        self._psf = psf

        return

    @property
    def psf(self):
        return self._psf

class StaticPSF(BasePSF):
    '''
    A static PSF between all exposures
    '''

    _req_params = []
    _opt_params = {}

    _name = 'static'
    is_static = True

    def __init__(self, config):
        '''
        See constructor of BaseShear
        '''

        super(StaticPSF, self).__init__(config)

        return

def build_psf(psf_type, psf_config):
    '''
    psf_type: str
        The name of the psf type
    kwargs: dict
        All args needed for the called psf constructor
    '''

    if psf_type in PSF_TYPES:
        # User-defined psf construction
        return PSF_TYPES[psf_type](psf_config)
    else:
        raise ValueError(f'{psf_type} is not a valid psf type!')

# allow for a few different conventions
PSF_TYPES = {
    'default': StaticPSF,
    'static': StaticPSF,
    }
