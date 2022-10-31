from abc import abstractmethod
import galsim

from superbit_lensing import utils

class BaseShear(object):

    _req_params = []
    _opt_params = {}

    _name = None

    def __init__(self, config):
        '''
        config: dict
            The `shear` config entries of an ImSim config
        '''

        self.config = utils.parse_config(
            config, self._req_params, self._opt_params, self._name
            )

        return

    @abstractmethod
    def lens(self):
        pass

class ConstantShear(BaseShear):
    '''
    Simple, constant shear field across focal plane
    '''

    _req_params = []
    _opt_params = {
        'g1': None,
        'g2': None,
        'e1': None,
        'e2': None,
        }

    def __init__(self, config):
        '''
        See constructor of BaseShear
        '''

        super(ConstantShear, self).__init__(config)

        # should only pass one type
        if self.config['g1'] is None:
            if self.config['e1'] is None:
                raise ValueError('Must pass either (g1,g2) or (e1,e2)!')
            assert 'e2' in self.config
            self.shear_type = 'e'
        else:
            assert 'g2' in self.config
            self.shear_type = 'g'

        return

    def lens(self, obj):
        '''
        obj: galsim.GSObject
            The GalSim object to lens
        '''

        if shear_type == 'e':
            e1, e2 = self.config['e1'], self.config['e2']
            sheared = obj.Shear(e1=e1, e2=e2)
        elif shear_type == 'g':
            g1, g2 = self.config['g1'], self.config['g2']
            sheared = obj.Shear(g1=g1, g2=g2)
        else:
            # shouldn't happen for now!
            raise ValueError(f'{shear_type} is not currentlys supported ' +
                             'for ConstantShear!')

        return sheared

class NFWShear(BaseShear):
    '''
    Basic wrapper around galsim NFW shear classes
    '''

    _req_params = ['mass', 'z', 'concentration']
    _opt_params = {}
    _name = 'nfw'

    def lens(self, obj):
        '''
        obj: galsim.GSObject
        '''
        pass

def build_shear(shear_type, shear_config):
    '''
    shear_type: str
        The name of the shear type
    kwargs: dict
        All args needed for the called shear constructor
    '''

    if shear_type in SHEAR_TYPES:
        # User-defined shear construction
        return SHEAR_TYPES[shear_type](shear_config)
    else:
        raise ValueError(f'{shear_type} is not a valid shear type!')

# allow for a few different conventions
SHEAR_TYPES = {
    'default': NFWShear,
    'nfw': NFWShear,
    'constant': ConstantShear,
    }
