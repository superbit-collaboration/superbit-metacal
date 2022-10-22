import galsim
from argparse import ArgumentParser

from superbit_lensing import utils
from superbit_lensing.galsim.imsim_config import ImSimConfig
import superbit_lensing.galsim.grid

import ipdb

def parse_args():
    parser = ArgumentParser()

    parser.add_argument('config_file', type=str,
                        help='Configuration file for mock sims')
    parser.add_argument('-run_name', type=str, default=None,
                        help='Name of mock simulation run')
    parser.add_argument('-outdir', type=str,
                        help='Output directory of simulated files')
    parser.add_argument('-ncores', type=int, default=1,
                        help='Number of cores to use for multiproessing')
    parser.add_argument('--clobber', action='store_true', default=False,
                        help='Turn on to overwrite existing files')
    parser.add_argument('--vb', action='store_true', default=False,
                        help='Turn on for verbose prints')

    return parser.parse_args()

class GridTestConfig(ImSimConfig):
    # _req_params = [??]
    pass

class ImSimRunner(object):
    def __init__(self, args):
        for key, val in vars(args).items():
            setattr(self, key, val)

        # setup logger
        logfile = f'imsim.log'
        log = utils.setup_logger(logfile, logdir=self.outdir)
        logprint = utils.LogPrint(log, self.vb)

        # setup config
        config = utils.read_yaml(self.config_file)

        # check for inconsistencies between command line options & config
        cmd_line_pars = {
            'run_name': self.run_name,
            'outdir': self.outdir,
            'ncores': self.ncores,
            'clobber': self.clobber,
            'vb': self.vb
            }

        for key, value in cmd_line_pars.items():
            if (key in config) and (config[key] != value):
                if value is not None:
                    config_val = config[key]
                    if (config_val is None) or (config_val == ''):
                        config_val = 'None'

                    logprint(f'Warning: passed value for {key} does not ' +
                            f'match config value of {config_val}; using ' +
                            f'command line value of {str(value)}')
            config[key] = value

        self.config = GridTestConfig(config)
        self.logprint = logprint

        # simulated properties for each class of objects will be stored in
        # the following
        self.objects= {
            'gals': Galaxies(),
            'cluster': ClusterGalaxies(),
            'stars': Stars()
        }

        return

    def assign_positions(self):

        ps = self.config['position_sampling'].copy()

        if isinstance(ps, str):
            if ps == 'random':
                # TODO
                pass
        elif isinstance(ps, dict):
            grid_type = ps.pop('type')
            extra_kwargs = {
                'Npix_x': self.config['image_xsize'],
                'Npiy_y': self.config['image_ysize'],
            }
            pos = grid.build_grid(ps, **kwargs)
            # TODO
        else:
            raise TypeError('position_sampling must either be a str or dict!')

        return

    def set_nobjects(self):
        if self.config['pos_sampling'] == 'random':
            self.

    def go(self):

        self.set_nobjects()

        self.assign_positions()

        return

class SourceClass(object):
    '''
    Base class for a class of simulated source (e.g. galaxy, star, etc.)
    '''

    def __init__(self):
        return

    def assign_positions(self, config):
        '''
        Assign source positions according to the run config
        config: dict
            A position sampling config, taken from the main config
        '''
        pass

    def set_positions(self, pos_list):
        '''
        Set source positions with an explicit list. Useful if source positions
        are coupled between source classes, such as with a MixedGrid

        pos_list: list
            A list of source positions
        '''
        pass

    # ...

class Galaxies(object):
    obj_type = 'galaxy'

    def __init__(self):
        return

class ClusterGalaxies(object):
    obj_type = 'cluster_galaxy'

    def __init__(self):
        return

class Stars(object):
    obj_type = 'stars'

    def __init__(self):
        return

class GridTestRunner(ImSimRunner):
    def __init__(self, *args, **kwargs):
        '''
        See ImSimRunner
        '''

        super(GridTestRunner, self).__init__(*args, **kwargs)

        # sanity check a few things
        if self.config['position_sampling'] == 'random':
            self.logprint('Position sampling set to random. Are you sure you ' +
                          'are running a grid test?')

        # ...

        return

def main(args):

    runner = GridTestRunner(args)

    runner.go()

    runner.logprint('Done!')

    return

if __name__ == '__main__':
    args = parse_args()
    main(args)
