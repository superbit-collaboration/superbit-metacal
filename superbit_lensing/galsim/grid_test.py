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
        self.objects = {
            'gals': Galaxies(self.config['galaxies']),
            'cluster': ClusterGalaxies(self.config['cluster']),
            'stars': Stars(self.config['stars]'])
        }

        return

    def go(self):

        self.setup_image()

        self.generate_objects()

        return

    def setup_image(self):
        self.setup_wcs()

        # ...

        return

    def setup_wcs(self):
        # self.wcs = ...
        return

    def generate_objects(self):

        self.assign_positions()

        # ...

        return

    def assign_positions(self):
        '''
        General parsing of the position_sampling config
        '''

        ps = self.config['position_sampling'].copy()

        if isinstance(ps, str):
            if ps == 'random':
                for name, obj_class in self.objects.items():
                    obj_class.assign_random_positions(self.image)
            else:
                raise ValueError('position_sampling can only be a str if ' +
                                 'set to `random`!')

        elif isinstance(ps, dict):
            _allowed_objs = ['galaxies', 'cluster', 'stars']

            # used if at least one source type is on a grid
            bg = grid.BaseGrid()
            mixed_grid = None

            for obj_type, obj_list self.objects.items():
                if name not in _allowed_objs:
                    raise ValueError('position_sampling fields must be ' +
                                     f'drawn from {_allowed_objs}!')

                # position sampling config per object type:
                ps_obj = ps[obj_type].copy()
                pos_type = ps_obj.pop('type')

                if pos_type == 'random':
                    obj_list.assign_random_positions(self.image)

                elif pos_type in bg._valid_grid_types:
                    obj_list.assign_grid_positions(
                        self.image, pos_type, ps_obj
                        )

                elif pos_type in bg._valid_mixed_types:

                # grid_type = ps.pop('type')
                grid_type = ps['grid_type']
                grid_kwargs = {
                    'Npix_x': self.config['image_xsize'],
                    'Npiy_y': self.config['image_ysize'],
                }
                # pos = grid.build_grid(ps, **kwargs)

                    if mixed_grid is None:
                        N_inj_types = 0
                        inj_frac = {}
                        gtypes = set()
                        gspacing = set()

                        for inpt, opts in ps.items():
                            for key, s in {'grid_type':gtypes, 'grid_spacing':gspacing}.items():
                                try:
                                    s.add(opts[key])
                                except KeyError:
                                    # Only has to be present for one input type
                                    pass

                            if opts['type'] == 'MixedGrid':
                                N_inj_types += 1
                                inj_frac[inpt] = opts['inj_frac']
                            else:
                                raise ValueError('The mixed sampling type {}'.format(opts['type']) +
                                                    ' is not yet implemented in `generate_objects()`.')

                        for key, s in {'grid_type':gtypes, 'grid_spacing':gspacing}.items():
                            if len(s) != 1:
                                raise ValueError('For now, only one `{}` is allowed '.format(key) +
                                                    'for a mixed grid!')
                            # if key not in ps:
                            #     ps[key] = s.pop()

                        gtype = gtypes.pop()

                        mixed_grid = grid.MixedGrid(gtype, N_inj_types, inj_frac)

                        grid_kwargs = self._build_grid_kwargs(pstype, ps)
                        mixed_grid.build_grid(**grid_kwargs)

                        # Objects are assigned immediately since we set all injection
                        # fractions during construction. Otherwise would have to wait
                        self.pos[real] = mixed_grid.pos[input_type]

                        # NOTE: We ignore the inputted nobjects and use the correct grid value
                        # instead (user was already warned)
                        inj_nobjs = mixed_grid.nobjects[input_type]
                        self.nobjects[real] = inj_nobjs

                    else:
                        # NOTE: Below is what we would do if we hadn't already already
                        # ensured object assignment during MixedGrid construction
                        #
                        # if mixed_grid.assigned_objects is True:
                        #     self.pos[real] = mixed_grid.pos[input_type]
                        # else:
                        #     mixed_grid.add_injection(input_type, ps[input_type]['inj_frac'])
                        obj_list.set_Nobjs(mixed_grid.nobjs[obj_type])
                        obj_list.set_positions(mixed_grid.pos[obj_type])

                else:
                    # An error should have already occured, but just in case:
                    raise ValueError('Position sampling type {} is not valid!'.format(gtype))

        else:
            raise TypeError('position_sampling must either be a str or dict!')

        return

    def _assign_positions(self):
        pass

    def shear_objects(self):
        pass

    def _shear_objects(self):
        pass

class SourceClass(object):
    '''
    Base class for a class of simulated source (e.g. galaxy, star, etc.)
    '''

    obj_type = None

    def __init__(self, config):
        '''
        config: dict
            A configuration dictionary of all fields for the given source
            class
        '''

        self.config = config

        # nobjs will not be set for some position sampling schemes, such as
        # grids
        try:
            self.Nobjs = config['nobjs']
        except KeyError:
            self.Nobjs = None

        # params to be set later on
        self.pos = None
        self.shear = None

        return

    # def assign_positions(self, pos_type, pos_config, image):
    #     '''
    #     Assign source positions according to the run config

    #     pos_config: dict
    #         A position sampling config, taken from the main run config
    #     image: galsim.Image
    #         A galsim image object. Assigned positions are done relative
    #         to this image (which has bounds, wcs, etc.)
    #     '''

    #     # config parsing has been done by this stage

    #     pos_type = pos_config['type']

    #     if pos_type == 'random':
    #         self._assign_random_positions(image)
    #     else:
    #         self._assign_grid_positions(image)

    #     return

    def assign_random_positions(self, image):
        '''
        TODO: Add in dec correction. For now, just doing the simplest
        possible thing in image space. Long term we should make these
        (ra,dec) and store the image WCS

        image: galsim.Image
            A galsim image object. Assigned positions are done relative
            to this image (which has bounds, wcs, etc.)
        '''

        if self.Nobjs is None:
            raise KeyError(
                f'Must set nobjs for the {name} field if using ' +
                'random position sampling!'
                )

        shape = (2, self.Nobjs)
        self.pos = np.empty(shape)

        for i, Npix in enumerate(image.array.shape):
            self.pos[i] = np.random.rand(self.Nobjs) * Npix

        return

    def assign_grid_positions(self, image, grid_type, grid_kwargs):
        '''
        image: galsim.Image
            A galsim image object. Assigned positions are done relative
            to this image (which has bounds, wcs, etc.)
        grid_type: str
            The name of the grid type (see grid.py for options)
        grid_kwargs: dict
            A dictionary of kwargs needed to build the corresponding
            grid type
        '''

        grid_kwargs = self._build_grid_kwargs(pstype, ps)

        tile_grid = grid.build_grid(grid_type, **grid_kwargs)
        self.pos = tile_grid.pos

        # NOTE: We ignore the inputted nobjects and use the correct grid value
        # instead (user was already warned)
        inj_nobjs = np.shape(tile_grid.pos)[0]

        self.Nobjs = inj_nobjs

        return

    def set_positions(self, pos_list):
        '''
        Set source positions with an explicit list. Useful if source positions
        are coupled between source classes, such as with a MixedGrid

        pos_list: list
            A list of source positions
        '''
        pass

    # ...

class Galaxies(SourceClass):
    obj_type = 'galaxy'

    def __init__(self, config):
        super(Galaxies, self).__init__(config)
        return

class ClusterGalaxies(SourceClass):
    obj_type = 'cluster_galaxy'

    def __init__(self, config):
        super(ClusterGalaxies, self).__init__(config)
        return

class Stars(SourceClass):
    obj_type = 'stars'

    def __init__(self, config):
        super(Stars, self).__init__(config)
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
