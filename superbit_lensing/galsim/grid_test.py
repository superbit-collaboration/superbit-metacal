from abc import abstractmethod
import numpy as np
import galsim
import time
from astropy.table import Table
from argparse import ArgumentParser

from superbit_lensing import utils
from superbit_lensing.galsim.imsim_config import ImSimConfig
import superbit_lensing.galsim.shear as shear
import superbit_lensing.galsim.grid as grid

import ipdb

# TODO: Incorporate seeds!

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
    parser.add_argument('--overwrite', action='store_true', default=False,
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
            'overwrite': self.overwrite,
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
        self.objects = {}
        for obj_type in ['galaxies', 'cluster_galaxies', 'stars']:
            try:
                self.objects[obj_type] = build_objects(
                    obj_type, self.config[obj_type]
                    )
            except KeyError:
                self.logprint(f'No config entry for {obj_type}; skipping')

        # grab a few params important for running
        try:
            self.ncores = self.config['run_options']['ncores']
        except KeyError:
            self.ncores = 1

        self.nexp = self.config['observation']['nexp']

        self.images = None

        # TODO: We may want to make this different for each
        # exp in the future
        self.psf = None

        return

    def go(self):

        # TODO: loop over exposures!
        # for exp in range(self.nexp):
        # self.logprint(f'Starting exp {exp}')

        self.logprint('Setting up image...')
        self.setup_images()

        self.logprint('Setting up PSF...')
        self.setup_psf()

        self.logprint('Setting up lenser...')
        self.setup_shear()

        self.logprint('Generating objects...')
        self.generate_objects()

        self.logprint('Adding noise...')
        self.add_noise()

        self.logprint('Adding weights...')
        # self.add_weights()

        self.logprint('Adding masks...')
        # self.add_weights()

        self.logprint('Writing out image...')
        # self.write()

        return

    def setup_images(self):
        '''
        At this point, only a basic galsim Image with the correct size
        has been generated. We initialize the rest of the needed features
        such as the WCS here
        '''

        # first we define a "base image" which is never a real observation but
        # useful for defining a perfectly centered, non-dithered, ideal image
        # from which to define object positions during sampling
        self.base_image = galsim.Image(self.Nx, self.Ny)

        # Most properties will be set later on, but we will need at least
        # the image size for WCS initialization

        self.images = []

        # TODO: add dithers!
        for i in range(self.config['observation']['nexp']):
            self.images.append(
                galsim.Image(self.Nx, self.Ny)
                )

        # WCS is added to image in the method
        self.setup_wcs()

        # ...

        return

    def setup_wcs(self, theta=0.0):
        '''
        Setup the image WCS

        theta: float
            Rotation between image axes and sky coordinate axis. Defines the
            transformation from image coords to the tangent plane coords
        '''

        ra_unit = self.config['cluster']['center_ra_unit']
        dec_unit = self.config['cluster']['center_dec_unit']

        sky_center = galsim.CelestialCoord(
            ra=self.config['cluster']['center_ra'] * ra_unit,
            dec=self.config['cluster']['center_dec'] * dec_unit
        )

        pixel_scale = self.pixel_scale

        theta *= galsim.degrees

        dudx =  np.cos(theta) * pixel_scale
        dudy = -np.sin(theta) * pixel_scale
        dvdx =  np.sin(theta) * pixel_scale
        dvdy =  np.cos(theta) * pixel_scale

        for i, image in enumerate(self.images):
            affine = galsim.AffineTransform(
                dudx, dudy, dvdx, dvdy, origin=image.true_center
                )

            self.images[i].wcs = galsim.TanWCS(
                affine, sky_center, units=galsim.arcsec
                )

        # now do the same for the base image
        base_affine = galsim.AffineTransform(
            dudx, dudy, dvdx, dvdy, origin=self.base_image.true_center
        )

        self.base_image.wcs = galsim.TanWCS(
            base_affine, sky_center, units=galsim.arcsec
        )

        return

    def setup_psf(self):
        '''
        TODO: Implement different PSF options in config first
        '''

        self.psf = galsim.Gaussian(fwhm=0.24)

        return

    def setup_shear(self):

        shear_config = self.config['shear'].copy()
        shear_type = shear_config.pop('type')

        self.shear = shear.build_shear(shear_type, shear_config)

        return

    def generate_objects(self):

        # for grids, won't know Nobjs until we finish
        # assigning positions
        self.assign_positions()

        for obj_type, obj_list in self.objects.items():
            obj_list.generate_objects(ncores=self.ncores)

            # TODO: check that this alters the obj attributes in-place!

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
                    obj_class.assign_random_positions(self.base_image)
            else:
                raise ValueError('position_sampling can only be a str if ' +
                                 'set to `random`!')

        elif isinstance(ps, dict):
            _allowed_objs = ['galaxies', 'cluster_galaxies', 'stars']

            # used if at least one source type is on a grid
            bg = grid.BaseGrid()
            mixed_grid = None

            for obj_type, obj_list in self.objects.items():
                if obj_type not in _allowed_objs:
                    raise ValueError('position_sampling fields must be ' +
                                     f'drawn from {_allowed_objs}!')

                # position sampling config per object type:
                ps_obj = ps[obj_type].copy()
                pos_type = ps_obj['type']

                if pos_type == 'random':
                    obj_list.assign_random_positions(self.base_image)

                elif pos_type in bg._valid_grid_types:
                    obj_list.assign_grid_positions(
                        self.base_image, pos_type, ps_obj
                        )

                elif pos_type in bg._valid_mixed_types:
                    if mixed_grid is None:
                        N_inj_types = 0
                        inj_frac = {}
                        gtypes = set()
                        gspacing = set()

                        # MixedGrids have to be built with info across all
                        # simultaneously
                        for name, config in ps.items():
                            try:
                                gtypes.add(config['grid_type'])
                                gspacing.add(config['grid_spacing'])
                            except KeyError:
                                # Only has to be present for one input type
                                pass
                            if config['type'] == 'MixedGrid':
                                N_inj_types += 1
                                inj_frac[name] = config['fraction']

                        # can only have 1 unique value of each
                        unq = {
                            'grid_type':gtypes,
                            'grid_spacing':gspacing
                            }
                        for key, s in unq.items():
                            if len(s) != 1:
                                raise ValueError('Only one {key} is allowed ' +
                                                 'for a MixedGrid!')

                        gtype = gtypes.pop()

                        mixed_grid = grid.MixedGrid(
                            gtype, N_inj_types, inj_frac
                            )

                        grid_kwargs = grid.build_grid_kwargs(
                            gtype, ps_obj, self.base_image, self.pixel_scale
                            )
                        mixed_grid.build_grid(**grid_kwargs)

                        # Objects are assigned immediately since we set all injection
                        # fractions during construction. Otherwise would have to wait
                        self.pos = mixed_grid.pos[obj_type]
                        self.Nobjs = mixed_grid.nobjects[obj_type]
                        assert self.pos.shape[0] == self.Nobjs

                    else:
                        # NOTE: Below is what we would do if we hadn't already already
                        # ensured object assignment during MixedGrid construction
                        #
                        # if mixed_grid.assigned_objects is True:
                        #     self.pos[real] = mixed_grid.pos[obj_type]
                        # else:
                        #     mixed_grid.add_injection(obj_type, ps[obj_type]['inj_frac'])

                        obj_list.set_Nobjs(mixed_grid.nobjects[obj_type])
                        obj_list.set_positions(mixed_grid.pos[obj_type])

                else:
                    # An error should have already occured, but just in case:
                    raise ValueError('Position sampling type {} is not valid!'.format(gtype))

        else:
            raise TypeError('position_sampling must either be a str or dict!')

        return

    def add_noise(self):
        return

    def shear_objects(self):
        pass

    def _shear_objects(self):
        pass

    #---------------------------------
    # a few handy quick access funcs

    @property
    def Nx(self):
        return self.config['image']['image_xsize']

    @property
    def Ny(self):
        return self.config['image']['image_ysize']

    @property
    def pixel_scale(self):
        return self.config['image']['pixel_scale']

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
        self.set_Nobjs_from_config()

        # params to be set later on
        self.pos = None
        self.im_pos = None
        self.shear = None
        self.grid = None
        self.catalog = None

        return

    def set_Nobjs(self, Nobjs):

        self.Nobjs = Nobjs

        return

    def set_Nobjs_from_config(self):

        try:
            self.Nobjs = self.config['Nobjs']
        except KeyError:
            self.Nobjs = None

        return

    def assign_random_positions(self, image):
        '''
        TODO: Add in dec correction. For now, just doing the simplest
        possible thing in image space

        image: galsim.Image
            A galsim image object. Assigned positions are done relative
            to this image (which has bounds, wcs, etc.)
        '''

        self._check_Nobjs()

        if self.Nobjs > 0:
            shape = (self.Nobjs, 2)
            self.pos = np.zeros(shape)
            self.pos_im = np.zeros(shape)

            for i, Npix in enumerate(image.array.shape):
                self.im_pos[i] = np.random.rand(self.Nobjs) * Npix

            # NOTE: 1 for fits-like origin. That is how we initialized
            # the galsim Image / WCS
            self.pos = image.wcs.wcs_pix2world(self.im_pos, 1)

        return

    def assign_grid_positions(self, image, ps_type, grid_config):
        '''
        image: galsim.Image
            A galsim image object. Assigned positions are done relative
            to this image (which has bounds, wcs, etc.)
        ps_type: str
            The position sampling type (either a single grid
            or MixedGrid in this case)
        grid_config: dict
            The configuration dictionary for the grid of the given
            object type
        '''

        if ps_type == 'MixedGrid':
            grid_type = grid_config['grid_type']
        else:
            # in this case a single grid
            grid_type = ps_type

        grid_kwargs = grid.build_grid_kwargs(
            grid_type, grid_config, image
            )

        self.grid = grid.build_grid(grid_type, **grid_kwargs)
        self.pos = tile_grid.pos
        self.im_pos = tile_grid.im_pos

        inj_nobjs = np.shape(tile_grid.pos)[0]

        self.Nobjs = inj_nobjs

        return

    def set_positions(self, pos_list):
        '''
        Set source positions with an explicit list. Useful if source positions
        are coupled between source classes, such as with a MixedGrid

        pos_list: np.ndarray (2xNobjs)
            An array of object positions # TODO: image or physical?
        '''

        self.pos = pos_list

        return

    def _check_Nobjs(self):

        if self.Nobjs is None:
            raise KeyError(
                f'Must set nobjs for the {self.obj_type} field if using ' +
                'random position sampling!'
                )

        return

    def generate_objects(self, image, ncores=1):
        '''
        TODO

        image: galsim.Image
            The GalSim image to add objects to
        ncores: int
            The number of processes to use when batching jobs
        '''

        seeds = self.config['seeds']

        start = time.time()

        if ncores == 1:
            # something like:
            # self.get_make_obj_args(batch_indices, k)
            raise NotImplementedError('Serial processing not yet implemented!')

        else:
            with Pool(ncores) as pool:
                # Create batches
                batch_indices = utils.setup_batches(self.Nobjs, ncores)

                # TODO: implement truth cats!
                # filled_image, truth_catalog = self.combine_objs(
                # filled_image = self.combine_objs(
                self.obj_stamps = self.collate_objs(
                    pool.starmap(
                        self.make_obj_runner,
                        ([
                            batch_indices[k],
                            self.config,
                            image,
                            self.psf,
                            self.pos,
                            self.shear,
                            galsim.UniformDeviate(
                                seeds['galobj_seed']+k+1
                                ),
                            logprint,
                        ] for k in range(ncores))
                    )
                )

        dt = time.time() - start
        logprint(f'Total time for {self.obj_type} injections: {dt:.1f}s')

        # TODO: implement truth cats!
        return filled_image#, truth_catalog

    # def get_make_obj_args(self, batch_indices, k):
    #     '''
    #     batch_indices: list
    #         A list of batch obj indices
    #     k: int
    #         The core number for multiprocessing
    #     '''

        # return []

    @staticmethod
    def collate_objs(make_obj_outputs):
        '''
        Process the multiprocessing returns of make_obj_runner()

        make_obj_outputs: list
            A list of len==ncores, each filled with (i, stamp_i, truth_i)
            tuples for each object
        '''

        # flatten N=Ncore outputs into 1 list
        self.obj_list = [item for sublist in make_obj_outputs
                         for item in sublist]

        return

    # TODO: rework this to work off of self.obj_stamps
    @staticmethod
    def combine_objs(obj_list, filled_image, truth_catalog, exp_num):
        '''
        Fill the passed image with objects from the object list.
        Multiprocessing-friendly.

        image: galsim.Image
            The GalSim image to add objects to
        obj_list: list of tuples
            The collated make_obj_runner outputs (i, stamp, truth)

        exp_num: int
            The exposure number. Only add to truth table if equal to 1
        '''

        # flatten outputs into 1 list
        make_obj_outputs = [item for sublist in make_obj_outputs
                            for item in sublist]

        for i, stamp, truth in make_obj_outputs:

            if (stamp is None) or (truth is None):
                continue

            # Find the overlapping bounds:
            bounds = stamp.bounds & filled_image.bounds

            # Finally, add the stamp to the full image.
            try:
                filled_image[bounds] += stamp[bounds]
            except galsim.errors.GalSimBoundsError as e:
                print(e)

            this_flux = np.sum(stamp.array)

            if exp_num == 1:
                row = [i, truth.cosmos_index, truth.x, truth.y,
                    truth.ra, truth.dec,
                    truth.g1, truth.g2,
                    truth.mu,truth.z,
                    this_flux, truth.fwhm, truth.mom_size,
                    truth.n, truth.hlr, truth.scale_h_over_r,
                    truth.obj_class
                    ]
                truth_catalog.addRow(row)

        return filled_image, truth_catalog

    @staticmethod
    def make_obj_runner(batch_indices, config, logprint, **kwargs):
        '''
        Handles the batch running of make_obj() over multiple cores

        batch_indices: list, np.ndarray
            The list or array of batch indices (ints)
        config: dict
            The object class config
        logprint: utils.LogPrint
            An ImSim LogPrint instance
        kwargs: list
            The keyword args to pass to make_obj()
        '''

        res = []
        for i in batch_indices:
            res.append(self.make_obj(
                i, config, logprint, **kwargs)
                       )

        return res

    @staticmethod
    def make_obj(i, config, logprint, **kwargs):
        '''
        Make a single object of the given class type. The base class version
        only does simple parsing; derived classes must implement a
        _make_obj() method

        i: int
            The index of the object to make
        config: dict
            The object class config
        logprint: utils.LogPrint
            An ImSim LogPrint instance
        kwargs: list
            The keyword args to pass to _make_obj()
        '''

        try:
            obj_index = int(i) # just in case
            logprint(f'Starting {obj_type} {i}')
            kwargs['obj_index'] = i
            stamp, truth = self._make_obj(config, logprint, **kwargs)
            logprint(f'{obj_type} {i} completed succesfully')

        except galsim.errors.GalSimError as e:
            logprint(f'{obj_type} {i} has failed with error {e}\nSkipping...')
            return i, None, None

        return i, stamp, truth

    @abstractmethod
    def _make_obj(*args, **kwargs):
        '''
        Each subclass must implement!
        '''
        pass

class CircleGalaxies(SourceClass):
    obj_type = 'galaxy'
    gal_type = 'circle'

    _req_fields = []
    _opt_fields = {
        'flux_min': 1e2,
        'flux_max': 1e4,
        'hlr_min': 0.1, # arcsec
        'hlr_max': 5, # arcsec
        'n_min': 0.3, # sersic index
        'n_max': 6.2, # sersic index
        'z_min': 0.0,
        'z_max': 3.0,
        }

    def __init__(self, config):
        super(CircleGalaxies, self).__init__(config)
        return

    @staticmethod
    def _make_obj(obj_index, config, logprint, image, pos, shear, ud):
        '''
        Static method that plays well with multiprocessing & does
        no config parsing or type checking

        obj_index: int
            Object index
        config: dict
            The object class config
        logprint: utils.LogPrint
            An ImSim LogPrint instance
        image: galsim.Image
            The GalSim image that we will render onto
        pos: list, tuple, np.array
            The (ra,dec) world position of the obj
        shear: shear.py class
            A shear instance
        ud: galsim.UniformDeviate
            The deviate to use for random sampling
        '''

        # Sample basic sersic parameters
        flux_min = config['flux_min']
        flux_max = config['flux_max']

        hlr_min = config['hlr_min']
        hlr_max = config['hlr_max']

        n_min = config['n_min']
        n_max = config['n_max']

        # Sersic class requires index 0.3 <= n <= 6.2
        if (n_min < 0.3):
            n_min = 0.3
            logprint.debug(f'n_min of {n_min} is too small; ' +
                           'setting to 0.3')
        if (n_max > 6.2):
            n_max = 6.2
            logprint.debug(f'n_max of {n_max} is too large; ' +
                           'setting to 6.2')

        flux = np.random.uniform(flux_min, flux_max)
        hlr = np.random.uniform(hlr_min, hlr_max)
        n = np.random.uniform(n_min, n_max)

        obj = galsim.Sersic(
            n=n, flux=flux, half_light_radius=hlr
            )

        # apply a random rotation
        theta = ud() * 2.0 * np.pi * galsim.radians
        obj = obj.rotate(theta)

        logprint.debug(f'galaxy z={z}; flux={flux}; hlr={hlr} ' + \
                       f'index={n}')

        # don't assign intrinsic shape for circle class
        # g1, g2 = ...
        # obj.shear(...)

        # *DO* shear it though!
        obj = shear.lens(obj)

        obj_stamp = _render_object(obj, psf, image, pos)

        # TODO: handle truth class
        truth = None

        return obj_stamp, truth

class COSMOSGalaxies(SourceClass):
    '''
    Create galaxies from one of the GalSim COSMOS catalogs

    NOTE: This is what we used for earlier SuperBIT pipeline sims
    '''

    obj_type = 'galaxy'
    gal_type = 'cosmos'

    def __init__(self, config):
        super(COSMOSGalaxies, self).__init__(config)
        return

    @staticmethod
    def _make_obj(config, image, pos, catalog, ud):
        '''
        config: dict
            The main ImSim configuration dictionary for
            this run
        image: galsim.Image instance
            The image to draw the object onto. Must have a
            defined WCS
        pos: tuple, np.ndarray (1x2)
            The (ra, dec) position of the object to draw in
            world coords
        catalog: np.rec_array, astropy.Table
            The COSMOS catalog of galaxies to draw from
        ud: galsim.UniformDeviate
            A GalSim uniform deviate instance (NOTE: is mp safe)
        '''

        wcs = image.wcs

        if not isinstance(pos, galsim.CelestialCoord):
            world_pos = galsim.CelestialCoord(pos[0], pos[1])

        image_pos = wcs.toImage(world_pos)

        # TODO: sort this out!
        # We also need this in the tangent plane, which we call "world coordinates" here.
        # This is still an x/y corrdinate
        # uv_pos = affine.toWorld(image_pos)
        # logprint.debug('created galaxy position')

        #------------------------------------------------------------
        # Draw a galaxy from scratch
        # NOTE: units of config['detector']['gain'] is assumed to be
        # be e-/ADU.

        index = int(
            np.floor(ud() * len(catalog))
            )

        z = catalog[index]['ZPDF']
        flux = catalog[index][sbparams.bandpass] *\
               sbparams.exp_time / sbparams.gain

        phi = catalog[index]['c10_sersic_fit_phi'] * galsim.radians
        q = catalog[index]['c10_sersic_fit_q']

        # Cosmos HLR is in units of HST pix, convert to arcsec.
        half_light_radius = catalog[index]['c10_sersic_fit_hlr'] *\
                            0.03*np.sqrt(q)
        n = catalog[index]['c10_sersic_fit_n']

        logprint.debug(f'galaxy i={index} z={gal_z} flux={gal_flux} ' + \
                       f'hlr={half_light_radius} sersic_index={n}')

        # Sersic class requires index n >= 0.3
        if (n < 0.3):
            n = 0.3

        gal = galsim.Sersic(n = n,
                            flux = gal_flux,
                            half_light_radius = half_light_radius)

        gal = gal.shear(q = q, beta = phi)
        logprint.debug('created galaxy')

        ## Apply a random rotation
        theta = ud()*2.0*np.pi*galsim.radians
        gal = gal.rotate(theta)

        ## Apply a random rotation
        theta = ud()*2.0*np.pi*galsim.radians
        gal = gal.rotate(theta)

        ## Get the reduced shears and magnification at this point
        try:
            nfw_shear, mu = nfw_lensing(nfw, uv_pos, gal_z)
            g1=nfw_shear.g1; g2=nfw_shear.g2
            gal = gal.lens(g1, g2, mu)
        except galsim.errors.GalSimError:
            logprint(f'could not lens galaxy at z = {gal_z}, setting default values...')
            g1 = 0.0; g2 = 0.0
            mu = 1.0

        final = galsim.Convolve([psf, gal])

        logprint.debug('Convolved star and PSF at galaxy position')

        stamp = final.drawImage(wcs=wcs.local(image_pos))
        stamp.setCenter(image_pos.x,image_pos.y)
        logprint.debug('drew & centered galaxy!')
        galaxy_truth=truth()
        galaxy_truth.cosmos_index = index
        galaxy_truth.ra=ra.deg; galaxy_truth.dec=dec.deg
        galaxy_truth.x=image_pos.x; galaxy_truth.y=image_pos.y
        galaxy_truth.g1=g1; galaxy_truth.g2=g2
        galaxy_truth.mu = mu; galaxy_truth.z = gal_z
        galaxy_truth.flux = stamp.added_flux
        galaxy_truth.n = n; galaxy_truth.hlr = half_light_radius
        #galaxy_truth.inclination = inclination.deg # storing in degrees for human readability
        galaxy_truth.scale_h_over_r = q
        galaxy_truth.obj_class = 'gal'

        logprint.debug('created truth values')

        try:
            galaxy_truth.fwhm=final.calculateFWHM()
        except galsim.errors.GalSimError:
            logprint.debug('fwhm calculation failed')
            galaxy_truth.fwhm=-9999.0

        try:
            galaxy_truth.mom_size=stamp.FindAdaptiveMom().moments_sigma
        except galsim.errors.GalSimError:
            logprint.debug('sigma calculation failed')
            galaxy_truth.mom_size=-9999.

        logprint.debug('stamp made, moving to next galaxy')

        return stamp, galaxy_truth

class ClusterGalaxies(SourceClass):
    obj_type = 'cluster_galaxies'

    def __init__(self, config):
        super(ClusterGalaxies, self).__init__(config)
        return

class COSMOSClusterGalaxies(SourceClass):
    obj_type = 'cosmos_cluster_galaxies'

    def __init__(self, config):
        super(COSMOSClusterGalaxies, self).__init__(config)
        return

class Stars(SourceClass):
    obj_type = 'stars'

    def __init__(self, config):
        super(Stars, self).__init__(config)
        return

class GAIAStars(SourceClass):
    obj_type = 'gaia'

    def __init__(self, config):
        super(GAIAStars, self).__init__(config)
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

def _render_obj(obj, psf, image, pos):
    '''
    Helper function to render the object onto a stamp given
    a few generic pieces computed in each subclass's _make_obj()

    obj: galsim.GSObject
        The GalSim object to render
    psf: galsim.GSObject
        The GalSim PSF to convolve with
    image: galsim.Image
        The GalSim image to draw onto
    pos: list, tuple, np.array
        The (ra,dec) world position of the obj
    '''

    final = galsim.Convolve([psf, obj])

    ra, dec = pos[0], pos[1]
    world_pos = galsim.CelestialCoord(ra, dec)
    image_pos = image.wcs.toImage(world_pos)

    # render stamp
    obj_stamp = final.drawImage(
        wcs=image.wcs.local(image_pos)
        )
    obj_stamp.setCenter(image_pos.x, image_pos.y)

    return obj_stamp

def build_objects(obj_type, obj_config):
    '''
    obj_type: str
        The name of the object class to build
    obj_config: dict
        A configuration dictionary that contains all needed
        fields to create the corresponding object class type
    '''

    # don't want to edit the original dict
    obj_config = obj_config.copy()

    allowed_obj_types = {
        'galaxies': GALAXY_TYPES,
        'cluster_galaxies': CLUSTER_GALAXY_TYPES,
        'stars': STAR_TYPES,
    }

    obj_type = obj_type.lower()
    if obj_type not in allowed_obj_types.keys():
        raise ValueError(f'obj_type must be one of {allowed_obj_types.keys()}!')

    try:
        class_type = obj_config.pop('type')

    except KeyError as e:
        raise KeyError(f'Must set a `type` for field {obj_type}!')

    try:
        allowed = allowed_obj_types[obj_type]
        return allowed[class_type](obj_config)

    except KeyError as e:
        raise KeyError(f'{class_type} not a valid option for {obj_type}!')

GALAXY_TYPES = {
    'default': COSMOSGalaxies,
    'cosmos' : COSMOSGalaxies,
    'circle' : CircleGalaxies,
    }
CLUSTER_GALAXY_TYPES = {
    'default': COSMOSClusterGalaxies,
    'cosmos' : COSMOSClusterGalaxies,
    }
STAR_TYPES = {
    'default': GAIAStars,
    'gaia': GAIAStars,
    # 'simple': TODO
}

def main(args):

    runner = GridTestRunner(args)

    runner.go()

    runner.logprint('Done!')

    return

if __name__ == '__main__':
    args = parse_args()
    main(args)
