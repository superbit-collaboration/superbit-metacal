from abc import abstractmethod
from multiprocessing import Pool
from copy import deepcopy
import numpy as np
import galsim
from astropy.table import Table, Row
import os
import time
from glob import glob

import grid

from superbit_lensing import utils

import ipdb

class SourceClass(object):
    '''
    Base class for a class of simulated source (e.g. galaxy, star, etc.)
    '''

    obj_type = None

    # fields that can be set in ImSim config[obj_type]
    _req_fields = []
    _opt_fields = {'Nobjs': None}

    def __init__(self, config, seed=None):
        '''
        config: dict
            A configuration dictionary of all fields for the given source
            class
        seed: int
            The seed for generating the source class, if desired
        '''

        self.config = config

        self.parse_config()

        # nobjs will not be set for some position sampling schemes, such as
        # grids
        self.set_Nobjs_from_config()

        self.seed = seed

        # params to be set later on
        self.pos = None
        self.im_pos = None
        self.pos_unit = None
        self.shear = None
        self.grid = None
        self.catalog = None

        # will store the object stamps for each exposure
        self.obj_list = {}

        return

    def parse_config(self):

        utils.parse_config(
            self.config, req=self._req_fields, opt=self._opt_fields,
            allow_unregistered=True
            )

        return

    def set_Nobjs(self, Nobjs):

        self.Nobjs = Nobjs

        return

    def set_Nobjs_from_config(self):

        try:
            self.Nobjs = self.config['Nobjs']
        except KeyError:
            self.Nobjs = 0

        return

    def set_seed(self, seed):
        '''
        seed: int
            The seed for this source class
        '''

        if self.seed is not None:
            raise AttriuteError(f'Seed for obj class {self._name} is ' +
                                'already set!')

        if not isinstance(seed, int):
            raise TypeError('Passed object seed must be an int!')

        self.seed = seed

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
            self.im_pos = np.zeros(shape)

            for i, Npix in enumerate(image.array.shape):
                self.im_pos[:,i] = np.random.rand(self.Nobjs) * Npix

            # NOTE: 1 for fits-like origin. That is how we initialized
            # the galsim Image / WCS
            # TODO: check this bit
            lenpos=len(self.im_pos)
            posIm = [galsim.PositionD(self.im_pos[i,:]) for i in range(lenpos)]
            self.pos = [image.wcs.toWorld(pos) for pos in posIm]
            self.pos_unit = galsim.radians

        return

    def assign_grid_positions(self, image, ps_type, grid_config, pixel_scale):
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
        pixel_scale: float
            The image pixel scale
        '''

        rng = np.random.default_rng(self.seed)

        if ps_type == 'MixedGrid':
            grid_type = grid_config['grid_type']
        else:
            # in this case a single grid
            grid_type = ps_type

        grid_kwargs = grid.build_grid_kwargs(
            grid_type, grid_config, image, pixel_scale, rng=rng
            )

        self.grid = grid.build_grid(grid_type, **grid_kwargs)
        self.pos = self.grid.pos
        self.pos_unit = self.grid.pos_unit
        self.im_pos = self.grid.im_pos

        inj_nobjs = np.shape(self.grid.pos)[0]

        self.Nobjs = inj_nobjs

        return

    def assign_mixed_grid_positions(self, mixed_grid):
        '''
        Assign positions from an existing MixedGrid

        mixed_grid: grid.MixedGrid
            A MixedGrid instance that has already assigned
            positions for each obj_type
        '''

        rng = np.random.default_rng(self.seed)

        self.set_Nobjs(mixed_grid.nobjects[self.obj_type])
        self.set_positions(
            mixed_grid.pos[self.obj_type],
            mixed_grid.im_pos[self.obj_type],
            mixed_grid.pos_unit,
        )
        self.grid = mixed_grid
        assert self.pos.shape[0] == self.Nobjs

        return

    def set_positions(self, pos_list, im_pos, pos_unit):
        '''
        Set source positions with an explicit list. Useful if source positions
        are coupled between source classes, such as with a MixedGrid

        pos_list: np.ndarray (2xNobjs)
            An array of object positions in *world* coords
        pos_list: np.ndarray (2xNobjs)
            An array of object positions in *image* coords
        pos_unit: galsim.Angle
            A GalSim angle unit class for the position list
        '''

        self.pos = pos_list
        self.im_pos = im_pos
        self.pos_unit = pos_unit

        return

    def _check_Nobjs(self):

        if self.Nobjs is None:
            raise KeyError(
                f'Must set nobjs for the {self.obj_type} field if using ' +
                'random position sampling!'
                )

        return

    def generate_objects(self, exp, band, run_config, image, psf, shear,
                         logprint, ncores=1):
        '''
        exp: int
            Internal exposure number
        band: str
            Name of bandpass filter to use for generated objects
        run_config: dict
            The configuration dictionary for the whole ImSim run
        image: galsim.Image
            The GalSim image to add the object stamps onto
            NOTE: This image may have a different WCS than the one
            used for position sampling, which is accounted for
        psf: galsim.GSObject
            The GalSim object representing the (for now, static) PSF
        shear: shear.py class
            The lenser to use for objects
        logprint: utils.LogPrint
            The LogPrint instance to use
        ncores: int
            The number of processes to use when batching jobs
        '''

        if self.seed is None:
            seed = utils.generate_seeds(1)
            self.set_seed(obj_seed)
            logprint(f'Generated {self.obj_type}_seed={obj_seed}')

        # if the object list already has an entry for the current band,
        # something has gone awry!
        assert band not in self.obj_list
        self.obj_list[band] = {}

        start = time.time()

        batch_indices = utils.setup_batches(self.Nobjs, ncores)

        if ncores == 1:
            logprint('Generating objects in serial')
            # Only one batch, and don't need to collate
            self.obj_list[band][exp] = self.make_obj_runner(
                *self.get_make_obj_args(
                    batch_indices, band, run_config, image, psf, shear,
                    logprint, 0
                    )
                )

        else:
            logprint(f'Generating objects in parallel; ncores={ncores}')
            with Pool(ncores) as pool:
                # Create batches
                batch_indices = utils.setup_batches(self.Nobjs, ncores)
                seeds = utils.generate_seeds(ncores, master_seed=self.seed)
                argsss =  [self.get_make_obj_args(
                                batch_indices, band, run_config, image, psf,
                                shear, logprint, k
                                ) for k in range(ncores)]
                self.collate_objs(
                    exp,
                    band,
                    pool.starmap(self.make_obj_runner,argsss)
                )

        dt = time.time() - start
        logprint(f'Total time for {self.obj_type} injections: {dt:.1f}s')
        logprint(f'Time per object: {dt/self.Nobjs:.2f}s')

        return

    def get_make_obj_args(self, batch_indices, band, run_config, image, psf,
                          shear, logprint, k):
        '''
        batch_indices: list
            A list of batch obj indices
        band: str
            Name of bandpass filter to use for generated objects
        run_config: dict
            The main ImSim run configuration dictionary
        image: galsim.Image
            The GalSim image to add the object stamps onto
            NOTE: This image may have a different WCS than the one
            used for position sampling, which is accounted for
        psf: galsim.GSObject
            The GalSim object representing the (for now, static) PSF
        shear: shear.py class
            The lenser to use for objects
        logprint: utils.LogPrint
            The LogPrint instance to use
        k: int
            The core number for multiprocessing

        NOTE: pool.starmap wont work w/ kwargs
        '''

        args = [
            batch_indices[k],
            band,
            run_config,
            self.config,
            image,
            self.pos,
            self.pos_unit,
            psf,
            shear,
            galsim.UniformDeviate(self.seed),
            logprint
            ]

        return args

    def generate_objects_from_exp(self, base_exp, new_exp, band, image):
        '''
        Instead of running the full generate_objects() method for new_exp,
        use the object stamps created in base_exp but update the stamp
        positions in image coords using the corresponding exposure image.
        This is a useful thing to do if your PSF & WCS are static across
        all exposures, as there is no reason to re-compute the stamps

        bas_exp: int
            The index of the exposure whose stamps you want to grab
        new_exp: int
            The index of the new exposure we're assigning object stamps to
        band: str
            Name of bandpass filter to use for generated objects
        image: galsim.Image
            The GalSim image corresponding to new_exp that we will draw the
            stamps onto
        '''

        # a tuple of (i, stamp_i, truth_i) for all objects i
        self.obj_list[band][new_exp] = deepcopy(
            self.obj_list[band][base_exp]
            )

        # update the stamp centers using new image WCS
        unit = self.pos_unit
        for i, stamp, truth in self.obj_list[band][new_exp]:
            if (stamp is None) or (truth is None):
                continue
            ra, dec = self.pos[i].ra, self.pos[i].dec
            world_pos = galsim.CelestialCoord(ra, dec)
            image_pos = image.wcs.toImage(world_pos)

            try:
                stamp.setCenter(image_pos.x, image_pos.y)
            except Exception as e:
                self.logprint(e)
                self.logprint('skipping')
                # TODO: check that in-place alteration was done!

        return

    def collate_objs(self, exp, band, make_obj_outputs):
        '''
        Process the multiprocessing returns of make_obj_runner()

        exp: int
            Internal exposure number
        band: str
            Name of bandpass filter to use for generated objects
        make_obj_outputs: list
            A list of len==ncores, each filled with (i, stamp_i, truth_i)
            tuples for each object
        '''

        # flatten N=Ncore outputs into 1 list
        self.obj_list[band][exp] = [item for sublist in make_obj_outputs
                                    for item in sublist]

        return

    @classmethod
    def make_obj_runner(cls, batch_indices, *args, **kwargs):
        '''
        Handles the batch running of make_obj() over multiple cores

        batch_indices: list, np.ndarray
            The list or array of batch indices (ints)
        args: list
            The positional args to pass to make_obj(). Setup in
            get_make_obj_args()
        '''

        res = []
        for i in batch_indices:
            res.append(cls.make_obj(
                i, *args, **kwargs)
                       )

        return res

    @classmethod
    def make_obj(cls, i, *args):
        '''
        Make a single object of the given class type. The base class version
        only does simple parsing; derived classes must implement a
        _make_obj() method

        i: int
            The index of the object to make
        args: list
            The positional args to pass to make_obj()
        logprint: utils.LogPrint
            An ImSim LogPrint instance (must be last entry in args list!)
        '''

        logprint = args[-1]

        try:
            logprint(f'Starting {cls._name} {cls.obj_type} {i}')
            stamp, truth = cls._make_obj(i, *args)
            logprint(f'{cls._name} {cls.obj_type} {i} completed succesfully')

        except galsim.errors.GalSimError as e:
            logprint(f'{cls._name} {cls.obj_type} {i} has failed with error: ' +
                     f'{e}Skipping...')
            return i, None, None

        return i, stamp, truth

    @abstractmethod
    def _make_obj(*args, **kwargs):
        '''
        Each subclass must implement!
        '''
        pass

    @classmethod
    def _render_obj(cls, obj, psf, image, pos):
        '''
        Helper function to render the object onto a stamp given
        a few generic pieces computed in each subclass's _make_obj()

        obj: galsim.GSObject
            The GalSim object to render
        psf: galsim.GSObject
            The GalSim PSF to convolve with
        image: galsim.Image
            The GalSim image to draw onto
        pos: list, tuple
            The (ra,dec) world position of the obj *including*
            a galsim.Angle unit for each entry
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

class CircleGalaxies(SourceClass):
    obj_type = 'galaxies'
    _name = 'circle'
    _obj_class = 'gal'

    _req_fields = []
    _opt_fields = {
        'flux_min': 5e1,
        'flux_max': 1e5,
        'hlr_min': 0.1, # arcsec
        'hlr_max': 2, # arcsec
        'n_min': 0.3, # sersic index
        'n_max': 6.2, # sersic index
        'z_min': 0.0,
        'z_max': 3.0,
        }

    def __init__(self, config, seed=None):
        '''
        See SourceClass
        '''
        super(CircleGalaxies, self).__init__(config, seed)
        return

    @classmethod
    def _make_obj(cls, obj_index, band, run_config, obj_config, image, pos,
                  pos_unit, psf, shear, ud, logprint):
        '''
        Static method that plays well with multiprocessing & does
        no config parsing or type checking

        obj_index: int
            Object index
        band: str
            Name of bandpass filter to use for generated objects
        run_config: dict
            The ImSim run config
        config: dict
            The object class config
        logprint: utils.LogPrint
            An ImSim LogPrint instance
        image: galsim.Image
            The GalSim image that we will render onto
        pos: list, tuple, np.array
            The (ra,dec) world position of the obj
            NOTE: Right now, this is actually the full position
            list due to an annoyance. Can fix in the future
        pos_unit: galsim.Angle
            A GalSim angle unit for the position list
        shear: shear.py class
            A shear instance
        ud: galsim.UniformDeviate
            The deviate to use for random sampling
        '''

        # NOTE: see docstring above
        pos = pos[obj_index]

        # Sample basic sersic parameters
        flux_min = obj_config['flux_min']
        flux_max = obj_config['flux_max']

        hlr_min = obj_config['hlr_min']
        hlr_max = obj_config['hlr_max']

        n_min = obj_config['n_min']
        n_max = obj_config['n_max']

        z_min = obj_config['z_min']
        z_max = obj_config['z_max']

        # Sersic class requires index 0.3 <= n <= 6.2
        if (n_min < 0.3):
            n_min = 0.3
            logprint.debug(f'n_min of {n_min} is too small; ' +
                           'setting to 0.3')
        if (n_max > 6.2):
            n_max = 6.2
            logprint.debug(f'n_max of {n_max} is too large; ' +
                           'setting to 6.2')

        flux = (flux_max - flux_min) * ud() + flux_min
        hlr = (hlr_max - hlr_min) * ud() + hlr_min
        n = (n_max - n_min) * ud() + n_min
        z = (z_max - z_min) * ud() + z_min

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
        obj, lens_pars = shear.lens(obj, return_lens_pars=True)
        g1, g2, mu = lens_pars['g1'], lens_pars['g2'], lens_pars['mu']
        

        #pos = [p*pos_unit for p in pos]
        pos = [pos.ra, pos.dec]
        obj_stamp = super(CircleGalaxies, cls)._render_obj(
            obj, psf, image, pos
            )

        # Create corresponding row in truth table
        truth = Table()
        truth['obj_index'] = obj_index,
        truth['ra'] = pos[0].deg,
        truth['dec'] = pos[1].deg,
        truth[f'flux_{band}'] = flux,
        truth['hlr'] = hlr,
        truth['g1'] = 0,
        truth['g2'] = 0,
        truth['theta'] = theta.deg,
        truth['theta_unit'] = 'deg',
        truth['n'] = n,
        truth['g1'] = g1,
        truth['g2'] = g2,
        truth['mu'] = mu,
        truth['z'] = z,
        truth['obj_class'] = 'gal'

        return obj_stamp, truth

class COSMOSGalaxies(SourceClass):
    '''
    Create galaxies from one of the GalSim COSMOS catalogs

    NOTE: This is what we used for earlier SuperBIT pipeline sims
    '''

    obj_type = 'galaxies'
    _name = 'cosmos'
    _obj_class = 'gal'

    def __init__(self, config, seed=None):
        '''
        See SourceClass
        '''
        super(COSMOSGalaxies, self).__init__(config, seed=seed)
        return

    @staticmethod
    def _make_obj(run_config, band, config, image, pos, catalog, ud):
        '''
        TODO: Update!

        run_config: dict
            The ImSim run config
        band: str
            Name of bandpass filter to use for generated objects
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

class COSMOSClusterGalaxies(SourceClass):
    obj_type = 'cluster_galaxies'
    _name = 'cosmos_cluster_galaxies'
    _obj_class = 'gal'

    def __init__(self, config, seed=None):
        '''
        See SourceClass
        '''
        super(COSMOSClusterGalaxies, self).__init__(config, seed=seed)
        return

class GAIAStars(SourceClass):
    obj_type = 'stars'
    _name = 'gaia'
    _obj_class = 'star'

    def __init__(self, config, seed=None):
        '''
        See SourceClass
        '''

        super(GAIAStars, self).__init__(config, seed=seed)

        if self.config['cat_file'] is not None:
            self.cat_name = self.config['cat_file']
        else:
            try:
                gaia_dir = self.config['gaia_dir']
            except KeyError:
                gaia_dir = './'

            gaia_files = os.path.join(gaia_dir, 'GAIA*.csv')
            gaia_cats = glob(gaia_files)
            sample_gaia_rng = np.random.default_rng(self.seed)

            self.cat_name = sample_gaia_rng.choice(gaia_cats)
            self.catalog = Table.read(self.cat_name)

        return

    def get_make_obj_args(self, *args):
        '''
        see SourceClass

        Need to add the GAIA catalog to the args
        '''

        orig_args = super(GAIAStars, self).get_make_obj_args(*args)

        # logprint is always last
        new_args = orig_args[:-1] + [self.catalog] + [orig_args[-1]]

        return new_args

    @classmethod
    def _make_obj(cls, obj_index, band, run_config, obj_config, image, pos,
                  pos_unit, psf, shear, ud, catalog, logprint):
        '''
        Static method that plays well with multiprocessing & does
        no config parsing or type checking

        obj_index: int
            Object index
        band: str
            Name of bandpass filter to use for generated objects
        run_config: dict
            The ImSim run config
        config: dict
            The object class config
        image: galsim.Image
            The GalSim image that we will render onto
        pos: list, tuple, np.array
            The (ra,dec) world position of the obj
            NOTE: Right now, this is actually the full position
            list due to an annoyance. Can fix in the future
        pos_unit: galsim.Angle
            A GalSim angle unit for the position list
        psf: galsim.GSObject
            The GalSim object representing the (for now, static) PSF
        shear: shear.py class
            A shear instance
        ud: galsim.UniformDeviate
            The deviate to use for random sampling
        catalog: astropy.Table, np.recarray
            The GAIA catalog to sample from
        logprint: utils.LogPrint
            An ImSim LogPrint instance
        '''

        # NOTE: see docstring above
        pos = pos[obj_index]

        # to fit some old conventions
        if 'crates' in band:
            band = band.replace('crates_', '')

        # randomly sample catalog
        gaia_index = int(ud() * len(catalog))

        flux = catalog[f'bitflux_electrons_{band}'][gaia_index]

        exp_time = run_config['observation']['exp_time']
        gain = run_config['detector']['gain']
        flux *= exp_time / gain

        obj = galsim.DeltaFunction(flux=flux)

        # No shear for stars!
        # obj = shear.lens(obj)

        #pos = [p*pos_unit for p in pos]
        pos = [pos.ra, pos.dec]
        obj_stamp = super(GAIAStars, cls)._render_obj(
            obj, psf, image, pos
            )

        # fwhm = obj_stamp.calculateFWHM()
        # mom = obj_stamp.FindAdaptiveMom().moments_sigma

        # Create corresponding row in truth table
        truth = Table()
        truth['obj_index'] = obj_index,
        truth['ra'] = pos[0].deg,
        truth['dec'] = pos[1].deg,
        truth[f'flux_{band}'] = flux,
        # truth['fwhm'] = fwhm,
        truth['obj_class'] = 'star'

        return obj_stamp, truth

def build_objects(obj_type, config, seed=None):
    '''
    obj_type: str
        The name of the object class to build
    config: dict
        A configuration dictionary that contains all needed
        fields to create the corresponding object class type
    seed: int
        A seed to set for the object constructor
    '''

    # don't want to edit the original dict
    config = config.copy()

    allowed_obj_types = {
        'galaxies': GALAXY_TYPES,
        'cluster_galaxies': CLUSTER_GALAXY_TYPES,
        'stars': STAR_TYPES,
    }

    obj_type = obj_type.lower()
    if obj_type not in allowed_obj_types.keys():
        raise ValueError(f'obj_type must be one of {allowed_obj_types.keys()}!')

    try:
        class_type = config.pop('type')

    except KeyError as e:
        raise KeyError(f'Must set a `type` for field {obj_type}!')

    try:
        allowed = allowed_obj_types[obj_type]
        return allowed[class_type](config, seed=seed)

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
