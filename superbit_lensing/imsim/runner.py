import numpy as np
import galsim
import fitsio
import os
from astropy.table import Table, vstack
from argparse import ArgumentParser

from superbit_lensing import utils

from config import ImSimConfig
import shear
import grid
from objects import build_objects

import ipdb

# TODO: double-check for full seed coverage!
# TODO: implement dithers!

class ImSimRunner(object):
    def __init__(self, args):
        for key, val in vars(args).items():
            setattr(self, key, val)

        # check for inconsistencies between command line options & config
        cmd_line_pars = {
            'run_name': self.run_name,
            'outdir': self.outdir,
            'ncores': self.ncores,
            'overwrite': self.overwrite,
            'vb': self.vb
            }

        config = utils.read_yaml(self.config_file)
        try:
            run_options = config['run_options']
        except KeyError:
            run_options = {}

        for key, value in cmd_line_pars.items():
            if (key in run_options):
                config_val = run_options[key]
                if (config_val != value):
                    if value is not None:
                        # Command line overrules config run_options
                        if (config_val is None) or (config_val == ''):
                            config_val = 'None'

                        print(f'Warning: passed value for {key} does not ' +
                            f'match config value of {config_val}; using ' +
                            f'command line value of {str(value)}')
                    else:
                        # in this case, use the config value
                        setattr(self, key, config_val)

        self.config = ImSimConfig(config)

        self.setup_outdir()

        # setup logger
        logfile = f'imsim.log'
        log = utils.setup_logger(logfile, logdir=self.outdir)
        logprint = utils.LogPrint(log, self.vb)

        self.logprint = logprint

        self.setup_seeds()

        # simulated properties for each class of objects will be stored in
        # the following
        self.objects = {}
        for obj_type in ['galaxies', 'cluster_galaxies', 'stars']:
            try:
                self.objects[obj_type] = build_objects(
                    obj_type, self.config[obj_type], self.seeds[obj_type]
                    )
            except KeyError:
                self.logprint(f'No config entry for {obj_type}; skipping')

        # grab a few params important for running
        self.nexp = self.config['observation']['nexp']

        self.images = None
        self.weights = None
        self.masks = None

        self.psfs = None

        return

    def setup_seeds(self):

        # need seeds for noise, dithering, and the 3 possible source classes
        _seed_types = ['noise', 'dithering', 'galaxies', 'cluster_galaxies',
                       'stars', 'master']
        Nseeds = len(_seed_types) - 1 # ignore master seed
        master_seed = None
        self.seeds = {}

        try:
            config_seeds = self.config['seeds']
            try:
                master_seed = config_seeds['master']
                if (master_seed is None) or (eval(master_seed) is None):
                    master_seed = None
            except KeyError:
                pass

            seeds = utils.generate_seeds(Nseeds, master_seed=master_seed)

            for seed_type, seed in config_seeds.items():
                if seed_type == 'master':
                    continue
                if (seed_type not in _seed_types):
                    raise ValueError(f'{seed_type} is not a valid seed type!')

                if (seed is None) or (eval(seed) is None):
                    self.logprint(f'{seed_type} seed not passed; generating')
                    self.seeds[seed_type] = seeds.pop()
                else:
                    self.seeds[seed_type] = seed

        except KeyError:
            self.logprint('No seeds passed; master seed will set based ' +
                          'on current time')
            seeds = utils.generate_seeds(Nseeds)

        for seed_type in _seed_types:
            if seed_type == 'master':
                continue
            if seed_type not in self.seeds:
                self.logprint(f'{seed_type} seed not passed; generating')
                self.seeds[seed_type] = seeds.pop()

        self.logprint('Using the following seeds:')
        for name, seed in self.seeds.items():
            self.logprint(f'{name}: {seed}')

        return

    def setup_outdir(self):
        '''
        While we will accept config['run_options']['basedir'] to be
        backwards compatible, we will prioritize the new
        config['output']['outdir']
        '''

        try:
            basedir = self.config['run_options']['basedir']
            if basedir == 'None':
                basedir = None
        except KeyError:
            basedir = None

        try:
            outdir = self.config['output']['outdir']
            if outdir == 'None':
                outdir = None
        except KeyError:
            outdir = None

        # can't setup logprint until outdir is setup
        if (basedir is not None) and (outdir is not None):
            print('Both basedir and outdir are set; prioritizing ' +
                          'outdir for ImSim running')
            self.outdir = outdir

        elif (basedir is None) and (outdir is None):
            print('No outdir set; using current working directory')
            self.outdir = os.getcwd()

        # now for two unambiguous cases
        elif basedir is not None:
            self.outdir = basedir
        else:
            self.outdir = outdir

        # shouldn't happen, but just in case
        if self.outdir == 'None':
            self.outdir = None

        utils.make_dir(self.outdir)

        return

    def go(self):

        self.logprint('Setting up images...')
        self.setup_images()

        self.logprint('Setting up PSFs...')
        self.setup_psfs()

        self.logprint('Setting up lenser...')
        self.setup_shear()

        self.logprint('Generating objects...')
        self.generate_objects()

        self.logprint('Adding objects to images...')
        self.fill_images()

        self.logprint('Adding noise...')
        self.add_noise()

        self.logprint('Adding weights...')
        self.add_weights()

        self.logprint('Adding masks...')
        self.add_masks()

        self.logprint('Writing out images...')
        self.write_images()

        self.logprint('Building truth catalog...')
        self.build_truth_cat()

        self.logprint('Writing truth catalog...')
        self.write_truth_cat()

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

        # Useful if you don't want to remake object stamps for each exposure
        self.static_wcs = True

        return

    def setup_psfs(self):
        '''
        TODO: Implement different PSF options in config & psf.py
        For now, we just do what is done in the original imsims,
        i.e. Gaussian jitter from gondola + optics
        '''

        jitter_fwhm = self.config['psf']['jitter_fwhm']
        use_optics = self.config['psf']['use_optics']

        # first the jitter component from gondola instabilities
        jitter = galsim.Gaussian(flux=1, fwhm=jitter_fwhm)

        if use_optics is True:
            # next, define Zernicke polynomial component for the optics
            lam = self.config['filter']['lam']
            diam = self.config['telescope']['diameter']
            obscuration = self.config['telescope']['obscuration']
            nstruts = self.config['telescope']['nstruts']
            strut_angle = self.config['telescope']['strut_angle']
            strut_thick = self.config['telescope']['strut_thick']

            # NOTE: aberrations were definined for lam = 550, and close to the
            # center of the camera. The PSF degrades at the edge of the FOV
            lam_over_diam = lam * 1.e-9 / diam    # radians
            lam_over_diam *= 206265.

            aberrations = np.zeros(38)             # Set the initial size.
            aberrations[0] = 0.                       # First entry must be zero
            aberrations[1] = -0.00305127
            aberrations[4] = -0.02474205              # Noll index 4 = Defocus
            aberrations[11] = -0.01544329             # Noll index 11 = Spherical
            aberrations[22] = 0.00199235
            aberrations[26] = 0.00000017
            aberrations[37] = 0.00000004

            optics = galsim.OpticalPSF(
                lam=lam,
                diam=diam,
                obscuration=obscuration,
                nstruts=nstruts,
                strut_angle=strut_angle,
                strut_thick=strut_thick,
                aberrations=aberrations
                )

            psf = galsim.Convolve([jitter_psf, optics])

            self.logprint(f'Calculated lambda over diam = {lam_over_diam} arcsec')
            self.logprint('use_optics is True; convolving telescope optics PSF profile')

        else:
            psf = jitter
            self.logprint('use_optics is False; using jitter-only PSF')

        # NOTE: As stated above, we are just making a static PSF for now
        self.psfs = []
        for i in range(len(self.images)):
            self.psfs.append(psf)

        self.static_psf = True

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

        Nexp = self.config['observation']['nexp']
        for exp in range(Nexp):
            self.logprint(f'Generating stamps for exposure {exp+1} of {Nexp}')
            if ((self.static_psf is False) and (self.static_wcs is False)) or \
               (exp == 0):
                for obj_type in self.objects:
                    self.objects[obj_type].generate_objects(
                        exp,
                        self.config,
                        self.images[exp],
                        self.psfs[exp],
                        self.shear,
                        self.logprint,
                        ncores=self.ncores,
                        )
            else:
                # In this case, a static PSF means we can just
                # use the same stamps, after we've accounted for
                # the new image's WCS
                self.logprint('PSF & WCS are static; updating existing ' +
                              'stamps with new image positions')
                for obj_type in self.objects:
                    self.objects[obj_type].generate_objects_from_exp(
                        0, exp, self.images[exp]
                        )

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
                        obj_list.assign_mixed_grid_positions(
                            mixed_grid
                            )

                    else:
                        # mixed_grid already created & positions assigned
                        obj_list.assign_mixed_grid_positions(
                            mixed_grid
                            )

                else:
                    # An error should have already occured, but just in case:
                    raise ValueError('Position sampling type {} is not valid!'.format(gtype))

        else:
            raise TypeError('position_sampling must either be a str or dict!')

        Nobjs_all = 0
        for obj_type, obj in self.objects.items():
            Nobjs = obj.Nobjs
            self.logprint(f'{Nobjs} {obj_type} with assigned positions')
            Nobjs_all += Nobjs

        self.logprint(f'Total of {Nobjs_all} objects with assigned positions')

        return

    def fill_images(self):
        '''
        Fill each exposure with the generated objects

        NOTE: This will only work for a static PSF as written, can
        generalize in the future
        '''

        N = len(self.images)
        for i, image in enumerate(self.images):
            self.logprint(f'Filling image {i+1} of {N}')
            for obj_type, obj in self.objects.items():
                self.logprint(f'Adding {obj_type}...')
                image = self._fill_image(
                    image, obj.obj_list[i], self.logprint
                    )
            self.images[i] = image

        return

    @staticmethod
    def _fill_image(image, obj_list, logprint):
        '''
        NOTE: This version is for a non-static PSF. Not currently used

        Fill the passed image with objects from the object list.
        Multiprocessing-friendly.

        image: galsim.Image
            The GalSim image to add objects to
        obj_list: list of tuples
            The collated make_obj_runner outputs (i, stamp, truth)
        logprint: utils.LogPrint
            A LogPrint instance
        '''

        for i, stamp, truth in obj_list:

            if (stamp is None) or (truth is None):
                continue

            # Find the overlapping bounds:
            bounds = stamp.bounds & image.bounds

            # Finally, add the stamp to the full image.
            try:
                image[bounds] += stamp[bounds]
            except galsim.errors.GalSimBoundsError as e:
                logprint(e)

        return image

    def add_noise(self):
        '''
        TODO: Generalize a bit! For now, we do the same thing as
        the original imsim generation
        '''

        exp_time = self.config['observation']['exp_time']

        gain = self.config['detector']['gain']

        sky_bkg = self.config['noise']['sky_bkg']
        read_noise = self.config['noise']['read_noise']
        dark_current = self.config['noise']['dark_current']

        Nim = len(self.images)
        noise_seed = self.seeds['noise']
        noise_seeds = utils.generate_seeds(Nim, master_seed=noise_seed)

        for i, z in enumerate(zip(self.images, noise_seeds)):
            self.logprint(f'Adding noise for image {i+1} of {Nim}')

            image, seed = z[0], z[1]

            dark_noise = dark_current * exp_time
            image += dark_noise

            noise = galsim.CCDNoise(
                sky_level=sky_bkg,
                gain=gain,
                read_noise=read_noise,
                rng=galsim.BaseDeviate(seed)
                )

            image.addNoise(noise)

        return

    def add_weights(self, default_val=1, dtype=np.dtype('f4')):
        '''
        NOTE: Right now, do the simplest thing: equal weights across image

        ext_name: str
            The name of the extension to build & save as an attribute
        default_val: int, float
            The default value to fill the extension plane with
        dtype: numpy.dtype
            A numpy dtype to set, if desired
        '''

        self._add_ext('weights', default_val, dtype=dtype)

        return

    def add_masks(self, default_val=0, dtype=np.dtype('i4')):
        '''
        NOTE: Right now, do the simplest thing: equal masks across image

        ext_name: str
            The name of the extension to build & save as an attribute
        default_val: int, float
            The default value to fill the extension plane with
        dtype: numpy.dtype
            A numpy dtype to set, if desired
        '''

        self._add_ext('masks', default_val, dtype=dtype)

        return

    def _add_ext(self, ext_name, default_val, dtype=None):
        '''
        ext_name: str
            The name of the extension to build & save as an attribute
        default_val: int, float
            The default value to fill the extension plane with
        dtype: numpy.dtype
            A numpy dtype to set, if desired
        '''

        ext = []
        for i, image in enumerate(self.images):
            ext_array = np.empty(image.array.shape, dtype=dtype)
            ext_array.fill(default_val)
            ext.append(ext_array)

        setattr(self, ext_name, ext)

        return

    def write_images(self):
        '''
        At this stage, images are ready for writing. Any additional
        extensions such as weights or masks are expected to have been
        run by this stage
        '''

        outdir = self.outdir
        run_name = self.run_name
        overwrite = self.overwrite

        for i, image in enumerate(self.images):
            # to match the old imsim convention:
            outnum = str(i).zfill(3)
            fname = f'{run_name}_{outnum}.fits'
            outfile = os.path.join(outdir, fname)

            if os.path.exists(outfile):
                self.logprint(f'{outfile} already exists')
                if overwrite is True:
                    self.logprint('Deleting as overwrite=True...')
                    os.remove(outfile)
                else:
                    self.logprint('Skipping as overwrite=False...')
                    continue

            try:
                with fitsio.FITS(outfile, 'rw') as out:
                    # no longer can pass an extension explicitly. We instead
                    # create each extension in order, with the following def:
                    # ext0: image
                    # ext1: weight
                    # ext2: mask

                    out.write(image.array)

                    try:
                        weight = self.weights[i]
                        out.write(weight)
                    except Exception:
                        logprint(f'Weight writing failed for image {i}; ' +
                                 'skipping')
                    try:
                        mask = self.masks[i]
                        out.write(mask)
                    except Exception:
                        logprint(f'Mask writing failed for image {i}; ' +
                                 'skipping')
            except OSError as e:
                self.logprint(e)
                self.logprint(f'Skipping writing for image {i}')

        return

    def build_truth_cat(self):

        Nprocessed = 0

        truth_cats = []
        for obj_type, obj in self.objects.items():
            # NOTE: the truth val is the same across all exposures
            for i, stamp, truth in obj.obj_list[0]:
                # add a unique ID to each obj
                if (stamp is None) or (truth is None):
                    continue

                truth['id'] = truth[f'obj_index'] + Nprocessed
                truth_cats.append(truth)

            Nprocessed += obj.Nobjs

        self.truth_cat = vstack(truth_cats)

        return

    def write_truth_cat(self):

        outdir = self.outdir
        run_name = self.config['run_options']['run_name']
        overwrite = self.config['run_options']['overwrite']

        fname = f'{run_name}_truth.fits'
        outfile = os.path.join(outdir, fname)

        if os.path.exists(outfile):
            # other case is handled below
            if overwrite is True:
                self.logprint(f'{outfile} already exists')
                self.logprint('Deleting as overwrite=True...')
                os.remove(outfile)

        try:
            self.truth_cat.write(outfile, overwrite=overwrite)

        except OSError as e:
            self.logprint(e)
            self.logprint('Skipping writing for image {i}')

        return

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
