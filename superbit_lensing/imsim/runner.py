import numpy as np
import galsim
import fitsio
import os
from astropy.table import Table, vstack, hstack, join
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

        self.setup_bands()
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

    def setup_bands(self):
        '''
        The config may specify either a single band implicitly or a
        dictionary of band names and corresponding central wavelengths
        '''

        bands = self.config['bandpasses']

        self.band_indx = {}

        bindx = 0
        for b, bdict in bands.items():
            if not isinstance(b, str):
                raise TypeError('Band names must be strings!')
            if not isinstance(bdict, dict):
                raise TypeError('The bandpasses field must have a dict for '
                                'each band entry!')
            if 'lam' not in bdict:
                raise ValueError('Each bandpass field must have `lam`!')

            lam = bdict['lam']
            if (not isinstance(lam, (int, float))) or (lam <= 0):
                raise ValueError('Central wavelength lam must be positive!')

            bands[b]['index'] = bindx
            self.band_indx[b] = bindx
            bindx += 1

        self.bands = bands
        self.Nbands = len(bands)

        bnames = list(self.bands.keys())
        p = 's' if self.Nbands > 1 else ''
        self.logprint(f'Registered {self.Nbands} band{p}: {bnames}')

        self.logprint('Checking for band consistency throughout config...')

        sky_bkg = self.config['noise']['sky_bkg']
        if len(sky_bkg) != self.Nbands:
            raise ValueError('There must be the same number of entries for '
                             'bandpasses and sky backgrounds!')
        for b, val in sky_bkg.items():
            if b not in self.bands:
                raise ValueError(f'{b} in sky_bkg not a registed band!')
            if not isinstance(val, (int, float)):
                raise TypeError('Sky backgroudn values must be floats!')

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

        if self.Nbands > 1:
            # need to create independent seeds for noise & dithering
            noise_seeds = utils.generate_seeds(self.Nbands)
            dither_seeds = utils.generate_seeds(self.Nbands)

            self.logprint('As Nbands > 1, generating independent noise '
                          'and dithering seeds for each:')
            self.logprint(f'noise: {noise_seeds}')
            self.logprint(f'dithering: {dither_seeds}')
            self.seeds['noise'] = noise_seeds
            self.seeds['dithering'] = dither_seeds
        else:
            # store the noise & dither seeds as a list anyway to match
            # formatting for multi-band case
            self.seeds['noise'] = [self.seeds['noise']]
            self.seeds['dithering'] = [self.seeds['dithering']]

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
        '''
        Main simulation runner. Will setup all necessary components across
        all bands, but will call fill_images() separately
        '''

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

        # self.logprint('Wr')

        self.logprint('Writing out images...')
        self.write_images()

        self.logprint('Building truth catalog...')
        # TODO: current multi-band refactor point!
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

        # we will keep a list of images for each band
        self.images = {}

        # TODO: add dithers!
        for b in self.bands:
            self.images[b] = []
            for i in range(self.config['observation']['nexp']):
                self.images[b].append(
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

        for b in self.bands:
            for i, image in enumerate(self.images[b]):
                affine = galsim.AffineTransform(
                    dudx, dudy, dvdx, dvdy, origin=image.true_center
                    )

                self.images[b][i].wcs = galsim.TanWCS(
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

        # NOTE: We are just making a static PSF (per band) for now
        self.psfs = {}

        jitter_fwhm = self.config['psf']['jitter_fwhm']
        use_optics = self.config['psf']['use_optics']

        # first the jitter component from gondola instabilities
        jitter = galsim.Gaussian(flux=1, fwhm=jitter_fwhm)

        if use_optics is True:
            self.logprint('use_optics is True; convolving telescope optics PSF profile')

            # next, define Zernicke polynomial component for the optics
            diam = self.config['telescope']['diameter']
            obscuration = self.config['telescope']['obscuration']
            nstruts = self.config['telescope']['nstruts']
            strut_angle = self.config['telescope']['strut_angle']
            strut_thick = self.config['telescope']['strut_thick']

            # NOTE: aberrations were definined for lam = 550, and close to the
            # center of the camera. The PSF degrades at the edge of the FOV
            # TODO: generalize once we have computed this for all filters;
            # see issue #103
            aberrations = np.zeros(38)             # Set the initial size.
            aberrations[0] = 0.                       # First entry must be zero
            aberrations[1] = -0.00305127
            aberrations[4] = -0.02474205              # Noll index 4 = Defocus
            aberrations[11] = -0.01544329             # Noll index 11 = Spherical
            aberrations[22] = 0.00199235
            aberrations[26] = 0.00000017
            aberrations[37] = 0.00000004

            for band in self.bands:
                self.psfs[band] = []

                lam = self.config['bandpass']['lam']
                lam_over_diam = lam * 1.e-9 / diam    # radians
                lam_over_diam *= 206265.

                self.logprint(f'Calculated lambda over diam = '
                              f'{lam_over_diam} arcsec for band {band}')

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

                # NOTE: as stated above, static per band for now
                for i in range(len(self.images)):
                    self.psfs[band].append(psf)

        else:
            self.logprint('use_optics is False; using jitter-only PSF')
            psf = jitter

            for band in self.bands:
                self.psfs[band] = []
                for i in range(len(self.images)):
                    self.psfs[band].append(psf)

        self.static_psf = True

        return

    def setup_shear(self):

        shear_config = self.config['shear'].copy()
        shear_type = shear_config.pop('type')

        self.shear = shear.build_shear(shear_type, shear_config)

        return

    def generate_objects(self):
        '''
        Generate a list of object stamps for all registered bands

        TODO: Now that we are multiband, it would make more sense to
        generate a list of GSObjects instead of stamps, particularly for
        truth catalog generation. However, this will require a fairly
        significant refactor
        '''

        # for grids, won't know Nobjs until we finish
        # assigning positions
        self.assign_positions()

        Nexp = self.config['observation']['nexp']
        for band in self.bands:
            self.logprint(f'Starting band {band}')
            for exp in range(Nexp):
                self.logprint(f'Generating stamps for exposure {exp+1} ' +
                              f'of {Nexp}')
                if ((self.static_psf is False) and \
                    (self.static_wcs is False)) or \
                    (exp == 0):
                    for obj_type in self.objects:
                        self.objects[obj_type].generate_objects(
                            exp,
                            band,
                            self.config,
                            self.images[band][exp],
                            self.psfs[band][exp],
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
                            0, exp, band, self.images[band][exp]
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

        for band in self.bands:
            self.logprint(f'Starting band {band}')

            N = len(self.images[band])
            for i, image in enumerate(self.images[band]):
                self.logprint(f'Filling image {i+1} of {N}')
                for obj_type, obj in self.objects.items():
                    self.logprint(f'Adding {obj_type}...')
                    image = self._fill_image(
                        image, obj.obj_list[band][i], self.logprint
                        )
                self.images[band][i] = image

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

        read_noise = self.config['noise']['read_noise']
        dark_current = self.config['noise']['dark_current']

        for band in self.bands:
            self.logprint(f'Starting band {band}')

            Nim = len(self.images[band])
            noise_seed = self.seeds['noise'][self.band_indx[band]]
            noise_seeds = utils.generate_seeds(Nim, master_seed=noise_seed)
            self.logprint(f'Using master noise seed of {noise_seed}')

            sky_bkg = self.config['noise']['sky_bkg'][band]
            self.logprint(f'Using sky background of {sky_bkg}')

            for i, z in enumerate(zip(self.images[band], noise_seeds)):
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

        all_ext = {}
        for band in self.bands:
            self.logprint(f'Starting band {band}')

            ext = []
            for i, image in enumerate(self.images[band]):
                ext_array = np.empty(image.array.shape, dtype=dtype)
                ext_array.fill(default_val)
                ext.append(ext_array)

            all_ext[band] = ext

        setattr(self, ext_name, all_ext)

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

        for band in self.bands:
            self.logprint(f'Starting band {band}')

            Nim = len(self.images[band])
            for i, image in enumerate(self.images[band]):
                # to match the old imsim convention:
                outnum = str(i).zfill(3)
                fname = f'{run_name}_{outnum}_{band}.fits'
                outfile = os.path.join(outdir, fname)

                self.logprint(f'Writing image {i+1} of {Nim} to {outfile}')

                if os.path.exists(outfile):
                    self.logprint(f'{outfile} already exists')
                    if overwrite is True:
                        self.logprint('Deleting as overwrite=True...')
                        os.remove(outfile)
                    else:
                        self.logprint('Skipping as overwrite=False...')
                        continue

                try:
                    # NOTE: we do this in a somewhat strange way to
                    # automatically save the WCS to the header
                    images = [image]

                    try:
                        weight = galsim.Image(self.weights[band][i])
                        weight.wcs = image.wcs
                        images.append(weight)

                    except Exception:
                        self.logprint(f'Weight writing failed for image {i}; ' +
                                      'skipping')
                    try:
                        mask = galsim.Image(self.masks[band][i])
                        mask.wcs = image.wcs
                        images.append(mask)

                    except Exception:
                        self.logprint(f'Mask writing failed for image {i}; ' +
                                      'skipping')

                    galsim.fits.writeMulti(images, outfile)

                    # add a few header extras
                    ipdb.set_trace()
                    self.extend_header(outfile)

                except OSError as e:
                    self.logprint(e)
                    self.logprint(f'Skipping writing for image {i}')

        return

    def extend_header(self, outfile):
        '''
        Add a few extra expected keys to the simulated header
        '''

        fits = fitsio.FITS(outfile, 'rw')

        self._extend_sci_header(fits)
        self._extend_wgt_header(fits)
        self._extend_msk_header(fits)

        h = fitsio.read_header(outfile)
        print(h)
        ipdb.set_trace()

        return

    def _extend_sci_header(self, fits):
        fits[0].write_key(
            'GAIN', self.config['detector']['gain'], comment='e-/ADU'
            )

        return

    def _extend_wgt_header(self, fits):
        # fits[1] = ...
        pass

    def _extend_msk_header(self, fits):
        # fits[2] = ...
        pass

    def build_truth_cat(self):
        '''
        Build a table of truth values for all rendered objects

        NOTE: the truth values are the same across all exposures,
        but there can be slight differences across bands (e.g. flux).
        This makes the following process messy to handle cases where
        a GalSim rendering failure happens for only a subset of bands
        '''

        joined_truth_cats = None

        # not all object types will have the same truth cols, and
        # i can't seem to get
        mask_val = -1000

        for band in self.bands:
            self.logprint(f'Starting band {band}')

            truth_cats = []
            for obj_type, obj in self.objects.items():
                self.logprint(f'Building truth catalog for {obj_type}...')

                # NOTE: we don't worry about rendering failures messing up the
                # indexing here, as (for now) the only stamp changes between
                # exposures is centering offsets to account for dithering.
                # TODO: generalize when needed!
                if (self.static_psf is False) or (self.static_wcs is False):
                    raise Exception('Truth catalog generation for non-' +
                                    'static WCS or PSF is not yet implemented!')

                # truth values are the same across exposures for a given band
                for i, stamp, truth in obj.obj_list[band][0]:
                    if (stamp is None) or (truth is None):
                        truth = None
                        continue

                    truth_cats.append(truth)

            # TODO: Would really like to have masked Tables merge
            # correctly, but having trouble getting the masking to
            # work...
            truth_cats = vstack(truth_cats).filled(mask_val)

            if joined_truth_cats is None:
                joined_truth_cats = truth_cats
            else:
                # the following is complex as it seems to fail to
                # do outer merges for masked tables...
                # left_mask = joined_truth_cats.mask.as_array()
                # d = [(dtype[0], dtype[1]) for dtype in left_mask.dtype]
                # print(d)
                # left_mask.dtype = [(col, bool) for col in joined_truth_cats.columns]
                # left_mask.dtype = [(col, bool) for col in left_mask.columns]
                # right_mask = truth_cats.mask.as_array()
                # right_mask.dtype = bool
                joined_truth_cats = join(
                    joined_truth_cats,
                    truth_cats,
                    join_type='outer',
                    )

        # add a unique ID to each obj
        joined_truth_cats['id'] = np.arange(len(joined_truth_cats))

        assert len(joined_truth_cats) == len(
            np.unique(joined_truth_cats['id'])
            )

        self.truth_cat = joined_truth_cats

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
