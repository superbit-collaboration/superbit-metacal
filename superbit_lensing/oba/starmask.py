from pathlib import Path
from glob import glob
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
import fitsio
import matplotlib.pyplot as plt

from superbit_lensing import utils
from bitmask import OBA_BITMASK, OBA_BITMASK_DTYPE
import superbit_lensing.oba.oba_io as oba_io

import ipdb

class StarmaskRunner(object):
    '''
    Runner class for generating the brigh star mask given calibrated,
    astrometrically-registered SuperBIT science images for the
    onboard analysis (OBA)

    NOTE: At this stage, input calibrated images should have the
    following structure:

    ext0: SCI (calibrated & background-subtracted)
    ext1: WGT (weight; 0 if masked, 1/sky_var otherwise)
    ext2: MSK (mask; see bitmask.py for def)
    ext3: BKG (background)
    '''

    _name = 'starmask'

    # The following class variables define the relationship between a bright
    # star's flux and the mask component sizes, in pixels. This is based on
    # parameterized fits to high-resolution images of the SB PSF model,
    # both on the diffraction spike (to set rectangular masks) and off (to
    # set the aperture mask). The model used was the following:
    # y = a + b*(x - c)^d
    # NOTE: the defined model must be a monotonic function!
    _mask_aperture_min_rad = 0.564 # radius in arcsec
    _mask_aperture_max_rad = 60 # radius in arcsec
    _mask_aperture_model_a = 0.0 # arcsec
    # _mask_aperture_model_b = 0.01082124
    _mask_aperture_model_b = 0.00468124
    # _mask_aperture_model_c = -2.20266
    _mask_aperture_model_c = -1.15214821
    # _mask_aperture_model_d = -2.5
    _mask_aperture_model_d = -2.49815

    _mask_spike_min_len = 1.41 # len (from center) in arcsec
    _mask_spike_max_len = None # len (from center) in arcsec
    _mask_spike_width = 8
    # _mask_spike_model_b = 0.00633375
    # _mask_spike_model_c = -1.75300
    # _mask_spike_model_d = -2.25

    # NOTE: older model
    # _mask_spike_model_a = 0.0
    # _mask_spike_model_b = 0.00118493
    # _mask_spike_model_c = 0.06926362
    # _mask_spike_model_d = -1.94172

    # NOTE: trying out replaced spike model
    _mask_spike_model_a = 0
    _mask_spike_model_b = 10635.139 / 7855385
    # _mask_spike_model_c = -0.017945
    _mask_spike_model_c = -0.0692659
    _mask_spike_model_d = -2

    # Default estimated image noise properties, if none are passed.
    # These come from Ajay's paper:
    # https://arxiv.org/pdf/2010.05145.pdf
    _default_sky_bkg_u = 0.061392424904074046 # ADU/s
    _default_sky_bkg_b = 0.15119519454026975 # ADU/s
    _default_sky_bkg_g = 0.1521530552782947 # ADU/s
    _default_sky_bkg_r = 0.08715579486254806 # ADU/s
    _default_sky_bkg_lum = 0.2450453811510681 # ADU/s
    _default_sky_bkg_nir = 0.18627355436655327 # ADU/s

    pixel_scale = 0.141 # arcsec / pixel
    gain_key = 'GAIN'

    def __init__(self, run_dir, gaia_cat, bands, target_name=None,
                 flux_col_base='flux_adu', flux_threshold=1e6,
                 sci_ext=0, wgt_ext=1, msk_ext=2, bkg_ext=3):
        '''
        run_dir: pathlib.Path
            The OBA run directory for the given target
        gaia_cat: pathlib.Path
            The path to a reference GAIA catalog that covers the
            target field, with stellar fluxes in ADU/s precomputed
            for SuperBIT filters
        bands: list of str's
            A list of band names
        target_name: str
            The name of the target. Default is to use the end of
            run_dir
        flux_col_base: str
            The base of the column name for the fluxes to use in
            the passed gaia_cat; will have "_{band}_s" appended to it
            NOTE: As specified above, flux units must be ADU/s!
        flux_threshold: float
            The total flux threshold for what stars to mask (in ADU)
        sci_ext: int
            The fits extension of the sci image
        wgt_ext: int
            The fits extension of the weight image
        msk_ext: int
            The fits extension of the mask image
        bkg_ext: int
            The fits extension of the background image
        '''

        args = {
            'run_dir': (run_dir, Path),
            'gaia_cat': (gaia_cat, Path),
            'bands': (bands, list),
            'flux_col_base': (flux_col_base, str),
            'flux_threshold': (flux_threshold, (int, float)),
            'sci_ext': (sci_ext, int),
            'wgt_ext': (wgt_ext, int),
            'msk_ext': (msk_ext, int),
            'bkg_ext': (bkg_ext, int),
        }

        for name, tup in args.items():
            val, allowed_types = tup
            utils.check_type(name, val, allowed_types)
            setattr(self, name, val)

        self.bands = bands
        for b in self.bands:
            utils.check_type(b, b, str)

        if target_name is None:
            target_name = run_dir.name

        utils.check_type('target_name', target_name, str)
        self.target_name = target_name

        # this dictionary will store the sci_cal image paths indexed by band
        self.images = {}

        # this dictionary will store a list of bright stars to be masked,
        # indexed by sci image path
        self.bright_stars = {}

        # this dictionary will store the estimated background noise per image,
        # indexed by sci image path
        self.sky_noise = {}

        # we leave this definition to the OBA_BITMASK
        self.mask_val = OBA_BITMASK['bright_star']

        # NOTE: can generalize if needed
        self.ra_tag  = 'ra'
        self.dec_tag = 'dec'
        self.flux_tag_base = 'flux_adu' # this is for actual ADU given exp time

        # NOTE: this will get setup later when we know the target pos!
        self.stars = None
        self.Nstars = 0

        return

    def go(self, logprint, rerun=False, overwrite=False):
        '''
        Create masks for bright GAIA stars in the target field. Requires
        images to have been astrometrically registered with a WCS in the
        image headers

        Currently planned steps:

        (1) Register input images & check for WCS solution
        (2) Setup a conservative stellar catalog to consider, built from GAIA
            and truncated for the target position. Guess at SB fluxes
        (3) Determine bright star list to mask for the given input GAIA
            catalog and target field, as well as estimated bkg noise
        (4) Build mask templates given star mag/flux in the relevant
            SuperBIT filters
        (5) Apply masks to the WGT & MSK extension of the input images

        logprint: utils.LogPrint
            A LogPrint instance for simultaneous logging & printing
        rerun: bool
            Set to re-run the astrometry if a WCS solution is already present
            in the image headers
        overwrite: bool
            Set to overwrite existing files
        '''

        logprint('Gathering images...')
        self.gather_images(logprint)

        logprint('Setting up star catalog...')
        self.setup_star_cat(logprint)

        logprint('Finding bright stars to mask...')
        self.find_bright_stars(logprint)

        logprint('Building mask templates...')
        self.build_mask_templates(logprint)

        logprint('Updating image weights & masks...')
        self.update_exts(logprint)

        return

    def gather_images(self, logprint):
        '''
        logprint: utils.LogPrint
            A LogPrint instance for simultaneous logging & printing
        '''

        for band in self.bands:
            logprint(f'Starting band {band}')

            cal_dir = (self.run_dir / band / 'cal/').resolve()
            bindx = oba_io.band2index(band)

            self.images[band] = glob(
                str(cal_dir / f'{self.target_name}*_{bindx}_*_cal.fits')
                )

            Nimages = len(self.images[band])
            logprint(f'Found {Nimages} images')

            # to keep consistent convention with other modules, store as Paths
            for i, image in enumerate(self.images[band]):
                image = Path(image)
                self.images[band][i] = image

        return

    def setup_star_cat(self, logprint):
        '''
        Setup the star catalog to be used given the input gaia_cat as a base.
        Make band and position cuts to dramatically reduce memory load

        logprint: utils.LogPrint
            A LogPrint instance for simultaneous logging & printing
        '''

        logprint(f'Setting up the star catalog given input {self.gaia_cat}')

        stars = Table.read(self.gaia_cat)
        logprint(f'Input GAIA catalog has {len(stars)} stars')

        # NOTE: don't have a super robust way to do this, but assume the
        # TRG_{RA/DEC} is constant across images (or at least very close),
        # so just pick one
        for band in self.bands:
            if len(self.images[band]) != 0:
                image = self.images[self.bands[0]][0]
                break
        hdr = fitsio.read_header(str(image))
        target_ra, target_dec = hdr['TRG_RA'], hdr['TRG_DEC']
        logprint(f'Target is at ({target_ra:.5f}, {target_dec:.5f})')

        # first, remove stars not sufficiently near the target
        target_pos = SkyCoord(ra=target_ra*u.deg, dec=target_dec*u.deg)

        # NOTE: already has unit (deg)
        ra = stars[self.ra_tag]
        dec = stars[self.dec_tag]

        stellar_pos = SkyCoord(ra=ra, dec=dec)

        separation = target_pos.separation(stellar_pos)

        # we keep stars that are within 1.5x the longer CCD length
        max_sep = 0.384 * 1.5 * u.deg
        logprint(f'Max separation: {max_sep:.6f}')
        stars = stars[separation < max_sep]
        logprint(f'{len(stars)} stars remaining after separation cut')

        logprint('Estimating SuperBIT fluxes for GAIA stars')
        stars = self._add_sb_star_fluxes(stars, logprint)

        self.stars = stars
        self.Nstars = len(stars)

        return

    def _add_sb_star_fluxes(self, stars, logprint):
        '''
        Make a best-guess of the SuperBIT stellar fluxes in e/s
        given the 2 GAIA bands

        stars: astropy.Table
            A table of GAIA stars
        logprint: utils.LogPrint
            A LogPrint instance for simultaneous logging & printing
        '''

        # we use this for the OBA sims, but useful here as well
        from superbit_lensing.oba.sims import instrument as inst
        from superbit_lensing.oba.sims import photometry as phot

        camera = inst.Camera('imx455')
        telescope = inst.Telescope('superbit')
        bandpass = inst.Bandpass('bandpass')
        bandpass.wavelengths = camera.wavelengths

        piv_dict = {
            'u': 395.35082727585194,
            'b': 476.22025867791064,
            'g': 596.79880208687230,
            'r': 640.32425638312820,
            'nir': 814.02475812251110,
            'lum': 522.73829660009810
        }

        for band in self.bands:
            logprint(f'Starting band {band}')
            piv_wave = piv_dict[band]

            if piv_wave > 640:
                gaia_mag = 'phot_rp_mean_mag'
            else:
                gaia_mag = 'phot_bp_mean_mag'

            bandpass_transmission = phot.get_transmission(band=band)

            # Find counts to add for the star
            logprint('Computing abmag_to_mean_fnu...')
            mean_fnu_star_mag = phot.abmag_to_mean_fnu(
                abmag=stars[gaia_mag].value
                )

            logprint('Computing mean_flambda_from_mean_fnu...')
            mean_flambda = phot.mean_flambda_from_mean_fnu(
                mean_fnu=mean_fnu_star_mag,
                bandpass_transmission=bandpass_transmission,
                bandpass_wavelengths=bandpass.wavelengths
                )

            logprint('Computing crate_from_mean_flambda...')
            crate_electrons_pix = phot.crate_from_mean_flambda(
                mean_flambda=mean_flambda,
                illum_area=telescope.illum_area.value,
                bandpass_transmission=bandpass_transmission,
                bandpass_wavelengths=bandpass.wavelengths
                )

            # store as electrons per second, to be converted to ADU per image
            # given the header GAIN
            stars[f'flux_electrons_{band}_s'] = crate_electrons_pix

        return stars

    def find_bright_stars(self, logprint):
        '''
        logprint: utils.LogPrint
            A LogPrint instance for simultaneous logging & printing
        '''

        Nstars = self.Nstars

        thresh = self.flux_threshold

        for band in self.bands:
            logprint(f'Starting band {band}')

            images = self.images[band]
            gaia_flux_col = f'flux_electrons_{band}_s' # e-/s
            sb_flux_col = f'{self.flux_tag_base}_{band}' # ADU

            for image in images:
                im_pars = oba_io.parse_image_file(image, 'sci')
                exp_time = im_pars['exp_time']
                gain = fitsio.read_header(str(image))[self.gain_key]

                # estimate the stellar flux in this band with the given gain
                # and exposure time
                star_fluxes = self.stars[gaia_flux_col] / gain * exp_time

                # only grab stars above the set threshold
                bright_indices = np.where(star_fluxes > thresh)
                bright_stars = self.stars[bright_indices]
                bright_stars[sb_flux_col] = star_fluxes[bright_indices]
                self.bright_stars[image] = bright_stars

                Nbright = len(self.bright_stars[image])
                logprint(f'Found {Nbright} stars of {Nstars} in image {image.name}')

        return

    def build_mask_templates(self, logprint):
        '''
        TODO: Come up with a more empirical model!

        logprint: utils.LogPrint
            A LogPrint instance for simultaneous logging & printing
        '''

        # TODO/NOTE: For quick testing, we make a very simple template
        # relation w/ flux

        pass

    def update_exts(self, logprint):
        '''
        Update the image weights & masks

        logprint: utils.LogPrint
            A LogPrint instance for simultaneous logging & printing
        '''

        sci_ext = self.sci_ext
        wgt_ext = self.wgt_ext
        msk_ext = self.msk_ext
        bkg_ext = self.bkg_ext

        for band in self.bands:
            logprint(f'Starting band {band}')

            Nimages = len(self.images[band])
            for k, image in enumerate(self.images[band]):
                logprint(f'Starting image {image.name} ({k+1} of {Nimages})')

                with fitsio.FITS(str(image), 'rw') as fits:
                    sci_hdr = fits[sci_ext].read_header()
                    wgt_hdr = fits[wgt_ext].read_header()
                    msk_hdr = fits[msk_ext].read_header()

                    msk = fits[msk_ext].read()
                    wgt = fits[wgt_ext].read()
                    bkg = fits[bkg_ext].read()

                    wcs = WCS(sci_hdr)

                    stars = self.bright_stars[image]

                    # estimate the sky var of this image using the median of
                    # the inv_wgt**2
                    # NOTE: want to ignore pixels w/ no weight
                    sky_var = 1. / wgt[wgt != 0]
                    sky_noise = np.median(np.sqrt(sky_var))
                    sky_level = np.median(bkg)
                    logprint(f'Using an estimated sky noise of {sky_noise:.3f} '
                             f'ADU; sky level = {sky_level:.2f}')

                    Nstars = len(stars)
                    for i, star in enumerate(stars):
                        flux = star[f'{self.flux_tag_base}_{band}'] # ADU
                        logprint(f'Starting bright star {i+1} of {Nstars}; ' +
                                 f'flux={flux:.1f} ADU')
                        self._add_star_mask(star, msk, wgt, wcs, band, sky_noise)

                    # now update FITs weight & mask extensions
                    fits[wgt_ext].write(msk, header=msk_hdr)
                    fits[msk_ext].write(msk, header=msk_hdr)

        return

    def _add_star_mask(self, star, msk, wgt, wcs, band, noise, origin=0):
        '''
        Add a bright star mask for the given star on a given image

        star: np.recarray / astropy.Table row
            The properties of the star
        msk: np.ndarray
            The image's mask array that we want to update
        wgt: np.ndarray
            The image's weight array that we want to update (new masks as 0's)
        wcs: astropy.wcs.WCS
            The image's WCS
        band: str
            The name of the image band
        noise: float
            The estimated sky noise of the image, in ADU
        origin: int
            The origin to use for the WCS
        '''

        # the size of the mask depends on a re-scaling of a conservative model
        # of the PSF shape by the stellar flux, and comparing it to the
        # estimated noise level of the the image of a given band
        flux = star[f'{self.flux_tag_base}_{band}'] # ADU
        aperture_size, spike_size, spike_width = self._get_mask_sizes(flux, noise)

        # The search radius to check for masking around a stellar
        # position, in pixels

        search_radius = int(np.ceil(np.max([
            aperture_size, spike_size
        ])))

        mask_val = self.mask_val

        ra = star[self.ra_tag] * u.deg
        dec = star[self.dec_tag] * u.deg
        star_world = SkyCoord(ra, dec)

        # origin is 0 as we will be working on numpy arrays
        star_im = wcs.all_world2pix(
            # star_world.ra.deg, star_world.dec.deg, 0
            ra, dec, origin
        )

        # rounded to nearest pixel
        ra_im_pix = int(np.round(star_im[0]))
        dec_im_pix = int(np.round(star_im[1]))

        # NOTE: Nx & Ny are flipped due to numpy vs. FITS
        # indexing conventions
        Nx = msk.shape[1]
        Ny = msk.shape[0]
        # startx = origin
        # starty = origin
        # endx = startx + Nx
        # endy = starty + Ny
        # x = np.arange(startx, endx)
        # y = np.arange(starty, endy)
        # X, Y = np.meshgrid(x, y)

        rad = search_radius # pix
        startx = np.max([0, ra_im_pix - rad])
        endx = np.min([Nx, ra_im_pix + rad])
        starty = np.max([0, dec_im_pix - rad])
        endy = np.min([Ny, dec_im_pix + rad])

        # will be easier if the stamp indexing stars from 0
        stamp_Nx = endx - startx
        stamp_Ny = endy - starty

        # x = np.arange(0, stamp_Nx, dtype=int)
        # y = np.arange(0, stamp_Ny, dtype=int)
        x = np.arange(startx, endx, dtype=int)
        y = np.arange(starty, endy, dtype=int)

        if (len(x) == 0) or (len(y) == 0):
            # no pixels overlap with the image
            return

        # NOTE: by using ij indexing, we can treat the
        # index elements as array(x,y) -> array[i,j]
        X, Y = np.meshgrid(x, y, indexing='ij')

        # start with aperture mask
        diff_x = X - star_im[0]
        diff_y = Y - star_im[1]
        dist = np.sqrt(diff_x**2 + diff_y**2)
        in_aperture = np.where(dist < aperture_size)

        # account for stamp offset, and index flip
        in_aperture = (
            in_aperture[1] + starty,
            in_aperture[0] + startx,
            )
        msk[in_aperture] = mask_val

        # now do spike mask
        right = np.where(
            (diff_x > 0) & (diff_x < spike_size) &
            (abs(diff_y) < spike_width/2.)
            )
        top = np.where(
            (diff_y > 0) & (diff_y < spike_size) &
            (abs(diff_x) < spike_width/2.)
            )
        left = np.where(
            (diff_x > -spike_size) & (diff_x < 0) &
            (abs(diff_y) < spike_width/2.)
            )
        bottom = np.where(
            (diff_y > -spike_size) & (diff_y < 0) &
            (abs(diff_x) < spike_width/2.)
            )

        # account for stamp offset, and index flip
        right = (
            right[1] + starty,
            right[0] + startx,
            )
        left = (
            left[1] + starty,
            left[0] + startx,
            )
        top = (
            top[1] + starty,
            top[0] + startx,
            )
        bottom = (
            bottom[1] + starty,
            bottom[0] + startx,
            )

        msk[right] = mask_val
        msk[left] = mask_val
        msk[top] = mask_val
        msk[bottom] = mask_val

        # for the weights, make them 0 inside the star mask
        wgt[in_aperture] = 0.
        wgt[right] = 0.
        wgt[left] = 0.
        wgt[top] = 0.
        wgt[bottom] = 0.

        return

    def _get_mask_sizes(self, flux, noise):
        '''
        Determine the stellar mask size given the stellar flux using the fitted
        relation defined at the top of the class

        The size of the mask depends on a re-scaling of a conservative model
        of the PSF shape by the stellar flux, and comparing it to the
        estimated noise level of the the image of a given band

        flux: float
            The value of the stellar flux
        noise: float
            The mean value of the sky noise in the image

        Returns: 3-tuple
            Return the following tuple: (aperture_size, spike_len, spike_width)
        '''

        aperture_size = self._compute_aperture_rad(flux, noise)
        spike_len = self._compute_spike_len(flux, noise)

        spike_width = self._mask_spike_width

        return (aperture_size, spike_len, spike_width)

    @classmethod
    def _compute_spike_len(cls, flux, noise):
        '''
        Compute the stellar aperture mask radius using fits to the following
        model:

        star_profile = flux * [a + b*(radius-c)^d]

        where radius is the distance from the stellar centroid to a position
        on the spike

        flux: float
            The stellar flux in a given band, in ADU
        noise: float
            The mean value of the sky noise in the image
        '''

        a = cls._mask_spike_model_a
        b = cls._mask_spike_model_b
        c = cls._mask_spike_model_c
        d = cls._mask_spike_model_d

        min_len = cls._mask_spike_min_len
        max_len = cls._mask_spike_max_len

        if max_len is None:
            # use a *very* conservative default guess of 2 arcmin
            max_len = 120. # arcsec

        # sample at each pixel
        pix_scale = cls.pixel_scale
        max_pixel = int(np.ceil(max_len / pix_scale))
        pixels = np.arange(1, max_pixel+1)
        radii = pixels * pix_scale
        Nradii = len(radii)

        # use our conservative model estimate of the smoothed PSF profile,
        # scaled by the stellar flux
        psf_profile = lambda r, a, b, c, d: a + b*(r - c)**d
        star_profile = flux * psf_profile(radii, a, b, c, d)

        # we want to compare the star profile (*not* along the diffraction
        # spikes) to the noise level of the given band

        r0, y0 = cls.find_intersection(
            radii, star_profile, np.array(Nradii*[noise])
            )

        if r0 is None:
            spike_len = 0
        else:
            spike_len = r0

        if min_len is not None:
            if spike_len < min_len:
                spike_len = min_len
        if max_len is not None:
            if spike_len > max_len:
                spike_len = max_len

        return spike_len

    @classmethod
    def _compute_aperture_rad(cls, flux, noise):
        '''
        Compute the stellar aperture mask radius using fits to the following
        model:

        star_profile = flux * [a + b*(radius-c)^d]

        where radius is the distance from the stellar centroid to a position
        *off* of the spike

        flux: float
            The stellar flux in a given band, in ADU
        noise: float
            The mean value of the sky noise in the image
        '''

        a = cls._mask_aperture_model_a
        b = cls._mask_aperture_model_b
        c = cls._mask_aperture_model_c
        d = cls._mask_aperture_model_d

        min_rad = cls._mask_aperture_min_rad
        max_rad = cls._mask_aperture_max_rad

        if max_rad is None:
            # use a *very* conservative default guess of 0.5 arcmin
            max_rad = 30. # arcsec

        # sample at each pixel
        pix_scale = cls.pixel_scale
        max_pixel = int(np.ceil(max_rad / pix_scale))
        pixels = np.arange(1, max_pixel+1)
        radii = pixels * pix_scale
        Nradii = len(radii)

        # use our conservative model estimate of the smoothed PSF profile,
        # scaled by the stellar flux
        psf_profile = lambda r, a, b, c, d: a + b*(r - c)**d
        star_profile = flux * psf_profile(radii, a, b, c, d)

        # we want to compare the star profile (*not* along the diffraction
        # spikes) to the noise level of the given band

        r0, y0 = cls.find_intersection(
            radii, star_profile, np.array(Nradii*[noise])
            )

        if r0 is None:
            aperture_rad = 0
        else:
            aperture_rad = r0

        if min_rad is not None:
            if aperture_rad < min_rad:
                aperture_rad = min_rad
        if max_rad is not None:
            if aperture_rad > max_rad:
                aperture_rad = max_rad

        return aperture_rad

    @staticmethod
    def find_intersection(x, y1, y2):
        '''
        Find the intersection (x0, y0) for the two input curves sampled at
        the same points.

        NOTE: Assumes only one intersection given the modeling choices for the
        stellar masking
        '''

        indx = np.argwhere(np.diff(np.sign(y1 - y2))).flatten()

        if len(indx) == 0:
            # no intersection found! This is more likely to happen for
            # lower flux than higher flux stars
            return None, None
        elif len(indx) > 1:
            # NOTE: this shouldn't happen for a monotonic modelling function!
            Nintersections = len(indx)
            raise ValueError(f'{len(indx)} intersections were found - are you '
                             'sure you defined a monotonic function for the PSF '
                             'profile?')

        # linearly interpolate between the two estimates
        x0 = x[indx][0]
        y0_1 = y1[indx][0]
        y0_2 = y2[indx][0]
        y0 = np.mean([y0_1, y0_2])

        return x0, y0
