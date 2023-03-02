from pathlib import Path
from astropy.wcs import WCS
import astropy.units as u
import fitsio

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

    # the following defines the relationship between the a bright star's
    # flux and the mask component sizes, in pixels
    _mask_aperture_min_rad = 10 # radius
    _mask_aperture_max_rad = 100 # radius
    _mask_aperture_slope = 1 # pixels / ADU

    _mask_spike_min_len = 20
    _mask_spike_max_len = 200
    _mask_spike_slope = 1 # pixels / ADU

    def __init__(self, run_dir, gaia_cat, bands, target_name=None,
                 flux_col_base='flux_adu', flux_threshold=1e4,
                 msk_ext=2):
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
            the passed gaia_cat; will have "_{band}" appended to it
            NOTE: As specified above, flux units must be ADU/s!
        flux_threshold: float
            The flux threshold (in ADU )
        msk_ext: int
            The fits extension of the mask image
        '''

        args = {
            'run_dir': (run_dir, Path),
            'gaia_cat': (gaia_cat, Path),
            'bands': (bands, list),
            'flux_col_base': (flux_col_base, str),
            'flux_threshold': (flux_threshold, (int, float)),
            'msk_ext': (msk_ext, int)
        }

        for name, tup in args.items():
            val, allowed_types = tup
            utils.check_type(name, val, allowed_types)
            setattr(self, name, val)

        if target_name is None:
            target_name = run_dir.name

        utils.check_type('target_name', target_name, str)
        self.target_name = target_name

        self.stars = fitsio.read(str(self.gaia_cat))

        # this dictionary will store the sci_cal image paths indexed by band
        self.images = {}

        # this dictionary will store a list of bright stars to be masked,
        # indexed by sci image path
        self.bright_stars = {}

        # we leave this definition to the OBA_BITMASK
        self.mask_val = OBA_BITMASK['bright_star']

        # NOTE: can generalize if needed
        self.ra_tag  = 'RA_ICRS'
        self.dec_tag = 'DE_ICRS'

        return

    def go(self, logprint, rerun=False, overwrite=False):
        '''
        Create masks for bright GAIA stars in the target field. Requires
        images to have been astrometrically registered with a WCS in the
        image headers

        Currently planned steps:

        (1) Register input images & check for WCS solution
        (2) Determine bright star list to mask for the given input GAIA
            catalog and target field
        (3) Build mask templates given star mag/flux in the relevant
            SuperBIT filters
        (4) Apply masks to the MSK extension of the input images

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

        logprint('Finding bright stars to mask...')
        self.find_bright_stars(logprint)

        logprint('Building mask templates...')
        self.build_mask_templates(logprint)

        logprint('Updating image masks...')
        self.update_masks(logprint)

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

    def find_bright_stars(self, logprint):
        '''
        logprint: utils.LogPrint
            A LogPrint instance for simultaneous logging & printing
        '''

        thresh = self.flux_threshold

        for band in self.bands:
            logprint(f'Starting band {band}')

            images = self.images[band]
            flux_col = f'{self.flux_col_base}_{band}'

            for image in images:
                im_pars = oba_io.parse_image(image, 'sci')
                exp_time = im_pars['exp_time']

                star_fluxes = self.stars[flux_col] * exp_time

                bright_stars = self.stars[star_fluxes > thresh]
                self.bright_stars[image] = bright_stars

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

    def update_masks(self, logprint):
        '''
        logprint: utils.LogPrint
            A LogPrint instance for simultaneous logging & printing
        '''

        msk_ext = self.msk_ext

        for band in self.bands:
            logprint(f'Starting band {band}')

            for image in self.images[band]:
                logprint(f'Starting image {image.name}')

                hdr = fitsio.read_header(str(image))
                msk = fitsio.read(str(image), ext=msk_ext)

                wcs = WCS(hdr)

                stars = self.bright_stars[image]

                for star in stars:
                    self._add_mask(star, msk, wcs, band)

        return

    def _add_mask(self, star, msk, wcs, band):
        '''
        Add a bright star mask for the given star on a given image

        star: np.recarray / astropy.Table row
            The properties of the star
        msk: np.ndarray
            The image's mask array that we want to update
        wcs: astropy.wcs.WCS
            The image's WCS
        band: str
            The name of the image band
        '''

        mask_val = self.mask_val

        ra = star[self.ra_tag] * u.deg
        dec = star[self.dec_tag] * u.deg
        star_world = SkyCoord(ra, dec)

        ipdb.set_trace()
        # origin is 0 as we will be working on numpy arrays
        star_im = wcs.world_to_image(star_world, origin=0)

        flux = star[f'{self.flux_col_base}_{band}']

        # NOTE: Nx & Ny are flipped due to numpy vs. FITS
        # indexing conventions
        Nx = image.shape[1]
        Ny = image.shape[0]

        startx = np.round(wcs.origin.x)
        starty = np.round(wcs.origin.y)
        endx = startx + Nx
        endy = starty + Ny

        x = np.arange(startx, endx)
        y = np.arange(starty, endy)
        X, Y = np.meshgrid(x, y)

        # start with aperture mask
        diff_x = X - star_im[0]
        diff_y = Y - star_im[1]
        dist = np.sqrt(diff_x**2 + diff_y**2)
        msk[dist < aperture_size] = mask_val

        return
