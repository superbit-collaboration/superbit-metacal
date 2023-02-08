from pathlib import Path
from glob import glob
from astropy.wcs import WCS
import fitsio

from superbit_lensing import utils
from superbit_lensing.oba.oba_io import band2index

import ipdb

class AstrometryRunner(object):
    '''
    Runner class for registering each calibrated sci image to an
    astrometric solution for the SuperBIT onboard analysis (OBA)

    NOTE: At this stage, input calibrated images should have the
    following structure:

    ext0: SCI (calibrated & background-subtracted)
    ext1: WGT (weight; 0 if masked, 1/sky_var otherwise)
    ext2: MSK (mask; 1 if masked, 0 otherwise)
    ext3: BKG (background)
    '''

    def __init__(self, run_dir, bands, target_name=None):
        '''
        run_dir: pathlib.Path
            The OBA run directory for the given target
        bands: list of str's
            A list of band names
        target_name: str
            The name of the target. Default is to use the end of
            run_dir
        '''

        args = {
            'run_dir': (run_dir, Path),
            'bands': (bands, list),
        }

        for name, tup in args.items():
            val, allowed_types = tup
            utils.check_type(name, val, allowed_types)
            setattr(self, name, val)

        if target_name is None:
            target_name = run_dir.name

        utils.check_type('target_name', target_name, str)
        self.target_name = target_name

        # this dictionary will store the sci_cal image paths indexed by band
        self.images = {}

        # this dictionary will store the WCS solution for each image, indexed
        # by sci_cal filename
        self.wcs_solutions = {}

        return

    def go(self, logprint, rerun=False, overwrite=False):
        '''
        Solve for the astrometric solution of each image. There may already
        be a WCS solution in the image headers from the image-checker, but
        can optionally ignore it and rerun on the fully calibrated images

        Currently planned steps:

        (1) Register input images
        (2) Check for existing WCS solution
        (3) Run Astrometry.net
        (4) If unsuccessful, modify image (e.g. filtering) and try again

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

        # NOTE: check for existing WCS solutions is done inside method
        logprint('Registering images...')
        self.register_images(logprint)

        return

    def gather_images(self, logprint):
        '''
        logprint: utils.LogPrint
            A LogPrint instance for simultaneous logging & printing
        '''

        for band in self.bands:
            logprint(f'Starting band {band}')

            cal_dir = (self.run_dir / band / 'cal/').resolve()
            bindx = band2index(band)

            self.images[band] = glob(
                str(cal_dir / f'{self.target_name}*_{bindx}_*_cal.fits')
                )

            Nimages = len(self.images[band])
            logprint(f'Found {Nimages} images')

            # to keep consistent convention with other modules, store as Paths
            for i, image in enumerate(self.images[band]):
                image = Path(image)
                self.images[band][i] = image
                self.wcs_solutions[image] = None

        return

    def register_images(self, logprint, rerun=False):
        '''
        Register all target images using Astrometry.net

        If rerun is False, check to see if target images already have
        an existing WCS solution

        logprint: utils.LogPrint
            A LogPrint instance for simultaneous logging & printing
        rerun: bool
            Set to re-run the astrometry if a WCS solution is already present
            in the image headers
        '''

        if rerun is True:
            logprint('Ignoring existing WCS solutions in image headers as ' +
                     'rerun is True')

        for band in self.bands:
            logprint(f'Starting band {band}')

            images = self.images[band]

            Nimages = len(images)
            for i, image in enumerate(images):
                image_name = image.name
                logprint(f'Starting {image_name}; {i+1} of {Nimages}')

                if rerun is False:
                    wcs = self.check_for_wcs(image)

                    if wcs is not None:
                        logprint('Existing WCS header found in image header; ' +
                                 'skipping')
                        # while we'll add it to the dict, don't write to header
                        # as it already exists there
                        self.wcs_solutions[image] = wcs
                        continue

                # TODO: Implement actual astrometry.net running!
                logprint('WARNING: Astrometric registration not yet implemented!')
                # wcs = ...
                # self.wcs_solutions[image] = wcs

        return

    def check_for_wcs(self, image_file, wcs_ext=0):
        '''
        Check image_file header for an existing WCS solution to inherit without
        re-running Astrometry.net

        image_file: str, pathlib.Path
            Path of fits image_file file
        wcs_ext: int
            The fits extension whose header may contain the WCS solution
        '''

        if isinstance(image_file, Path):
            image_file = str(image_file)

        hdr = fitsio.read_header(image_file, ext=wcs_ext)

        req_keys = ['CRVAL1', 'CRVAL2', 'CRPIX1', 'CRPIX2', 'CTYPE1', 'CTYPE2']

        for key in req_keys:
            if key not in hdr.keys():
                return None
        
        return WCS(image_file)

