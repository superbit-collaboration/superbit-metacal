from pathlib import Path
from glob import glob
from astropy.wcs import WCS
import fitsio

from superbit_lensing import utils

import ipdb

class CoaddRunner(object):
    '''
    Runner class for coadding each calibrated, bkg-subtracted
    sci image for each band (as well as a detection band)
    for the SuperBIT onboard analysis (OBA)

    NOTE: At this stage, input calibrated images should have a
    WCS solution in the header and the following structure:

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

        return

    def go(self, logprint, overwrite=False):
        '''
        Make coadd images for each band, as well as a composite detection band
        using SWarp

        Steps:

        (1) Gather all target calibrated, bkg-subtracted sci images
        (2) Check that all imgages have a WCS solution
        (3) Run SWarp for all target images of a given band, generating
            maximal coadd images (i.e. full extent of all single-epoch
            exposures, which will have non-uniform depth)
        (4) Create a detection coadd from a chosen subset of single-band coadds

        logprint: utils.LogPrint
            A LogPrint instance for simultaneous logging & printing
        overwrite: bool
            Set to overwrite existing files
        '''

        logprint('Gathering calibrated sci frames...')
        self.gather_images(logprint)

        logprint('Checking for image WCS solutions...')
        self.check_for_wcs(logprint)

        # TODO: implement!
        # logprint('Making single-band coadd images...')

        # TODO: implement!
        # logprint('Making detection coadd image...')

        return


    def gather_images(self, logprint):
        '''
        logprint: utils.LogPrint
            A LogPrint instance for simultaneous logging & printing
        '''

        for band in self.bands:
            logprint(f'Starting band {band}')

            cal_dir = (self.run_dir / band / 'cal/').resolve()

            self.images[band] = glob(
                str(cal_dir / f'{self.target_name}*_{band}_*_cal.fits')
                )

            Nimages = len(self.images[band])
            logprint(f'Found {Nimages} images')

            # to keep consistent convention with other modules, store as Paths
            for i, image in enumerate(self.images[band]):
                image = Path(image)
                self.images[band][i] = image

        return

    def check_for_wcs(self, logprint):
        '''
        Make sure each input image has a WCS solution before coadding
        '''

        for band in self.bands:
            for image in self.images[band]:
                im_name = image.name
                hdr = fitsio.read_header(str(image))

                try:
                    wcs = WCS(hdr)

                except KeyError as e:
                    raise KeyError(f'WCS for image {im_name} not found!')

                if wcs is None:
                    raise ValueError(f'WCS for image {im_name} is None!')

        return
