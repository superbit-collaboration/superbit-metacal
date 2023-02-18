from pathlib import Path

import ipdb

from superbit_lensing import utils
from superbit_lensing.cookiecutter import CookieCutter

class CookieCutterRunner(object):
    '''
    Runner class for the CookieCutter format, which takes image cutouts
    of sources from an input detection catalog (typically SExtractor)
    as well as object metadata, including information needed to re-
    construct the original images where they intersect with the stamps

    While CookieCutter can handle a variety of inputs, we are only
    going to save the following for the SuperBIT onboard analysis:

    (1) Image cutouts of the sources from the RAW SCI images (int16)
    (2) A bitmask containing masking & segmentation information
    (3) Metadata such as ID, position, boxsize, and image info needed
        to reconstruct the original images in the locations of the
        cutouts. We will also save summary statistics of the sky
        background per-stamp rather than save the expensive weightmaps

    NOTE: At this stage, input images should have a
    WCS solution in the header and the following structure:

    Single-epoch images:
    ext0: SCI (calibrated & background-subtracted)
    ext1: WGT (weight; 0 if masked, 1/sky_var otherwise)
    ext2: MSK (mask; 1 if masked, 0 otherwise)
    ext3: BKG (background)

    Coadd images:
    ext0: SCI (calibrated & background-subtracted)
    ext1: WGT (weight; 0 if masked, 1/sky_var otherwise)
    ext2: SEG (segmentation; 0 if sky, NUMBER if pixel is assigned to an obj)
    '''

    def __init__(self, config_file, run_dir, bands, target_name=None,
                 sci_ext=0, wgt_ext=1, msk_ext=2, bkg_ext=3,
                 coadd_sci_ext=0, coadd_wgt_ext=1, coadd_seg_ext=2):
        '''
        config_file: pathlib.Path
            The filepath of the base CookieCutter config
        run_dir: pathlib.Path
            The OBA run directory for the given target
        bands: list of str's
            A list of band names to make CookieCutter files for
        target_name: str
            The name of the target. Default is to use the end of
            run_dir
        sci_ext: int
            The science frame fits extension
        wgt_ext: int
            The weight frame fits extension
        msk_ext: int
            The mask frame fits extension
        bkg_ext: int
            The background frame fits extension
        coadd_sci_ext: int
            The science frame fits extension for coadds
        coadd_wgt_ext: int
            The weight frame fits extension for coadds
        coadd_seg_ext: int
            The segmentation frame fits extension for coadds
        '''

        args = {
            'config_file': (config_file, Path),
            'run_dir': (run_dir, Path),
            'bands': (bands, list),
            'sci_ext': (sci_ext, int),
            'wgt_ext': (wgt_ext, int),
            'msk_ext': (wgt_ext, int),
            'bkg_ext': (wgt_ext, int),
            'coadd_sci_ext': (sci_ext, int),
            'coadd_wgt_ext': (wgt_ext, int),
            'coadd_seg_ext': (seg_ext, int),
        }

        for name, tup in args.items():
            val, allowed_types = tup
            utils.check_type(name, val, allowed_types)
            setattr(self, name, val)

        if target_name is None:
            target_name = run_dir.name

        utils.check_type('target_name', target_name, str)
        self.target_name = target_name

        for band in self.bands:
            utils.check_type('band', band, str)

        # TODO: ...

        # this dictionary will store dict's of image filepaths
        # indexed by extensions for each set of observations, indexed
        # by band
        self.images = {}

        return

    def go(self, logprint, overwrite=False):
        '''
        Make a CookieCutter output file for each band

        Steps:

        (1) Register all single-epoch and coadd images needed for
            the CookieCutter
        (2) Create the CookieCutter config files given the set of
            registered images
        (3) Run the CookieCutter on all bands
            NOTE: We ignore the single-band coadds for OBA

        logprint: utils.LogPrint
            A LogPrint instance for simultaneous logging & printing
        overwrite: bool
            Set to overwrite existing files
        '''

        logprint('Gathering input images...')
        self.gather_images(logprint)

        logprint('Creating CookieCutter config files...')
        self.make_configs(logprint)

        logprint('Running CookieCutter...')
        self.run_cookie_cutter(logprint, overwrite=overwrite)

        return

    def gather_images(self, logprint):
        '''
        TODO: ...

        logprint: utils.LogPrint
            A LogPrint instance for simultaneous logging & printing
        '''

        for band in self.bands:
            logprint(f'Starting band {band}')

        return

    def create_configs(self, logprint):
        '''
        TODO: ...

        logprint: utils.LogPrint
            A LogPrint instance for simultaneous logging & printing
        '''

        for band in self.bands:
            logprint(f'Starting band {band}')

            # TODO: write...

        return

    def run_cookie_cutter(self, logprint, overwrite):
        '''
        TODO: ...

        logprint: utils.LogPrint
            A LogPrint instance for simultaneous logging & printing
        overwrite: bool
            Set to overwrite existing files
        '''

        for band in self.bands:
            logprint(f'Starting band {band}')

        return
