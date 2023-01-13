from pathlib import Path
from glob import glob
import os
import numpy as np
import fitsio

from superbit_lensing import utils

import ipdb

class MaskingRunner(object):
    '''
    Runner class for generating the mask images given calibrated
    SuperBIT science images for the onboard analysis (OBA)

    NOTE: At this stage, input calibrated images should have the
    following structure:

    ext0: SCI (calibrated)
    ext1: WGT (weight; 0 if masked, 1 otherwise)
    ext2: MSK (mask; 1 if masked, 0 otherwise)
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

        # this dictionary will store the mask arrays for each image, indexed
        # by sci_cal filename
        self.masks = {}

        return

    def go(self, logprint, overwrite=False):
        '''
        Determine the final mask for each single-epoch exposure. The
        hot pixel mask is already determined during the calibrations
        step and is used as the starting mask. This is the place
        for more sophisticated masking such as to deal with cosmic
        rays and satellite trails

        Currently planned steps:

        (1) Initialize masks from calibrated image msk ext's
        (2) Cosmic rays
        (3) Satellite trails
        (4) ...

        logprint: utils.LogPrint
            A LogPrint instance for simultaneous logging & printing
        overwrite: bool
            Set to overwrite existing files
        '''

        logprint('Initializing masks...')
        self.initialize_masks(logprint)

        logprint('Masking cosmic rays...')
        self.mask_cosmic_rays(logprint)

        logprint('Masking satellite trails...')
        self.mask_satellite_trails(logprint)

        return

    def initialize_masks(self, logprint, msk_ext=2):
        '''
        Initialize mask for each image using the MSK extension

        logprint: utils.LogPrint
            A LogPrint instance for simultaneous logging & printing
        msk_ext: int
            The fits extension of the mask plane for the cal images to build
            the full mask for
        '''

        for band in self.bands:
            logprint(f'Starting band {band}')

            cal_dir = (self.run_dir / band / 'cal/').resolve()

            cal_files = glob(
                str(cal_dir / f'{self.target_name}*_{band}_*_cal.fits')
                )

            for cal_file in cal_files:
                msk = fitsio.read(cal_file, ext=msk_ext)
                cal_name = Path(cal_file).name

                # shouldn't happen, but check to be sure
                cal_file = Path(cal_file)
                if cal_file in self.masks:
                    cal_name = cal_file.name
                    raise ValueError(f'{cal_name} already has a ' +
                                     'registered mask!')
                self.masks[cal_file] = msk

        return

    def mask_cosmic_rays(self, logprint):
        '''
        TODO: Run a cosmic ray finder on each cal image and combine mask
        with self.masks entry
        '''

        logprint('\nWARNING: Cosmic ray masking not yet implemented!\n')

        return

    def mask_satellite_trails(self, logprint):
        '''
        TODO: Run a satellite trail finder on each cal image and combine mask
        with self.masks entry
        '''

        logprint('\nWARNING: Satellite trail masking not yet implemented!\n')

        return
