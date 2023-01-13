from pathlib import Path
from glob import glob
import os
import numpy as np
import fitsio

from superbit_lensing import utils

import ipdb

class BackgroundRunner(object):
    '''
    Runner class for estimating the background light for calibrated
    SuperBIT images for the onboard analysis (OBA)

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

        # TODO: ...

        return

    def go(self, logprint, overwrite=False):
        '''
        Estimate the background of each calibrated science image
        (now with full masks) using SExtractor

        Currently planned steps:

        (1) Register all target calibrated sci images
        (2) Run SExtractor on each image
        (3) ...

        logprint: utils.LogPrint
            A LogPrint instance for simultaneous logging & printing
        overwrite: bool
            Set to overwrite existing files
        '''

        logprint('Registering calibrated sci frames...')
        self.register_images(logprint)

        logprint('Estimating image backgrounds...')
        self.estimate_background(logprint)

        logprint('Collating outputs...')
        self.collate(logprint, overwrite=overwrite)

        return

    def register_images(self, logprint):
        '''
        logprint: utils.LogPrint
            A LogPrint instance for simultaneous logging & printing
        '''

        logprint('\nWARNING: Image registration not yet implemented!\n')

    def estimate_background(self, logprint):
        '''
        logprint: utils.LogPrint
            A LogPrint instance for simultaneous logging & printing
        '''

        logprint('\nWARNING: Background estimation not yet implemented!\n')

        return

    def collate(self, logprint, overwrite=False):
        '''
        logprint: utils.LogPrint
            A LogPrint instance for simultaneous logging & printing
        overwrite: bool
            Set to overwrite existing files
        '''

        logprint('\nWARNING: Background collation not yet implemented!\n')

        return



