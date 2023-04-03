from pathlib import Path
from glob import glob
from time import time
import os
import numpy as np
import fitsio
from lacosmic import lacosmic

from superbit_lensing import utils
from bitmask import OBA_BITMASK, OBA_BITMASK_DTYPE
from superbit_lensing.oba.oba_io import band2index

import ipdb

class MaskingRunner(object):
    '''
    Runner class for generating the mask images given calibrated
    SuperBIT science images for the onboard analysis (OBA)

    NOTE: At this stage, input calibrated images should have the
    following structure:

    ext0: SCI (calibrated)
    ext1: WGT (weight; 0 if masked, 1 otherwise)
    ext2: MSK (mask; see bitmask.py for def)
    '''

    _name = 'masking'

    # by default, do *all* masking steps
    # NOTE: hot pixels & "inactive region" masks are handled in cals
    _default_mask_types = [
        'cosmic_rays',
        'satellites'
    ]

    def __init__(self, run_dir, bands, target_name=None, mask_types=None,
                 cr_contrast=5, sci_ext=0, msk_ext=2):
        '''
        run_dir: pathlib.Path
            The OBA run directory for the given target
        bands: list of str's
            A list of band names
        target_name: str
            The name of the target. Default is to use the end of
            run_dir
        mask_types: list of str's
            A list of mask types to include in the masking scheme. See the
            class variable `default_mask_types` for the full list
        cr_contrast: int, float
            The contrast value to use for cosmic ray detection
            NOTE: This param is called f_lim in the La Cosmic paper; see
            https://ui.adsabs.harvard.edu/abs/2001PASP..113.1420V/abstract
        sci_ext: int
            The science frame fits extension
        wgt_ext: int
            The mask frame fits extension
        '''

        args = {
            'run_dir': (run_dir, Path),
            'bands': (bands, list),
            'cr_contrast': (cr_contrast, (int, float)),
            'sci_ext': (sci_ext, int),
            'msk_ext': (msk_ext, int),
        }

        for name, tup in args.items():
            val, allowed_types = tup
            utils.check_type(name, val, allowed_types)
            setattr(self, name, val)

        if target_name is None:
            target_name = run_dir.name

        utils.check_type('target_name', target_name, str)
        self.target_name = target_name

        default_mtypes = self._default_mask_types
        if mask_types is None:
            mask_types = default_mtypes

        utils.check_type('mask_types', mask_types, list)
        for mtype in mask_types:
            utils.check_type('mask_type', mtype, str)
            if mtype not in default_mtypes:
                raise ValueError(f'{mtype} not a registered mask type! ' +
                                 f'Must be one of {default_mtypes}')
        self.mask_types = mask_types

        # this dictionary will store a list of calibrated image paths indexed
        # by band
        self.images = {}

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

        NOTE: The above used to be individual steps, but we now do them
        in succession to minimize memory footprint for the QCC

        logprint: utils.LogPrint
            A LogPrint instance for simultaneous logging & printing
        overwrite: bool
            Set to overwrite existing files
        '''

        logprint('Gathering input calibrated images...')
        self.gather_images(logprint)

        # NOTE: The following used to be separate calls across all images, but
        # this can require a large memory footprint. So we do the loop here
        # instead to prioritize memory over repeated computation
        i = 1
        for band in self.bands:
            logprint(f'Starting band {band}')

            images = self.images[band]
            Nfiles = len(images)

            i = 1
            for image in images:
                logprint(f'Starting {image}; ({i} of {Nfiles})')

                logprint('Reading original mask...')
                mask = fitsio.read(str(image), ext=self.msk_ext)

                if 'cosmic_rays' in self.mask_types:
                    logprint('Masking cosmic rays...')
                    mask = self.mask_cosmic_rays(image, mask, logprint)
                else:
                    logprint('Skipping cosmic ray masking given config file')

                if 'satellites' in self.mask_types:
                    logprint('Masking satellite trails...')
                    mask = self.mask_satellite_trails(image, mask, logprint)
                else:
                    logprint('Skipping satellite trail masking given config file')

                logprint('Updating image mask...')
                self.update_mask(image, mask, logprint)

                i += 1

        return

    def gather_images(self, logprint):
        '''
        Register all input images whose masks we will update

        logprint: utils.LogPrint
            A LogPrint instance for simultaneous logging & printing
        '''

        msk_ext = self.msk_ext

        for band in self.bands:
            logprint(f'Starting band {band}')

            cal_dir = (self.run_dir / band / 'cal/').resolve()
            bindx = band2index(band)

            cal_files = glob(
                str(cal_dir / f'{self.target_name}*_{bindx}_*_cal.fits')
                )

            Nimages = len(cal_files)
            logprint(f'Found {Nimages} calibrated images')
            self.images[band] = cal_files

        return

    def mask_cosmic_rays(self, image_file, mask, logprint):
        '''
        Run a cosmic ray finder on each cal image and combine mask
        with self.masks entry

        image_file: pathlib.Path
            The path of the input image
        mask: np.ndarray
            The existing image mask. Need masking info so we don't treat
             hot pixels, etc. as cosmic rays
        logprint: utils.LogPrint
            A LogPrint instance for simultaneous logging & printing
        '''

        sci_ext = self.sci_ext
        msk_ext = self.msk_ext

        cosmic_val = OBA_BITMASK['cosmic_ray']

        name = Path(image_file).name
        logprint(f'Running lacosmic on {name}')

        data, hdr = fitsio.read(
            str(image_file), ext=sci_ext, header=True
            )
        gain = hdr['GAIN']

        # la cosmic won't understand our bitmask
        la_mask = mask.astype(bool)

        start = time()

        # TODO: can we move some of these pars to a config?
        data_cr_corr, cr_mask = lacosmic(
            data=data.astype(np.float32),
            contrast=self.cr_contrast,
            cr_threshold=6,
            neighbor_threshold=6,
            mask=la_mask,
            effective_gain=gain, # e-/ADU (0.343 for SB)
            # TODO: generalize the readnoise val!
            readnoise=2.52, # e- RMS
            maxiter=2
            )

        end = time()
        dT = end - start

        logprint(f'Cosmic ray masking took {dT:.2f} s')

        # OR cosmic ray mask with the existing mask on cal file
        mask[cr_mask == True] += cosmic_val

        return mask

    def mask_satellite_trails(self, image_file, mask, logprint):
        '''
        TODO: Run a satellite trail finder on each cal image and combine mask
        with self.masks entry
        '''

        logprint('\nWARNING: Satellite trail masking not yet implemented!\n')

        return mask

    def update_mask(self, image_file, new_mask, logprint):
        '''
        Update the mask extension of the input image given the new mask

        image_file: pathlib.Path
            The path of the input image
        new_mask: np.ndarray
            The new image mask to write
        logprint: utils.LogPrint
            A LogPrint instance for simultaneous logging & printing
        '''

        msk_ext = self.msk_ext

        with fitsio.FITS(str(image_file), 'rw') as fits:
            # overwrites existing image
            fits[msk_ext].write(new_mask)

        return
