from pathlib import Path
from glob import glob
from copy import copy
import os
import numpy as np
import fitsio

from superbit_lensing import utils
from bitmask import OBA_BITMASK, OBA_BITMASK_DTYPE
import oba_io

import ipdb

class CalsRunner(object):
    '''
    Runner class for calibrating raw SuperBIT science images
    for the onboard analysis (OBA)

    NOTE: At this stage, input raw images should have the
    following structure:

    ext0: SCI (raw)

    After cals has finished, the (new) calibrated images will have
    the following structure:

    ext0: SCI (calibrated)
    ext1: WGT (weight; 0 if masked, 1 otherwise)
    ext2: MSK (mask; see bitmask.py for def)
    '''

    # smallest numpy will allow is 1 byte
    _mask_dtype = OBA_BITMASK_DTYPE

    # NOTE: this is the numpy array shape, not FITS shape!
    _image_shape = (6422, 9600)

    # the "inactive region" mask accounts for the rows & cols that are not
    # exposed to light
    # NOTE: these index slices are for the 0-indexed numpy arrays representing
    # the mask, *not* the 1-indexed FITS rows/columns!
    _inactive_reg_slices = [
        np.s_[:, 22:23+1], # vertical colum *near* left edge, but not on it
        np.s_[6389:, :]    # ~30 rows on the top of the detector
        ]

    def __init__(self, run_dir, darks_dir, flats_dir, bands, target_name=None,
                 cal_dtype=np.dtype('float64'), allow_negatives=True):
        '''
        run_dir: pathlib.Path
            The OBA run directory for the given target
        darks_dir: pathlib.Path
            The directory location of the master dark frames
        flats_dir: pathlib.Path
            The directory location of the master dark frames
        bands: list of str's
            A list of band names
        target_name: str
            The name of the target. Default is to use the end of
            run_dir
        cal_dtype: numpy.dtype
            A string signifying the numpy dtype of the output cal image
        allow_negatives: bool
            Set to allow negative numbers in the calibration dtype (e.g.
            BITPIX = -64 instead of 64)
        '''

        args = {
            'run_dir': (run_dir, Path),
            'darks_dir': (darks_dir, Path),
            'flats_dir': (flats_dir, Path),
            'bands': (bands, list),
            'cal_dtype': (cal_dtype, np.dtype),
            'allow_negatives': (allow_negatives, bool),
        }

        for name, tup in args.items():
            val, allowed_types = tup
            utils.check_type(name, val, allowed_types)
            setattr(self, name, val)

        # make sure each band name is a str
        for i, b in enumerate(self.bands):
            utils.check_type('band_{i}', b, str)

        if target_name is None:
            target_name = run_dir.name

        utils.check_type('target_name', target_name, str)
        self.target_name = target_name

        # this dictionary will store the calibration frames associated
        # with each raw exposure, indexed by raw sci filename
        self.cals = {}

        # this dictionary will store the calibrated image arrays, indexed
        # by raw sci filename
        self.calibrated = {}

        # this dictionary will store the hot pixel mask for each master dark
        # frame (plus the "inactive region" mask), indexed by the dark filename
        self.pixel_masks = {}

        self.inactive_reg_mask = self._make_inactive_region_mask()

        return

    def _make_inactive_region_mask(self):
        '''
        Mask the rows & cols of the CCD that are not exposed to light
        '''

        unmasked = OBA_BITMASK['unmasked']
        inactive_val = OBA_BITMASK['inactive_region']

        inactive_reg_mask = unmasked * np.ones(
            self._image_shape, dtype=self._mask_dtype
            )

        for s in self._inactive_reg_slices:
            inactive_reg_mask[s] = inactive_val

        return inactive_reg_mask

    def go(self, logprint, overwrite=False):
        '''
        Run all steps to convert raw SCI frames to calibrated frames.
        The current calibration steps require master dark & flat
        images. The basic procedure is as follows:

        (1) Assign the corresponding master dark & flat for each image
        (2) Create hot pixel mask using master dark (in addition to
            the "inactive region" mask)
            NOTE: The current plan is to have a static master flat
        (3) Basic calibration: Cal = (Raw - Dark) / Flat
        (4) Write out calibrated images w/ original headers
                - Includes wgt & msk image collation

        logprint: utils.LogPrint
            A LogPrint instance for simultaneous logging & printing
        overwrite: bool
            Set to overwrite existing files
        '''

        logprint('Assigning calibration frames...')
        self.assign_cals(logprint)

        logprint('Creating hot pixel + "inactive region" masks...')
        self.make_image_masks(logprint)

        logprint('Applying calibrations to raw images...')
        self.apply_cals(logprint)

        logprint('Writing out calibrated images...')
        self.write_calibrated_images(logprint, overwrite=overwrite)

        logprint('Calibrations completed!')

        return

    def assign_cals(self, logprint):
        '''
        Assign the associated calibration frames to each raw iamge

        logprint: utils.LogPrint
            A LogPrint instance for simultaneous logging & printing
        '''

        ignore = '[!_]'

        for b in self.bands:
            logprint(f'Starting band {b}')

            band_dir = (self.run_dir / b)

            # want to ignore any existing OBA files from previous runs
            images = oba_io.get_raw_files(
                band_dir, self.target_name, band=b
                )

            Nimages = len(images)
            for i, image in enumerate(images):
                image = Path(image)
                image_base = Path(image).name
                logprint(f'Grabbing cals for {image_base}; {i+1} of {Nimages}')
                dark, flat = self.get_image_cals(image)
                logprint(f'Using master dark file {dark.name}')
                logprint(f'Using master flat file {flat.name}')

                self.cals[image] = {}
                self.cals[image]['dark'] = dark
                self.cals[image]['flat'] = flat

        return

    def get_image_cals(self, image_file):
        '''
        Grab the master calibration frames associated to the passed
        image file

        image_file: str
            The filepath of the raw image to assign cals to
        '''

        image_pars = oba_io.parse_image_file(
            Path(image_file), image_type='sci'
            )

        dark = self._get_image_dark(image_pars)
        flat = self._get_image_flat(image_pars)

        for name, cal in zip(['dark', 'flat'], [dark, flat]):
            if cal is None:
                raise ValueError(f'No master {name} found for {image_file}!')

        return dark, flat

    def _get_image_dark(self, image_pars):
        '''
        Find the master dark closest in time to the passed image file

        image_pars: dict
            The image file parameters, parsed by oba_io
        '''

        utc = image_pars['utc']
        im_type = 'cal'

        # can set additional requirements to match on
        req = {
            'exp_time': image_pars['exp_time']
        }

        dark = oba_io.closest_file_in_time(
            im_type, self.darks_dir, utc, req=req
            )

        return dark

    def _get_image_flat(self, image_pars):
        '''
        Find the master flat closest in time to the passed image file

        NOTE: The current plan is to have a single, static master flat,
        though this method will work for more

        image_pars: dict
            The image file parameters, parsed by oba_io
        '''

        utc = image_pars['utc']
        im_type = 'cal'

        # can set additional requirements to match on
        req = {
            'exp_time': image_pars['exp_time']
        }

        flat = oba_io.closest_file_in_time(
            im_type, self.flats_dir, utc, req=req
            )

        return flat

    def make_image_masks(self, logprint, threshold=1000):
        '''
        Create the hot pixel & inactive region masks for each master dark
        used for cals. Builds ontop of the inactive region mask set in
        constructor. See bitmask.py for details

        logprint: utils.LogPrint
            A LogPrint instance for simultaneous logging & printing
        threshold: int
            The threshold value to use when defining a hot pixel for a
            given master dark
        '''

        utils.check_type('threshold', threshold, int)
        logprint(f'Using a hot pixel threshold value of {threshold}')

        unmasked_val = OBA_BITMASK['unmasked']
        hot_pix_val = OBA_BITMASK['hot_pixel']

        for sci_file, cals in self.cals.items():

            dark_file = cals['dark']
            dark = fitsio.read(str(dark_file))

            if dark.shape != self._image_shape:
                raise ValueError(f'dark shape {dark.shape} does not match ' +
                                 f'the SuperBIT image shape {self._image_shape}')

            # start with the inactive region mask as the base
            mask = self.inactive_reg_mask.copy()

            bad_pix = np.where(dark > threshold)
            mask[bad_pix] = hot_pix_val

            shape = mask.shape
            Nbad = len(mask[mask != unmasked_val])
            bad_frac = Nbad / (shape[0] * shape[1])
            bad_perc = 100 * bad_frac
            logprint(f'Mask for {dark_file.name} has {Nbad} bad pixels; ' +
                     f'{bad_perc:.2f}%')

            self.pixel_masks[dark_file] = mask

        return

    def apply_cals(self, logprint):
        '''
        Apply basic calibrations given the registered cal frames
        to each raw sci image:

        Cal = (Raw - Dark) / Flat

        NOTE: Operation is performed for all pixels, regardless of
        the pixel mask

        logprint: utils.LogPrint
            A LogPrint instance for simultaneous logging & printing
        '''

        for image_file, cals in self.cals.items():
            # NOTE: If we don't cast to higher precision here, will cause
            # overflow as the raw files are u16!
            dark = fitsio.read(cals['dark']).astype(self.cal_dtype)
            flat = fitsio.read(cals['flat']).astype(self.cal_dtype)
            raw = fitsio.read(image_file).astype(self.cal_dtype)

            self.calibrated[image_file] = ((raw - dark) / flat)

        return

    def write_calibrated_images(self, logprint, overwrite=False):
        '''
        Write out new calibrated images, inheriting the original
        fits file headers

        The calibrated image fits extensions are as follows:
        0: SCI (calibrated)
        1: WGT (weight; 0 if masked, 1 otherwise)
        2: MSK (mask; 1 if masked, 0 otherwise)

        logprint: utils.LogPrint
            A LogPrint instance for simultaneous logging & printing
        overwrite: bool
            Set to overwrite existing files
        '''

        i = 1
        Nfiles = len(self.calibrated)
        for raw_file, cal in self.calibrated.items():
            raw_dir = raw_file.parent
            cal_dir = raw_dir / 'cal/'

            cal_name= raw_file.name.replace(
                '.fits', '_cal.fits'
                )
            cal_file = cal_dir / cal_name

            logprint(f'Writing {cal_file}; ({i} of {Nfiles})')

            if os.path.exists(cal_file):
                logprint(f'Warning: {cal_file} already exists')
                if overwrite is True:
                    logprint(f'Removing as overwrite is True')
                    os.remove(cal_file)
                else:
                    raise OSError(f'{cal_file} already exists ' +
                                  'and overwrite is False!')

            # we want to inherit the header of the raw file
            cal_hdr = copy(fitsio.read_header(raw_file))
            cal_hdr['IMTYPE'] = 'SCI_CAL'

            # NOTE: itemsize is in bytes
            bitpix = str(self.cal_dtype.itemsize * 8)
            if self.allow_negatives is True:
                bitpix = f'-{bitpix}'
            cal_hdr['BITPIX'] =  bitpix

            # TODO: do something more robust here!
            # we do this hacky thing as the hen sims have BZERO=2^15...
            if cal_hdr['BZERO'] != 0:
                logprint(f'WARNING: BZERO = {cal_hdr["BZERO"]}; are you ' +
                         'sure that is correct? Setting to zero for now...')
            cal_hdr['BZERO'] =  0

            # create the weight & mask image for the calibrated file,
            # based off of the hot pixel mask created for the associated
            # dark file
            wgt, wgt_hdr = self._make_wgt_image(raw_file)
            msk, msk_hdr = self._make_msk_image(raw_file)

            with fitsio.FITS(cal_file, 'rw') as fits:
                fits.write(cal, header=cal_hdr)
                fits.write(wgt, header=wgt_hdr)
                fits.write(msk, header=msk_hdr)

            i += 1

        return

    def _make_wgt_image(self, raw_image):
        '''
        Create the weight image for the input raw_file. Very simple
        at this stage:
            0: Masked
            1: Not masked

        NOTE: Must be run after cals have been assigned!

        raw_image: pathlib.Path
            The path of the raw image that we will create the
            associated weight image for
        '''

        dark_file = self.cals[raw_image]['dark']
        pixel_mask = self.pixel_masks[dark_file]

        # At this stage, the weight map is just the logical not of
        # the hot pixel mask
        # NOTE: While we want the weight map to be floats in the future,
        # we save disk space by using ints
        wgt = (~pixel_mask).astype(np.uint8)

        wgt_hdr = {
            'IMTYPE': 'WEIGHT'
        }

        return wgt, wgt_hdr

    def _make_msk_image(self, raw_image):
        '''
        Create the mask image for the input raw_file. Just the hot
        pixel mask & inactive region at this stage - more complex
        features like cosmic rays and satellite trails are handled later

        NOTE: Must be run after cals have been assigned!

        raw_image: pathlib.Path
            The path of the raw image that we will create the
            associated weight image for
        '''

        dark_file = self.cals[raw_image]['dark']

        # NOTE: dtype set in bitmask.py
        msk = self.pixel_masks[dark_file]

        msk_hdr = {
            'IMTYPE': 'MASK'
        }

        return msk, msk_hdr
