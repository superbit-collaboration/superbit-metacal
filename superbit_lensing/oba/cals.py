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

    _name = 'cals'

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

    # use this if the SATURATE key is not present in image headers
    _default_saturate = 65535
    _saturate_key = 'SATURATE'

    def __init__(self, run_dir, darks_dir, flats_dir, bands, target_name=None,
                 cal_dtype=np.dtype('float64'), allow_negatives=True,
                 hp_threshold=1000, ignore_flats=False):
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
        hp_threshold: int
            The threshold value to use when defining a hot pixel for a
            given master dark
        ignore_flats: bool
            Set to ignore flat field corrections (i.e. assume a uniform
            pixel response). This may be desired as the SuperBIT flats
            have spatially-varying properties due to inconsistencies in
            the filter wheel positions
        '''

        args = {
            'run_dir': (run_dir, Path),
            'darks_dir': (darks_dir, Path),
            'flats_dir': (flats_dir, Path),
            'bands': (bands, list),
            'cal_dtype': (cal_dtype, np.dtype),
            'allow_negatives': (allow_negatives, bool),
            'hp_threshold': (hp_threshold, int),
            'ignore_flats': (ignore_flats, bool),
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

        return

    def go(self, logprint, overwrite=False):
        '''
        Run all steps to convert raw SCI frames to calibrated frames.
        The current calibration steps require master dark & flat
        images. The basic procedure is as follows:

        (1) Assign the corresponding master dark & flat for each image
        (2) Create hot pixel mask using master dark (in addition to
            the "inactive region" & saturation mask)
        (3) Basic calibration: Cal = (Raw - Dark) / Flat
        (4) Write out calibrated images w/ original headers
                - Includes wgt & msk image collation

        NOTE: The above used to be individual steps, but we now do them
        in succession to minimize memory footprint for the QCC

        logprint: utils.LogPrint
            A LogPrint instance for simultaneous logging & printing
        overwrite: bool
            Set to overwrite existing files
        '''

        logprint('Assigning calibration frames...')
        self.assign_cals(logprint)

        # NOTE: The following used to be separate calls across all images, but
        # this can require a large memory footprint. So we do the loop here
        # instead to prioritize memory over repeated computation
        i = 1
        Nfiles = len(self.cals)
        for sci_file, cals in self.cals.items():
            logprint(f'Starting {sci_file}; ({i} of {Nfiles})')

            logprint('Creating hot pixel + "inactive region" masks...')
            dark_file = cals['dark']

            mask = self.make_image_mask(sci_file, dark_file, logprint)

            logprint('Applying calibrations to raw image...')
            cal = self.apply_cals(sci_file, logprint)

            logprint('Writing out calibrated image...')
            self.write_calibrated_image(
                sci_file, cal, mask, logprint, overwrite=overwrite
                )

            i += 1

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

                try:
                    logprint(f'Grabbing cals for {image_base}; {i+1} of {Nimages}')
                    dark, flat = self.get_image_cals(image)

                    logprint(f'Using master dark file {dark.name}')

                    if self.ignore_flats is True:
                        assert flat is None
                        logprint(f'Using uniform flat file as ignore_flats is True')
                    else:
                        logprint(f'Using master flat file {flat.name}')

                    self.cals[image] = {}
                    self.cals[image]['dark'] = dark
                    self.cals[image]['flat'] = flat

                except ValueError as e:
                    logprint(e)
                    logprint('Removing image from OBA consideration')

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
                if (name == 'flat') and self.ignore_flats:
                    # ok only if the ignore_flats setting is True
                    continue
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

        NOTE: Flats now have band-dependence

        image_pars: dict
            The image file parameters, parsed by oba_io
        '''

        if self.ignore_flats is True:
            return None

        utc = image_pars['utc']
        im_type = 'cal'

        # can set additional requirements to match on
        req = {
            'band': image_pars['band'],
        }

        flat = oba_io.closest_file_in_time(
            im_type, self.flats_dir, utc, req=req
            )

        return flat

    def make_image_mask(self, sci_file, dark_file, logprint):
        '''
        Create the hot pixel & saturation mask for the given sci & dark images
        used for cals. Builds ontop of the inactive region mask. See bitmask.py
        for pixel mask details

        sci_file: pathlib.Path
            The path to the given science file
        dark_file: pathlib.Path
            The path to the corresponding dark calibration file
        logprint: utils.LogPrint
            A LogPrint instance for simultaneous logging & printing
        '''

        threshold = self.hp_threshold
        logprint(f'Using a hot pixel threshold value of {threshold}')

        unmasked_val = OBA_BITMASK['unmasked']
        hot_pix_val = OBA_BITMASK['hot_pixel']
        saturated_val = OBA_BITMASK['saturated']

        sci, hdr = fitsio.read(str(sci_file), header=True)
        dark = fitsio.read(str(dark_file))

        if dark.shape != self._image_shape:
            raise ValueError(f'dark shape {dark.shape} does not match ' +
                             f'the SuperBIT image shape {self._image_shape}')
        # we have to check this one due to the inactive region mask
        if sci.shape != self._image_shape:
            raise ValueError(f'sci shape {sci.shape} does not match ' +
                             f'the SuperBIT image shape {self._image_shape}')

        # look to see if the image specifies its saturation point
        # (might instead be the start of the NL response)
        try:
            saturate = hdr[self._saturate_key]
        except KeyError:
            saturate = self._default_saturate

        # start with the inactive region mask as the base
        mask = self._make_inactive_region_mask()

        hot_pix = np.where(dark > threshold)
        mask[hot_pix] = hot_pix_val
        saturated_pix = np.where(sci >= saturate)
        mask[saturated_pix] = saturated_val

        shape = mask.shape
        Nbad = len(mask[mask != unmasked_val])
        bad_frac = Nbad / (shape[0] * shape[1])
        bad_perc = 100 * bad_frac
        logprint(f'Mask for {dark_file.name} has {Nbad} bad pixels; ' +
                 f'{bad_perc:.2f}%')
        logprint(f'Hot pixels: {len(mask[hot_pix])}')
        logprint(f'Saturated pixels: {len(mask[saturated_pix])}')

        return mask

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

    def apply_cals(self, raw_file, logprint):
        '''
        Apply basic calibrations given the registered cal frames
        to each raw sci image:

        Cal = (Raw - Dark) / Flat

        NOTE: Operation is performed for all pixels, regardless of
        the pixel mask

        NOTE: The division by the flat field is skipped if ignore_flats
        is set to True

        raw_file: pathlib.Path
            The path to the input raw image file to calibrate
        logprint: utils.LogPrint
            A LogPrint instance for simultaneous logging & printing
        '''

        cals = self.cals[raw_file]

        # NOTE: If we don't cast to higher precision here, will cause
        # overflow as the raw files are u16!
        dark = fitsio.read(cals['dark']).astype(self.cal_dtype)
        raw = fitsio.read(raw_file).astype(self.cal_dtype)

        if self.ignore_flats is True:
            calibrated = raw - dark
        else:
            flat = fitsio.read(cals['flat']).astype(self.cal_dtype)

            # NOTE: we definitely don't want to divide by 0, so check
            # master flat for 0's or negatives. These are likely flagged
            # pixels so we don't worry too much about it
            bad = flat[flat <= 0]
            if len(bad) > 0:
                logprint(f'Found {len(bad)} flat pixels <=0; setting to 1')
                flat[flat <= 0] = 1.

            calibrated = ((raw - dark) / flat)

        return calibrated

    def write_calibrated_image(self, raw_file, cal, msk, logprint,
                               overwrite=False):
        '''
        Write out new calibrated image, inheriting the original
        fits file headers

        The calibrated image fits extensions are as follows:
        0: SCI (calibrated)
        1: WGT (weight; 0 if masked, 1 otherwise)
        2: MSK (mask; 1 if masked, 0 otherwise)

        raw_file: pathlib.Path
            The path to the raw image file
        cal: np.ndarray
            The calibrated image to write
        msk: np.ndarray
            The mask of the calibrated image to write
        logprint: utils.LogPrint
            A LogPrint instance for simultaneous logging & printing
        overwrite: bool
            Set to overwrite existing files
        '''

        raw_dir = raw_file.parent
        cal_dir = raw_dir / 'cal/'

        cal_name= raw_file.name.replace(
            '.fits', '_cal.fits'
            )
        cal_file = cal_dir / cal_name

        logprint(f'Writing to {cal_file}')

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
        wgt, wgt_hdr = self._make_wgt_image(msk)
        msk, msk_hdr = self._make_msk_image(msk)

        with fitsio.FITS(cal_file, 'rw') as fits:
            fits.write(cal, header=cal_hdr)
            fits.write(wgt, header=wgt_hdr)
            fits.write(msk, header=msk_hdr)

        return

    def _make_wgt_image(self, mask):
        '''
        Create the weight image for the input raw_file. Very simple
        at this stage:
            0: Masked
            1: Not masked

        NOTE: Must be run after cals have been assigned!

        mask: np.ndarray
            The pixel mask of a given image
        '''

        # NOTE: While we want the weight map to be floats in the future,
        # we save disk space by using ints
        wgt = np.zeros(mask.shape, dtype=np.uint8)
        wgt[mask == 0] = 1

        wgt_hdr = {
            'IMTYPE': 'WEIGHT'
        }

        return wgt, wgt_hdr


    def _make_msk_image(self, mask):
        '''
        Create the mask image for the input raw_file. Just the hot
        pixel mask & inactive region at this stage - more complex
        features like cosmic rays and satellite trails are handled later

        NOTE: Must be run after cals have been assigned!
        NOTE: dtype set in bitmask.py

        mask: np.ndarray
            The pixel mask of a given image
        '''

        mask_hdr = {
            'IMTYPE': 'MASK'
        }

        for key, val in OBA_BITMASK.items():
            mask_hdr[key] = val

        return mask, mask_hdr
