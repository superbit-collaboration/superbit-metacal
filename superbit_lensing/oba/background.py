from pathlib import Path
from glob import glob
import os
import numpy as np
import fitsio
from astropy.io import fits

from superbit_lensing import utils
from superbit_lensing.oba.oba_io import band2index

import ipdb

# TODO: Handle output cat location!!

class BackgroundRunner(object):
    '''
    Runner class for estimating the background light for calibrated
    SuperBIT images for the onboard analysis (OBA).

    NOTE: At this stage, input calibrated images should have the
    following structure:

    ext0: SCI (calibrated)
    ext1: WGT (weight; 0 if masked, 1 otherwise)
    ext2: MSK (mask; see bitmask.py for def)

    The runner overwrites the calibrated SCI image with the background-
    subtracted image and updates the weights with the background RMS field:

    ext0: SCI (calibrated & bkg-subtracted)
    ext1: WGT (weight; 0 if masked, 1/var(bkg)^2 otherwise)
    ext2: MSK (mask; 1 if masked, 0 otherwise)
    ext3: BKG (background)
    '''

    _name = 'background'

    def __init__(self, run_dir, bands, se_config, target_name=None,
                 sci_ext=0, wgt_ext=1, msk_ext=2, bkg_ext=3):
        '''
        run_dir: pathlib.Path
            The OBA run directory for the given target
        bands: list of str's
            A list of band names
        se_config: pathlib.Path
            The file path of the SExtractor config file to use for
            background estimation
        target_name: str
            The name of the target. Default is to use the end of
            run_dir
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
            'bands': (bands, list),
            'se_config': (se_config, Path),
            'sci_ext': (sci_ext, int),
            'wgt_ext': (wgt_ext, int),
            'msk_ext': (msk_ext, int),
            'bkg_ext': (bkg_ext, int),
        }

        for name, tup in args.items():
            val, allowed_types = tup
            utils.check_type(name, val, allowed_types)
            setattr(self, name, val)

        if target_name is None:
            target_name = run_dir.name

        utils.check_type('target_name', target_name, str)
        self.target_name = target_name

        # will be a dictionary of calibrated sci images indexed
        # by band
        self.images = {}

        # will be a dictionary of background check-images indexed
        # by input calibrated sci images
        self.backgrounds = {}

        return

    def go(self, logprint, overwrite=False):
        '''
        Estimate the background of each calibrated science image
        (now with full masks) using SExtractor

        Currently planned steps:

        (1) Gather all target calibrated sci images
        (2) Run SExtractor on each image, generating a background-
            subtracted image, a background image, and background RMS map
        (3) Update weight map WGT with bkg rms map
        (4) Collate new bkg-subtracted images with updated extensions:
            - ext0: SUB (calibrated, bkg-subtracted SCI)
            - ext1: WGT (weight map, including bkg-rms)
            - ext2: MSK (bitmask)
            - ext3: BKG (background map)

        logprint: utils.LogPrint
            A LogPrint instance for simultaneous logging & printing
        overwrite: bool
            Set to overwrite existing files
        '''

        logprint('Gathering calibrated sci frames...')
        self.gather_images(logprint)

        logprint('Estimating image backgrounds...')
        self.estimate_background(logprint, overwrite=overwrite)

        logprint('Collating outputs...')
        self.collate(logprint)

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

            # to keep consistent convention with other modules, store as Paths
            for i, image in enumerate(self.images[band]):
                self.images[band][i] = Path(image)

        return

    def estimate_background(self, logprint, overwrite=False):
        '''
        logprint: utils.LogPrint
            A LogPrint instance for simultaneous logging & printing
        overwrite: bool
            Set to overwrite existing files
        '''

        for band in self.bands:
            logprint(f'Starting band {band}')

            images = self.images[band]
            Nimages = len(images)

            for i, image in enumerate(images):
                image_name = image.name
                logprint(f'Starting image {image_name}; {i+1} of {Nimages}')

                cmd = self.setup_se_cmd(image, band, overwrite=overwrite)
                self._run_sextractor(cmd, logprint)

        return

    def setup_se_cmd(self, image_file, band, overwrite=False):
        '''
        Generate the SExtractor cmd given the image filename and band

        image_file: pathlib.Path
            The filepath of the image to run SExtractor on
        band: str
            The name of the band
        overwrite: bool
            Set to overwrite existing files
        '''

        if isinstance(image_file, Path):
            image_file = str(image_file)

        # get either the base config or a band-modified one, depending
        # on the circumstance
        config_file = self._get_se_config(band)

        base = f'sex {image_file}[0] -c {config_file}'

        options = self._get_se_cmd_args(image_file, band, overwrite=overwrite)

        cmd = f'{base} {options}'

        return cmd

    def _get_se_config(self, band):
        '''
        Grab or generate a SExtractor config file for the given band based on
        the base config passed in the constructor. Update any values that
        are band-specific

        NOTE: For now, we are using just a base config and overwriting any
        relevant keys on the command line. But keeping the functionality here

        band: str
            The name of the band
        '''

        return self.se_config

    def _get_se_cmd_args(self, image_file, band, overwrite=False):
        '''
        Generate the SExtractor command-line args to overwrite any
        band-specific fields for the single-epoch background estimation
        runs

        image_file: str, pathlib.Path
            The filepath of the image to run SExtractor on
        band: str
            The name of the band
        overwrite: bool
            Set to overwrite existing files
        '''

        utils.check_type('image_file', image_file, (str, Path))

        if isinstance(image_file, str):
            image_file = Path(image_file)

        # want to save one dir up from the `band/cal/` dir
        cat_dir = image_file.parents[1] / 'cat/'
        utils.make_dir(str(cat_dir))

        cat_name = image_file.name.replace('.fits', '_cat.fits')
        cat_file = cat_dir / cat_name

        if cat_file.is_file():
            if overwrite is False:
                raise OSError('Catalog already exists and overwrite is False!' +
                            f'\n{str(cat_file)}')
            else:
                cat_file.unlink()

        image_file = str(image_file)
        cat_file = str(cat_file)

        wgt_type = 'MAP_WEIGHT'
        wgt_image = f'{image_file}[1]'

        # NOTE: This is probably already in the config file, but let's be sure
        # -BACKGROUND: Background-subtracted image
        # BACKGROUND: The background solution used
        # BACKGROUND_RMS: The background noise map
        checkimage_type = '-BACKGROUND,BACKGROUND,BACKGROUND_RMS'

        sub_name = image_file.replace('.fits', '_sub.fits')
        bkg_name = image_file.replace('.fits', '_bkg.fits')
        bkg_rms_name = image_file.replace('.fits', '_bkg_rms.fits')

        checkimage_name = f'{sub_name},{bkg_name},{bkg_rms_name}'

        # keep track of these output filenames for later collation
        self.backgrounds[Path(image_file)] = {
            'sub': Path(sub_name),
            'bkg': Path(bkg_name),
            'bkg_rms': Path(bkg_rms_name),
        }

        # now setup a few additional default configuration files
        config_dir = Path(utils.MODULE_DIR) / 'oba/configs/sextractor/'

        # this sets the photometric parameters that SExtractor computes
        param_file = str(config_dir / 'sb_sextractor.param')

        # this sets the detection filter
        filter_file = str(config_dir / 'default.conv')

        # this sets the neural network for the star classifier
        nnw_file = str(config_dir / 'default.nnw')

        args = {
            'CATALOG_NAME': cat_file,
            'WEIGHT_TYPE': wgt_type,
            'WEIGHT_IMAGE': wgt_image,
            'CHECKIMAGE_TYPE': checkimage_type,
            'CHECKIMAGE_NAME': checkimage_name,
            'PARAMETERS_NAME': param_file,
            'FILTER_NAME': filter_file,
            'STARNNW_NAME': nnw_file,
        }

        cmd_args = ''
        for name, val in args.items():
            cmd_args += f' -{name} {val}'

        return cmd_args

    def _run_sextractor(self, cmd, logprint):
        '''
        cmd: str
            The SExtractor command to run
        logprint: utils.LogPrint
            A LogPrint instance for simultaneous logging & printing
        '''

        logprint()
        logprint(f'SExtractor cmd:\n{cmd}')
        logprint()

        try:
            rc = utils.run_command(cmd)

            logprint(f'SExtractor run completed successfully')
            logprint()

        except Exception as e:
            logprint()
            logprint('WARNING: SExtractor failed with the following ' +
                          f'error:')
            raise e

        return

    def collate(self, logprint):
        '''
        Collate the produced SExtractor check-images into a single multi-
        extension fits file, as well as update the SCI_CAL image with
        the background-subtracted image (overwrite check is done earlier)

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
            for i, image in enumerate(self.images[band]):
                image_name = image.name
                logprint(f'Collating {image_name}; {i+1} of {Nimages}')

                sub_file = self.backgrounds[image]['sub']
                bkg_file = self.backgrounds[image]['bkg']
                rms_file = self.backgrounds[image]['bkg_rms']

                sub = fitsio.read(str(sub_file))
                bkg = fitsio.read(str(bkg_file))
                rms = fitsio.read(str(rms_file))

                # NOTE: couldn't get fitsio to work correctly for
                # updating ext images in-place...
                with fits.open(str(image), mode='update') as f:
                    # update sci image with the background-subtracted imaged
                    logprint('Updating SCI image...')
                    f[sci_ext].data = sub
                    f[sci_ext].name = 'SCI'

                    # update weight map given the bkg_rms field
                    logprint('Updating WGT image...')
                    wgt = f[wgt_ext].data
                    new_wgt = self._update_wgt_map(wgt, rms)
                    f[wgt_ext].data = new_wgt
                    f[wgt_ext].name = 'WGT'

                    f[msk_ext].name = 'MSK'

                    # add bkg image to file
                    logprint('Adding BKG image...')
                    bkg_hdr = fits.Header({
                        'IMTYPE': 'BACKGROUND'
                    })
                    bkg_hdu = fits.ImageHDU(bkg, header=bkg_hdr)
                    f.append(bkg_hdu)
                    f[bkg_ext].name = 'BKG'

                    # updates file
                    f.flush()

                # now cleanup check-images
                for fname in [sub_file, bkg_file, rms_file]:
                    fname.unlink()

        return

    def _update_wgt_map(self, wgt, bkg_rms):
        '''
        Update the existing `cal` weight map (at this stage just 1's and 0's)
        with the background rms map

        wgt: np.array
            The existing weight map to update
        bkg_rms: np.array
            The SExtractor BACKGROUND_RMS check-image
        '''

        # the initial weight map saved during the call to cal.py is just
        # the complement of the mask. Thus the weight map should now be
        # just the background RMS map with masked pixels given zero weight
        new_wgt = 1. / bkg_rms**2
        new_wgt[wgt == 0] = 0

        return new_wgt



