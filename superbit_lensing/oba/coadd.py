from pathlib import Path
import os
from glob import glob
from astropy.wcs import WCS
import fitsio

from superbit_lensing import utils
from superbit_lensing.coadd import SWarpRunner

import ipdb

class CoaddRunner(object):
    '''
    Runner class for coadding each calibrated, bkg-subtracted
    sci image for each band (as well as a detection band)
    for the SuperBIT onboard analysis (OBA).

    NOTE: At this stage, input calibrated images should have a
    WCS solution in the header and the following structure:

    ext0: SCI (calibrated & background-subtracted)
    ext1: WGT (weight; 0 if masked, 1/sky_var otherwise)
    ext2: MSK (mask; 1 if masked, 0 otherwise)
    ext3: BKG (background)
    '''

    def __init__(self, config_file, run_dir, bands, det_bands,
                 target_name=None, sci_ext=0, wgt_ext=1):
        '''
        config_file: pathlib.Path
            The filepath of the base SWarp config
        run_dir: pathlib.Path
            The OBA run directory for the given target
        bands: list of str's
            A list of band names to make coadds for
        det_bands: list of str's
            A list of band names to use for the detection coadd
        target_name: str
            The name of the target. Default is to use the end of
            run_dir
        sci_ext: int
            The science frame fits extension
        wgt_ext: int
            The weight frame fits extension
        '''

        args = {
            'config_file': (config_file, Path),
            'run_dir': (run_dir, Path),
            'bands': (bands, list),
            'det_bands': (bands, list),
            'sci_ext': (sci_ext, int),
            'wgt_ext': (wgt_ext, int)
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

        # this dictionary will store the coadd image filepaths (sci & wgt)
        # indexed by both band and ext
        self.coadds = {}

        self.outfile_base = f'{target_name}_coadd'

        # these dicts are lists of image files (str's not Path's!) indexed
        # by band, including FITS extension.These are passed directly
        # to SWarp
        self.sci_images = {}
        self.wgt_images = {}

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

        logprint('Making single-band coadd images...')
        self.make_coadds(logprint, overwrite=overwrite)

        # TODO: implement!
        # logprint('Making detection coadd image...')

        logprint('Collating coadd extensions...')
        self.collate_extensions(logprint)

        return

    def gather_images(self, logprint=None):
        '''
        logprint: utils.LogPrint
            A LogPrint instance for simultaneous logging & printing
        '''

        # have to add this edge case in case the standard
        # get_{sci/wgt}_images() gets called
        if logprint is None:
            logprint = print

        sci_ext = self.sci_ext
        wgt_ext = self.wgt_ext

        for band in self.bands:
            logprint(f'Starting band {band}')

            cal_dir = (self.run_dir / band / 'cal/').resolve()

            self.images[band] = glob(
                str(cal_dir / f'{self.target_name}*_{band}_*_cal.fits')
                )

            # useful format for later SWarp args
            self.sci_images[band] = [
                f'{image}[{sci_ext}]' for image in self.images[band]
                ]

            self.wgt_images[band] = [
                f'{image}[{wgt_ext}]' for image in self.images[band]
                ]

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

    def make_coadds(self, logprint, overwrite=False):
        '''
        Make a coadd image using SWarp for each band

        logprint: utils.LogPrint
            A LogPrint instance for simultaneous logging & printing
        overwrite: bool
            Set to True to overwrite existing coadd files
        '''

        for band in self.bands:
            logprint(f'Starting band {band}')
            self.coadds[band] = {}

            outdir = (self.run_dir / band / 'coadd/').resolve()
            outfile = outdir / f'{self.outfile_base}_{band}.fits'

            if outfile.is_file():
                if overwrite is False:
                    raise OSError(f'{outfile} already exists and '
                                  'overwrite is False!')
                else:
                    logprint(f'{outfile} exists; deleting as ' +
                                  'overwrite is True')
                    outfile.unlink()

            self._run_swarp(logprint, band, outfile)

            self.coadds[band]['sci'] = outfile
            # self.coadds[band]['wgt'] = outfile.replace('.fits', '.wgt.fits')
            self.coadds[band]['wgt'] = Path(
                outdir / outfile.name.replace('.fits', '.wgt.fits')
                )

        if len(self.coadds) != len(self.bands):
            logprint('WARNING: The number of produced coadds does not ' +
                          'equal the number of passed bands; something ' +
                          'likely has failed!')

        return

    def _run_swarp(self, logprint, band, outfile, detection=False):
        '''
        logprint: utils.LogPrint
            A LogPrint instance for simultaneous logging & printing
        band: str
            The band to make a coadd of
        outfile: pathlib.Path
            The path of the output coadd file
        detection: bool
            Set to True to indicate you are making a detection image
        '''

        if band not in self.bands:
            # make sure band is in self.bands, except for the detection image
            if detection is False:
                raise ValueError(f'{band} is not a registered band!')

        swarp_cmd = self._setup_swarp_cmd(
            band, outfile, detection=detection
            )

        logprint()
        logprint(f'SWarp cmd: {swarp_cmd}')
        os.system(swarp_cmd)
        logprint(f'SWarp completed for band {band}')
        logprint()

        # move any extra created files if needed
        outdir = outfile.parent
        if os.getcwd() != outdir:
            logprint('Cleaning up local directory...')
            cmd = f'mv *.xml *.fits {outdir}'
            logprint()
            logprint(cmd)
            os.system(cmd)
            logprint()

        return

    def _setup_swarp_cmd(self, band, outfile, detection=False):
        '''
        band: str
            The band to make a coadd of
        outfile: pathlib.Path
            The path of the output coadd file
        detection: bool
            Set to True to indicate you are making a detection image
        '''

        outdir = outfile.parent

        # a few common cmd args
        config_arg = '-c ' + str(self.config_file)
        weight_outfile = Path(
            outdir / outfile.name.replace('.fits', '.wgt.fits')
            )
        resamp_arg = '-RESAMPLE_DIR ' + str(outdir)
        outfile_arg = '-IMAGEOUT_NAME '+ str(outfile) + ' ' +\
                      '-WEIGHTOUT_NAME ' + str(weight_outfile)

        if detection is False:
            # normal coadds are made from resampling from all single-epoch
            # exposures (& weights) for a given band & target
            sci_im_args = ' '.join(self.sci_images[band])
            wgt_im_args = ','.join(self.wgt_images[band])

            image_args = f'{sci_im_args} -WEIGHT_IMAGE {wgt_im_args}'

            # use config value for single-band
            ctype_arg = ''

        else:
            # detection coadds resample from the single-band coadds
            sci_im_list = [self.coadds[b]['sci'] for b in self.det_bands]
            wgt_im_list = [self.coadds[b]['wgt'] for b in self.det_bands]

            sci_im_args = ' '.join(sci_im_list)
            wgt_im_args = ','.join(wgt_im_list)

            image_args = f'{sci_im_args} -WEIGHT_IMAGE {wgt_im_args}'

            # DES suggests using AVERAGE instead of CHI2 or WEIGHTED
            ctype_arg = '-COMBINE_TYPE AVERAGE'

        cmd = ' '.join([
            'swarp ', image_args, resamp_arg, outfile_arg, config_arg, ctype_arg
            ])

        return cmd

    def collate_extensions(self, logprint):
        '''
        Collate all single-extension coadd images into a multi-extension
        FITS file per band

        logprint: utils.LogPrint
            A LogPrint instance for simultaneous logging & printing
        '''

        for band, coadds in self.coadds.items():
            logprint(f'Collating coadd extensions for {band} band')
            sci_file = coadds['sci']
            wgt_file = coadds['wgt']

            wgt, wgt_hdr = fitsio.read(str(wgt_file), header=True)

            with fitsio.FITS(str(sci_file), 'rw') as fits:
                # adds 1 to the extension number, so do it in order
                # (sci, wgt)
                fits.write(wgt, header=wgt_hdr)

            # now cleanup old wgt files
            wgt_file.unlink()

        return
