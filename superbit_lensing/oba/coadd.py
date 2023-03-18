from pathlib import Path
import os
from glob import glob
import numpy as np
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import fitsio

from superbit_lensing import utils
from superbit_lensing.coadd import SWarpRunner
from superbit_lensing.oba.oba_io import band2index

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
    ext2: MSK (mask; see bitmask.py for def)
    ext3: BKG (background)

    The output coadd images will have the following structure:
    Coadd images:
    ext0: SCI (calibrated & background-subtracted)
    ext1: WGT (weight; 0 if masked, 1/sky_var otherwise)
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
            'det_bands': (det_bands, list),
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

        # this keeps track of any bands that have no input images to coadd,
        # in case you still requested it
        # NOTE: won't work if it is a detection band!
        self.skip = []

        self.outfile_base = f'{target_name}_coadd'

        # these dicts are lists of image files (str's not Path's!) indexed
        # by band, including FITS extension. These are passed directly
        # to SWarp
        self.sci_images = {}
        self.wgt_images = {}

        # these dicts store a tuple of the bounding box of all images taken of
        # a given band, indexed by band (first world, then coadd pixels)
        self.ra_bounds = {}
        self.dec_bounds = {}
        self.x_bounds = {}
        self.y_bounds = {}

        # this dict stores the actual (Nx, Ny) size of each coadd image,
        # indexed by band
        self.coadd_size = {}

        return

    def go(self, logprint, overwrite=False):
        '''
        Make coadd images for each band, as well as a composite detection band
        using SWarp

        Steps:

        (1) Gather all target calibrated, bkg-subtracted sci images
        (2) Check that all images have a WCS solution
        (3) Determine coadd image sizes
        (4) Run SWarp for all target images of a given band, generating
            maximal coadd images (i.e. full extent of all single-epoch
            exposures, which will have non-uniform depth)
        (5) Create a detection coadd from a chosen subset of single-band coadds
        (6) Collate new coadd checkimages

        logprint: utils.LogPrint
            A LogPrint instance for simultaneous logging & printing
        overwrite: bool
            Set to overwrite existing files
        '''

        logprint('Gathering calibrated sci frames...')
        self.gather_images(logprint)

        logprint('Checking for image WCS solutions...')
        self.check_for_wcs(logprint)

        logprint('Determining coadd image sizes...')
        self.determine_coadd_sizes(logprint)

        logprint('Making single-band coadd images...')
        self.make_coadds(logprint, overwrite=overwrite)

        logprint('Making detection image...')
        self.make_detection_image(logprint, overwrite=overwrite)

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
            bindx = band2index(band)

            self.images[band] = glob(
                str(cal_dir / f'{self.target_name}*_{bindx}_*_cal.fits')
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

            if Nimages > 0:
                # to keep consistent convention with other modules, store as Paths
                for i, image in enumerate(self.images[band]):
                    image = Path(image)
                    self.images[band][i] = image
            else:
                logprint(f'Adding {band} to the skip list')

                if band in self.det_bands:
                    raise OSError(f'Cannot skip band {band} as it is a ' +
                                  'detection band!')

                self.skip.append(band)

        return

    def check_for_wcs(self, logprint):
        '''
        Make sure each input image has a WCS solution before coadding
        '''

        for band in self.bands:
            for image in self.images[band]:
                im_name = image.name
                hdr = fitsio.read_header(str(image))

                skip = False
                try:
                    wcs = WCS(hdr)

                except KeyError as e:
                    skip = True

                if wcs is None:
                    skip = True

                if skip is True:
                    logprint(f'WCS for image {im_name} not found! Will skip ' +
                             'during coaddition')

                    self.images[band].remove(image)

                    sci_image = f'{image}[{sci_ext}]'
                    wgt_image = f'{image}[{wgt_ext}]'
                    self.sci_images[band].remove(str(sci_image))
                    self.wgt_images[band].remove(str(wgt_image))

        return

    def determine_coadd_sizes(self, logprint):
        '''
        For a set of single-epoch images, determine the coadd image size

        NOTE: Images not in the detection coadd can be determined by SWarp
        automatically, but all detection images must be the same size

        logprint: utils.LogPrint
            A LogPrint instance for simultaneous logging & printing
        '''

        for band in self.bands:
            logprint(f'Starting band {band}')
            if band in self.skip:
                logprint(f'Skipping as no images were found; size=(None None)')
                ra_bounds[band] = (None, None)
                dec_bounds[band] = (None, None)
                continue

            images = self.images[band]

            min_ra = None
            max_ra = None
            min_dec = None
            max_dec = None

            # First, determine the bounding box of min/max RA & DEC
            for image in images:
                im, hdr = fitsio.read(str(image), header=True)

                wcs = WCS(hdr)
                im_shape = im.shape

                # NOTE: np index ordering is opposite of FITS!
                Nx, Ny = im_shape[1], im_shape[0]

                corners = [
                    (0, 0),
                    (0, Ny),
                    (Nx, 0),
                    (Nx, Ny)
                ]

                for corner in corners:
                    x, y = corner

                    world_pos = wcs.pixel_to_world(x, y)
                    ra = world_pos.ra.value
                    dec = world_pos.dec.value

                    if min_ra is None:
                        min_ra = ra
                    else:
                        min_ra = np.min([ra, min_ra])
                    if max_ra is None:
                        max_ra = ra
                    else:
                        max_ra = np.max([ra, max_ra])

                    if min_dec is None:
                        min_dec = dec
                    else:
                        min_dec = np.min([dec, min_dec])
                    if max_dec is None:
                        max_dec = dec
                    else:
                        max_dec = np.max([dec, max_dec])

            ra_bounds = (min_ra, max_ra)
            dec_bounds = (min_dec, max_dec)
            self.ra_bounds[band] = ra_bounds
            self.dec_bounds[band] = dec_bounds

            logprint(f'RA bounds: ({min_ra:.6f}, {max_ra:.6f})')
            logprint(f'DEC bounds: ({min_dec:.6f}, {max_dec:.6f})')

            # Next, we use the bounding box in world coords to determine how
            # large each coadd needs to be in a single image plane of our
            # given pixel scale
            coadd_corners = [
                (ra_bounds[0], dec_bounds[0]),
                (ra_bounds[0], dec_bounds[1]),
                (ra_bounds[1], dec_bounds[0]),
                (ra_bounds[1], dec_bounds[1])
            ]

            # NOTE: a bit hacky since cornish isn't currently building. Use the
            # first image per band to determine the pixel values at the
            # boundaries. Some of these will be off of the image, but it should
            # give us an accurate estimate of the total coadd image size
            im, hdr = fitsio.read(str(self.images[band][0]), header=True)
            wcs = WCS(hdr)

            min_x = None
            max_x = None
            min_y = None
            max_y = None
            for corner in coadd_corners:
                ra, dec = corner

                im_pos = wcs.world_to_pixel(SkyCoord(ra*u.deg, dec*u.deg))
                x, y = im_pos

                if min_x is None:
                    min_x = x
                else:
                    min_x = np.min([x, min_x])
                if max_x is None:
                    max_x = x
                else:
                    max_x = np.max([x, max_x])

                if min_y is None:
                    min_y = y
                else:
                    min_y = np.min([y, min_y])
                if max_y is None:
                    max_y = y
                else:
                    max_y = np.max([y, max_y])

            self.x_bounds[band] = (min_x, max_x)
            self.y_bounds[band] = (min_y, max_y)

            Nx = int(np.ceil(max_x - min_x))
            Ny = int(np.ceil(max_y - min_y))
            self.coadd_size[band] = (Nx, Ny)

            logprint(f'X bounds: ({min_x:.6f}, {max_x:.6f})')
            logprint(f'Y bounds: ({min_y:.6f}, {max_y:.6f})')
            logprint(f'Image size: ({Nx}, {Ny})')

        # lastly, we need to homogenize the coadd sizes for all images used in
        # the detection coadd
        logprint(f'Homogenizing image sizes for the detection bands ' +
                 f'{self.det_bands} (selecting maximum bounding box)')

        det_xsize = 0
        det_ysize = 0
        for band in self.det_bands:
            xsize, ysize = self.coadd_size[band]
            det_xsize = np.max([xsize, det_xsize])
            det_ysize = np.max([ysize, det_ysize])

        for band in self.det_bands:
            self.coadd_size[band] = (det_xsize, det_ysize)

        logprint(f'Detection image size: ({det_xsize}, {det_ysize})')

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
            if band in self.skip:
                logprint('Skipping as no images were found')
                continue

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
            self.coadds[band]['wgt'] = outdir / outfile.name.replace(
                '.fits', '.wgt.fits'
                )

        if len(self.coadds) != (len(self.bands) - len(self.skip)):
            logprint('WARNING: The number of produced coadds does not ' +
                     'equal the number of passed bands (minus skips); ' +
                     'something likely has failed!')

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

        if band in self.det_bands:
            xsize, ysize = self.coadd_size[band]
            size_arg = f'-IMAGE_SIZE {xsize},{ysize}'
        else:
            # single-band coadds *not* used in the detection image do not have
            # to have the same size, so let SWarp decide automatically
            size_arg = '-IMAGE_SIZE 0'

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
            sci_im_list = [str(self.coadds[b]['sci']) for b in self.det_bands]
            wgt_im_list = [str(self.coadds[b]['wgt']) for b in self.det_bands]

            sci_im_args = ' '.join(sci_im_list)
            wgt_im_args = ','.join(wgt_im_list)

            image_args = f'{sci_im_args} -WEIGHT_IMAGE {wgt_im_args}'

            # DES suggests using AVERAGE instead of CHI2 or WEIGHTED
            ctype_arg = '-COMBINE_TYPE AVERAGE'

        cmd = ' '.join([
            'swarp ',
            image_args,
            resamp_arg,
            outfile_arg,
            config_arg,
            ctype_arg,
            size_arg
            ])

        return cmd

    def make_detection_image(self, logprint, overwrite=False):
        '''
        Make a coadd image using SWarp for each band

        logprint: utils.LogPrint
            A LogPrint instance for simultaneous logging & printing
        overwrite: bool
            Set to True to overwrite existing coadd files
        '''

        for band in self.det_bands:
            if band not in self.coadds:
                raise ValueError('Cannot make detection image until all '
                                 'following single-band coadds are done: '
                                 f'{self.det_bands}')

        band = 'det'
        self.coadds[band] = {}

        outdir = self.run_dir / band / 'coadd/'

        outfile = outdir / f'{self.target_name}_coadd_{band}.fits'

        if outfile.is_file():
            if overwrite is False:
                raise OSError(f'{outfile} already exists and '
                                'overwrite is False!')
            else:
                logprint(f'{outfile} exists; deleting as ' +
                                'overwrite is True')
                outfile.unlink()

        self._run_swarp(logprint, band, outfile, detection=True)

        self.coadds[band]['sci'] = outfile
        self.coadds[band]['wgt'] = outdir / outfile.name.replace(
            '.fits', '.wgt.fits'
            )

        return

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
                # Name the main coadd image
                fits[0].write_key('EXTNAME', 'SCI')

                # adds 1 to the extension number, so do it in order
                # (sci, wgt)
                fits.create_image_hdu(
                    img=wgt,
                    dtype='i1',
                    dims=wgt.shape,
                    extname='WGT'
                    )

            # now cleanup old wgt files
            wgt_file.unlink()

        return
