from pathlib import Path
from glob import glob
import numpy as np
import fitsio

import ipdb

from superbit_lensing import utils
from superbit_lensing.cookiecutter import CookieCutter
from oba_io import band2index
from bitmask import OBA_BITMASK, OBA_BITMASK_DTYPE

class OutputRunner(object):
    '''
    Runner class for the CookieCutter output format, which takes image
    cutouts of sources from an input detection catalog (typically SExtractor)
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
    ext2: MSK (mask; see bitmask.py for def)
    ext3: BKG (background)

    Coadd images:
    ext0: SCI (calibrated & background-subtracted)
    ext1: WGT (weight; 0 if masked, 1/sky_var otherwise)
    ext2: SEG (segmentation; 0 if sky, NUMBER if pixel is assigned to an obj)
    '''

    _out_sci_dtype_default = np.dtype('uint16')
    _out_msk_dtype_default = OBA_BITMASK_DTYPE

    # The following dict defines the correspondance between our internal keys
    # to the desired output format config keys, in this case the CookieCutter.
    # Each internal key is used to map a given raw SCI exposure to its
    # corresponding set of value-added images (e.g. mask, background, etc.)
    # and their extensions
    cookiecutter_keymap = {
        'sci_file': 'image_file',
        'sci_ext': 'image_ext',
        'wgt_file': 'weight_file',
        'wgt_ext': 'weight_ext',
        'msk_file': 'mask_file',
        'msk_ext': 'mask_ext',
        # 'skyvar_file': 'skyvar_file',
        # 'skyvar_ext': 'skyvar_ext',
        'bkg_file': 'background_file',
        'bkg_ext': 'background_ext',
        'seg_file': 'segmentation_file',
        'seg_ext': 'segmentation_ext',
    }

    def __init__(self, run_dir, bands, target_name=None,
                 sci_ext=0, wgt_ext=1, msk_ext=2, bkg_ext=3,
                 coadd_sci_ext=0, coadd_wgt_ext=1, coadd_seg_ext=2,
                 id_tag='NUMBER', boxsize_tag='boxsize',
                 ra_tag='ALPHAWIN_J2000', dec_tag='DELTAWIN_J2000',
                 ra_unit='deg', dec_unit='deg',
                 out_sci_dtype=None, out_msk_dtype=None,
                 make_center_stamp=True, center_stamp_size=512):
        '''
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
        id_tag: str
            The name of the ID column in the detection catalog
        boxsize_tag: str
            The name of the boxsize column in the detection catalog
            NOTE: CookieCutter will write a boxsize col w/ this name
            if not present in the catalog
        ra_tag: str
            The name of the RA column in the detection catalog
        dec_tag: str
            The name of the DEC column in the detection catalog
        ra_unit: str
            The unit of the RA column in the detection catalog
        dec_unit: str
            The unit of the DEC column in the detection catalog
        out_sci_dtype: str
            The numpy dtype for the output SCI stamps
        out_msk_dtype: str
            The numpy dtype for the output MSK stamps
        make_center_stamp: bool
            Set to True to make a single large stamp at the target center
        center_stamp_size: int
            If making a central stamp, set the size (for now, must be square)
        '''

        args = {
            'run_dir': (run_dir, Path),
            'bands': (bands, list),
            'sci_ext': (sci_ext, int),
            'wgt_ext': (wgt_ext, int),
            'msk_ext': (msk_ext, int),
            'bkg_ext': (bkg_ext, int),
            'coadd_sci_ext': (coadd_sci_ext, int),
            'coadd_wgt_ext': (coadd_wgt_ext, int),
            'coadd_seg_ext': (coadd_seg_ext, int),
            'id_tag': (id_tag, str),
            'boxsize_tag': (boxsize_tag, str),
            'ra_tag': (ra_tag, str),
            'dec_tag': (dec_tag, str),
            'ra_unit': (ra_unit, str),
            'dec_unit': (dec_unit, str),
            'make_center_stamp': (make_center_stamp, bool),
            'center_stamp_size': (center_stamp_size, int),
        }

        for name, tup in args.items():
            val, allowed_types = tup
            utils.check_type(name, val, allowed_types)
            setattr(self, name, val)

        if target_name is None:
            target_name = run_dir.name

        utils.check_type('target_name', target_name, str)
        self.target_name = target_name

        dtype_args = {
            'out_sci_dtype': out_sci_dtype,
            'out_msk_dtype': out_msk_dtype,
        }
        for name, val in dtype_args.items():
            if val is not None:
                utils.check_type(name, val, np.dtype)
            else:
                default = getattr(self, f'_{name}_default')
                setattr(self, name, default)

        for band in self.bands:
            utils.check_type('band', band, str)

        # this dictionary will store dict's of image filepaths & their
        # extensions for each set of observations, indexed
        # by band & raw image filepath
        self.images = {}

        # this dictionary will store the CookieCutter config filepaths,
        # indexed by band
        self.config_files = {}

        # if no images are found for a given band, add it to the skip list
        self.skip = []

        self.det_coadd = self.run_dir / f'det/coadd/{target_name}_coadd_det.fits'
        self.det_cat = self.run_dir / f'det/cat/{target_name}_coadd_det_cat.fits'

        # these get set if you want to make a center stamp at the target position
        self.target_ra = None
        self.target_dec = None

        return

    def go(self, logprint, overwrite=False):
        '''
        Run the output generation step of the OBA. Make a CookieCutter
        output file for each band

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

        out_sci_dtype = self.out_sci_dtype
        out_msk_dtype = self.out_msk_dtype
        logprint(f'Using output SCI dtype of {out_sci_dtype}')
        logprint(f'Using output MSK dtype of {out_msk_dtype}')

        logprint('Gathering input images...')
        self.gather_images(logprint)

        logprint('Creating CookieCutter config files...')
        self.create_configs(logprint, overwrite=overwrite)

        logprint('Running CookieCutter...')
        self.run_cookie_cutter(logprint, overwrite=overwrite)

        return

    def gather_images(self, logprint):
        '''
        For each band, we create a nested dictionary indexed by
        raw sci image that keeps track of each corresponding
        value-added image (e.g. SCI, MSK, WGT, COADD_SEG, etc.)
        and extension

        logprint: utils.LogPrint
            A LogPrint instance for simultaneous logging & printing
        '''

        target_name = self.target_name

        sci_ext = self.sci_ext
        wgt_ext = self.wgt_ext
        msk_ext = self.msk_ext
        bkg_ext = self.bkg_ext
        coadd_sci_ext = self.coadd_sci_ext
        coadd_wgt_ext = self.coadd_wgt_ext
        coadd_seg_ext = self.coadd_seg_ext

        # the dict keys for each raw image entry in self.images[band]
        keymap = self.cookiecutter_keymap

        # the detection coadd will be useful for all bands to grab the segmap
        det_coadd_name = self.det_coadd.name

        if not self.det_coadd.is_file():
            raise OSError('Could not find det coadd {str(self.det_coadd})!')

        if self.make_center_stamp is True:
            target_ra = None
            target_dec = None

        for band in self.bands:
            logprint(f'Starting band {band}')
            self.images[band] = {}

            # unlike most modules, our base unit is the RAW image
            band_dir = (self.run_dir / band).resolve()
            cal_dir = band_dir / 'cal/'

            bindx = band2index(band)

            images = glob(
                str(band_dir / f'{target_name}*_{bindx}_*.fits')
                )

            if len(images) == 0:
                logprint(f'WARNING: Zero raw images found; skipping')
                self.skip.append(band)

            for image in images:
                image = Path(image)
                raw_name = image.name
                cal_name = raw_name.replace('.fits', '_cal.fits')

                # NOTE: We use RAW_SCI instead of CAL_SCI for our main cutout
                # NOTE: Later we will set the input dir in the CookieCutter
                # config to be the root OBA dir for the given target, so we
                # only keep pathing info relative to that dir
                image_map = {
                    'sci_file': f'{band}/{raw_name}',
                    'sci_ext': sci_ext,
                    'wgt_file': f'{band}/cal/{cal_name}',
                    'wgt_ext': wgt_ext,
                    'msk_file': f'{band}/cal/{cal_name}',
                    'msk_ext': msk_ext,
                    'bkg_file': f'{band}/cal/{cal_name}',
                    'bkg_ext': bkg_ext,
                    'seg_file': f'det/coadd/{det_coadd_name}',
                    'seg_ext': coadd_seg_ext,
                    }

                self.images[band][Path(image)] = image_map

                if self.make_center_stamp is True:
                    hdr = fitsio.read_header(str(image))

                    if target_ra is None:
                        target_ra = hdr['TRG_RA']
                    else:
                        if hdr['TRG_RA'] != target_ra:
                            raise ValueError('Inconsistent target RA values ' +
                                             'between image headers!')
                    if target_dec is None:
                        target_dec = hdr['TRG_DEC']
                    else:
                        if hdr['TRG_DEC'] != target_dec:
                            raise ValueError('Inconsistent target DEC values ' +
                                             'between image headers!')

        self.target_ra = target_ra
        self.target_dec = target_dec

        return

    def create_configs(self, logprint, overwrite=False):
        '''
        Given the earlier raw image -> value-added image mapping,
        create the CookieCutter configs necessary to run for each band

        logprint: utils.LogPrint
            A LogPrint instance for simultaneous logging & printing
        '''

        run_dir = self.run_dir
        target_name = self.target_name

        # config entries that are band-independent
        base_config = self._build_base_config()

        for band in self.bands:
            logprint(f'Starting band {band}')

            if band in self.skip:
                logprint(f'Skipping as no images were found')
                continue

            band_dir = (run_dir / band).resolve()

            outdir = band_dir / 'out/'
            outfile = f'{target_name}_{band}_cutouts.fits'
            utils.make_dir(outdir)

            # set band-dependent quantities
            config = base_config.copy()
            config['output']['dir'] = str(outdir)
            config['output']['filename'] = str(outfile)
            config['output']['overwrite'] = overwrite

            # set minimal segmentation values according to OBA bitmask
            config['segmentation']

            # a dictionary indexed by raw SCI filepath
            images = self.images[band]
            config['images'] = {}

            for image, image_map in images.items():
                logprint(f'Writing configuration for {image.name}')

                # now for the per-image configuration
                config['images'][image.name] = {}
                for key, cckey in self.cookiecutter_keymap.items():
                    config['images'][image.name][cckey] = image_map[key]

            # write one CookieCutter config per band
            config_outfile = outdir / f'{target_name}_{band}_cutouts.yaml'
            if config_outfile.is_file():
                if overwrite is False:
                    raise OSError(f'{str(config_outfile)} already exists and ' +
                                  'overwrite is False!')
                else:
                    config_outfile.unlink()

            utils.write_yaml(config, str(config_outfile))
            logprint(f'Wrote CookieCutter config to {str(config_outfile)}')
            self.config_files[band] = config_outfile

        return

    def _build_base_config(self):
        '''
        Build the base CookieCutter config that is band-independent

        NOTE: This is done through the opt args for the class for now,
        but we could require a config instead
        '''

        # grab some of the needed segmask values from the OBA bitmask
        oba_obj = OBA_BITMASK['seg_obj']
        oba_neighbor = OBA_BITMASK['seg_neighbor']

        # we won't be using the full paths as we are setting a input dir
        det_cat = f'det/cat/{self.det_cat.name}'

        base_config = {
            'input': {
                'dir': str(self.run_dir),
                'catalog': str(det_cat),
                'catalog_ext': 1,
                'id_tag': self.id_tag,
                'boxsize_tag': self.boxsize_tag,
                'ra_tag': self.ra_tag,
                'dec_tag': self.dec_tag,
                'ra_unit': self.ra_unit,
                'dec_unit': self.dec_unit,
            },
            'segmentation': {
                # this will allow us to only use 3 values for the segmap
                'type': 'minimal',
                'obj': oba_obj,
                'neighbor': oba_neighbor,
            },
            'output': {
                'sci_dtype': str(self.out_sci_dtype),
                'msk_dtype': str(self.out_msk_dtype),
                'make_center_stamp': self.make_center_stamp,
                'center_stamp_size': self.center_stamp_size,
                # The default is (None, None)
                'center_stamp_pos': [self.target_ra, self.target_dec]
            },
            }

        return base_config

    def run_cookie_cutter(self, logprint, overwrite=False):
        '''
        TODO: ...

        logprint: utils.LogPrint
            A LogPrint instance for simultaneous logging & printing
        overwrite: bool
            Set to overwrite existing files
        '''

        for band in self.bands:
            logprint(f'Starting band {band}')

            if band in self.skip:
                logprint('Skipping as no images were found')
                continue

            config_file = str(self.config_files[band])
            logprint(f'Using config file {config_file}')

            # NOTE: Can pass either the config file or the loaded config
            cutter = CookieCutter(config=config_file, logprint=logprint)

            logprint('Starting...')
            cutter.go()

        return
