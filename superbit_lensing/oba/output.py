from pathlib import Path
from glob import glob

import ipdb

from superbit_lensing import utils
from superbit_lensing.cookiecutter import CookieCutter
from oba_io import band2index

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
                 out_sci_dtype='i2', out_msk_dtype='i1'):
        '''
        # TODO: remove if we don't end up needing it!
        # config_file: pathlib.Path
        #     The filepath of the base CookieCutter config
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
        '''

        args = {
            # 'config_file': (config_file, Path),
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
            'out_sci_dtype': (out_sci_dtype, str),
            'out_msk_dtype': (out_msk_dtype, str),
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

        # this dictionary will store dict's of image filepaths & their
        # extensions for each set of observations, indexed
        # by band & raw image filepath
        self.images = {}

        self.det_coadd = self.run_dir / f'det/coadd/{target_name}_coadd_det.fits'
        self.det_cat = self.run_dir / f'det/cat/{target_name}_coadd_det_cat.fits'

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

            band_dir = (run_dir / band).resolve()

            outdir = band_dir / 'out/'
            outfile = f'{target_name}_{band}_cutouts.fits'
            utils.make_dir(outdir)

            # set band-dependent quantities
            config = base_config.copy()
            config['output']['dir'] = str(outdir)
            config['output']['filename'] = str(outfile)
            config['output']['overwrite'] = overwrite

            # a dictionary indexed by raw SCI filepath
            images = self.images[band]

            for image, image_map in images.items():
                logprint(f'Writing configuration for {image.name}')

                # now for the per-image configuration
                config[image.name] = {}
                for key, cckey in self.cookiecutter_keymap.items():
                    image_map[cckey] = image_map[key]
                    config[image.name][cckey] = image_map[key]

            # write one CookieCutter config per band
            config_outfile = outdir / f'{target_name}_{band}_cookiecutter.yaml'
            if config_outfile.is_file():
                if overwrite is False:
                    raise OSError(f'{str(config_outfile)} already exists and ' +
                                  'overwrite is False!')
                else:
                    config_outfile.unlink()

            utils.write_yaml(config, str(config_outfile))
            logprint(f'Wrote CookieCutter config to {str(config_outfile)}')

        return

    def _build_base_config(self):
        '''
        Build the base CookieCutter config that is band-independent

        NOTE: This is done through the opt args for the class for now,
        but we could require a config instead
        '''

        # we won't be using the full paths as we are setting a input dir
        det_cat = f'cat/{self.det_cat.name}'

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
            'output': {
                'sci_dtype': self.out_sci_dtype,
                'msk_dtype': self.out_msk_dtype,
            },
            }

        return base_config

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
