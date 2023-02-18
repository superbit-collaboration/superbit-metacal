from pathlib import Path
from copy import deepcopy

from superbit_lensing import utils
from oba_io import IOManager, BAND_INDEX
from preprocess import PreprocessRunner
from cals import CalsRunner
from masking import MaskingRunner
from background import BackgroundRunner
from astrometry import AstrometryRunner
from coadd import CoaddRunner
from detection import DetectionRunner
# from output import CookieCutterRunner
# from cleanup import CleanupRunner

import ipdb

class OBARunner(object):
    '''
    Runner class for the SuperBIT onboard analysis. For now,
    only analyzes cluster lensing targets
    '''

    # fields for the config file
    _req_fields = [
        'modules'
        ]
    _opt_fields = {
        'masking': None,
        }

    # if no config is passed, use the default modules list
    _default_modules = [
        'preprocessing',
        'cals',
        'masking',
        'background',
        'astrometry',
        'coadd',
        'detection',
        'cookiecutter',
        'cleanup'
        # ...
    ]

    # The QCC uses ints for the band when writing out the
    # exposure filenames
    _bindx = BAND_INDEX

    _allowed_bands = _bindx.keys()

    # TODO: Determine which bands we will use in the detection image!
    det_bands = [
        'b',
        'lum',
        # ...
    ]

    def __init__(self, config_file, io_manager, target_name,
                 bands, det_bands, logprint, test=False):
        '''
        config_file: str
            The OBA configuration file
        io_manager: oba_io.IOManager
            An IOManager instance that defines all relevant OBA
            path information
        target_name: str
            The name of the target to run the OBA on
        bands: list of str's
            A list of band names to process
        det_bands: list of str's
            A list of band names to use when creating detection image
        logprint: utils.LogPrint
            A LogPrint instance for simultaneous logging & printing
        test: bool
            Set to indicate that this is a test run. Useful for skipping
            some checks to speedup test run
        '''

        inputs = {
            'config_file': (config_file, Path),
            'io_manager': (io_manager, IOManager),
            'target_name': (target_name, str),
            'bands': (bands, list),
            'det_bands': (det_bands, list),
            'logprint': (logprint, utils.LogPrint),
            'test': (test, bool)
        }
        for name, tup in inputs.items():
            var, cls = tup
            utils.check_type(name, var, cls)
            setattr(self, name, var)

        self.parse_config()
        self.parse_bands()

        # set relevant dirs for the oba of the given target
        self.set_dirs()

        # setup & register the various config files that will get used
        # during many of the OBA processing steps
        self.setup_configs()

        return

    def parse_config(self):
        # TODO: Do additional parsing!
        if self.config_file is not None:
            config = utils.read_yaml(self.config_file)
            self.config = utils.parse_config(
                config, self._req_fields, self._opt_fields
                )
        else:
            # make a very simple config consistent with a
            # real data run
            self.config = self._make_default_config()

        self.modules = self.config['modules']

        return

    def _make_default_config(self):
        '''
        Make a basic config if one is not passed
        '''

        self.config = {}
        self.config['modules'] = _default_modules

        return

    def parse_bands(self):
        '''
        Band variables already passed initial type checks
        '''

        if len(self.bands) == 0:
            raise ValueError('Must pass at least one band!')
        if len(self.det_bands) == 0:
            raise ValueError('Must pass at least one det band!')

        inputs = {
            'bands': self.bands,
            'det_bands': self.det_bands
        }
        for name, bnds in inputs.items():
            for b in bnds:
                if not isinstance(b, str):
                    raise TypeError(f'{name} must be filled with str\'s!')

        return

    def set_dirs(self):

        self.set_raw_dir()
        self.set_cals_dirs()
        self.set_run_dir()
        self.set_out_dir()

        return

    def set_raw_dir(self):
        '''
        Determine the raw data directory for the target given the
        registered IOManager
        '''

        self.raw_dir = self.io_manager.RAW_DATA

        return


    def set_cals_dirs(self):
        '''
        Determine the calibration directories for the target given the
        registered IOManager
        '''

        self.darks_dir = self.io_manager.DARKS
        self.flats_dir = self.io_manager.FLATS

        return

    def set_run_dir(self):
        '''
        Determine the temporary run directory given the registered
        IOManager
        '''

        oba_dir = self.io_manager.OBA_DIR

        self.run_dir = oba_dir / self.target_name

        return

    def set_out_dir(self):
        '''
        Determine the permanent output directory for the target
        given the registered IOManager
        '''

        oba_results = self.io_manager.OBA_RESULTS

        self.out_dir = oba_results / self.target_name

        return

    def setup_configs(self):
        '''
        Many steps of the OBA will require input configs, such as for SWarp
        and SExtractor. Manage the registration of these config files here
        '''

        # will hold the filepaths for all config files, indexed by type
        self.configs = {}

        configs_dir = Path(utils.MODULE_DIR) / 'oba/configs/'
        sex_dir = configs_dir / 'sextractor/'
        swarp_dir = configs_dir / 'swarp/'


        # NOTE: for now, just a single config, but can be updated for
        # multi-band if needed
        self.configs['swarp'] = swarp_dir / 'swarp.config'

        self.configs['sextractor'] = {}

        for band in self.bands:
            self.configs['sextractor'][band] = sex_dir / \
                f'sb_sextractor_{band}.config'

        # treat `det` as a derived band
        self.configs['sextractor']['det'] = sex_dir / \
                'sb_sextractor_det.config'

        # Some extra SExtractor config files
        self.configs['sextractor']['param'] = sex_dir / 'sb_sextractor.param'
        self.configs['sextractor']['filter'] = sex_dir / 'default.conv'
        self.configs['sextractor']['nnw'] = sex_dir / 'default.nnw'

        # for now, a single config for bkg estimation
        self.configs['sextractor']['bkg'] = sex_dir / \
            'sb_sextractor_bkg.config'

        return

    def go(self, overwrite=False):
        '''
        Run all of the required on-board analysis steps in the following order:

        0) Preprocessing
        1) Basic calibrations
        2) Masking
        3) Background estimation
        4) Astrometry
        5) Coaddition (single-band & detection image)
        6) Source detection
        7) Cookie-Cutter (output MEDS-like format)
        8) Compression & cleanup (TODO)

        NOTE: you can choose which subset of these steps to run using the
        `modules` field in the OBA config file, but in most instances this
        should only be done for testing

        overwrite: bool
            Set to overwrite existing files
        '''

        target = self.target_name

        self.logprint(f'\nStarting onboard analysis for target {target}')

        self.logprint('\nStarting preprocessing')
        self.run_preprocessing(overwrite=overwrite)

        self.logprint('\nStarting image calibrations')
        self.run_calibrations(overwrite=overwrite)

        self.logprint('\nStarting image masking')
        self.run_masking(overwrite=overwrite)

        self.logprint('\nStarting background estimation')
        self.run_background(overwrite=overwrite)

        self.logprint('\nStarting astrometric registration')
        self.run_astrometry(overwrite=overwrite)

        self.logprint('\nStarting coaddition')
        self.run_coaddition(overwrite=overwrite)

        self.logprint('\nStarting source detection')
        self.run_detection(overwrite=overwrite)

        self.logprint('\nStarting cookie cutter ')
        self.run_cookie_cutter(overwrite=overwrite)

        # TODO: Current refactor point!
        # self.logprint('\nStarting cleanup')
        # self.run_cleanup(overwrite=overwrite)

        self.logprint(f'\nOnboard analysis completed for target {target}')

        return

    def run_preprocessing(self, overwrite=False):
        '''
        Do basic setup of temp oba run directory, copy raw sci
        frames to that dir, and uncompress files

        overwrite: bool
            Set to overwrite existing files
        '''

        if 'preprocessing' not in self.modules:
            self.logprint('Skipping preprocessing given config modules')
            return

        runner = PreprocessRunner(
            self.raw_dir,
            self.run_dir,
            self.out_dir,
            self.bands,
            target_name=self.target_name
            )

        if self.test is True:
            skip_decompress = True
        else:
            skip_decompress = False

        runner.go(
            self.logprint, overwrite=overwrite, skip_decompress=skip_decompress
            )

        return

    def run_calibrations(self, overwrite=False):
        '''
        overwrite: bool
            Set to overwrite existing files
        '''

        if 'cals' not in self.modules:
            self.logprint('Skipping image calibrations given config modules')
            return

        runner = CalsRunner(
            self.run_dir,
            self.darks_dir,
            self.flats_dir,
            self.bands,
            target_name=self.target_name
            )

        runner.go(self.logprint, overwrite=overwrite)

        return

    def run_masking(self, overwrite=False):
        '''
        overwrite: bool
            Set to overwrite existing files
        '''

        if 'masking' not in self.modules:
            self.logprint('Skipping image masking given config modules')
            return


        # TODO: right now we only do this for the masking step, but can
        # generalize for each module
        try:
            mask_types = self.config['masking']['types']
        except KeyError:
            mask_types = None

        runner = MaskingRunner(
            self.run_dir,
            self.bands,
            target_name=self.target_name,
            mask_types=mask_types
            )

        runner.go(self.logprint, overwrite=overwrite)

        return

    def run_background(self, overwrite=True):
        '''
        overwrite: bool
            Set to overwrite existing files
        '''

        if 'background' not in self.modules:
            self.logprint('Skipping background estimation given config modules')
            return

        runner = BackgroundRunner(
            self.run_dir,
            self.bands,
            self.configs['sextractor']['bkg'],
            target_name=self.target_name
            )

        runner.go(self.logprint, overwrite=overwrite)

        return

    def run_astrometry(self, overwrite=True, rerun=False):
        '''
        overwrite: bool
            Set to overwrite existing files
        rerun: bool
            Set to rerun astrometry even if WCS is in image header
        '''

        if 'astrometry' not in self.modules:
            self.logprint('Skipping astrometry estimation given config modules')
            return

        runner = AstrometryRunner(
            self.run_dir,
            self.bands,
            target_name=self.target_name
            )

        runner.go(self.logprint, overwrite=overwrite, rerun=rerun)

        return

    def run_coaddition(self, overwrite=True):
        '''
        overwrite: bool
            Set to overwrite existing files
        '''

        if 'coadd' not in self.modules:
            self.logprint('Skipping coaddition given config modules')
            return

        runner = CoaddRunner(
            self.configs['swarp'],
            self.run_dir,
            self.bands,
            self.det_bands,
            target_name=self.target_name
            )

        runner.go(self.logprint, overwrite=overwrite)

        return

    def run_detection(self, overwrite=True):
        '''
        overwrite: bool
            Set to overwrite existing files
        '''

        if 'detection' not in self.modules:
            self.logprint('Skipping detection given config modules')
            return

        runner = DetectionRunner(
            self.configs['sextractor']['det'],
            self.run_dir,
            target_name=self.target_name
            )

        runner.go(self.logprint, overwrite=overwrite)

        return

    def run_cookie_cutter(self, overwrite=True):
        '''
        overwrite: bool
            Set to overwrite existing files
        '''

        if 'cookiecutter' not in self.modules:
            self.logprint('Skipping cookiecutter given config modules')
            return

        # runner = CookieCutterRunner(
        #     self.run_dir,
        #     target_name=self.target_name
        #     )

        # runner.go(self.logprint, overwrite=overwrite)

        return

    def run_cleanup(self, overwrite=True):
        '''
        overwrite: bool
            Set to overwrite existing files
        '''

        if 'cleanup' not in self.modules:
            self.logprint('Skipping cleanup given config modules')
            return

        # runner = cleanupRunner(
        #     self.run_dir,
        #     target_name=self.target_name
        #     )

        # runner.go(self.logprint, overwrite=overwrite)

        return
