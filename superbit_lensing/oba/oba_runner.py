from pathlib import Path
from copy import deepcopy

from superbit_lensing import utils
from config import OBAConfig
from oba_io import IOManager, BAND_INDEX
from preprocess import PreprocessRunner
from cals import CalsRunner
from masking import MaskingRunner
from background import BackgroundRunner
from astrometry import AstrometryRunner
from starmask import StarmaskRunner
from coadd import CoaddRunner
from detection import DetectionRunner
from output import OutputRunner
from cleanup import CleanupRunner

import ipdb

class OBARunner(object):
    '''
    Runner class for the SuperBIT onboard analysis. For now,
    only analyzes cluster lensing targets
    '''

    # The QCC uses ints for the band when writing out the
    # exposure filenames
    _bindx = BAND_INDEX

    _allowed_bands = _bindx.keys()

    def __init__(self, config_file, io_manager, logprint, test=False):
        '''
        config_file: str
            The OBA configuration file. See OBAConfig class for definition
        io_manager: oba_io.IOManager
            An IOManager instance that defines all relevant OBA
            path information
        logprint: utils.LogPrint
            A LogPrint instance for simultaneous logging & printing
        test: bool
            Set to indicate that this is a test run. Useful for skipping
            some checks to speedup test run
        '''

        inputs = {
            'config_file': (config_file, Path),
            'io_manager': (io_manager, IOManager),
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
        # if self.config_file is not None:
        #     config = utils.read_yaml(self.config_file)
        #     self.config = utils.parse_config(
        #         config, self._req_fields, self._opt_fields
        #         )
        # else:
        #     # make a very simple config consistent with a
        #     # real data run
        #     self.config = self._make_default_config()

        # NOTE: all parsing has been moved into the new OBAConfig class,
        # including default setting. See class for config def details
        self.config = OBAConfig(self.config_file)

        self.modules = self.config['modules']

        if self.modules is None:
            self.modules = []

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

    @property
    def target_name(self):
        return self.config['run_options']['target_name']

    @property
    def bands(self):
        return self.config['run_options']['bands']

    @property
    def det_bands(self):
        return self.config['run_options']['det_bands']

    def go(self, overwrite=False):
        '''
        Run all of the required on-board analysis steps in the following order:

        0) Preprocessing
        1) Basic calibrations
        2) Masking
        3) Background estimation
        4) Astrometry
        5) Bright star mask (needs astrometry)
        6) Coaddition (single-band & detection image)
        7) Source detection
        8) Cookie-Cutter (output MEDS-like cutout format)
        9) Compression & cleanup

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

        self.logprint('\nStarting bright star masking')
        self.run_starmask(overwrite=overwrite)

        self.logprint('\nStarting coaddition')
        self.run_coaddition(overwrite=overwrite)

        self.logprint('\nStarting source detection')
        self.run_detection(overwrite=overwrite)

        self.logprint('\nStarting output generation')
        self.run_output(overwrite=overwrite)

        self.logprint('\nStarting cleanup')
        self.run_cleanup(overwrite=overwrite)

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


        # had to rework this to have a consistent config depth
        cosmics = self.config['masking']['cosmic_rays']
        satellites = self.config['masking']['satellites']

        mask_types = []
        mtypes = ['cosmic_rays', 'satellites']
        for mtype in mtypes:
            if self.config['masking'][mtype] is True:
                mask_types.append(mtype)

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

    def run_astrometry(self, overwrite=True):
        '''
        overwrite: bool
            Set to overwrite existing files
        '''

        if 'astrometry' not in self.modules:
            self.logprint('Skipping astrometry estimation given config modules')
            return

        rerun = self.config['astrometry']['rerun']
        search_radius = self.config['astrometry']['search_radius']

        runner = AstrometryRunner(
            self.run_dir,
            self.bands,
            target_name=self.target_name,
            search_radius=search_radius,
            )

        runner.go(self.logprint, overwrite=overwrite, rerun=rerun)

        return

    def run_starmask(self, overwrite=True):
        '''
        overwrite: bool
            Set to overwrite existing files
        '''

        if 'starmask' not in self.modules:
            self.logprint('Skipping bright star masking given config modules')
            return

        # rerun = self.config['astrometry']['rerun']
        # search_radius = self.config['astrometry']['search_radius']

        gaia_filename = self.io_manager.gaia_filename
        gaia_cat = self.io_manager.GAIA_DIR / gaia_filename

        runner = StarmaskRunner(
            self.run_dir,
            gaia_cat,
            self.bands,
            target_name=self.target_name,
            )

        runner.go(self.logprint, overwrite=overwrite)

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

    def run_output(self, overwrite=True):
        '''
        overwrite: bool
            Set to overwrite existing files
        '''

        if 'output' not in self.modules:
            self.logprint('Skipping output given config modules')
            return

        runner = OutputRunner(
            self.run_dir,
            self.bands,
            target_name=self.target_name
            )

        runner.go(self.logprint, overwrite=overwrite)

        return

    def run_cleanup(self, overwrite=True):
        '''
        overwrite: bool
            Set to overwrite existing files
        '''

        if 'cleanup' not in self.modules:
            self.logprint('Skipping cleanup given config modules')
            return
        else:
            clean_oba_dir =  self.config['cleanup']['clean_oba_dir']

        runner = CleanupRunner(
            self.run_dir,
            self.out_dir,
            self.bands,
            target_name=self.target_name,
            clean_oba_dir=clean_oba_dir,
            )

        runner.go(self.logprint, overwrite=overwrite)

        return
