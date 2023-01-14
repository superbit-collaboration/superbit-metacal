from pathlib import Path
from copy import deepcopy
import os

from superbit_lensing import utils
from oba_io import IOManager
from preprocess import PreprocessRunner
from cals import CalsRunner
from masking import MaskingRunner
from background import BackgroundRunner

import ipdb

class OBARunner(object):
    '''
    Runner class for the SuperBIT onboard analysis. For now,
    only analyzes cluster lensing targets
    '''

    # fields for the config file
    _req_fields = ['include']
    _opt_fields = {}

    # if no config is passed, use the default include list
    _default_include= [
        'preprocessing',
        'cals',
        'masking',
        'background',
        'coadd',
        'detection',
        # ...
    ]

    # The QCC uses ints for the band when writing out the
    # exposure filenames
    _bindx = {
        'u': 0,
        'b': 1,
        'g': 2,
        'dark': 3,
        'r': 4,
        'nir': 2,
        'lum': 3
    }

    _allowed_bands = _bindx.keys()

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

        self.include = self.config['include']

        return

    def _make_default_config(self):
        '''
        Make a basic config if one is not passed
        '''

        self.config = {}
        self.config['include'] = _default_include

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

        # for now, we only analyze cluster lensing targets
        self.raw_dir = self.io_manager.RAW_CLUSTERS / self.target_name

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

        # for now, we only analyze cluster lensing targets
        clusters_dir = self.io_manager.OBA_CLUSTERS

        self.run_dir = clusters_dir / self.target_name

        return

    def set_out_dir(self):
        '''
        Determine the permanent output directory for the target
        given the registered IOManager
        '''

        # for now, we only analyze cluster lensing targets
        oba_results = self.io_manager.OBA_RESULTS / 'clusters'

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

        # TODO: Need to sort out params, filter, etc. and set!
        self.configs['swarp'] = {}
        self.configs['sextractor'] = {}

        for band in self.bands:
            self.configs['sextractor'][band] = sex_dir / \
                f'sb_sextractor_{band}.config'

            # NOTE: for now, just a single config, but can be updated
            self.configs['swarp'][band] = swarp_dir / 'swarp.config'

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
        7) MEDS-maker
        8) Compression & cleanup

        NOTE: you can choose which subset of these steps to run using the
        `include` field in the OBA config file, but in most instances this
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

        self.logprint(f'\nOnboard analysis completed for target {target}')

        return

    def run_preprocessing(self, overwrite=False):
        '''
        Do basic setup of temp oba run directory, copy raw sci
        frames to that dir, and uncompress files

        overwrite: bool
            Set to overwrite existing files
        '''

        if 'preprocessing' not in self.include:
            self.logprint('Skipping preprocessing given config include')
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

        if 'cals' not in self.include:
            self.logprint('Skipping image calibrations given config include')
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

        if 'masking' not in self.include:
            self.logprint('Skipping image masking given config include')
            return

        runner = MaskingRunner(
            self.run_dir,
            self.bands,
            target_name=self.target_name
            )

        runner.go(self.logprint, overwrite=overwrite)

        return

    def run_background(self, overwrite=True):
        '''
        overwrite: bool
            Set to overwrite existing files
        '''

        if 'background' not in self.include:
            self.logprint('Skipping background estimation given config include')
            return

        runner = BackgroundRunner(
            self.run_dir,
            self.bands,
            self.configs['sextractor']['bkg'],
            target_name=self.target_name
            )

        runner.go(self.logprint, overwrite=overwrite)

        return
