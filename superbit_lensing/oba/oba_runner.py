from pathlib import Path
from copy import deepcopy
import os

from superbit_lensing import utils
from oba_io import IOManager
from preprocess import PreprocessRunner

import ipdb

class OBARunner(object):
    '''
    Runner class for the SuperBIT onboard analysis. For now,
    only analyzes cluster lensing targets
    '''

    _req_fields = []
    _opt_fields = {}

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
                 bands, det_bands, logprint):
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
        '''

        inputs = {
            # 'config_file' TODO: add if needed later!
            'io_manager': (io_manager, IOManager),
            'target_name': (target_name, str),
            'bands': (bands, list),
            'det_bands': (det_bands, list),
            'logprint': (logprint, utils.LogPrint),
        }
        for name, tup in inputs.items():
            var, cls = tup
            utils.check_type(name, var, cls)
            setattr(self, name, var)

        self.parse_config()
        self.parse_bands()

        # set relevant dirs for the oba of the given target
        self.set_dirs()

        return

    def parse_config(self):
        pass

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

        overwrite: bool
            Set to overwrite existing files
        '''

        target = self.target_name

        self.logprint(f'Starting onboard analysis for target {target}')

        self.logprint('Starting preprocessing...')
        self.run_preprocessing(overwrite=overwrite)

        self.logprint('Applying image calibrations...')
        self.run_calibrations(overwrite=overwrite)

        self.logprint(f'Onboard analysis completed for target {target}')

        return

    def run_preprocessing(self, overwrite=False):
        '''
        Do basic setup of temp oba run directory, copy raw sci
        frames to that dir, and uncompress files

        overwrite: bool
            Set to overwrite existing files
        '''

        runner = PreprocessRunner(
            self.raw_dir,
            self.run_dir,
            self.out_dir,
            self.bands,
            target_name=self.target_name
            )

        runner.go(self.logprint, overwrite=overwrite)

        return

    def run_calibrations(self, overwrite=False):
        '''
        overwrite: bool
            Set to overwrite existing files
        '''

        runner = CalsRunner(
            self.run_dir,
            self.bands,
            target_name=self.target_name
            )

        runner.go(self.logprint, overwrite=overwrite)

        return
