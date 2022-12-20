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

    def __init__(self, config_file, io_manager, logprint):
        '''
        config_file: str
            The OBA configuration file
        io_manager: oba_io.IOManager
            An IOManager instance that defines all relevant OBA
            path information
        logprint: utils.LogPrint
            A LogPrint instance for simultaneous logging & printing
        '''

        self.parse_config()
        self.parse_bands()

        if not isinstance(io_manager, IOManager):
            raise TypeError('io_manager must be a IOManager instance!')
        self.io_manager = io_manager

        if not isinstance(logprint, utils.LogPrint):
            raise TypeError('logprint must be a LogPrint instance!')
        self.logprint = logprint

        # TODO: something about target name...
        self.target_name = None

        # set relevant dirs for the oba of the given target
        self.set_dirs()

        return

    def parse_config(self):
        pass

    def parse_bands(self):
        pass

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

    def go(self):
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

        TODO: ...
        '''

        target = self.target_name

        self.logprint('Starting onboard analysis for target {target}')

        self.logprint('Starting preprocessing...')
        self.run_preprocessing()

        self.logprint('Onboard analysis completed for target {target}')

        return

    def run_preprocessing(self):
        '''
        Do basic setup of temp oba run directory, copy raw sci
        frames to that dir, and uncompress files
        '''

        runner = PreprocessRunner(
            self.raw_dir,
            self.run_dir,
            self.out_dir,
            self.bands,
            target_name=self.target_name
            )

        return

