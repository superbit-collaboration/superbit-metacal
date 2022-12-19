import os
from pathlib import Path

from superbit_lensing import utils

import ipdb

class IOManager(object):
    '''
    A small class that handles all filepath definitions for
    the SuperBIT on-board analysis (OBA). While the actual
    run directories are already fixed, this I/O manager can
    assign all paths relative to a local directory for testing
    if desired

    QCC dir structure:

    ----------------------------------------------------------------------
    Raw data

    Root data dir:
    RAW_DATA: /data/bit/science_images/

    Raw cluster science images:
    RAW_CLUSTERS: RAW_DATA/clusters/

    Raw wl science images for a given target:
    RAW_CLUSTERS/{TARGET_NAME}/

    ----------------------------------------------------------------------
    On-board analysis

    Root oba dir:
    OBA_DIR: /home/bit/oba_temp/

    Root oba dir for clusters:
    OBA_CLUSTERS: OBA_DIR/clusters/

    Temporary analysis dir for a given cluster target:
    OBA_CLUSTERS/{TARGET_NAME}/

    Temporary analysis files per band for a given target:
    OBA_CLUSTERS/{TARGET_NAME}/{BAND}/

    Possible bands are [‘u’, ‘b’, ‘g’, ‘r’, ‘nir’, ‘lum’, ‘det’]

    NOTE: ‘det’ is not an actual filter, but a combination of bands (derived)

    Temporary output analysis files (coadds, MEDS, & logs for now)
    for a given band & target:
    OBA_CLUSTERS/{TARGET_NAME}/out/

    ----------------------------------------------------------------------
    Permanent OBA outputs

    OBA results root dir:
    OBA_RESULTS: /data/bit/oba_results/

    OBA results for clusters:
    OBA_RESULTS/clusters/

    OBA results for a given cluster target:
    OBA_RESULTS/clusters/{TARGET_NAME}/

    ----------------------------------------------------------------------
    Testing on a local device (i.e. *not* qcc)

    While all of the above paths are fixed for the qcc, you may want to run
    tests locally on a different machine. In this case, use the optional
    constructor arg `root_dir` to prepend all paths with your desired dir

    ----------------------------------------------------------------------
    Example useage:

    root_dir = /my/favorite/dir/

    io_manager = IOManager(root_dir=root_dir)

    # returns {root_dir}/home/bit/oba_temp/
    oba_dir = io_manager.OBA_DIR
    ...

    '''

    # Default is registered to the above hard-coded qcc paths
    # Will update

    _registered_dir_names = [
        'RAW_DATA',
        'RAW_CLUSTERS',
        'OBA_DIR',
        'OBA_CLUSTERS',
        'OBA_RESULTS',
        ]

    def __init__(self, root_dir=None):
        '''
        root_dir: str
            The root directory that defines all other SuperBIT
            OBA filepaths relative to. This is useful if you want
            to test or simulate a real run locally.
        '''

        if root_dir is not None:
            if not isinstance(root_dir, (str, Path)):
                raise TypeError('root_dir must be a str or pathlib.Path!')
        else:
            # will allow for uniform dir setting
            root_dir = '/'

        self.root_dir = Path(root_dir).resolve()

        # only makes dir if it does not yet exist
        utils.make_dir(self.root_dir)

        self.registered_dirs = {}
        self.register_dirs()

        return

    def register_dirs(self):
        '''
        Register the needed directories given the defaults
        and passed root_dir, if desired
        '''

        # defaults for qcc (root / added later)
        _registered_defaults = {
            'RAW_DATA': 'data/bit/science_images/',
            'RAW_CLUSTERS': 'data/bit/science/images/clusters/',
            'OBA_DIR': 'home/bit/oba_temp/',
            'OBA_CLUSTERS': 'home/bit/oba_temp/clusters/',
            'OBA_RESULTS': 'data/bit/oba_results/',
            }

        self.registered_dirs = {}

        for name, default in _registered_defaults.items():
            # works for all cases as default root_dir is "/"
            self.registered_dirs[name] = self.root_dir / default

        return

    def _check_dir(self, name):
        if name not in self.registered_dirs:
            raise ValueError(f'{name} is not registered yet!')
        return self.registered_dirs[name]

    @property
    def RAW_DATA(self):
        name = 'RAW_DATA'
        return self._check_dir(name)

    @property
    def RAW_CLUSTERS(self):
        name = 'RAW_CLUSTERS'
        return self._check_dir(name)

    @property
    def OBA_DIR(self):
        name = 'OBA_DIR'
        return self._check_dir(name)

    @property
    def OBA_CLUSTERS(self):
        name = 'OBA_CLUSTERS'
        return self._check_dir(name)

    @property
    def OBA_RESULTS(self):
        name = 'OBA_RESULTS'
        return self._check_dir(name)

    def print_dirs(self, logprint=None):
        '''
        Print out all registered directories

        logprint: utils.LogPrint
            A LogPrint instance if you want to print out to a log
        '''

        if logprint is None:
            logprint = print

        for dname, dval in self.registered_dirs.items():
            logprint(f'{dname}: {str(dval)}')

        return
