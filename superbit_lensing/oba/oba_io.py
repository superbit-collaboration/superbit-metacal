import os
from pathlib import Path
from glob import glob
import numpy as np

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
    Calibration data

    Darks (one master per day):
    DARKS: /data/bit/master_darks/

    Flats (for now, one master flat):
    FLATS: /data/bit/master_flats/

    ----------------------------------------------------------------------
    Raw data

    Root data dir:
    RAW_DATA: /data/bit/science_images/

    NOTE: See below
    RAW_TARGET: RAW_DATA

    NOTE: The following is for the old definition; *all* exposures
    (other than cals) will now be pointed to RAW_DATA
    # Raw wl science images for a given target (if passed):
    # RAW_TARGET: RAW_DATA/{TARGET_NAME}/

    ----------------------------------------------------------------------
    On-board analysis

    Root oba dir:
    OBA_DIR: /home/bit/oba_temp/

    NOTE: Only if a target_name is passed:
    Temporary analysis dir for a given target name:
    OBA_TARGET: OBA_DIR/{TARGET_NAME}/

    Temporary analysis files per band for a given target:
    OBA_DIR/{TARGET_NAME}/{BAND}/

    Possible bands are [‘u’, ‘b’, ‘g’, ‘r’, ‘nir’, ‘lum’, ‘det’]

    NOTE: ‘det’ is not an actual filter, but a combination of bands (derived)

    Temporary output analysis files (FITS, configs, & logs for now)
    for a given band & target:
    OBA_DIR/{TARGET_NAME}/out/

    ----------------------------------------------------------------------
    GAIA catalog(s)

    We put a curated GAIA catalog here that has stellar positions and
    estimated fluxes for each SuperBIT filter in ADU/s; used for bright
    star masking

    GAIA_DIR: /data/bit/gaia/

    ----------------------------------------------------------------------
    Permanent OBA outputs

    OBA results root dir:
    OBA_RESULTS: /data/bit/oba_results/

    OBA results for a given target:
    OBA_RESULTS/{TARGET_NAME}/

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

    # alternatively, can instantiate for a given target:
    io_manager = IOManager(root_dir=root_dir, target_name=target_name)

    # returns {root_dir}/data/bit/science_images/{target_name}/
    target_dir = io_manager.RAW_TARGET

    '''

    # Default is registered to the above hard-coded qcc paths
    # Will update

    _registered_dir_names = [
        'CAL_DATA',
        'RAW_DATA',
        'RAW_TARGET',
        'GAIA_DIR',
        'OBA_DIR',
        'OBA_TARGET',
        'OBA_RESULTS',
        ]

    gaia_filename = 'gaia_superbit_fluxes.fits'

    def __init__(self, root_dir=None, target_name=None):
        '''
        root_dir: str
            The root directory that defines all other SuperBIT
            OBA filepaths relative to. This is useful if you want
            to test or simulate a real run locally.
        '''

        if root_dir is not None:
            utils.check_type('root_dir', root_dir, (str, Path))
        else:
            # will allow for uniform dir setting
            root_dir = '/'

        self.root_dir = Path(root_dir).resolve()

        if target_name is not None:
            utils.check_type('target_name', target_name, str)
        self.target_name = target_name

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
            'DARKS': 'data/bit/master_darks/',
            'FLATS': 'data/bit/master_flats/',
            'RAW_DATA': 'data/bit/science_images/',
            # NOTE: old definition
            'RAW_TARGET': None,
            'GAIA_DIR': 'data/bit/gaia/',
            'OBA_DIR': 'home/bit/oba_temp/',
            'OBA_TARGET': None,
            'OBA_RESULTS': 'data/bit/oba_results/',
            }

        self.registered_dirs = {}

        for name, default in _registered_defaults.items():
            if default is not None:
                # works for all cases as default root_dir is "/"
                self.registered_dirs[name] = self.root_dir / default
            else:
                self.registered_dirs[name] = None

        # the IO manager is target agnostic, but if a target name is passed
        # we can add some convenience dirs
        if self.target_name is not None:
            target_defaults = {
                # NOTE: old definition!
                # TODO: Relook at this!
                'RAW_TARGET': self.registered_dirs['RAW_DATA'],
                'OBA_TARGET': self.registered_dirs['OBA_DIR'] / self.target_name
            }
            for name, path in target_defaults.items():
                self.registered_dirs[name] = path

        return

    def _check_dir(self, name):
        if name not in self.registered_dirs:
            raise ValueError(f'{name} is not registered yet!')
        return self.registered_dirs[name]

    @property
    def DARKS(self):
        name = 'DARKS'
        return self._check_dir(name)

    @property
    def FLATS(self):
        name = 'FLATS'
        return self._check_dir(name)

    @property
    def RAW_DATA(self):
        name = 'RAW_DATA'
        return self._check_dir(name)

    @property
    def RAW_TARGET(self):
        name = 'RAW_TARGET'
        return self._check_dir(name)

    @property
    def GAIA_DIR(self):
        name = 'GAIA_DIR'
        return self._check_dir(name)

    @property
    def OBA_DIR(self):
        name = 'OBA_DIR'
        return self._check_dir(name)

    @property
    def OBA_TARGET(self):
        name = 'OBA_TARGET'
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

#------------------------------------------------------------------------------
# general I/O functions

def parse_image_file(image_file, image_type):
    '''
    Return a dictionary of SCI image parameters given a filename.
    Raw sci images and calibration images have different filename
    conventions:

    SCI: {TARGET_NAME}_{BAND_INDEX}_{EXP_TIME}_{UNIX_TIME}.fits
    CAL: master_{TYPE}_{EXP_TIME}_{UNIX_TIME}.fits

    image_file: pathlib.Path
        The filepath of the raw image. Can be a processed image as
        long as it still follows the standard raw image filename
        convention
    image_type: str
        The type of image being searched for. Can be one of:
        ['sci', 'cal']
    '''

    # while not officially supported, try to handle a passed str
    if isinstance(image_file, str):
        image_file = Path(image_file)

    utils.check_type('image_file', image_file, Path)
    utils.check_type('image_type', image_type, str)

    allowed_image_types = ['sci', 'cal']
    if image_type not in allowed_image_types:
        raise ValueError(f'image_type {image_type} is not valid! ' +
                         f'Must be one of {allowed_image_types}')

    parse_funcs = {
        'sci': parse_sci_image_file,
        'cal': parse_cal_image_file
    }

    return parse_funcs[image_type](image_file)

def parse_sci_image_file(image_file):
    '''
    Return a dictionary of SCI image parameters given a filename

    Raw sci image filename convention:
    {TARGET_NAME}_{EXP_TIME}_{BAND_INDEX}_{UTC}.fits

    image_file: pathlib.Path
        The filepath of the raw image. Can be a processed image as
        long as it still follows the standard raw image filename
        convention
    '''

    # while not officially supported, try to handle a passed str
    if isinstance(image_file, str):
        image_file = Path(image_file)

    name = image_file.name
    features = name.split('_')

    # remove file ext
    features[-1] = features[-1].replace('.fits', '')

    if features[-1] == 'cal':
        # in this case, we're dealing with a calibrated image file
        offset = 1
    else:
        # should be a raw
        offset = 0

    # We define them relative to the end to allow for _'s in a target_name
    im_pars = {
        'target_name': '_'.join(features[0:-3-offset]),
        'band': index2band(int(features[-3-offset])),
        'exp_time': int(features[-2-offset]),
        'utc': int(features[-1-offset])
        }

    return im_pars

def parse_cal_image_file(image_file):
    '''
    Return a dictionary of calibration image parameters given a filename

    Raw calibration image filename convention:
    master_{TYPE}_{EXP_TIME}_{UTC}.fits

    image_file: pathlib.Path
        The filepath of the calibration image
        convention
    '''

    # while not officially supported, try to handle a passed str
    if isinstance(image_file, str):
        image_file = Path(image_file)

    name = image_file.name
    features = name.split('_')

    # remove file ext
    features[-1] = features[-1].replace('.fits', '')

    # for a calibration frame, the 0th feature is just the str "master"
    im_pars = {
        'cal_type': features[1].lower(),
        'exp_time': int(features[2]),
        'utc': int(features[3])
        }

    return im_pars

def closest_file_in_time(image_type, search_dir, utc, req=None):
    '''
    Find the closest file in time, given a search directory and utc

    NOTE: Will work correctly only for sci & cal images

    image_type: str
        The type of image being searched for. Can be one of:
        ['sci', 'cal']
    search_dir: pathlib.Path
        The directory to conduct the search for a file
    utc: int
        The UTC time to the closest second
    req: dict
        A set of requirements that must be met to be a valid match,
        e.g. the same exposure time. Fields must be those registered
        in the above image parser functions
    '''

    files = glob(str(search_dir.resolve())+'/*')
    Nfiles = len(files)

    if Nfiles == 0:
        return None

    times = np.zeros(Nfiles, dtype=int)
    good_indices = []
    for i, fname in enumerate(files):
        im_pars = parse_image_file(fname, image_type)

        # Check if this is a valid image to consider as
        # a match. For example, ignore files w/ a different
        # exposure time
        if req is not None:
            good = True
            for name, val in req.items():
                if im_pars[name] != val:
                    good = False
                    # only one failed req is enough
                    # continue
            if good is True:
                good_indices.append(i)
        else:
            good_indices.append(i)

        times[i] = im_pars['utc']

    # don't consider any files that don't meet requirements
    files = [files[i] for i in good_indices]
    times = np.array([times[i] for i in good_indices], dtype=int)
    assert len(files) == len(times)

    if len(files) == 0:
        return None

    # best file match is the one that minimizes abs difference in UTC
    indx = np.argmin(abs(times - utc))

    return Path(files[indx])

def get_raw_files(search_dir, target_name, band=None):
    '''
    Grab all raw sci exposures in a given search dir

    search_dir: pathlib.Path
        The directory in which to do the search
    target_name: str
        The name of the sci target
    band: str
        The name of the desired band (defaults to all)
    '''

    utils.check_type('search_dir', search_dir, Path)
    utils.check_type('target_name', target_name, str)

    if band is None:
        band_str = ''
    else:
        utils.check_type('band', band, str)
        bindx = band2index(band)
        band_str = f'_{bindx}_'

    # for raw files, we want to ignore any temporary OBA files
    # that may have gotten written out to the same dir
    ignore = '[!'
    for suffix in OBA_FILE_SUFFIXES:
        ignore += f'{suffix},'
    ignore += ']'

    fname_base = f'{target_name}*{band_str}*{ignore}.fits'
    image_base = search_dir / fname_base

    files = glob(
        str(image_base.resolve())
        )

    return files

#------------------------------------------------------------------------------
# The SuperBIT convention for band indx <-> str mapping
BAND_INDEX = {
    0: 'u',
    1: 'b',
    2: 'g',
    3: 'dark',
    4: 'r',
    5: 'nir',
    6: 'lum',
}

def band2index(band):
    '''
    Convert a band str to the appropriate indx

    band: str
        The band str to convert
    '''

    inv = {v: k for k, v in BAND_INDEX.items()}

    return inv[band]

def index2band(indx):
    '''
    Convert a band str to the appropriate indx

    indx: int
        The band index to convert to a band str
    '''

    return BAND_INDEX[indx]

# Various suffixes that get appended to temporary OBA files
OBA_FILE_SUFFIXES = [
    '_cal',
    # TODO: finish!
]
