from pathlib import Path
import os
from glob import glob
import shutil

from superbit_lensing import utils
from superbit_lensing.oba.oba_io import band2index

import ipdb

class TestPrepper(object):
    '''
    A class that handles any needed setup to turn the outputs of simulted
    data into the expected format for real data on flight, e.g.
    compressing image files
    '''

    _compression_method = 'bzip2'
    _compression_args = '-zk' # forces compression, keep orig file
    _compression_ext = 'bz2'

    def __init__(self, target_name, bands, skip_existing=True):
        '''
        target_name: str
            The name of the target to run the OBA on
        bands: list of str's
            A list of band names
        skip_existing: bool
            Set to skip existing compressed files
        '''

        utils.check_type('target_name', target_name, str)
        self.target_name = target_name

        utils.check_type('bands', bands, list)
        for b in bands:
            utils.check_type('band', b, str)
        self.bands = bands

        utils.check_type('skip_existing', skip_existing, bool)
        self.skip_existing = skip_existing

        return

    def go(self, io_manager, overwrite=None, logprint=None):
        '''
        Handle any necessary test preparation on simulated
        inputs to the OBA module. For now, just the following:

        (1) Copy simulation files to target_dir
        (2) Compress simulated image files

        io_manager: oba_io.IOManager
            An IOManager instance that defines all relevant OBA
            path information
        overwrite: bool
            Set to overwrite existing files
        logprint: utils.LogPrint
            A LogPrint instance for simultaneous logging & printing
        '''

        if logprint is None:
            logprint = print

        logprint(f'Starting test prepper for target {self.target_name}')

        target_dir = io_manager.RAW_TARGET

        # not all IO managers are registered to a particular target
        if target_dir is None:
            # NOTE: Old definition!
            # target_dir = io_manager.RAW_CLUSTERS / self.target_name
            target_dir = io_manager.RAW_DATA

        target_dir = target_dir.resolve()
        source_dir = str(target_dir / 'imsim')
        target_dir = str(target_dir)
        logprint(f'Using raw target dir {target_dir}')

        if not os.path.exists(target_dir):
            raise OSError(f'{target_dir} does not exist!')

        logprint('Copying simulated images...')
        self.copy_images(source_dir, target_dir, logprint)

        logprint('Compressing image files...')
        self.compress_images(target_dir, logprint, overwrite=overwrite)

        logprint('\nCompleted test setup\n')

        return

    def copy_images(self, source_dir, target_dir, logprint):
        '''
        source_dir: str
            The path to the simulated images
        target_dir: str
            The path to the raw target directory
        logprint: utils.LogPrint
            A LogPrint instance for simultaneous logging & printing
        overwrite: bool
            Set to overwrite existing compressed files
        '''

        search = str(Path(source_dir) / '*[!truth].fits')
        images = glob(search)

        for image in images:
            shutil.copy(image, target_dir)

        return

    def compress_images(self, target_dir, logprint, overwrite=False):
        '''
        target_dir: str
            The path to the raw target directory
        logprint: utils.LogPrint
            A LogPrint instance for simultaneous logging & printing
        overwrite: bool
            Set to overwrite existing compressed files
        '''

        cmethod = self._compression_method
        cext = self._compression_ext

        for b in self.bands:
            logprint(f'Starting band {b}')

            im_list = self._get_im_list(target_dir, b)
            Nim = len(im_list)
            logprint(f'Found {Nim} images for band {b}')
            for i, im in enumerate(im_list):
                cim = f'{im}.{cext}'
                if os.path.exists(cim):
                    if self.skip_existing is True:
                        logprint(f'{i}: {cim} already exists; skipping')
                        continue
                    else:
                        logprint(f'{i}: {cim} already exists; deleting')
                        os.remove(cim)

                logprint(f'Compressing {im} using {cmethod}; {i} of {Nim}')
                self._compress_file(im, logprint)

        return

    def _get_im_list(self, target_dir, band):

        bindx = band2index(band)
        exp = f'{target_dir}/{self.target_name}*_{bindx}_[!truth]*.fits'
        im_list = glob(exp)

        return im_list

    def _compress_file(self, filename, logprint):

        cmethod = self._compression_method
        cargs = self._compression_args
        cmd = f'{cmethod} {cargs} {filename}'

        logprint(f'cmd = {cmd}')

        utils.run_command(cmd, logprint=logprint)

        return

class HenSimsTestPrepper(TestPrepper):
    '''
    Prepper for realistic SuperBIT sims stored in `hen`
    '''

    def __init__(self, target_name, bands, run_name='quicktest',
                 skip_existing=True, strehl_ratio=100, sim_dir=None):
        '''
        target_name: str
            The name of the target to run the OBA on
        bands: list of str's
            A list of band names
        skip_existing: bool
            Set to skip existing compressed files
        strehl_ratio: int, float
            The Strehl ratio of the sims (100 is perfect focus)
        '''

        utils.check_type('run_name', run_name, str)
        self.run_name = run_name

        utils.check_type('strehl_ratio', strehl_ratio, (int, float))
        self.strehl_ratio = strehl_ratio

        if sim_dir is None:
            sim_dir = Path(
                '/home/sweveret/repos/superbit-metacal/tests/ajay/'
                )
        utils.check_type('sim_dir', sim_dir, Path)
        self.sim_dir = sim_dir

        super(HenSimsTestPrepper, self).__init__(
            target_name, bands, skip_existing=skip_existing
            )

        return

    def go(self, io_manager, overwrite=None, logprint=None):
        '''
        Handle any necessary test preparation on simulated
        inputs to the OBA module. For now, just the following:

        (1) Copy simulation files to target_dir
        (2) Compress simulated image files

        io_manager: oba_io.IOManager
            An IOManager instance that defines all relevant OBA
            path information
        overwrite: bool
            Set to overwrite existing files
        logprint: utils.LogPrint
            A LogPrint instance for simultaneous logging & printing
        '''

        if logprint is None:
            logprint = print

        logprint(f'Starting test prepper for target {self.target_name}')

        target_dir = io_manager.RAW_TARGET

        # not all IO managers are registered to a particular target
        if target_dir is None:
            # NOTE: Old definition!
            # target_dir = io_manager.RAW_CLUSTERS / self.target_name
            target_dir = io_manager.RAW_DATA

        target_dir = target_dir.resolve()
        source_dir = self.sim_dir / self.run_name / self.target_name
        target_dir = str(target_dir)
        logprint(f'Using raw target dir {target_dir}')

        if not os.path.exists(target_dir):
            raise OSError(f'{target_dir} does not exist!')

        logprint('Copying simulated images...')

        # NOTE: unlike ImSim, the sims on `hen` are in band-specific
        # subdirectories, so copy across all bands
        for band in self.bands:
            band_source_dir = os.path.join(
                source_dir, band, str(self.strehl_ratio)
                )
            self.copy_images(band_source_dir, target_dir, logprint)

        logprint('Compressing image files...')
        self.compress_images(target_dir, logprint, overwrite=overwrite)

        logprint('\nCompleted test setup\n')

        return

def make_test_prepper(test_type, *args, **kwargs):
    '''
    obj_type: str
        Type of test to run
    config: dict
        A configuration dictionary that contains all needed
        fields to create the corresponding object class type
    seed: int
        A seed to set for the object constructor
    '''

    test_type = test_type.lower()
    if test_type not in PREPPER_TYPES.keys():
        raise ValueError(
            f'obj_type must be one of {PREPPER_TYPES.keys()}!'
            )

    try:
        return PREPPER_TYPES[test_type](*args,**kwargs)

    except KeyError as e:
        raise KeyError(
            f'{test_type} not a valid option for {PREPPER_TYPES.keys()}!'
            )

PREPPER_TYPES = {
    'imsim': TestPrepper,
    'hen' : HenSimsTestPrepper,
}
