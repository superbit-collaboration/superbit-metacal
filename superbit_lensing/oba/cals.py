from pathlib import Path

from superbit_lensing import utils

class CalsRunner(object):
    '''
    Runner class for calibrating raw SuperBIT science images
    for the onboard analysis (OBA)

    '''

    def __init__(self, run_dir, bands, target_name):
        '''
        run_dir: pathlib.Path
            The OBA run directory for the given target
        cal_dir: pathlib.Path
            The directory location of the 
        bands: list of str's
            A list of band names
        target_name: str
            The name of the target
        '''

        args = {
            'run_dir': (run_dir, Path),
            'bands': (bands, list),
            'target_name': (target_name, str)
        }

        for name, tup in args.items():
            val, allowed_types = tup
            utils.check_type(name, val, allowed_types)
            setattr(self, name, val)

        # make sure each band name is a str
        for i, b in enumerate(self.bands):
            utils.check_type('band_{i}', b, str)

        return

    def go(self, logprint, overwrite=False):
        '''
        Run all steps to convert raw SCI frames to calibrated frames.
        The current calibration steps require master dark & flat
        images. The basic procedure is as follows:

        (1) Find the corresponding master dark & flat for each image
        (2) Create hot pixel mask using master dark
            NOTE: The current plan is to have a static master flat
        (3) Basic calibration: Cal = (Raw - Dark) / Flat
        (4) Write out calibrated images w/ original headers

        logprint: utils.LogPrint
            A LogPrint instance for simultaneous logging & printing
        overwrite: bool
            Set to overwrite existing files
        '''

        return
