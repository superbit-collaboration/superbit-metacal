from pathlib import Path
import os
from glob import glob
import numpy as np
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import fitsio

from superbit_lensing import utils
from superbit_lensing.coadd import SWarpRunner
from superbit_lensing.oba.oba_io import band2index
from detection import DetectionRunner

import ipdb


def main():
    '''
        config_file: pathlib.Path
            The filepath of the base SExtractor config
        run_dir: pathlib.Path
            The OBA run directory for the given target
        target_name: str
            The name of the target. Default is to use the end of
            run_dir
        sci_ext: int
            The science frame fits extension
        wgt_ext: int
            The weight frame fits extension
    '''
    config_file = Path('superbit_lensing/oba/configs/sextractor/sb_sextractor_lum.config')
    run_dir = Path('tests/euclid/euclid_test/Abell2813')
    bands = ['lum']
    det_bands = ['lum']
    target_name = "Abell2813"
    
    
    log = utils.setup_logger("euclid_detect_test.log", logdir="./")

    # we won't know whether to run in verbose mode until after config
    # parsing; start verbose and reset when we know
    logprint = utils.LogPrint(log, True)
    
    runner = DetectionRunner(
            config_file,
            run_dir,
            target_name=target_name,
            )
    runner.go(logprint=logprint)
    
    return 0
    
    
    

if __name__ == '__main__':
    #args = parse_args()
    rc = main()

    if rc == 0:
        print('\neuclid_detect_runner.py completed without error\n')
    else:
        print(f'\neuclid_detect_runner.py failed with rc={rc}\n')