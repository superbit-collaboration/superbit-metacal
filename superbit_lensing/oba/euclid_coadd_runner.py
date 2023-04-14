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
from coadd import CoaddRunner

import ipdb


def main():
    '''
        config_file: pathlib.Path
            The filepath of the base SWarp config
        run_dir: pathlib.Path
            The OBA run directory for the given target
        bands: list of str's
            A list of band names to make coadds for
        det_bands: list of str's
            A list of band names to use for the detection coadd
        target_name: str
            The name of the target. Default is to use the end of
            run_dir
        sci_ext: int
            The science frame fits extension
        wgt_ext: int
            The weight frame fits extension
        combine_type: str
            The SWarp combine type to use for single-band exposures
        det_combine_type: str
            The SWarp combine type to use for the detection image
    '''
    config_file = Path('superbit_lensing/oba/configs/swarp/swarp.config')
    run_dir = Path('tests/euclid/euclid_test/Abell2813')
    bands = ['vis']
    det_bands = ['vis']
    target_name = "Abell2813"
    
    
    log = utils.setup_logger("euclid_coadd_test.log", logdir="./")

    # we won't know whether to run in verbose mode until after config
    # parsing; start verbose and reset when we know
    logprint = utils.LogPrint(log, True)
    
    runner = CoaddRunner(
            config_file,
            run_dir,
            bands,
            det_bands,
            target_name=target_name,
            #combine_type=combine_type,
            #det_combine_type=det_combine_type
            )
    runner.go(logprint=logprint)
    
    cutter = EuclidOutputRunner(
            run_dir,
            bands,
            target_name=target_name,
            make_center_stamp=False,
            make_2d=False,
            )
    cutter.go(logprint=logprint)
    
    
    return 0
    
    
    

if __name__ == '__main__':
    #args = parse_args()
    rc = main()

    if rc == 0:
        print('\neuclid_coadd_runner.py completed without error\n')
    else:
        print(f'\neuclid_coadd_runner.py failed with rc={rc}\n')
