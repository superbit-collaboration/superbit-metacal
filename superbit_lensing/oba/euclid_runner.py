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
from detection import DetectionRunner
from output import OutputRunner

import ipdb

class EuclidOutputRunner(OutputRunner):
    # The following dict defines the correspondance between our internal keys
    # to the desired output format config keys, in this case the CookieCutter.
    # Each internal key is used to map a given raw SCI exposure to its
    # corresponding set of value-added images (e.g. mask, background, etc.)
    # and their extensions
    cookiecutter_keymap = {
        'sci_file': 'image_file',
        'sci_ext': 'image_ext',
        # NOTE: for perfect images, right now, skip the rest
        # 'wgt_file': 'weight_file',
        # 'wgt_ext': 'weight_ext',
        #'msk_file': 'mask_file',
        #'msk_ext': 'mask_ext',
        # 'skyvar_file': 'skyvar_file',
        # 'skyvar_ext': 'skyvar_ext',
        #'bkg_file': 'background_file',
        #'bkg_ext': 'background_ext',
        #'seg_file': 'segmentation_file',
        #'seg_ext': 'segmentation_ext',
    }

def main():
    config_file_swarp = Path('superbit_lensing/oba/configs/swarp/swarp.config')
    run_dir = Path('tests/euclid/test_obj_wcs/Abell2813')
    bands = ['vis']
    det_bands = ['vis']
    target_name = "Abell2813"
    
    
    log = utils.setup_logger("euclid_coadd_test.log", logdir="./")

    # we won't know whether to run in verbose mode until after config
    # parsing; start verbose and reset when we know
    logprint = utils.LogPrint(log, True)
    
    coadder = CoaddRunner(
            config_file_swarp,
            run_dir,
            bands,
            det_bands,
            target_name=target_name,
            )
    coadder.go(logprint=logprint)
    
    config_file_se = Path('superbit_lensing/oba/configs/sextractor/eu_sextractor_vis.config')
    detector = DetectionRunner(
            config_file_se,
            run_dir,
            target_name=target_name,
            )
    detector.go(logprint=logprint)
    
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
        print('\neuclid_runner.py completed without error\n')
    else:
        print(f'\neuclid_runner.py failed with rc={rc}\n')
