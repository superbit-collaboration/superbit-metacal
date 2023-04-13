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
    '''
        run_dir: pathlib.Path
            The OBA run directory for the given target
        bands: list of str's
            A list of band names to make CookieCutter files for
        target_name: str
            The name of the target. Default is to use the end of
            run_dir
        sci_ext: int
            The science frame fits extension
        wgt_ext: int
            The weight frame fits extension
        msk_ext: int
            The mask frame fits extension
        bkg_ext: int
            The background frame fits extension
        coadd_sci_ext: int
            The science frame fits extension for coadds
        coadd_wgt_ext: int
            The weight frame fits extension for coadds
        coadd_seg_ext: int
            The segmentation frame fits extension for coadds
        id_tag: str
            The name of the ID column in the detection catalog
        boxsize_tag: str
            The name of the boxsize column in the detection catalog
            NOTE: CookieCutter will write a boxsize col w/ this name
            if not present in the catalog
        ra_tag: str
            The name of the RA column in the detection catalog
        dec_tag: str
            The name of the DEC column in the detection catalog
        ra_unit: str
            The unit of the RA column in the detection catalog
        dec_unit: str
            The unit of the DEC column in the detection catalog
        out_sci_dtype: str
            The numpy dtype for the output SCI stamps
        out_msk_dtype: str
            The numpy dtype for the output MSK stamps
        make_center_stamp: bool
            Set to True to make a single large stamp at the target center
        center_stamp_size: int
            If making a central stamp, set the size (for now, must be square)
        make_2d: bool
            Set to make a 2D CookieCutter output file as well
    '''
    run_dir = Path('tests/euclid/euclid_test/Abell2813')
    bands = ['lum']
    target_name = "Abell2813"
    
    
    log = utils.setup_logger("euclid_detect_test.log", logdir="./")

    # we won't know whether to run in verbose mode until after config
    # parsing; start verbose and reset when we know
    logprint = utils.LogPrint(log, True)
    
    runner = EuclidOutputRunner(
            run_dir,
            bands,
            target_name=target_name,
            make_center_stamp=False,
            make_2d=False,
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