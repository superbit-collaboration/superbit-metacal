import os,sys
from pathlib import Path
import glob
import esutil as eu
import meds
from argparse import ArgumentParser
import superbit_lensing.utils as utils
from superbit_lensing.medsmaker.superbit import medsmaker_mocks as medsmaker

import ipdb

def parse_args():

    parser = ArgumentParser()

    parser.add_argument('mock_dir', type=str,
                        help='Directory containing mock data')
    parser.add_argument('outfile', type=str,
                        help='Name of output MEDS file')
    parser.add_argument('-outdir', type=str, default=None,
                        help='Output directory for MEDS file')
    parser.add_argument('-run_name', action='store', type=str, default=None,
                        help='Name of mock simulation run')
    parser.add_argument('-fname_base', action='store', type=str, default=None,
                        help='Image filename base [ims=fname_base_XXX.fits]')
    parser.add_argument('-psf_mode', action='store', choices=['piff', 'psfex'], default='piff',
                        help='model exposure PSF using either piff or psfex')
    parser.add_argument('-psf_seed', type=int, default=None,
                        help='Seed for chosen PSF estimation mode')
    parser.add_argument('-master_dark', type=str, default=None,
                        help='Name of master dark file to subtract')
    parser.add_argument('-master_flat', type=str, default=None,
                        help='Name of master flat file to subtract')
    parser.add_argument('--meds_coadd', action='store_true', default=False,
                        help='Set to keep coadd cutout in MEDS file')
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='Set to overwrite files')
    parser.add_argument('--source_select', action='store_true', default=False,
                        help='Set to select sources during MEDS creation')
    parser.add_argument('--select_truth_stars', action='store_true', default=False,
                        help='Set to match against truth catalog for PSF model fits')
    parser.add_argument('--vb', action='store_true', default=False,
                        help='Verbosity')

    return parser.parse_args()

def main(args):

    mock_dir = args.mock_dir
    outfile = args.outfile
    outdir = args.outdir
    run_name = args.run_name
    filename_base = args.fname_base
    psf_mode = args.psf_mode
    psf_seed = args.psf_seed
    master_dark = args.master_dark
    master_flat = args.master_flat
    use_coadd = args.meds_coadd
    overwrite = args.overwrite
    source_selection = args.source_select
    select_truth_stars = args.select_truth_stars
    vb = args.vb

    if args.outdir is None: outdir = mock_dir

    logfile = 'medsmaker.log'
    log = utils.setup_logger(logfile, logdir=outdir)
    logprint = utils.LogPrint(log, vb=vb)

    # Define some file names
    bpm = os.path.join(mock_dir,'mask_files/forecast_mask.fits')
    combined_mask_file = os.path.join(outdir, 'combined_mask.fits')
    image_files = glob.glob(os.path.join(mock_dir, filename_base) +\
                        '*[!truth,mcal,.sub,*_cal,mock_coadd].fits')

    logprint(f'Using science frames: {image_files}')

    # This is the output MEDS file
    outfile = os.path.join(outdir, outfile)

    # Set up BITMeasurement object
    logprint('Setting up BITMeasurement object...')
    bm = medsmaker.BITMeasurement(image_files=image_files,
                                    data_dir=mock_dir,
                                    run_name=run_name,
                                    outdir=outdir,
                                    log=log,
                                    overwrite=overwrite,
                                    vb=vb
                                    )

    # Load calibration data and masks
    logprint('Loading calibration data...')
    bm.setup_calib_data(master_dark=master_dark,
                        master_flat=master_flat,
                        bpm=bpm
                        )


    bm._set_all_paths_debug(run_name, psf_mode=psf_mode)

    '''
    # Do a minimal data reduction
    logprint('Quick-reducing single-exposures...')
    bm.quick_reduce()

    # Get mask(s) for science images
    logprint(f'Getting {combined_mask_file}...')
    bm.get_combined_mask(combined_mask_file)

    # Make single-exposure catalogs
    logprint('Making single-exposure catalogs...')
    im_cats = bm.make_exposure_catalogs(weight_file=combined_mask_file)

    # Special function to create a combined weight-mask image
    logprint('Making combined weights...')
    bm.make_combined_weight()

    # Combine images, make a detection catalog.
    logprint('Making coadd & its catalog...')
    bm.make_coadd_catalog()

    # Build a PSF model for each image.
    logprint('Making PSF models...')

    bm.make_psf_models(
        select_truth_stars=select_truth_stars,
        im_cats=im_cats,
        use_coadd=use_coadd,
        psf_mode=psf_mode,
        psf_seed=psf_seed
        )

    '''
    logprint('Making MEDS...')

    # Make the image_info struct.
    image_info = bm.make_image_info_struct(use_cal=True, use_coadd=use_coadd)

    # Make the object_info struct.
    obj_info = bm.make_object_info_struct()

    # Make the MEDS config file.
    meds_config = bm.make_meds_config(use_coadd=use_coadd)

    # Create metadata for MEDS
    magzp = 30.
    meta = bm.meds_metadata(magzp, use_coadd)

    # Finally, make and write the MEDS file.
    medsObj = meds.maker.MEDSMaker(
        obj_info, image_info, config=meds_config,
        psf_data=bm.psf_models, meta_data=meta
        )

    logprint(f'Writing to {outfile}')
    medsObj.write(outfile)

    logprint('Done!')

    return 0

if __name__ == '__main__':
    args = parse_args()
    rc = main(args)

    if rc !=0:
        print(f'process_mocks failed w/ return code {rc}!')
