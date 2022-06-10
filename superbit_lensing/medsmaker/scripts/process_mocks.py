import os,sys
from pathlib import Path
import glob
import esutil as eu
import meds
from argparse import ArgumentParser
import superbit_lensing.utils as utils
from superbit_lensing.medsmaker.superbit import medsmaker_mocks as medsmaker

import pdb, pudb, traceback

## Get the location of the main Medmaker superbit package.
# filepath = Path(os.path.realpath(__file__))
# sbdir = filepath.parents[1]
# sbdir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# sys.path.insert(0,str(sbdir))

parser = ArgumentParser()

parser.add_argument('mock_dir', type=str,
                    help='Directory containing mock data')
parser.add_argument('outfile', type=str,
                    help='Name of output MEDS file')
parser.add_argument('-outdir', type=str, default=None,
                    help='Output directory for MEDS file')
parser.add_argument('-fname_base', action='store', type=str, default=None,
                    help='Basename of mock image files')
parser.add_argument('-run_name', action='store', type=str, default=None,
                    help='Name of mock simulation run')
parser.add_argument('-meds_coadd', action='store_true', default=False,
                    help='Set to keep coadd cutout in MEDS file')
parser.add_argument('-psf_mode', action='store', choices=['piff', 'psfex'], default='piff',
                    help='model exposure PSF using either piff or psfex')
parser.add_argument('--clobber', action='store_true', default=False,
                    help='Set to overwrite files')
parser.add_argument('--source_select', action='store_true', default=False,
                    help='Set to select sources during MEDS creation')
parser.add_argument('--select_truth_stars', action='store_true', default=False,
                    help='Set to match against truth catalog for PSF model fits')
parser.add_argument('--vb', action='store_true', default=False,
                    help='Verbosity')

def main():
    args = parser.parse_args()
    mock_dir = args.mock_dir
    outfile = args.outfile
    outdir = args.outdir
    run_name = args.run_name
    psf_mode = args.psf_mode
    use_coadd = args.meds_coadd
    clobber = args.clobber
    source_selection = args.source_select
    select_truth_stars = args.select_truth_stars
    vb = args.vb

    if args.outdir is None:
        outdir = mock_dir

    logfile = 'medsmaker.log'
    logdir = outdir
    log = utils.setup_logger(logfile, logdir=logdir)
    logprint = utils.LogPrint(log, vb=vb)

    if args.fname_base is None:
        fname_base = run_name
    else:
        fname_base = args.fname_base


    science = glob.glob(os.path.join(mock_dir, fname_base)+'*[!truth,mcal,.sub,mock_coadd].fits')
    logprint(f'Science frames: {science}')

    outfile = os.path.join(outdir, outfile)

    logprint('Setting up configuration...')
    bm = medsmaker.BITMeasurement(
        image_files=science, data_dir=mock_dir, run_name=run_name, log=log, vb=vb
        )

    bm.set_working_dir(path=outdir)
    bm.set_mask(mask_name='forecast_mask.fits',mask_dir=os.path.join(mock_dir,'mask_files'))
    bm.set_weight(weight_name='forecast_weight.fits',weight_dir=os.path.join(mock_dir,'weight_files'))

    # Combine images, make a catalog.
    logprint('Making coadd & its catalog...')
    bm.make_coadd_catalog(source_selection=source_selection)

    # Make single-exposure catalogs
    logprint('Making single-exposure catalogs...')
    im_cats = bm.make_exposure_catalogs()

    # Build a PSF model for each image.
    logprint('Making PSF models...')
    bm.make_psf_models(select_truth_stars=select_truth_stars,im_cats=im_cats, use_coadd=use_coadd,psf_mode=psf_mode)

    logprint('Making MEDS...')

    # Make the image_info struct.
    image_info = bm.make_image_info_struct(use_coadd=use_coadd)

    # Make the object_info struct.
    obj_info = bm.make_object_info_struct()

    # Make the MEDS config file.
    meds_config = bm.make_meds_config(use_coadd=use_coadd)

    # Create metadata for MEDS
    meta = bm._meds_metadata(magzp=30.0)
    # Finally, make and write the MEDS file.

    medsObj = meds.maker.MEDSMaker(obj_info, image_info, config=meds_config,
                                    psf_data=bm.psf_models, meta_data=meta)
    
    logprint(f'Writing to {outfile}')
    medsObj.write(outfile)
    
    # Remove spurious detections with 0 cutouts
    bm.filter_meds(outfile, clean=True, min_cutouts=1)
   
    """
    bm.run(clobber=clobber,source_selection = source_selection, select_stars = select_stars, outfile = outfile,psf_mode=psf_mode)
    """

    logprint('Done!')

    return 0

if __name__ == '__main__':
    rc = main()

    if rc !=0:
        raise Exception
