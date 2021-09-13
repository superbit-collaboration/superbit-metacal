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
parser.add_argument('outdir',type=str,
                    help='Output directory for MEDS file')
parser.add_argument('--fname_base', action='store', type=str, default=None,
                    help='Basename of mock image files')
parser.add_argument('--meds_coadd', action='store_true', default=False,
                    help='Set to keep coadd cutout in MEDS file')
parser.add_argument('--clobber', action='store_true', default=False,
                    help='Set to overwrite files')
parser.add_argument('--source_select', action='store_true', default=False,
                    help='Set to select sources during MEDS creation')
parser.add_argument('--select_stars', action='store_true', default=False,
                    help='Set to remove stars during source selection')
parser.add_argument('-v', '--verbose', action='store_true', default=False,
                    help='Verbosity')

def main():
    args = parser.parse_args()
    mock_dir = args.mock_dir
    outfile = args.outfile
    outdir=args.outdir
    use_coadd = args.meds_coadd
    clobber = args.clobber
    source_selection = args.source_select
    select_stars = args.select_stars
    data_dir = '/Users/jemcclea/Research/SuperBIT/mock_forecasting_data/'
    vb = args.verbose

    logfile = 'medsmaker.log'
    logdir = mock_dir
    log = utils.setup_logger(logfile, logdir=logdir)
    logprint = utils.LogPrint(log, vb=vb)

    if args.fname_base is None:
        fname_base = 'superbit_gaussJitter_'
    else:
        fname_base = args.fname_base

    science = glob.glob(os.path.join(mock_dir, fname_base)+'*[!truth,mcal,.sub].fits')
    logprint(f'Science frames: {science}')

    outfile = os.path.join(outdir, outfile)

    logprint('Setting up configuration...')
    bm = medsmaker.BITMeasurement(image_files=science, data_dir=data_dir, log=log, vb=vb)

    bm.set_working_dir(path=outdir)
    bm.set_path_to_psf(path=os.path.join(outdir, 'psfex_output'))

    # Make a mask.
    logprint('Making mask...')
    bm.make_mask(overwrite=clobber, mask_name='forecast_mask.fits')

    # Combine images, make a catalog.
    logprint('Making catalog...')
    bm.make_catalog(source_selection=source_selection)

    # Build a PSF model for each image.
    logprint('Making PSF models...')
    bm.make_psf_models(select_stars=select_stars, use_coadd=use_coadd)

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
                                    psf_data=bm.psfEx_models, meta_data=meta)

    logprint(f'Writing to {outfile}')
    medsObj.write(outfile)
    """
    bm.run(clobber=clobber,source_selection = source_selection, select_stars = select_stars, outfile = outfile)
    """

    logprint('Done!')

    return 0

if __name__ == '__main__':
    rc = main()

    if rc !=0:
        raise Exception
