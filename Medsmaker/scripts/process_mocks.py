import os,sys
import importlib.util
from pathlib import Path
import glob
import esutil as eu
import meds
from argparse import ArgumentParser
import pdb, pudb, traceback

## Get the location of the main Medmaker superbit package.
filepath = Path(os.path.realpath(__file__))
sbdir = filepath.parents[1]
# sbdir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0,str(sbdir))
from superbit import medsmaker_mocks as medsmaker

parser = ArgumentParser()

parser.add_argument('--mock_dir', action='store', type=str, default=None,
                    help='Directory containing mock data')
parser.add_argument('--fname_base', action='store', type=str, default=None,
                    help='Basename of mock image files')
parser.add_argument('--meds_coadd', action='store_true', default=False,
                    help='Set to keep coadd cutout in MEDS file')
parser.add_argument('-v', '--verbose', action='store', type=int, default=1,
                    help='Verbosity (0: None, default: 1)')

def vbprint(s, vb):
    if vb > 0:
        print(s)

    return

def main():
    args = parser.parse_args()
    use_coadd = args.meds_coadd
    vb = args.verbose

    ## Get either one or two hours' worth of exposures
    if args.mock_dir is None:
        mock_dir = os.path.join(sbdir.parents[0], 'GalSim/forecasting/cluster5')
    else:
        mock_dir = args.mock_dir
    if args.fname_base is None:
        fname_base = 'superbit_gaussJitter_'
    else:
        fname_base = args.fname_base

    science = glob.glob(os.path.join(mock_dir, fname_base)+'*[!.sub].fits')

    ## clobber is currently only called by make_mask()
    clobber = False
    source_selection = True
    select_stars = False
    outfile = os.path.join(mock_dir, 'cluster5_newshape.meds')

    try:
        bm = medsmaker.BITMeasurement(image_files=science, data_dir=mock_dir)

        bm.set_working_dir(path=mock_dir)
        bm.set_path_to_psf(path=os.path.join(mock_dir, 'psfex_output'))

        # Make a mask.
        vbprint('Making mask...', vb)
        bm.make_mask(overwrite=clobber, mask_name='forecast_weight.fits')

        # Combine images, make a catalog.
        vbprint('Making catalog...', vb)
        bm.make_catalog(source_selection=source_selection)

        # Build a PSF model for each image.
        vbprint('Making PSF models...', vb)
        bm.make_psf_models(select_stars=select_stars, use_coadd=use_coadd)

        vbprint('Making MEDS...', vb)

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

        vbprint(f'Writing to {outfile}', vb)
        medsObj.write(outfile)
        """
        bm.run(clobber=clobber,source_selection = source_selection, select_stars = select_stars, outfile = outfile)
        """

        vbprint('Done!', vb)

    except:
        vbprint('Exception has occured..', vb)
        thingtype, value, tb = sys.exc_info()
        traceback.print_exc()
        # pdb.post_mortem(tb)

if __name__ == '__main__':
    main()
