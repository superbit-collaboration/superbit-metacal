import os,sys
import importlib.util
import glob
import pdb, traceback
import esutil as eu
# Get the location of the main superbit package.
#dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
#sys.path.insert(0,dir)
#from superbit import medsmaker_debug as medsmaker
import medsmaker_debug as medsmaker

# Start by making a directory...
if not os.path.exists('../Data/calib'):
    os.mkdir('../Data/')
    os.mkdir('../Data/calib')

science = glob.glob('/Users/jemcclea/Research/GalSim/examples/output-debug/mockSuperbit_shear_300*.fits')
flats = glob.glob('/Users/jemcclea/Research/SuperBIT_2019/A2218/FlatImages/*')
biases = glob.glob('/Users/jemcclea/Research/SuperBIT_2019/A2218/BiasImages/*')
darks = glob.glob('/Users/jemcclea/Research/SuperBIT_2019/A2218/DarkImages/*')
try:
    bm = medsmaker.BITMeasurement(image_files=science)
    # The path names should be updated; as written the code also expects all
    # calibration files to be in the same directory
    
    bm.set_working_dir(path='debug3')
    bm.set_path_to_psf(path='debug3/psfex_output')

    """
    bm.make_mask(overwrite=False)
    # Combine images, make a catalog.
    bm.make_catalog(source_selection=True)
    # Build a PSF model for each image.
    bm.make_psf_models(gaia_select=False)
    # Make the image_info struct.
    image_info = bm.make_image_info_struct()
    # Make the object_info struct.
    obj_info = bm.make_object_info_struct()
    # Make the MEDS config file.
    meds_config = bm.make_meds_config()
    # Create metadata for MEDS
    meta = bm._meds_metadata(magzp=30.0)
    # Finally, make and write the MEDS file.
    pdb.set_trace()
    medsObj = meds.maker.MEDSMaker(obj_info,image_info,config=meds_config,psf_data = bm.psfEx_models,meta_data=meta)
    """
    
    bm.run(clobber=False,source_selection = True, select_from_gaia=False,outfile = "debug3/debug-0.206-PSFscale.meds")

except:
    thingtype, value, tb = sys.exc_info()
    traceback.print_exc()
    pdb.post_mortem(tb)
