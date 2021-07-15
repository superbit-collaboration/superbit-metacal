import os,sys
import importlib.util
import glob
import pdb, traceback
import esutil as eu
## Get the location of the main superbit package.
dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0,dir)
from superbit import medsmaker_mocks as medsmaker
import meds

## Get either one or two hours' worth of exposures
science = glob.glob('/users/jmcclear/data/superbit/superbit-metacal/GalSim/forecasting/cluster5/nobkg_newshape/round*/superbit_gaussJitter_00?.fits') 
#science.extend(science2)

## clobber is currently only called by make_mask()
 
clobber=False
source_selection=True
select_stars = False
outfile = "/users/jmcclear/data/superbit/superbit-ngmix/scripts/forecasting/cluster5/nobkg/cluster5_newshape.meds"

try:
    
    bm = medsmaker.BITMeasurement(image_files=science)

    bm.set_working_dir(path='/users/jmcclear/data/superbit/superbit-ngmix/scripts/forecasting/cluster5/nobkg')
    bm.set_path_to_psf(path='/users/jmcclear/data/superbit/superbit-ngmix/scripts/forecasting/cluster5/nobkg/psfex_output')

    # Make a mask.                                                                                                                                   
    bm.make_mask(overwrite=clobber,mask_name='forecast_weight.fits')
 
    # Combine images, make a catalog.                                                                                                                
    bm.make_catalog(source_selection=source_selection)
    
    # Build a PSF model for each image.                                                                                                              
    bm.make_psf_models(select_stars=select_stars)
    
    # Make the image_info struct.                                                                                                                    
    image_info = bm.make_image_info_struct()
    
    # Make the object_info struct.                                                                                                                   
    obj_info = bm.make_object_info_struct()
    
    # Make the MEDS config file.                                                                                                                     
    meds_config = bm.make_meds_config()
    
    # Create metadata for MEDS                                                                                                                       
    meta = bm._meds_metadata(magzp=30.0)
    # Finally, make and write the MEDS file.                                                                                                         
    
    medsObj = meds.maker.MEDSMaker(obj_info,image_info,config=meds_config,psf_data = bm.psfEx_models,meta_data=meta)
    medsObj.write(outfile)
    """
    bm.run(clobber=clobber,source_selection = source_selection, select_stars = select_stars, outfile = outfile)
    """    
except:
    thingtype, value, tb = sys.exc_info()
    traceback.print_exc()
    pdb.post_mortem(tb)
