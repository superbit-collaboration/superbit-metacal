import os,sys
import importlib.util
import glob
import pdb, traceback
import esutil as eu
# Get the location of the main superbit package.
dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0,dir)
from superbit import medsmaker_mocks as medsmaker

# Start by making a directory...
if not os.path.exists('../Data/calib'):
    os.mkdir('../Data/')
    os.mkdir('../Data/calib')


#science = glob.glob('/Users/jemcclea/Research/GalSim/examples/output-debug/mock_jitteronly_flight_jitter_only_oversampled_1x30000?.fits')
science = glob.glob('/Users/jemcclea/Research/GalSim/examples/output-jitter/round3/mock_superbit_flight_jitter_only_oversampled_1x30000?.fits')
flats = glob.glob('/Users/jemcclea/Research/SuperBIT_2019/A2218/FlatImages/*')
biases = glob.glob('/Users/jemcclea/Research/SuperBIT_2019/A2218/BiasImages/*')
darks = glob.glob('/Users/jemcclea/Research/SuperBIT_2019/A2218/DarkImages/*')
truth_file='/Users/jemcclea/Research/GalSim/examples/output-jitter/round3/truth_flight_jitter_only_oversampled_1x300007.dat'
try:
    bm = medsmaker.BITMeasurement(image_files=science, truth_file=truth_file)
    # The path names should be updated; as written the code also expects all
    # calibration files to be in the same directory


    bm.set_working_dir(path='/Users/jemcclea/Research/SuperBIT/superbit-ngmix/scripts/output-debug/optics_gaussJitter')
    bm.set_path_to_psf(path='/Users/jemcclea/Research/SuperBIT/superbit-ngmix/scripts/output-debug/optics_gaussJitter/psfex_output')

    bm.run(clobber=False,source_selection = True, outfile = "/Users/jemcclea/Research/SuperBIT/superbit-ngmix/scripts/output-debug/optics_gaussJitter/mock_optics_gaussJitter.meds")

except:
    thingtype, value, tb = sys.exc_info()
    traceback.print_exc()
    pdb.post_mortem(tb)
