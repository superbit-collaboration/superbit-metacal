import os
import superbit_lensing.utils as utils
from superbit_lensing.pipe import SuperBITPipeline

config_file='/Users/jemcclea/Research/SuperBIT/superbit-metacal/configs/pipe_test_real_sims.yaml'
log_file='psfex_test_real_sims.log'
logdir='/Users/jemcclea/Research/SuperBIT/forecasting_analysis/psfex_test/'

if not os.path.isdir(logdir):
    cmd='mkdir -p %s' % logdir
    os.system(cmd)


log = utils.setup_logger(log_file, logdir=logdir)
pipe = SuperBITPipeline(config_file, log)

rc = pipe.run()

assert(rc == 0)

