import os
import superbit_lensing.utils as utils
from superbit_lensing.pipe import SuperBITPipeline
import time

<<<<<<< HEAD
start_time = time.time()
config_file='./configs/test_psfex.yaml'
log_file='psfex_cutout_test.log'
logdir='/users/jmcclear/data/superbit/forecasting-analysis/psfex_cutout_tests'
=======
config_file='/Users/jemcclea/Research/SuperBIT/superbit-metacal/configs/pipe_test_real_sims.yaml'
log_file='psfex_test_real_sims.log'
logdir='/Users/jemcclea/Research/SuperBIT/forecasting_analysis/psfex_test/'
>>>>>>> cbae47bbfa4ede6725529f0bb868fd928c62da23

if not os.path.isdir(logdir):
    cmd='mkdir -p %s' % logdir
    os.system(cmd)


log = utils.setup_logger(log_file, logdir=logdir)
pipe = SuperBITPipeline(config_file, log)

rc = pipe.run()

assert(rc == 0)


end_time=time.time()
print("\n\n\n Pipeline execution time = %.2f\n\n\n" % (end_time - start_time))
