import os
import superbit_lensing.utils as utils
from superbit_lensing.pipe import SuperBITPipeline
import time

start_time = time.time()
config_file='./configs/do_forecast_sims.yaml'
log_file='cl5_v3_sims_omp8.log'
logdir='/users/jmcclear/data/superbit/forecasting-analysis/cluster5'

if not os.path.isdir(logdir):
    cmd='mkdir -p %s' % logdir
    os.system(cmd)


log = utils.setup_logger(log_file, logdir=logdir)
pipe = SuperBITPipeline(config_file, log)

rc = pipe.run()

assert(rc == 0)


end_time=time.time()
print("\n\n\n Pipeline execution time = %.2f\n\n\n" % (end_time - start_time))
