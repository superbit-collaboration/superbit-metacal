import os
import superbit_lensing.utils as utils
from superbit_lensing.pipe import SuperBITPipeline

config_file='/Users/jemcclea/Research/SuperBIT/superbit-metacal/configs/pipe_test_real_sims.yaml'
log_file='pipe_test_real_sims.log'
logdir='./pipe_sexparam_tests/sigthresh1.1_minarea5/'

if not os.path.isdir(logdir):
    cmd='mkdir -p %s' % logdir
    os.system(cmd)


log = utils.setup_logger(log_file, logdir=logdir)
pipe = SuperBITPipeline(config_file, log)

rc = pipe.run()

assert(rc == 0)

