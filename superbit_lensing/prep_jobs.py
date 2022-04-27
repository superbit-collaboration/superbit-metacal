import os
import shutil
from argparse import ArgumentParser

import utils
from config import make_cluster_configs

parser = ArgumentParser()

parser.add_argument('job_config', type=str,
                    help='Filepath to yaml configuration file for all jobs')
parser.add_argument('--fresh', action='store_true', default=False,
                    help='Clean test directory of old outputs')

class ClusterJob(object):
    '''
    This class stores all relevant info for a given cluster simulation job

    Right now it is not much more than a dict, but may want to increase
    complexity later
    '''

    _req_params = {'base_dir', 'mass', 'redshift', 'job_index', 'ncores',
                   'memory', }

    def __init__(self, job_dict):
        self.parse_job_dict(job_dict)
        self.job_dict = job_dict

        return

    def _parse_job_dict(self, job_dict):
        for name in self._req_params:
            if name not in job_dict:
                raise KeyError(f'The passed job_dict does not contain {name}!')

        return

def parse_config(config):
    '''
    Make sure the config satisfies requirements
    '''

    req = ['run_name', 'base_dir', 'nfw_dir', 'gs_base_config',
           'mass_bins', 'z_bins']

    # values are defaults if not present
    opt = {
        'realizations': 1,
        'ncores_per_job': 8,
        'memory_per_job': 64, # GB
    }

    for key in req:
        if key not in config.keys():
            raise ValueError(f'{key} must be in the run prep config!')

    for key, default in opt.items():
        if key not in config.keys():
            config[key] = default

    masses = config['mass_bins']
    redshifts = config['z_bins']

    for vals, name in zip([masses, redshifts], ['mass_bins', 'z_bins']):
        if isinstance(vals, list):
            for v in vals:
                if not isinstance(v, float):
                    raise TypeError(f'`{name}` entries must be a float!')
        elif isinstance(v, float):
            pass
        else:
            raise TypeError('`{name}` must be a float or list of floats!')

    if isinstance(realizations, list):
        for i in realizations:
            if not isinstance(i, int):
                raise TypeError('Each element of `realizations` must be an int!')
        else:
            if isinstance(realizations, int):
                # assume this gives the # of realizations
                realizations = [i for i in range(realizations)]
            else:
                raise TypeError('`realizations` must be an int or list!')

    return

def create_run_directories(config, fresh=False):
    '''
    Create all required run directories given input configuration
    '''

    base_dir = config['basedir']
    run_name = config['run_name']

    if os.path.exists(base_dir):
        if fresh is False:
            raise Exception(f'{base_dir} already exists; set `fresh` ' +\
                            'to True to overwrite all files!')
        else:
            print(f'{base_dir} already exists; cleaning dir as `fresh==True`')
            shutil.rmtree(base_dir)

    end_dir = base_dir.split('\\')[-1]
    if end_dir != base_dir:
        print(f'Warning: dir {end_dir} does not match base_name. Is that correct?')
        print('Continuing for now...')

    utils.make_dir(base_dir)

    masses = config['mass_bins']
    redshifts = config['z_bins']
    realizations = config['realizations']
    ncores = config['ncores_per_job']
    memory = config['memory_per_job']

    jobs = [] # list of ClusterJob's
    jindx = 0
    for m in masses:
        for z in redshifts:

            # We need to have a clean mapping of masses into
            # subdir names
            mantissa, exp = fexp(m)
            cl_name = f'cl_m{mantissa}e{exp}_z{z}'

            cluster_dir = os.path.join(base_dir, cl_name)
            utils.make_dir(cluster_dir)
            for r in realizations:
                # make subdirectory
                real_dir = os.path.join(cluster_dir, f'r{r}')
                utils.make_dir(real_dir)

                job_dict = {
                    'base_dir': real_dir,
                    'mass': m,
                    'z': z
                    'job_index': jindx
                    'mass_mantissa': mantissa,
                    'mass_exp': exp,
                    'ncores': ncores,
                    'memory': memory
                }
                jobs.append(ClusterJob(job_dict))
                jindx += 1

    return jobs

def fexp(x, precision=1):
    '''
    Returns the mantissa and exponent of float x in base 10
    for a given precision (i.e. # of digits after decimal)

    Acts like np.fexp, but for decimal numbers
    '''

    exp = int(np.log10(x))

    mantissa = round(x / 10**exp, precision)

    return mantissa, exp

def main(args):
    job_config = args.job_config
    fresh = args.fresh

    config = utils.read_yaml(job_config)

    create_run_directories(config, fresh=fresh)

    return 0

if __name__ == '__main__':
    args = parser.parse_args()
    rc = main(args)

    if rc == 0:
        print('\nScript completed without errors')
    else:
        print(f'\nScript failed with rc={rc}')
