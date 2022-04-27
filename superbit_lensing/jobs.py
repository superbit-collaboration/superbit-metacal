import os
import shutil
from numpy.random import SeedSequence, default_rng
from argparse import ArgumentParser

import utils
from config import make_cluster_configs

parser = ArgumentParser()

parser.add_argument('jobs_config', type=str,
                    help='Filepath to yaml configuration file for all jobs')
parser.add_argument('--fresh', action='store_true', default=False,
                    help='Clean test directory of old outputs')

class JobsManager(object):

    def __init__(self, config_file, fresh=False):
        '''
        A class that manages the creation of multiple CluserJob's

        config: str
            Filepath of a configuration dictionary for managing multiple jobs
        fresh: bool
            Set to True to clean out any existing runs in specified
            paths
        '''

        self.config_file = config_file
        self.config = utils.read_yaml(config_file)
        self.fresh = fresh
        self.jobs = None

        return

    def run(self):
        '''
        Create each individual ClusterJob along with the necessary
        directory structure, config files, etc.
        '''

        self.jobs = self.create_jobs()
        self.set_job_seeds()
        self.make_job_configs()

        return


    def parse_config(self):
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

        for key, default in self._opt.items():
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

    def create_jobs(self):
        '''
        Create all ClusterJob's and required run directories
        given input configuration

        Returns a list of ClusterJob's
        '''

        config = self.config
        fresh = self.fresh

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
                        'run_name': run_name,
                        'mass': m,
                        'z': z
                        'realization': r,
                        'job_index': jindx
                        'mass_mantissa': mantissa,
                        'mass_exp': exp,
                        'ncores': ncores,
                        'memory': memory
                    }
                    jobs.append(ClusterJob(job_dict))
                    jindx += 1

        return jobs

    @staticmethod
    def fexp(x, precision=1):
        '''
        Returns the mantissa and exponent of float x in base 10
        for a given precision (i.e. # of digits after decimal)

        Acts like np.fexp, but for decimal numbers
        '''

        exp = int(np.log10(x))
        mantissa = round(x / 10**exp, precision)

        return mantissa, exp

    def set_job_seeds(self):
        '''
        Safely set multiple, uncorrelated seeds across jobs

        See https://numpy.org/doc/stable/reference/random/parallel.html
        for details
        '''

        Njobs = len(self.jobs)

        # Set master seed of *all* job seed generation
        # to be local time in microseconds
        ss = SeedSequence(int(time.time()*1e6))

        child_seeds = ss.spawn(Njobs)
        streams = [default_rng(s) for s in child_seeds]

        for i in range(len(self.jobs)):
            job_seed = int(streams[i].random()*1e16)
            self.jobs[i]['gs_master_seed'] = job_seed

        return

    def make_job_configs(self):
        '''
        jobs: list
            A list of ClusterJob's
        config: dict
            The prep_jobs configuration dictionary
        '''

        gs_base_config = utils.read_yaml(self.config['gs_base_config'])

        for i in range(len(self.jobs)):
            # Make gs job config from base config
            self.jobs[i].make_gs_config(gs_base_config)

            # Make pipe config
            # TODO!!
            # ...

    return

class ClusterJob(object):
    '''
    This class stores all relevant info for a given cluster simulation job

    Right now it is not much more than a dict, but may want to increase
    complexity later
    '''

    _req_params = ['base_dir', 'run_name', 'mass', 'redshift', 'job_index',
                   'ncores', 'memory', 'realization']

    _opt_params = ['gs_master_seed']

    def __init__(self, job_config):
        self.parse_job_config(job_config)
        self._config = job_config

        return

    def _parse_job_config(self, job_config):
        for name in self._req_params:
            if name not in job_config:
                raise KeyError(f'The passed job_config does not contain {name}!')

        return

    def make_gs_config(self, gs_base_config):
        '''
        Make a job-specific GalSim config file given a base config
        '''

        gs_base = utils.read_yaml(gs_base_config)
        gs_config = gs_base.copy()

        base_dir = self._config['base_dir']
        run_dir = self._config['run_name']

        gs_name = f'{run_name}_gs_config.yaml'
        gs_filepath = os.path.join(base_dir, gs_name)
        self._config['gs_config'] = gs_filepath

        # update base GalSim config w/ needed job-specific changes
        updates = {
            'mass': self._config['mass'], # Msol / h
            'nfw_z_halo': self._config['z'],
            'outdir': self._config['base_dir'],
            'master_seed': self._config['gs_master_seed']
        }
        for key, val in updates.items():
            gs_config[key] = val

        utils.write_yaml(gs_config, gs_filepath)

        return

    def __setitem__(self, key, item):
        self._config[key] = item

    def __getitem__(self, key):
        return self._config[key]

    def __repr__(self):
        return repr(self._config)

    def __len__(self):
        return len(self._config)

    def __delitem__(self, key):
        del self._config[key]

    def __contains__(self, item):
        return item in self._config

    def __iter__(self):
        return iter(self._config)

def main(args):
    config = args.jobs_config
    fresh = args.fresh

    manager = JobsManager(config, fresh=fresh)
    manager.run()

    return 0

if __name__ == '__main__':
    args = parser.parse_args()
    rc = main(args)

    if rc == 0:
        print('\nScript completed without errors')
    else:
        print(f'\nScript failed with rc={rc}')
