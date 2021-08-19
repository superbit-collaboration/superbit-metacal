import numpy as np
import fitsio
import os
import yaml
import utils

class SuperBITPipeline(dict):

    _req_fields = {'run_options': ['run_name', 'order']}

    def __init__(self, config_file):
        self.config_file = config_file

        with open(config_file, 'r') as stream:
            self._config = yaml.safe_load(stream)

        self._check_run_config()

        return

    def _check_run_config(self):
        '''
        Make sure all required elements of a run config are present.
        '''

        print(self._config['run_options'].keys())

        for field, lst in self._req_fields.items():
            if field not in self._config.keys():
                raise KeyError(f'Field "{field}" must be present in run config!')
            for l in lst:
                if l not in self._config[field].keys():
                    raise KeyError(f'Must have an entry for "{l}" in field "{field}"!')

        # anything else?
        # ...

        return

    def run(self):
        pass

    def __getitem__(self, key):
        val = self._config.__getitem__(self, key)

        return val

    def __setitem__(self, key, val):

        self._config.__setitem__(self, key, val)

        return

def make_test_config(config_file='pipe_test.yaml', clobber=False):
    basedir = os.path.dirname(__file__)
    testdir = os.path.join(basedir, 'tests')
    filename = os.path.join(testdir, config_file)

    if not os.path.isdir(testdir):
        os.mkdir(testdir)

    if (clobber is True) or (not os.path.exists(filename)):
        with open(filename, 'w') as f:
            # Create dummy config file
            CONFIG = {
                'run_options': {
                    'run_name': 'pipe_test',
                    'order': {
                        'GalSim',
                        'MedsMaker',
                        'Metacalibration'
                    }
                },
                'GalSim': {
                    'config_file': 'test.yaml',
                    'config_dir': 'test'
                },
                'MedsMaker': {
                    'config_dir': 'astro_config'
                },
                'Metacalibration': {
                    'ngmix_config': 'test.yaml',
                    'config_dir': 'test'
                },
            }

            yaml.dump(CONFIG, f, default_flow_style=False)

    return filename

def main():

    config_file = make_test_config(clobber=True)

    pipe = SuperBITPipeline(config_file)

    pipe.run()

    return 0

if __name__ == '__main__':
    res = main()

    if res == 0:
        print('Tests have completed without errors')
