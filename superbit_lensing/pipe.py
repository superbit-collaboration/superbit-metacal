from abc import ABC, abstractmethod
import os
import yaml
import logging
import subprocess

import utils
from diagnostics import build_diagnostics

import pudb

class SuperBITModule(dict):
    '''
    Class for an arbitrary module in the SuperBIT weak lensing pipeline

    These modules are not intended to be run on their own, but as components
    called in the `SuperBITPipeline` class

    The pipeline handles the usual things like logging, verbosity, etc.
    '''

    _req_fields = []
    _opt_fields = []

    def __init__(self, name, config, set_defaults=False):

        self.name = name

        if not isinstance(config, dict):
            raise TypeError(f'The passed config for module {name} was not a dict!')

        self._config = config
        self._check_config(set_defaults=set_defaults)

        self.diagnostics = build_diagnostics(name, config)

        return

    def _check_config(self, set_defaults=False):
        '''
        Make sure all required elements of the module config are present.
        '''

        # Check that required fields are there
        for field in self._req_fields:
            if field not in self._config.keys():
                raise KeyError(f'Field "{field}" must be present in the {self.name} config!')

        # Make sure there are no surprises
        for field in self._config.keys():
            if (field not in self._req_fields) and (field not in self._opt_fields):
                raise KeyError(f'{field} is not a valid field for the {self.name} module!')

        # Make sure all fields are at least initialized to None
        if set_defaults is True:
            for field in self._opt_fields:
                if field not in self._config.keys():
                    self._config[field] = None

        # anything else?
        # ...

        return

    def _run_setup(self, logprint):
        logprint(f'Starting module {self.name}')

        # ...

        return

    def _setup_options(self, run_options):
        options = ''
        for opt in self._opt_fields:
            if opt in self._config:
                options += f' --{opt}={self._config[opt]}'

        vb = ' -v' if run_options['vb'] is True else ''
        options += vb

        return options

    def _run_cleanup(self, logprint):

        logprint(f'Module {self.name} completed succesfully')

        # ...
        return

    @abstractmethod
    def run(self, run_options, logprint):
        pass

    def run_diagnostics(self, run_options, logprint):
        self.diagnostics.run(run_options, logprint)

        return

    def _run_command(self, cmd, logprint):
        '''
        Run bash command
        '''

        logprint(f'\n{cmd}\n')

        # for live prints but no error handling:
        # process = subprocess.Popen(cmd.split(),
        #                            stdout=subprocess.PIPE,
        #                            stderr=subprocess.PIPE,
        #                            bufsize=1)

        # for line in iter(process.stdout.readline, b''):
        #     logprint(line.decode('utf-8').replace('\n', ''))

        # output, error = process.communicate()

        args = [cmd.split()]
        kwargs = {'stdout':subprocess.PIPE,
                  # TODO: check this!
                  # 'stderr':subprocess.PIPE,
                  'stderr':subprocess.STDOUT,
                  'bufsize':1}
        with subprocess.Popen(*args, **kwargs) as process:
            try:
                for line in iter(process.stdout.readline, b''):
                    logprint(utils.decode(line).replace('\n', ''))

                stdout, stderr = process.communicate()

            except:
                process.kill()
                raise
                # stdout, stderr = process.communicate()
                # logprint('\n'+utils.decode(stderr))

                # return 1
                # process.kill()

            rc = process.poll()

            if rc:
                # stdout, stderr = process.communicate()
                err = subprocess.CalledProcessError(rc,
                                                    process.args,
                                                    output=stdout,
                                                    stderr=stderr)

                logprint('\n'+utils.decode(err.stderr))
                raise err

                # return rc

        # try:
        #     process = subprocess.run(cmd.split(),
        #                              stdout=subprocess.PIPE,
        #                              stderr=subprocess.PIPE,
        #                              # stderr=subprocess.PIPE,
        #                              # stderr=subprocess.STDOUT,
        #                              bufsize=1,
        #                              check=True)
        # except subprocess.CalledProcessError as e:
        #     logprint(e.stderr)
        #     return 1

        # output, error = process.communicate()
        # rc = process.returncode

        # if rc != 0:
        #     logprint(f'call returned in error with rc={rc}:')
        #     logprint(output)
        #     logprint(error)
        #     return rc

        # process.wait()

        # now get the return code from the executed process
        # streamdata = process.communicate()[0]
        # res = process.communicate()
        rc = process.returncode

        # process.stdout.close()

        return rc

    def __getitem__(self, key):
        val = self._config.__getitem__(self, key)

        return val

    def __setitem__(self, key, val):

        self._config.__setitem__(self, key, val)

        return

class SuperBITPipeline(SuperBITModule):

    # We abuse the required fields variable a bit here, as
    # the pipeline module is atypical.
    _req_fields = ['run_options']
    _req_run_options_fields = ['run_name', 'order', 'vb']

    # _opt_fields = {get_module_types().keys()}
    _opt_fields = [] # Gets setup in constructor
    _opt_run_options_fields = ['ncores', 'run_diagnostics']

    def __init__(self, config_file, log=None):

        config = utils.read_yaml(config_file)

        self._setup_opt_fields()

        super(SuperBITPipeline, self).__init__('pipeline',
                                               config)
                                               # set_defaults=False)
        self._check_config()

        self.log = log
        self.vb = self._config['run_options']['vb']

        self.logprint = utils.LogPrint(log, self.vb)

        if self._config['run_options']['ncores'] is None:
            # use half available cores by default
            ncores = os.cpu_count() #// 2
            self._config['run_options']['ncores'] = ncores
            self.logprint(f'`ncores` was not set; using all available ({ncores})')

        col = 'run_diagnostics'
        if col in self._config['run_options']:
            self.do_diagnostics = self._config['run_options'][col]
        else:
            self.do_diagnostics = False
            self._config['run_options'][col] = False

        module_names = list(self._config['run_options']['order'])
        # module_names.remove('run_options') # Not really a module

        self.modules = []
        for name in module_names:
            self.modules.append(build_module(name,
                                             self._config[name],
                                             self.logprint))

        return

    def _check_config(self, set_defaults=False):
        '''
        Make sure all required elements of the module config are present.
        '''

        # super(SuperBITPipeline, self)._check_config(set_defaults=set_defaults)

        # The pipeline module required fields variable is a bit special
        for field in self._req_run_options_fields:
            if field not in self._config['run_options'].keys():
                raise KeyError(f'Must have an entry for "{field}" in the "run_options" ' + \
                'field for the {self.name} config!')

        if set_defaults is True:
            for field in self._opt_run_options_fields:
                if field not in self._config['run_options'].keys():
                    self._config['run_options'][field] = None

        return

    def _setup_opt_fields(cls):
        cls._opt_fields = MODULE_TYPES.keys()

        return

    def run(self):
        '''
        Run each module in order
        '''

        self.logprint('\nStarting pipeline run\n')
        for module in self.modules:
            rc = module.run(self._config['run_options'], self.logprint)

            if rc !=0:
                self.logprint.warning(f'Exception occured during {module.name}.run()')
                return 1

            if self.do_diagnostics is True:
                module.run_diagnostics(self._config['run_options'], self.logprint)

        self.logprint('\nFinished pipeline run')

        return 0

class GalSimModule(SuperBITModule):
    _req_fields = ['config_file', 'outdir']
    _opt_fields = ['config_dir', 'vb', 'use_mpi', 'use_srun', 'run_name',
                   'clobber']

    def __init__(self, name, config):
        super(GalSimModule, self).__init__(name, config)

        self.gs_config_path = None
        self.gs_config = None

        return

    def run(self, run_options, logprint):
        '''
        Relevant type checks and param init's have already
        taken place
        '''

        logprint(f'\nRunning module {self.name}\n')
        logprint(f'config:\n{self._config}')

        self.gs_config_path = os.path.join(self._config['config_dir'],
                                           self._config['config_file'])
        self.gs_config = utils.read_yaml(self.gs_config_path) # Do we need this?

        cmd = self._setup_run_command(run_options)

        rc = self._run_command(cmd, logprint)
        # rc = utils.run_command(cmd, logprint)

        return rc

    def _setup_run_command(self, run_options):

        galsim_dir = os.path.join(utils.MODULE_DIR, 'galsim')
        galsim_filepath = os.path.join(galsim_dir, 'mock_superBIT_data.py')

        outdir = self._config['outdir']
        base = f'python {galsim_filepath} {self.gs_config_path} --outdir={outdir}'

        # options = self._setup_options(run_options)
        options = ''

        if 'run_name' not in self._config:
            run_name = run_options['run_name']
            options += f' --run_name={run_name}'

        if 'clobber' in self._config:
            options += ' --clobber'

        if run_options['vb'] is True:
            options += ' -v'

        cmd = base + options

        ncores = run_options['ncores']
        if ncores > 1:
            if hasattr(self._config, 'use_mpi'):
                if self._config['use_mpi'] is True:
                    cmd = f'mpiexec -n {ncores} ' + cmd
            if hasattr(self._config, 'use_srun'):
                if self._config['use_srun'] is True:
                    cmd = f'srun --mpi=pmix ' + cmd
            else:
                cmd = cmd + f' --ncores={ncores}'

        return cmd

class MedsmakerModule(SuperBITModule):
    _req_fields = ['mock_dir', 'outfile']
    _opt_fields = ['fname_base', 'meds_coadd', 'outdir', 'clobber', 'source_select',
                   'cut_stars', 'vb']

    def __init__(self, name, config):
        super(MedsmakerModule, self).__init__(name, config)

        # ...

        return

    def run(self, run_options, logprint):
        logprint(f'\nRunning module {self.name}\n')
        logprint(f'config:\n{self._config}')

        cmd = self._setup_run_command(run_options)

        rc = self._run_command(cmd, logprint)

        return rc

    def _setup_run_command(self, run_options):

        mock_dir = self._config['mock_dir']
        outfile = self._config['outfile']
        outdir = self._config['outdir']

        filepath = os.path.join(utils.get_module_dir(),
                                'medsmaker',
                                'scripts',
                                'process_mocks.py')

        base = f'python {filepath} {mock_dir} {outfile}'

        options = self._setup_options(run_options)

        cmd = base + options

        return cmd

class MetacalModule(SuperBITModule):
    _req_fields = ['medsfile', 'outfile']
    _opt_fields = ['outdir','start', 'end', 'plot', 'n', 'vb']

    def __init__(self, name, config):
        super(MetacalModule, self).__init__(name, config)

        col = 'outdir'
        if col not in self._config:
            self._config[col] = os.getcwd()

        return

    def run(self, run_options, logprint):
        logprint(f'\nRunning module {self.name}\n')
        logprint(f'config:\n{self._config}')

        cmd = self._setup_run_command(run_options)

        rc = self._run_command(cmd, logprint)

        return rc

    def _setup_run_command(self, run_options):

        run_name = run_options['run_name']
        outdir = self._config['outdir']
        medsfile = self._config['medsfile']
        outfile = self._config['outfile']
        mcal_dir = os.path.join(utils.get_module_dir(),
                                'metacalibration')
        filepath = os.path.join(mcal_dir, 'ngmix_fit_superbit3.py')

        base = f'python {filepath} {medsfile} {outfile}'

        # Set up some default values that require the run config
        col = 'n'
        if col not in self._config:
            self._config['n'] = run_options['ncores']

        col = 'plot'
        if col not in self._config:
            self._config['plot'] = run_options['run_diagnostics']

        # options = f' --outdir={outdir} --plot={plot} --n={ncores}' + \
        #     self._setup_options(run_options)
        options = self._setup_options(run_options)

        cmd = base + options

        return cmd

class NgmixFitModule(SuperBITModule):
    _req_fields = ['medsfile', 'outfile', 'config']
    _opt_fields = ['outdir', 'start', 'end', 'n', 'clobber', 'vb']

    def __init__(self, name, config):
        super(NgmixFitModule, self).__init__(name, config)

        col = 'outdir'
        if col not in self._config:
            self._config[col] = os.getcwd()

        return

    def run(self, run_options, logprint):
        logprint(f'\nRunning module {self.name}\n')
        logprint(f'config:\n{self._config}')

        cmd = self._setup_run_command(run_options)

        rc = self._run_command(cmd, logprint)

        return rc

    def _setup_run_command(self, run_options):

        outdir = self._config['outdir']

        medsfile = self._config['medsfile']
        outfile = self._config['outfile']
        outfile = os.path.join(outdir, outfile)
        config = self._config['config']

        ngmix_dir = os.path.join(utils.get_module_dir(),
                                'ngmix_fit')
        filepath = os.path.join(ngmix_dir, 'ngmix_fit.py')

        base = f'python {filepath} '

        # While these are considered required for the pipeline, they are
        # technically optional args in ngmix_fit.py due to testing option
        base += f'--medsfile={medsfile} --outfile={outfile} --config={config}'

        # Setup some options that rquire the run config
        col = 'n'
        if col not in self._config:
            self._config['n'] = run_options['ncores']

        options = self._setup_options(run_options)

        cmd = base + options

        return cmd

class ShearProfileModule(SuperBITModule):
    _req_fields = []
    _opt_fields = []

    def __init__(self, name, config):
        super(ShearProfileModule, self).__init__(name, config)

        raise Exception('Not yet implemented!')

        col = 'outdir'
        if col not in self._config:
            self._config[col] = os.getcwd()

        return

    def run(self, run_options, logprint):
        logprint(f'\nRunning module {self.name}\n')
        logprint(f'config:\n{self._config}')

        cmd = self._setup_run_command(run_options)

        rc = self._run_command(cmd, logprint)

        return rc

    def _setup_run_command(self, run_options):

        outdir = self._config['outdir']

        medsfile = self._config['medsfile']
        outfile = self._config['outfile']
        outfile = os.path.join(outdir, outfile)
        config = self._config['config']

        ngmix_dir = os.path.join(utils.get_module_dir(),
                                'ngmix_fit')
        filepath = os.path.join(ngmix_dir, 'ngmix_fit.py')

        base = f'python {filepath} '

        # While these are considered required for the pipeline, they are
        # technically optional args in ngmix_fit.py due to testing option
        base += f'--medsfile={medsfile} --outfile={outfile} --config={config}'

        # Setup some options that rquire the run config
        col = 'n'
        if col not in self._config:
            self._config['n'] = run_options['ncores']

        options = self._setup_options(run_options)

        cmd = base + options

        return cmd

def build_module(name, config, logprint):
    name = name.lower()

    if name in MODULE_TYPES.keys():
        # User-defined input construction
        module = MODULE_TYPES[name](name, config)
    else:
        # Attempt generic input construction
        logprint(f'Warning: {name} is not a pre-defined module type.')
        logprint('Attempting generic module construction.')
        logprint('Module is not guaranteed to run succesfully.')

        module = SuperBITModule(name, config)

    return module

def make_test_ngmix_config(config_file='ngmix_test.yaml', outdir=None,
                           run_name=None, clobber=False):
    if outdir is not None:
        filename = os.path.join(outdir, config_file)

    if run_name is None:
        run_name = 'pipe_test'

    if (clobber is True) or (not os.path.exists(filename)):
        with open(filename, 'w') as f:
            CONFIG = {
                'gal': {
                    'model': 'bdf',
                },
                'psf': {
                    'model': 'gauss'
                },
                'priors': {
                    'T_range': [-1., 1.e3],
                    'F_range': [-100., 1.e9],
                    'g_sigma': 0.1,
                    'fracdev_mean': 0.5,
                    'fracdev_sigma': 0.1
                },
                'fit_pars': {
                    'method': 'lm',
                    'lm_pars': {
                        'maxfev':2000,
                        'xtol':5.0e-5,
                        'ftol':5.0e-5
                        }
                },
                'pixel_scale': 0.144, # arcsec / pixel
                'nbands': 1,
                'seed': 172396,
                'run_name': run_name
            }
            yaml.dump(CONFIG, f, default_flow_style=False)

    return filename

def make_test_config(config_file='pipe_test.yaml', outdir=None, clobber=False):
    if outdir is not None:
        filename = os.path.join(outdir, config_file)

    if (clobber is True) or (not os.path.exists(filename)):
        run_name = 'pipe_test'
        outdir = os.path.join(utils.TEST_DIR, run_name)
        medsfile = os.path.join(outdir, f'{run_name}_meds.fits')
        ngmix_test_config = make_test_ngmix_config('ngmix_test.yaml',
                                                   outdir=outdir,
                                                   run_name=run_name)
        with open(filename, 'w') as f:
            # Create dummy config file
            CONFIG = {
                'run_options': {
                    'run_name': run_name,
                    'outdir': outdir,
                    'vb': True,
                    'ncores': 8,
                    'run_diagnostics': True,
                    'order': [
                        'galsim',
                        'medsmaker',
                        'metacal',
                        'ngmix-fit'
                        ]
                },
                'galsim': {
                    'config_file': 'pipe_test.yaml',
                    # 'config_file': 'superbit_parameters_forecast.yaml',
                    'config_dir': os.path.join(utils.MODULE_DIR,
                                               'galsim',
                                               'config_files'),
                    'outdir': outdir,
                    'clobber': True
                },
                'medsmaker': {
                    'mock_dir': outdir,
                    'outfile': f'{run_name}_meds.fits',
                    'fname_base': run_name,
                    'outdir': outdir
                },
                'metacal': {
                    'medsfile': medsfile,
                    'outfile': f'{run_name}_mcal.fits',
                    'outdir': outdir,
                    'end': 100
                },
                'ngmix-fit': {
                    'medsfile': medsfile,
                    'outfile': f'{run_name}_ngmix.fits',
                    'config': ngmix_test_config,
                    'outdir': outdir,
                    'end': 100
                }
            }

            yaml.dump(CONFIG, f, default_flow_style=False)

    return filename

def get_module_types():
    return MODULE_TYPES

# NOTE: This is where you must register a new module
MODULE_TYPES = {
    'galsim': GalSimModule,
    'medsmaker': MedsmakerModule,
    'metacal': MetacalModule,
    'ngmix-fit': NgmixFitModule,
    'shear-profile': ShearProfileModule,
    }

