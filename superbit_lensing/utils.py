import logging
import os
import sys
import yaml
from astropy.table import Table
import superbit_lensing as sb
import numpy as np
import subprocess
import pdb, pudb

class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child
    https://stackoverflow.com/questions/4716533/how-to-attach-debugger-to-a-python-subproccess
    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

        return

class LogPrint(object):

    def __init__(self, log, vb):
        '''
        Requires a logging obj and verbosity level
        '''

        # Must be a Logger object
        assert(isinstance(log, logging.Logger))

        self.log = log
        self.vb = vb

        return

    def __call__(self, msg):
        '''
        treat it like print()
        e.g. lprint = LogPrint(...); lprint('message')
        '''
        self.log.info(msg)
        if self.vb is True:
            print(msg)

        return

    def debug(self, msg):
        '''
        don't print for a debug
        '''
        self.log.debug(msg)

        return

    def warning(self, msg):
        self.log.warning(msg)
        if self.vb is True:
            print(msg)

        return


class Logger(object):

    def __init__(self, logfile, logdir=None):
        if logdir is None:
            logdir = './'

        self.logfile = os.path.join(logdir, logfile)

        # only works for newer versions of python
        # log = logging.basicConfig(filename=logfile, level=logging.DEBUG)

        # instead:
        log = logging.getLogger()
        log.setLevel(logging.DEBUG)
        # log.setLevel(logging.ERROR)
        handler = logging.FileHandler(self.logfile, 'w', 'utf-8')
        handler.setFormatter(logging.Formatter('%(name)s %(message)s'))
        log.addHandler(handler)

        self.log = log

        return

    # other useful things?
    # ...

def setup_logger(logfile, logdir=None):
    '''
    Utility function if you just want the log and not the Logger object
    '''

    if logdir is not None:
        if not os.path.exists(logdir):
            os.mkdir(logdir)

    logger = Logger(logfile, logdir=logdir)

    return logger.log

def read_yaml(yaml_file):
    with open(yaml_file, 'r') as stream:
        return yaml.safe_load(stream)

# def check_req_params(params, values, defaults, config):
def check_req_params(config, params, defaults):
    '''
    Ensure that certain required parameters have their values set to
    something either than the default after a configuration file is read.
    This is needed to allow certain params to be set either on the command
    line or config file.

    config: An object that (potentially) has the param values stored as
    attributes
    params: List of required parameter names
    defaults: List of default values of associated params
    '''

    for param, default in zip(params, defaults):
        # Should at least be set by command line arg defaults, but double check:
        if (not hasattr(config, param)) or (getattr(config, param) == default):
            e_msg = f'Must set {param} either on command line or in passed config!'
            raise Exception(e_msg)

    return

def check_req_fields(config, req, name=None):
    for field in req:
        if not field in config:
            raise ValueError(f'{name}config must have field {field}')

    return

def check_fields(config, req, opt, name=None):
    '''
    req: list of required field names
    opt: list of optional field names
    name: name of config type, for extra print info
    '''
    assert isinstance(config, dict)

    if name is None:
        name = ''
    else:
        name = name + ' '

    if req is None:
        req = []
    if opt is None:
        opt = []

    # ensure all req fields are present
    check_req_fields(config, req, name=name)

    # now check for fields not in either
    for field in config:
        if (not field in req) and (not field in opt):
            raise ValueError(f'{field} not a valid field for {name}config!')

    return

def sigma2fwhm(sigma):
    c = np.sqrt(8.*np.log(2))
    return c * sigma

def fwhm2sigma(fwhm):
    c = np.sqrt(8.*np.log(2))
    return fwhm / c

def decode(msg):
    if isinstance(msg, str):
        return msg
    elif isinstance(msg, bytes):
        return msg.decode('utf-8')
    elif msg is None:
        return ''
    else:
        print(f'Warning: message={msg} is not a string or bytes')
        return msg

def run_command(cmd, logprint=None):

    if logprint is None:
        # Just remap to print then
        logprint = print

    args = [cmd.split()]
    kwargs = {'stdout':subprocess.PIPE,
              'stderr':subprocess.STDOUT,
              # 'universal_newlines':True,
              'bufsize':1}

    with subprocess.Popen(*args, **kwargs) as process:
        try:
            # for line in iter(process.stdout.readline, b''):
            for line in iter(process.stdout.readline, b''):
                logprint(decode(line).replace('\n', ''))

            stdout, stderr = process.communicate()

        except:
            logprint('')
            logprint('.....................ERROR....................')
            logprint('')

            logprint('\n'+decode(stderr))
            # try:
            #     logprint('\n'+decode(stderr))
            # except AttributeError:
            #     logprint('\n'+stderr)

            rc = process.poll()
            raise subprocess.CalledProcessError(rc,
                                                process.args,
                                                output=stdout,
                                                stderr=stderr)
            # raise subprocess.CalledProcessError(rc, cmd)

        rc = process.poll()

        # if rc:
        #     stdout, stderr = process.communicate()
        #     logprint('\n'+decode(stderr))
            # return 1

        if rc:
            stdout, stderr = process.communicate()
            logprint('\n'+decode(stderr))
            # raise subprocess.CalledProcessError(rc, cmd)
            raise subprocess.CalledProcessError(rc,
                                                process.args,
                                                output=stdout,
                                                stderr=stderr)

    # rc = popen.wait()

    # rc = process.returncode

    return rc

def ngmix_dict2table(d):
    '''
    convert the result of a ngmix fit to an astropy table
    '''

    # Annoying, but have to do this to make Table from scalars
    for key, val in d.items():
        d[key] = np.array([val])

    return Table(data=d)

def make_dir(d):
    '''
    Makes dir if it does not already exist
    '''

    if not os.path.exists(d):
        os.mkdir(d)

    return

def get_base_dir():
    '''
    base dir is parent repo dir
    '''
    module_dir = get_module_dir()
    return os.path.dirname(module_dir)

def get_module_dir():
    return os.path.dirname(__file__)

def get_test_dir():
    base_dir = get_base_dir()
    return os.path.join(base_dir, 'tests')

BASE_DIR = get_base_dir()
MODULE_DIR = get_module_dir()
TEST_DIR = get_test_dir()
