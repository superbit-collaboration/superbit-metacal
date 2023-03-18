'''
This script is for printing a SuperBIT OBA config (global or target) on the QCC
during flight operations. It is designed to work on the outputs of prep_oba.py
been installed
'''

from pathlib import Path
from argparse import ArgumentParser

from superbit_lensing import utils
from oba_io import IOManager

def parse_args():

    parser = ArgumentParser()

    parser.add_argument('target_name', type=str,
                        help='The name of the target whose config youd like ' +
                        'to print. For the global config, pass "global" ' +
                        '(ignores case)')

    # NOTE: not registered by the QCC, just for testing locally
    parser.add_argument('-root_dir', type=str, default=None,
                        help='Root directory for OBA run (if testing locally)')

    return parser.parse_args()

def main(args):

    target_name = args.target_name
    root_dir = args.root_dir

    #-----------------------------------------------------------------
    # Load in existing global or target OBA config

    if target_name.lower() == 'global':
        config_file = Path(utils.MODULE_DIR) / 'oba/configs/oba_global_config.yaml'
    else:
        # try to look for a pre-made target OBA config file produced by prep_oba.py
        io_manager = IOManager(root_dir=root_dir, target_name=target_name)
        config_file = io_manager.OBA_TARGET / f'{target_name}_oba.yaml'

    if not config_file.is_file():
        print(f'Failed: {config_file} not found (did you run prep_oba.py?)')
        return 1

    config = utils.read_yaml(config_file)

    for outer_key in config:
        print(f'{outer_key}:')
        if isinstance(config[outer_key], dict):
            for inner_key, val in config[outer_key].items():
                print(f'  {inner_key}: {val}')
        elif isinstance(config[outer_key], list):
            for item in config[outer_key]:
                print(f'  -{item}')
        else:
            print(config[outer_key])

    return 0

if __name__ == '__main__':
    args = parse_args()
    rc = main(args)
