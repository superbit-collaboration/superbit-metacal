'''
This script is for printing a SuperBIT OBA config (global, astromatic, or target)
on the QCC during flight operations. For a target config, it is designed to work
on the outputs of prep_oba.py

To use, simply do the following:

python print_oba_config {config_type}

where config_type can be one of the following (case insensitive except for target):

- target_name: The name of a target for its current config (requires running
  prep_oba.py first)
- global: The global OBA config that target configs are built from
- swarp: The global SWarp config
- sextractor_{band}: The global OBA SExtractor config for a given band

If running locally (i.e. *not* on the QCC), then pass the path to your local
test QCC using -root_dir
'''

from pathlib import Path
from argparse import ArgumentParser

from superbit_lensing import utils
from oba_io import IOManager

import ipdb

def parse_args():

    parser = ArgumentParser()

    parser.add_argument('config_type', type=str,
                        help='The name of the config that youd like to print. ' +
                        'For a target, pass the target_name. For the global ' +
                        'config, pass "global" (ignores case). Similarly for ' +
                        '"swarp" or "sextractor_{band}"')

    # NOTE: not registered by the QCC, just for testing locally
    parser.add_argument('-root_dir', type=str, default=None,
                        help='Root directory for OBA run (if testing locally)')

    return parser.parse_args()

def main(args):

    config_type = args.config_type
    root_dir = args.root_dir

    #-----------------------------------------------------------------
    # Load in existing global or target OBA config

    config_type = config_type.lower()
    is_target = False

    if config_type == 'global':
        config_file = Path(utils.MODULE_DIR) / 'oba/configs/oba_global_config.yaml'
    elif config_type == 'swarp':
        config_file = Path(utils.MODULE_DIR) / 'oba/configs/swarp/swarp.config'
    elif 'sextractor' in config_type:
        config_file = Path(utils.MODULE_DIR) / f'oba/configs/sextractor/sb_{config_type}.config'
    else:
        # try to look for a pre-made target OBA config file produced by prep_oba.py
        is_target = True
        target_name = config_type
        io_manager = IOManager(root_dir=root_dir, target_name=target_name)
        config_file = io_manager.OBA_TARGET / f'{target_name}_oba.yaml'

    if not config_file.is_file():
        msg = f'Failed: {config_file} not found'
        if is_target is True:
            msg += ' (did you run prep_oba.py?)'

        print(msg)
        return 1

    # OBA global & target configs are yaml files
    if (config_type == 'global') or is_target:
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

    # astromatic configs are txt files
    else:
        with open(config_file) as f:
            for line in f:
                # to save space, don't print additional new line
                if line[-1] == '\n':
                    line = line[0:-1]
                print(line)

    return 0

if __name__ == '__main__':
    args = parse_args()
    rc = main(args)
