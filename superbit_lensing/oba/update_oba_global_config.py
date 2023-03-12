import shutil
from pathlib import Path
from argparse import ArgumentParser
from glob import glob
import fitsio

from superbit_lensing import utils

import ipdb

'''
This script is used for updating the SuperBIT onboard analysis (OBA) global
config file using the QCC commander. It only updates one config value at a
time. As each config field (besides `modules`) is a 2-level dict, we use
the following scheme:

python update_oba_global_config.py outer_key, inner_key, new_val, val_type

outer_key:
  inner_key: old_val -> type(new_val)
'''

def parse_args():

    parser = ArgumentParser()

    parser.add_argument('outer_key', type=str,
                        help='Name of the outer OBA config key')
    parser.add_argument('inner_key', type=str,
                        help='Name of the inner OBA config key')
    parser.add_argument('new_val', type=str,
                        help='The new value for the OBA global config entry')
    parser.add_argument('val_type', type=str,
                        help='The explicit type that you want to save the ' +
                        'new config value as in the config')

    return parser.parse_args()

def main(args):

    #-----------------------------------------------------------------
    # Initial setup

    outer_key = args.outer_key
    inner_key = args.inner_key
    new_val = args.new_val
    val_type = args.val_type

    #-----------------------------------------------------------------
    # Load in existing global OBA config

    global_config_file = Path(utils.MODULE_DIR) / 'oba/configs/oba_global_config.yaml'
    global_config = utils.read_yaml(global_config_file)
    old_val = global_config[outer_key][inner_key]

    #-----------------------------------------------------------------
    # Handle the annoying case of bools
    if val_type == 'bool':
        new_val = new_val.title()
        if new_val == 'True':
            # this is ok as-is
            pass
        elif new_val == 'False':
            # this only works for an empty str
            new_val = ''
        else:
            raise ValueError('If val_type is bool, can only pass one of the ' +
                             'following: [True, False] (in any capitalization)')

    #-----------------------------------------------------------------
    # this grabs the relevant type operator for proper type casting
    casted_val = getattr(__builtins__, val_type)(new_val)
    global_config[outer_key][inner_key] = casted_val

    utils.write_yaml(global_config, global_config_file)

    print('Made the following update to the OBA global config:')
    print(f'{outer_key}:')
    print(f'  {inner_key}: {old_val} -> {casted_val}')

    return

if __name__ == '__main__':
    args = parse_args()
    main(args)
