'''
This script is used for updating the SuperBIT onboard analysis (OBA) global
config file using the QCC commander. It only updates one config value at a
time. As each config field (besides `modules`) is a 2-level dict, we *would*
use the following scheme:

python update_oba_global_config.py outer_key, inner_key, new_val, val_type

*HOWEVER*, the QCC commander can only pass 1 string. As all of these inputs
need to be strings (and cannot use commas), we do the following:

python update_oba_global_config.py "outer_key&inner_key&new_val&val_type"

where:

- outer_key: Name of the outer OBA config key
- inner_key: Name of the inner OBA config key
- new_val: The new value for the OBA global config entry
- val_type: The explicit type that you want to save the new config value as
            in the config

NOTE: ".", "_", "-", etc. are not acceptable delimiters as they may be part of
the new value you are trying to construct

NOTE: to pass a list or tuple of str's, do something like the following:

python update_oba_global_config.py outer_key&inner_key&['val1', 'val2']&list

output:
    print(outer_key: [inner_key: old_val -> type(new_val)])
'''

import shutil
from pathlib import Path
from argparse import ArgumentParser
from glob import glob
import fitsio

from superbit_lensing import utils

import ipdb

def parse_args():

    parser = ArgumentParser()

    parser.add_argument('update_tokens', type=str,
                        help='The update tokens in the format of: ' +
                        'outer_key.inner_key.new_val.val_type')

    return parser.parse_args()

def main(args):

    #-----------------------------------------------------------------
    # Initial setup

    update_tokens = args.update_tokens
    tokens = update_tokens.split('&')

    if len(tokens) != 4:
        print('Failed: The input must have the format of: ' +
              'outer_key&inner_key&new_val&val_type')
        return 1

    outer_key = tokens[0]
    inner_key = tokens[1]
    new_val = tokens[2]
    val_type = tokens[3]

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
            print('Failed: For val_type == bool, can only pass one of the ' +
                  'following: [True, False] (in any capitalization)')
            return 1

    #-----------------------------------------------------------------
    # this grabs the relevant type operator for proper type casting

    if val_type in ['list', 'tuple']:
        import ast
        casted_val = ast.literal_eval(new_val)
    else:
        casted_val = getattr(__builtins__, val_type)(new_val)

    global_config[outer_key][inner_key] = casted_val

    utils.write_yaml(global_config, global_config_file)

    print(f'{outer_key}: [{inner_key}: {old_val} -> {casted_val}]')

    return 0

if __name__ == '__main__':
    args = parse_args()
    rc = main(args)
