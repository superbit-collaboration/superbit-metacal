'''
This script is used for updating the SuperBIT onboard analysis (OBA) global
or specific target config file using the QCC commander. It only updates one
config value at a time. As each config field (besides `modules`) is a 2-level
dict, we *would* use the following scheme:

python update_oba_config.py target_name outer_key inner_key new_val val_type

*HOWEVER*, the QCC commander can only pass 1 string. As all of these inputs
need to be strings (and cannot use commas), we do the following:

python update_oba_config.py "target_name&outer_key&inner_key&new_val&val_type"

where:

- target_name: Name of the config to change (can be "global" or "{target_name}")
- outer_key: Name of the outer OBA config key
- inner_key: Name of the inner OBA config key
- new_val: The new value for the OBA global config entry
- val_type: The explicit type that you want to save the new config value as
            in the config

NOTE: ".", "_", "-", etc. are not acceptable delimiters as they may be part of
the new value you are trying to construct

NOTE: to pass a list or tuple of str's, do something like the following:

python update_oba_config.py "global&outer_key&inner_key&['val1', 'val2']&list"

output:
    print({target_name}: outer_key: [inner_key: old_val -> type(new_val)])
'''

from pathlib import Path
from argparse import ArgumentParser
from glob import glob

from superbit_lensing import utils
from oba_io import IOManager

import ipdb

def parse_args():

    parser = ArgumentParser()

    parser.add_argument('update_tokens', type=str,
                        help='The update tokens in the format of: ' +
                        '"target_name&outer_key&inner_key&new_val&val_type"')

    # NOTE: not registered by the QCC, just for testing locally
    parser.add_argument('-root_dir', type=str, default=None,
                        help='Root directory for OBA run (if testing locally)')

    return parser.parse_args()

def main(args):

    #-----------------------------------------------------------------
    # Initial setup

    update_tokens = args.update_tokens
    root_dir = args.root_dir

    tokens = update_tokens.split('&')

    if len(tokens) != 5:
        print('Failed: The input must have the format of: ' +
              'target_name&outer_key&inner_key&new_val&val_type')
        return 1

    target_name = tokens[0]
    outer_key = tokens[1]
    inner_key = tokens[2]
    new_val = tokens[3]
    val_type = tokens[4]

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
        return 2

    config = utils.read_yaml(str(config_file))

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
            return 3

    #-----------------------------------------------------------------
    # this grabs the relevant type operator for proper type casting

    if val_type.lower() == 'none':
        casted_val = None
    elif val_type in ['list', 'tuple']:
        import ast
        casted_val = ast.literal_eval(new_val)
    else:
        casted_val = getattr(__builtins__, val_type)(new_val)

    # NOTE: the `modules` field works differently than the others
    if outer_key == 'modules':
        # ignore inner key
        try:
            old_val = config[outer_key]
        except KeyError:
            config[outer_key] = []
            old_val = None
        inner_key = '(None)'

        config[outer_key] = casted_val
    else:
        try:
            old_val = config[outer_key][inner_key]
        except KeyError:
            config[outer_key] = {}
            old_val = None

        # if the new val is None, delete the old entry
        if (casted_val is None) and (old_val is not None):
            del config[outer_key][inner_key]
        else:
            config[outer_key][inner_key] = casted_val

    utils.write_yaml(config, config_file)

    print(f'{target_name} config: {outer_key}: [{inner_key}: {old_val} -> {casted_val}]')

    return 0

if __name__ == '__main__':
    args = parse_args()
    rc = main(args)
