'''
This script is used for updating the SuperBIT onboard analysis (OBA) global
astromatic config files (SWarp or SExtractor) using the QCC commander. It
only updates one config value at a time.

As each config field is of the format "KEY VALUE # COMMENT" (with arbitrary
number of spaces), we *would* use the following scheme:

python update_oba_astromatic_config.py config_type key new_val

*HOWEVER*, the QCC commander can only pass 1 string. As all of these inputs
need to be strings (and we cannot use commas), we do the following:

python update_oba_astromatic_config.py "config_type&config_file&key&new_val"

where:

- config_type: Name of the config type (also dir name:
               {MODULE_DIR}/oba/configs/{config_type}/ )
- config_file: Name of the astromatic config file *in* config_type dir
- key: Name of the astromatic config key whose value you want to change
- new_val: The new value for the config entry

NOTE: ".", "_", "-", etc. are not acceptable delimiters as they may be part of
the new value you are trying to construct

NOTE: The input str format is made to be consistent with "update_oba_config.py"

output:
    print(config_file: key: old_val -> new_val)
'''

import os
from pathlib import Path
from argparse import ArgumentParser
from glob import glob

from superbit_lensing import utils

import ipdb

def parse_args():

    parser = ArgumentParser()

    parser.add_argument('update_tokens', type=str,
                        help='The update tokens in the format of: ' +
                        '"config_type&config_file&key&new_val"')

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

    if len(tokens) != 4:
        print('Failed: The input must have the format of: ' +
              '"config_type&config_file&key&new_val"')
        return 1

    config_type = tokens[0]
    config_file = tokens[1]
    key = tokens[2]
    new_val = tokens[3]

    #-----------------------------------------------------------------
    # Initial input parsing

    # NOTE: The astromatic configs for OBA only live in one place:
    config_dir = Path(utils.MODULE_DIR) / 'oba/configs/'

    astromatic_config_types = ['swarp', 'sextractor']
    if config_type not in astromatic_config_types:
        print('Failed: the only accepted astromatic config types are: '+
              f'{astromatic_config_types}')
        return 1

    config_dir = config_dir / config_type
    config_file = config_dir / config_file

    if not config_file.is_file():
        print(f'Failed: {config_file.name} not found in {config_dir}')
        return 2

    #-----------------------------------------------------------------
    # Load in existing config and edit the requested line

    old_val = None
    found_key = False

    new_lines = []
    with open(config_file) as f:
        for line in f:
            # gets rid of excess spaces
            components = line.split()

            # just in case
            if len(components) == 0:
                continue

            # Will ignore comments
            if components[0] == key:
                old_val = components[1]
                components[1] = str(new_val)
                new_line = ' '.join(components)
                found_key = True
            else:
                new_line = line

            new_lines.append(new_line)

    if found_key is False:
        print(f'Failed: Did not find the key {key} in config {config_file.name}')
        return 3

    #-----------------------------------------------------------------
    # Write new config & replace old one

    tmp_file = config_dir / f'tmp_{config_file.name}'

    with open(str(tmp_file), 'x') as tmp:
        tmp.writelines(new_lines)

    config_file.unlink()

    tmp_file.rename(config_file)

    print(f'{config_file}: {key}: {old_val} -> {new_val}')

    return 0

if __name__ == '__main__':
    args = parse_args()
    rc = main(args)


