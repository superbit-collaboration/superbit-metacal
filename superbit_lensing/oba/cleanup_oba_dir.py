'''
This script is for cleaning up a temproary SuperBIT OBA dir on the QCC
during flight operations. We will not always run the OBA configured to
cleanup the temporary run directories while debugging

NOTE: Makes assumptions about the location of the OBA dir for a given
target name in accordance w/ prep_oba.py
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

    io_manager = IOManager(root_dir=root_dir, target_name=target_name)

    target_dir = io_manager.OBA_TARGET

    if target_dir.is_dir():
        utils.rm_tree(target_dir)
        print(f'Removed {target_name} OBA dir {target_dir}')
    else:
        print(f'No OBA dir found for {target_name} at {target_dir}')

    return 0

if __name__ == '__main__':
    args = parse_args()
    rc = main(args)
