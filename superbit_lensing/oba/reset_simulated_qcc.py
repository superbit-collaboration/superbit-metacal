from argparse import ArgumentParser
from pathlib import Path
from glob import glob

from superbit_lensing import utils
from oba_io import IOManager

import ipdb

'''
NOTE: This script is only to be run for local tests of the OBA on a local
version of the QCC. It's purpose is to remove test observations of a target
from the IOManager.RAW_TARGET dir as different simulations will have different
realizations of the target, and they will clobber one another if testing on
different simulation runs. The script will ask you to confirm the deletion,
and *will not* run on an actual QCC root dir
'''

def parse_args():

    parser = ArgumentParser()

    parser.add_argument('target_name', type=str,
                       help='Name of the target whose files we will delete')
    parser.add_argument('root_dir', type=str,
                        help='Root directory for the local, test QCC')

    return parser.parse_args()

def main(args):
    target_name = args.target_name
    root_dir = args.root_dir

    if root_dir == '/':
        raise ValueError('WARNING: the passed root_dir is the actual root_dir '
                         'of your system; are you sure you are not running on '
                         'the real QCC? You definitely do *not* want to do '
                         'that!')

    io_manager = IOManager(root_dir=root_dir, target_name=target_name)

    print('')
    print('WARNING: You are about to delete all OBA images for the target '
          f'{target_name} currently registered to the simulated QCC whose '
          f'root_dir is {root_dir}. Are you sure you want to proceed?')

    proceed = input(f'\nWill only proceed with input `yes`: ')

    if proceed != 'yes':
        print('Aborting')
        return 0

    raw_target_dir = io_manager.RAW_TARGET
    print(f'Confirmed; removing {target_name} images at {raw_target_dir}...')

    # includes compressed images
    search = str(raw_target_dir / f'{target_name}_*.fits*')
    target_images = glob(search)

    for image in target_images:
        print(f'Deleting image {image}')
        image = Path(image)
        image.unlink()

    return 0

if __name__ == '__main__':
    args = parse_args()

    rc = main(args)

    if rc == 0:
        print('\nreset_simulated_qcc.py completed without error\n')
    else:
        print(f'\nreset_simulated_qcc.py failed with rc={rc}\n')
