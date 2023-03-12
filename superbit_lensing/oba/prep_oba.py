import shutil
from pathlib import Path
from argparse import ArgumentParser
from glob import glob
import fitsio

from superbit_lensing import utils
from oba_io import IOManager, band2index

import ipdb

'''
This script is used for preparing the SuperBIT onboard analysis (OBA) on the
QCC flight computer. It expects a single string from the QCC commander that we
parse for all needed information:

1) ...

TODO: Write after talking w/ Emaad!
'''

def parse_args():

    parser = ArgumentParser()

    parser.add_argument('target_name', type=str,
                        help='Name of the target to prepare for the OBA')
    parser.add_argument('req_u_images', type=int,
                        help='Number of required u-band images for the OBA run ' +
                        '(-1 to not analyze band during OBA)')
    parser.add_argument('req_b_images', type=int,
                        help='Number of required b-band images for the OBA run ' +
                        '(-1 to not analyze band during OBA)')
    parser.add_argument('req_g_images', type=int,
                        help='Number of required g-band images for the OBA run ' +
                        '(-1 to not analyze band during OBA)')
    parser.add_argument('req_r_images', type=int,
                        help='Number of required r-band images for the OBA run ' +
                        '(-1 to not analyze band during OBA)')
    parser.add_argument('req_lum_images', type=int,
                        help='Number of required lum-band images for the OBA run ' +
                        '(-1 to not analyze band during OBA)')
    parser.add_argument('req_nir_images', type=int,
                        help='Number of required nir-band images for the OBA run ' +
                        '(-1 to not analyze band during OBA)')
    parser.add_argument('allow_unverified', type=str, choices=['True', 'False'],
                        help='Set to True to allow images not verified by the '
                        'image checker to be analyzed by the OBA. Otherwise False')

    # NOTE: Only use this if you are testing this script locally!
    parser.add_argument('-root_dir', type=str, default=None,
                        help='Root directory for OBA run (if testing locally)')

    return parser.parse_args()

def main(args):

    #-----------------------------------------------------------------
    # Initial setup

    target_name = args.target_name
    allow_unverified = args.allow_unverified
    root_dir = args.root_dir

    # annoying situation due to argparse rules + no optional args
    # for SB commander
    if allow_unverified == 'True':
        allow_unverified = True
    elif allow_unverified == 'False':
        allow_unverified = False

    req_images = {
        'u': args.req_u_images,
        'b': args.req_b_images,
        'g': args.req_g_images,
        'r': args.req_r_images,
        'lum': args.req_lum_images,
        'nir': args.req_nir_images,
    }

    # this will keep track of how many acceptable images are in RAW_DIR
    acceptable_images = {}

    io_manager = IOManager(target_name=target_name, root_dir=root_dir)

    # This keeps track of the actual bands we will use during the OBA
    # analysis. If a required number of bands is set to 0, then none
    # are required for a successful prepper run - but any images of that
    # band that *do* exist will still be analyzed. To skip the analysis
    # of any images for a given band, set the required number to -1
    bands = ['u', 'b', 'g', 'r', 'lum', 'nir']
    use_bands = bands.copy()

    #-----------------------------------------------------------------
    # For each band, collect all images in RAW_DATA and determine if
    # each is valid for OBA

    for band in bands:
        print(f'Checking band {band}')
        if req_images[band] == -1:
            # this sentinel value indicates that the band should be
            # ignored for the OBA of this target
            use_bands.remove(band)
            acceptable_images[band] = 0
            print(f'Ignoring {band} as required images set to -1')
            continue

        bindx = band2index(band)
        search = str(
            io_manager.RAW_DATA / f'{target_name}*_{bindx}_*.fits.bz2'
            )
        images = glob(search)

        print(f'Found {len(images)} to check')

        good_images = []
        for image in images:
            hdr = fitsio.read_header(image)

            try:
                image_quality = hdr['IMG_QUAL']
            except KeyError:
                # this shouldn't happen, but just in case
                raise KeyError('SuperBIT SCI images should have IMG_QUAL in ' +
                               'the FITS image header!')

            if image_quality == 'GOOD':
                good_images.append(image)
            elif image_quality == 'UNVERIFIED':
                if allow_unverified is True:
                    good_images.append(image)

        Ngood = len(good_images)
        if Ngood < req_images[band]:
            print(f'You requested {req_images[band]} images for band {band} ' +
                  f'but only {Ngood} were found using allow_unverified=' +
                  f'{allow_unverified}')
            return

        acceptable_images[band] = len(good_images)

    #-----------------------------------------------------------------
    # If the script has not ended yet, then the number of images requirements
    # have been satisfied. Create the OBA configuration file for the target

    # This has almost all configurable options already set
    base_file = Path(utils.MODULE_DIR) / 'oba/configs/oba_global_config.yaml'
    config = utils.read_yaml(base_file)

    # Now update with our target-specific configuration
    config['run_options']['target_name'] = target_name
    config['run_options']['bands'] = use_bands

    #-----------------------------------------------------------------
    # Write config to disk

    config_outdir = io_manager.OBA_TARGET
    # will make dirs recursively if needed
    utils.make_dir(config_outdir)

    config_outfile = str(
        config_outdir / f'{target_name}_oba.yaml'
        )
    utils.write_yaml(config, config_outfile)

    #-----------------------------------------------------------------
    # Now print out the config for the user to see & confirm

    print('OBA prepper completed succesfully')
    print('The following acceptable images were found for each desired band ' +
          f'using allow_unverified={allow_unverified}:')

    for band in use_bands:
        Ngood = acceptable_images[band]
        print(f'{band}: {Ngood}')

    print(f'The following OBA configuration for {target_name} has been saved ' +
          f'to: {config_outfile}')

    for outer_key in config:
        print(f'{outer_key}:')
        if isinstance(config[outer_key], dict):
            for inner_key, val in config[outer_key].items():
                print(f'  {inner_key}: {val}')
        elif isinstance(config[outer_key], list):
            for item in config[outer_key]:
                print(f'  {item}')
        else:
            print(config[outer_key])

    return

if __name__ == '__main__':
    args = parse_args()
    main(args)
