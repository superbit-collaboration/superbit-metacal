'''
This script is used for preparing the SuperBIT onboard analysis (OBA) on the
QCC flight computer. It expects a target name and a series of integers for
each SuperBIT band that define the following:

1) How many images are required per-band for this target to run the OBA
   (can be zero; will still analyze those bands & send down cutouts)
2) Whether to ignore a given band in the OBA (by setting to -1)

In addition, the final argument `allow_unverified` is used to allow for some
flexibility in what images can be accepted for the OBA. The fiducial plan is
for all raw SCI images to be examined by a image checker that will set the
header IMG_QUAL to one of ['GOOD', 'BAD', 'UNVERIFIED']. In the case that the
image checker does not work or is not run on some images, you can choose to
allow unverified images in the analysis
'''

import shutil
from pathlib import Path
from argparse import ArgumentParser
from glob import glob
import fitsio

from superbit_lensing import utils
from oba_io import IOManager, band2index

import ipdb

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
    parser.add_argument('req_nir_images', type=int,
                        help='Number of required nir-band images for the OBA run ' +
                        '(-1 to not analyze band during OBA)')
    parser.add_argument('req_lum_images', type=int,
                        help='Number of required lum-band images for the OBA run ' +
                        '(-1 to not analyze band during OBA)')

    # NOTE: We define the bool arg this way to make it work cleanly with the QCC commander
    parser.add_argument('allow_unverified', type=int, choices=[0, 1],
                        help='Set to 1 (True) to allow images not verified by the '
                        'image checker to be analyzed by the OBA. Otherwise 0 (False)')

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
    allow_unverified = bool(allow_unverified)

    req_images = {
        'u': args.req_u_images,
        'b': args.req_b_images,
        'g': args.req_g_images,
        'r': args.req_r_images,
        'nir': args.req_nir_images,
        'lum': args.req_lum_images,
    }

    # this will keep track of how many acceptable images are in RAW_DIR
    acceptable_images = {}

    io_manager = IOManager(target_name=target_name, root_dir=root_dir)

    # This keeps track of the actual bands we will use during the OBA
    # analysis. If a required number of bands is set to 0, then none
    # are required for a successful prepper run - but any images of that
    # band that *do* exist will still be analyzed. To skip the analysis
    # of any images for a given band, set the required number to -1
    bands = ['u', 'b', 'g', 'r', 'nir', 'lum']
    use_bands = bands.copy()

    #-----------------------------------------------------------------
    # For each band, collect all images in RAW_DATA and determine if
    # each is valid for OBA

    failed = False
    for band in bands:
        if req_images[band] == -1:
            # this sentinel value indicates that the band should be
            # ignored for the OBA of this target
            use_bands.remove(band)
            acceptable_images[band] = 0
            continue

        bindx = band2index(band)
        search = str(
            io_manager.RAW_DATA / f'{target_name}*_{bindx}_*.fits.bz2'
            )
        images = glob(search)

        good_images = []
        for image in images:
            hdr = fitsio.read_header(image)

            try:
                image_quality = hdr['IMG_QUAL']
            except KeyError:
                # this shouldn't happen, but just in case
                print('SuperBIT SCI images should have IMG_QUAL in ' +
                      'the FITS image header!')
                return 1

            if image_quality == 'GOOD':
                good_images.append(image)
            elif image_quality == 'UNVERIFIED':
                if allow_unverified is True:
                    good_images.append(image)

        Ngood = len(good_images)
        acceptable_images[band] = len(good_images)

        if Ngood < req_images[band]:
            failed = True

    #-----------------------------------------------------------------
    # Check if the number of acceptables images satisfies requirements

    base_msg = 'Acceptable images:'
    for band in bands:
        if band in acceptable_images:
            Nimages = acceptable_images[band]
        else:
            Nimages = -1
        base_msg += f' {Nimages}'

    if failed is True:
        print(f'Failed: {base_msg}')
        return 2
    else:
        print(base_msg)

    #-----------------------------------------------------------------
    # If the script has not ended yet, then the number of images requirements
    # have been satisfied. Create the OBA configuration file for the target

    # This has almost all configurable options already set
    base_file = Path(utils.MODULE_DIR) / 'oba/configs/oba_global_config.yaml'
    config = utils.read_yaml(base_file)

    # Now update with our target-specific configuration
    config['run_options']['target_name'] = target_name
    config['run_options']['bands'] = use_bands

    if allow_unverified is True:
        config['run_options']['min_image_quality'] = 'unverified'
    else:
        config['run_options']['min_image_quality'] = 'good'

    #-----------------------------------------------------------------
    # Do any sanity checks needed to error *before* running the OBA
    # to save headaches later on

    det_bands = config['run_options']['det_bands']

    for band in det_bands:
        Ngood = acceptable_images[band]
        if Ngood == 0:
            print(f'Failed: det_bands={det_bands} but there are 0 acceptable ' +
                  f'images for {band}')
            return 3

    #-----------------------------------------------------------------
    # Write config to disk

    config_outdir = io_manager.OBA_TARGET
    # will make dirs recursively if needed
    utils.make_dir(config_outdir)

    config_outfile = str(
        config_outdir / f'{target_name}_oba.yaml'
        )
    utils.write_yaml(config, config_outfile)

    print(f'Config at {config_outfile}')

    return 0

if __name__ == '__main__':
    args = parse_args()
    main(args)
