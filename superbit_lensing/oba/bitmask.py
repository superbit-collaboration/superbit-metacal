import numpy as np

'''
This file defines the bitmask used by the SuperBIT onboard analysis (OBA)
'''

OBA_BITMASK = {
    'unmasked': 0,
    'inactive_region': 1,
    'hot_pixel': 2,
    'cosmic_ray': 3,
    'satellite': 4,
    'bright_star': 5,

    # the following don't have a universal meaning in an actual image, but
    # are used to compress the source segmentation map in the CookieCutter
    'seg_obj': 6,
    'seg_neighbor': 7,
    # ...
}

# NOTE: this is the minimum number of *bits* needed to represent the bitmask
OBA_BITMASK_MIN_SIZE = int(np.ceil(np.log2(len(OBA_BITMASK))))

# NOTE: numpy doesn't like using dtypes less than 1 byte
OBA_BITMASK_DTYPE = np.dtype(f'u{OBA_BITMASK_MIN_SIZE//8 + 1}')
