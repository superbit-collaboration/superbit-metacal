import numpy as np

'''
This file defines the bitmask used by the SuperBIT onboard analysis (OBA)
'''

OBA_BITMASK = {
    'unmasked': 0,
    'inactive_region': 1,
    'hot_pixel': 2,
    'saturated': 4,
    'cosmic_ray': 8,
    'satellite': 16,
    'bright_star': 32,

    # the following don't have a universal meaning in an actual image, but
    # are used to compress the source segmentation map in the CookieCutter
    'seg_obj': 64,
    'seg_neighbor': 128,
    # ...
}

# NOTE: this is the minimum number of *bits* needed to represent the bitmask
OBA_BITMASK_MIN_SIZE = len(OBA_BITMASK) - 1

# NOTE: numpy doesn't like using dtypes less than 1 byte
OBA_BITMASK_DTYPE = np.dtype(f'u{int(np.ceil(OBA_BITMASK_MIN_SIZE / 8))}')
