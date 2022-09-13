import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table, vstack, hstack
import sys
import os
import glob
from argparse import ArgumentParser
import pdb


parser = ArgumentParser()

parser.add_argument('--shear_tables',type=str,default=None,
                    help = 'tables to read in: xxx_shear_profile_cat.fits')


def main():
    
    args = parser.parse_args()
    shear_tables = args.shear_tables

    if shear_tables is None:
        shear_tables = 'r*/*shear_profile_cat.fits'
    
    
    glob_tables = glob.glob(shear_tables)
    print(f'reading tables {glob_tables}')
    glob_tables.sort()

    alpha_arr = []
    sig_alpha_arr = []
    weight_arr = []

    for tabn in glob_tables:
        
        truth_name = tabn.replace('shear_profile_cat', 'truth')
        truth_cat = Table.read(truth_name, format='fits')
        nstars = len((truth_cat['obj_class'] == 'star').nonzero()[0])
        weight_arr.append(nstars)

        shear_tab = Table.read(tabn, format='fits')

        try:
            this_alpha = shear_tab.meta['ALPHA']
            this_sig_alpha = shear_tab.meta['sig_alpha']
            alpha_arr.append(this_alpha)
            sig_alpha_arr.append(this_sig_alpha)
            print(f'table {tabn} has shear bias {this_alpha:.4f} +/- {this_sig_alpha:.4f} + {nstars} stars')

        except KeyError as e:
            raise e


    print(f'\nmean alpha is {np.mean(alpha_arr):.4f} +/- {(np.std(alpha_arr)/np.sqrt(len(alpha_arr))):.4f}')

    weighted_mean = np.sum(np.array(alpha_arr)*np.array(weight_arr))/np.sum(weight_arr)
    print(f'\nweighted mean alpha = {weighted_mean}')

    print(f'\n\n\n a = {alpha_arr}')
    print(f'\n\n sig = {sig_alpha_arr}\n')
    
    return 0

if __name__ == "__main__":

    rc = main()

    if rc !=0:
        raise Exception
