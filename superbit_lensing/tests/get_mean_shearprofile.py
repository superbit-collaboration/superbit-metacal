from astropy.table import vstack, hstack, Table
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
from astropy.io import fits
import glob
from esutil import htm
import sys
import os
import astropy
import pdb

def get_catalogs(catnames):

    holding={}
    try:
        for i in np.arange(len(catnames)):
            tab=Table.read(catnames[i],format='fits',hdu=1)
            holding["tab{0}".format(i)] = tab
    except:
         for i in np.arange(len(catnames)):
            tab=Table.read(catnames[i],format='ascii')
            holding["tab{0}".format(i)] = tab

    all_catalogs=vstack([holding[val] for val in holding.keys()])

    return all_catalogs


def main():

    shearcat_names = 'r*/*_shear_profile_cat.fits'
    all_shearcats = glob.glob(shearcat_names)

    stacked_shear = get_catalogs(all_shearcats)
    
    stacked_shear.sort('midpoint_r')

    # Should probs update sigma/mean_sigma btw
    stacked_shear.write('./stacked_shear_profile_cats.fits',format='fits',overwrite=True)

    radii = np.unique(stacked_shear['midpoint_r'])

    # There must be a more elegant way to do this
    N = len(radii)
    counts = np.zeros(N)
    midpoint_r = np.zeros(N)
    gtan_mean = np.zeros(N)
    gtan_err = np.zeros(N)
    gcross_mean = np.zeros(N)
    gcross_err = np.zeros(N)

    nfw_mid_r = np.zeros(N)
    nfw_gtan_mean = np.zeros(N)
    nfw_gtan_err = np.zeros(N)
    nfw_gcross_mean = np.zeros(N)
    nfw_gcross_err = np.zeros(N)


    for i,radius in enumerate(radii):


        annulus = stacked_shear['midpoint_r'] == radius
        n = len(stacked_shear[annulus])

        midpoint_r[i] = radius
        counts[i] = np.mean(stacked_shear['counts'][annulus])
        gtan_mean[i] = np.mean(stacked_shear['mean_gtan'][annulus])
        gcross_mean[i] = np.mean(stacked_shear['mean_gcross'][annulus])

        gtan_err[i] = np.std(stacked_shear['mean_gtan'][annulus])
        gcross_err[i] = np.std(stacked_shear['mean_gcross'][annulus])

        nfw_gtan_mean[i] = np.mean(stacked_shear['mean_nfw_gtan'][annulus])
        nfw_gcross_mean[i] = np.mean(stacked_shear['mean_nfw_gcross'][annulus])

        nfw_gtan_err[i] = np.std(stacked_shear['mean_nfw_gtan'][annulus])
        nfw_gcross_err[i] = np.std(stacked_shear['mean_nfw_gcross'][annulus])



    table = Table()
    table.add_columns([counts, midpoint_r, gtan_mean, gcross_mean, gtan_err, gcross_err],
                          names=['counts', 'midpoint_r', 'mean_gtan', 'mean_gcross', 'err_gtan', 'err_gcross'])

    table.add_columns([nfw_gtan_mean, nfw_gcross_mean, nfw_gtan_err, nfw_gcross_err],
                          names=['mean_nfw_gtan', 'mean_nfw_gcross', 'err_nfw_gtan', 'err_nfw_gcross'])


    table.write('stacked_shear_values.fits', format='fits', overwrite=True)

if __name__ == "__main__":

    main()
