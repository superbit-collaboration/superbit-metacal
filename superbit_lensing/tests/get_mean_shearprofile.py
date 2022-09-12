from astropy.table import vstack, hstack, Table
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import glob
from esutil import htm
import sys
import os
import astropy
import pdb, ipdb
import collections
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument('-shear_cats',type=str,default=None,
                    help = 'tables to read in: xxx_shear_profile_cat.fits')
parser.add_argument('-annular_cats',type=str,default=None,
                    help = 'tables to read in: xxx_annular.fits')
parser.add_argument('-stackcat_name',type=str,default=None,
                    help = 'name of ouput stacked catalog')

class CatalogStacker():

    def __init__(self, cat_list=None):

        self.cat_list = cat_list
        self.alpha_list = []
        self.stacked_cat = None
        self.mean_a = None
        self.std_a = None
        self.avg_nobj = None
        self.num_alphas = None

    def _get_avg_nobj(self):
        '''
        Returns the average length of catalogs in cat_list.
        If cat_list is galaxy catalogs,  it's an average source density
        if cat_list is shear profile catalogs, it's the average number of radial bins
        '''

        len_stacked = len(self.stacked_cat)
        len_catlist = len(self.cat_list)
        avg_cat_len = len_stacked/len_catlist

        self.avg_nobj = avg_cat_len

        return


    def _get_alpha_stats(self):

        num_alphas  = len(self.alpha_list)
        mean_alpha = np.mean(self.alpha_list)
        std_alpha = np.std(self.alpha_list)/np.sqrt(num_alphas)

        self.mean_a = mean_alpha
        self.std_a = std_alpha
        self.num_alphas = num_alphas

        return


    def get_catalogs(self, catnames=None):
        '''
        Either stack the supplied catnames, or
        stack the catalogs already in self
        '''

        if catnames == None:
            catnames = self.cat_list

        assert isinstance(catnames, list), f"catnames should be a list, got {type(catnames)} instead"

        holding = {}

        for i in np.arange(len(catnames)):
            tab=Table.read(catnames[i],format='fits')
            holding["tab{0}".format(i)] = tab

            # Skip the alpha list if it doesn't exist
            try:
                self.alpha_list.append(tab.meta['ALPHA'])
            except KeyError:
                pass

        stacked_catalog = vstack([holding[val] for val in holding.keys()], metadata_conflicts='silent')
        stacked_catalog.meta = collections.OrderedDict()

        self.stacked_cat = stacked_catalog

        return


    def _get_cat_stats(self):

        self._get_avg_nobj()

        if (len(self.alpha_list) > 1):
            self._get_alpha_stats()

        return


    def run(self):

        # Concatenate all the catalogs
        self.get_catalogs()

        # Get summary statistics
        self._get_cat_stats()

        return


def main():

    args = parser.parse_args()
    shearcat_names = args.shear_cats
    annular_names = args.annular_cats
    stackcat_name = args.stackcat_name

    if shearcat_names is None:
        shearcat_names = 'r*/*_shear_profile_cat.fits'
    if annular_names is None:
        annular_names = 'r*/*annular.fits'
    if stackcat_name is None:
        stackcat_name = 'stacked_shear_profile_cats.fits'

    shearcat_list = glob.glob(shearcat_names)
    annular_list = glob.glob(annular_names)

    # Get source density
    all_annulars = CatalogStacker(annular_list)
    all_annulars.run()

    avg_n_sources = np.ceil(all_annulars.avg_nobj)

    print("")
    print(f"Avg number of galaxies in catalog is {avg_n_sources}")
    print("")

    # Get shear cats and also average alpha
    all_shears = CatalogStacker(shearcat_list)
    all_shears.run()

    stacked_shear = all_shears.stacked_cat

    stacked_shear.sort('midpoint_r')
    stacked_shear.write(stackcat_name,format='fits',overwrite=True)

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

    print(f'mean alpha = {all_shears.mean_a:.5f} +/- {all_shears.std_a/np.sqrt(all_shears.num_alphas):.4f}')
    print(f'std alpha = {all_shears.std_a:.5f}')

    table.meta['avg_alpha'] = all_shears.mean_a
    table.meta['std_alpha'] = all_shears.std_a
    table.meta['num_alphas'] = all_shears.num_alphas
    table.meta['mean_n_gals'] = avg_n_sources

    mean_shear_name = stackcat_name.replace('stacked','mean')
    table.write(mean_shear_name, format='fits', overwrite=True)


if __name__ == "__main__":

    main()
