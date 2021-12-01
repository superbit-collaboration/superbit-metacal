import sys
import getopt
import re
import os
import math
import numpy as np
from numpy import r_, c_
from numpy import linalg as la
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
#import nfwtools as nt
from astropy.io import fits
from astropy.table import Table


class Annular():


    def __init__(self,cat_info = None, annular_info = None):

        """
        :incat:   table that will be read in
        :outcat:  name of output shear profile catalog
        :g1/2:    reduced-shear components
        :gcross:  cross-shear (B-mode)
        :gtan:    tangential shear (E-mode)
        :x/y:     x and y positions
        :r:       distance of galaxy to NFW center
        :nbins:   number of radial bins for averaging (default = 5)
        :start/endrad: region to consider

        """

        self.cat_info = cat_info
        self.annular_info = annular_info
        self.g1 = None
        self.g2 = None
        self.gcross = None
        self.gtan = None
        self.x = None
        self.y = None
        self.r = None
        self.mu = None

        """
        self.xc = annular_info['nfw_center'][0]
        self.yc = annular_info['nfw_center'][1]
        self.startrad = annular_info['shear_args'][0]
        self.endrad = annular_info['shear_args'][1]
        self.nbins = annular_info['nbins']
        """


    def open_table(self,cat_info):

        try:
            tab = Table.read(cat_info['incat'],format='ascii')
        except:
            tab = Table.read(cat_info['incat'],format='csv')

        try:
            self.x = tab[cat_info['xy_args'][0]]
            self.y = tab[cat_info['xy_args'][1]]
            self.g1 = tab[cat_info['shear_args'][0]]
            self.g2 = tab[cat_info['shear_args'][1]]
            #self.mu = tab['nfw_mu']
        except:
            print("Could not load xy/g1g2 columns; check supplied column names?")
            pdb.set_trace()

        return

    def get_r_gtan(self,write=False):
        """
        Populates self.r with radial distance of galaxies from (xc,yc)
        and self.gtan/gcross with tangential and cross ellipticities

        Failed shape measurements with g1/g2 = -999999 get filtered out
        """

        xc = self.annular_info['nfw_center'][0]
        yc = self.annular_info['nfw_center'][1]

        #wg=(self.mu>1)
        g = np.sqrt(self.g1**2 + self.g2**2)
        std_g = np.std(g)
        wg = (g>0)
        print("## %d galaxies actually used for calculation" % len(wg.nonzero()[0]))

        g1=self.g1[wg] #*np.sqrt(2)
        g2=self.g2[wg] #*np.sqrt(2)
        x=self.x[wg]
        y=self.y[wg]

        self.r=np.sqrt( ((x-xc)**2.0) + ((y-yc)**2))

        phi = np.arctan2((y-yc),(x-xc))
        print("## Mean g: %f sigma_g: %f" % (np.mean(g),np.std(g)))
        self.gtan= -1.0*(g1*np.cos(2.0*phi) + g2*np.sin(2.0*phi))
        self.gcross = g1*np.sin(2.0*phi) - g2*np.cos(2.0*phi) # note that annular.c has opposite sign convention

        """
        self.gtan = self.gtan%/(2*np.std(g))
        self.gcross = self.gcross
        """
        if write==True:
            newtab=Table()
            newtab.add_columns([x,y,self.r,self.gtan,self.gcross],names=['x','y','r','gcross','gtan'])
            newtab.write('annular_inputs.csv',format='csv',overwrite=True)
        return

    def do_annular(self):

        """
        workhorse; this will contain the actual azimuthal binning
        """
        minrad = self.annular_info['rad_args'][0]
        maxrad = self.annular_info['rad_args'][1]
        num_bins = self.annular_info['nbins']

        number,bins = np.histogram(self.r,bins=num_bins,range=(minrad,maxrad))


        print("##\n r       n        gtan      err_gtan     gcross     err_gcross")


        for i in range(len(bins)-1):
            annulus = (self.r>=bins[i]) & (self.r<bins[i+1])
            midpoint_r = np.mean([bins[i],bins[i+1]])
            n = number[i]
            gtan_mean = np.mean(self.gtan[annulus])
            gcross_mean = np.mean(self.gcross[annulus])
            gtan_err = np.std(self.gtan[annulus])/np.sqrt(n)
            gcross_err = np.std(self.gcross[annulus])/np.sqrt(n)

            print("%.3f   %d   %f   %f   %f   %f" % (midpoint_r,n,gtan_mean,gtan_err,gcross_mean,gcross_err))

        return


    def run(self,write=False):

        self.open_table(self.cat_info)
        self.get_r_gtan(write=write)
        self.do_annular()

        return


def print_header(args):

    print("###\n### Shear profile calculation\n###")
    print("### created with:")
    print("### %s\n###" % (' '.join(args)))
    print("## r: midpoint radius of annulus")
    print("## n: number of objects for shear calculation")
    print("## gtan: tangential (E-mode) reduced-shear")
    print("## err_gtan: standard error of gtan")
    print("## gcross: cross (B-mode) reduced-shear")
    print("## err_gcross: standard error of gcross")

    return

def main(args):

    """
    TO DO: clean up input arguments, using -o --option format
    and make startrad/endrad options as well, rather than magic numbers

    """

    if ( ( len(args) < 3) or (args == '-h') or (args == '--help') ):
        print("\n### \n### annular_jmac is a routine which takes an ascii table and x/y/g1/g2 column names as its input and returns shear profiles")
        print("### Note that at present, annular_jmac assumes that the shear arguments [g1, g2] are *reduced shear*, not image ellipticities\n###\n")
        print(" python annular_jmac.py incat g1_arg g2_arg start_rad end_rad n_bins\n \n")

    else:
        pass

    # Define catalog args
    incat = args[1]
    g1_arg = args[2]
    g2_arg = args[3]
    startrad = float(args[4])
    endrad = float(args[5])
    num_bins = int(args[6])

    # Define annular args
    x_arg = 'X_IMAGE'
    y_arg = 'Y_IMAGE'
    #startrad = 180
    #endrad = 4000
    nfw_center = [5031,3353]
    #nfw_center = [3333,2227]
    #nbins =16

    print_header(args)

    cat_info={'incat':incat, 'xy_args':[x_arg,y_arg], 'shear_args':[g1_arg,g2_arg]}
    annular_info={'rad_args': [startrad, endrad],'nfw_center': nfw_center, 'nbins':num_bins}

    annular = Annular(cat_info,annular_info)
    annular.run(write=True)

    #pdb.set_trace()
    #print("annular successfully completed, enjoy your shear profile <3")


if __name__ == '__main__':

    """
    import pdb, traceback, sys
    try:
        main(sys.argv)
    except:
        thingtype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
    """
    main(sys.argv)
