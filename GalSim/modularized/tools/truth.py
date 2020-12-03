# Standard imports
import os
import galsim

__all__ = ["true_attributes", "truth_catalog_setup"]

class true_attributes:
    def __init__(self):
        '''
        class to store attributes of a mock galaxy or star
        :x/y: object position in full image
        :ra/dec: object position in WCS --> may need revision?
        :g1/g2: NFW shear moments
        :mu: NFW magnification
        :z: galaxy redshift
        '''
        self.x = None 
        self.y = None
        self.ra = None
        self.dec = None
        self.g1 = 0.0
        self.g2 = 0.0
        self.mu = 1.0
        self.z = 0.0

def truth_catalog_setup():
    """ Sets up the truth catalog.

    Returns:
        Galsim output catalog.
    """
    names = ['gal_num',
             'obj_type',   
             'x_image',
             'y_image',
             'ra',
             'dec',
             'g1_meas',
             'g2_meas',
             'nfw_mu',
             'redshift',
             'flux']             
    types = [int,
             str,   
             float,
             float,
             float,
             float,
             float,
             float,
             float,
             float,
             float]
              
    return galsim.OutputCatalog(names, types)

