# Standard imports
import galsim
import numpy as np

# Modularized Galsim code related imports
from object_maker import make_bkg_galaxy

__all__ = ["inject_bkg_galaxies"]

def inject_bkg_galaxies(sci_img,
                        sbparams, 
                        wcs,
                        jitter_psf,
                        affine,
                        fits_cat_data,
                        cosmos_cat,
                        nfw_halo,
                        sb_optical_psf,
                        bandpass,
                        truth_catalog,
                        logger):
    """ Injects background galaxies in the science image

    Args:
        sci_img ([type]): [description]
        sbparams ([type]): [description]
        wcs ([type]): [description]
        jitter_psf ([type]): [description]
        affine ([type]): [description]
        fits_cat_data ([type]): [description]
        cosmos_cat ([type]): [description]
        nfw_halo ([type]): [description]
        sb_optical_psf ([type]): [description]
        bandpass ([type]): [description]
        truth_catalog ([type]): [description]
        logger ([type]): [description]

    Returns:
        sci_img: Science image with the injected background galaxies
        truth_catalog: Truth catalog for the injected background galaxies.
    """
    
    print('')

    for i in range(sbparams.n_bkg_gal):
         # The usual random number generator using a 
         # different seed for each galaxy.
        ud = galsim.UniformDeviate(sbparams.galobj_seed+i+1)

        bkg_gal_stamp, bkg_gal_truth =  make_bkg_galaxy(
                                        ud=ud,
                                        wcs=wcs,
                                        jitter_psf=jitter_psf,
                                        affine=affine,
                                        fits_cat_data=fits_cat_data,
                                        cosmos_cat=cosmos_cat,
                                        nfw_halo=nfw_halo,
                                        sb_optical_psf=sb_optical_psf,
                                        bandpass=bandpass,
                                        sbparams=sbparams,
                                        logger=logger)

        # Find the overlapping pixels between the science image
        # and the background galaxy stamp created above
        stamp_overlap = bkg_gal_stamp.bounds & sci_img.bounds

        # Add the bkg galaxy stamp to the science image
        sci_img[stamp_overlap] += bkg_gal_stamp[stamp_overlap]
        bkg_gal_flux = np.sum(bkg_gal_stamp.array)

        # Create a row to append to truth catalog
        obj_type = "bkg_galaxy"
        truth_row = [i,
                     obj_type,
                     bkg_gal_truth.x,
                     bkg_gal_truth.y,
                     bkg_gal_truth.ra,
                     bkg_gal_truth.dec,
                     bkg_gal_truth.g1,
                     bkg_gal_truth.g2,
                     bkg_gal_truth.mu,
                     bkg_gal_truth.z,
                     bkg_gal_flux]

        # Append row to the truth catalog
        truth_catalog.addRow(truth_row)
        logger.info("Background galaxy: {} created".format(i))
    
    return sci_img, truth_catalog
