# Standard imports
import galsim
import numpy as np

# Modularized Galsim code related imports
from object_maker import make_star

__all__ = ["inject_stars"]

def inject_stars(sci_img,
                 sbparams,
                 wcs,
                 jitter_psf,
                 truth_catalog,
                 sb_optical_psf,
                 affine,
                 logger):
    """Injects stars into the science image

    Args:
        sci_img ([type]): science image
        sbparams ([type]): SuperBIT parameters
        wcs ([type]): World Coordinate System
        jitter_psf ([type]): estimate of the jitter psf
        truth_catalog ([type]): Truth catalog to store the "true" information
                                of each injected object
        sb_optical_psf ([type]): Estimate of the SuperBIT optical PSF.
                                 Currently only valid for the LUM band.
        affine ([type]): affine
        logger ([type]): For logging.

    Returns:
        sci_img: Science image with the injected stars.
        truth_catalog: Truth catalog with the information from the injected
                       stars
    """
    print('')
    
    for i in range(sbparams.n_stars):
        ud = galsim.UniformDeviate(sbparams.stars_seed+i+1)

        star_stamp, star_truth = make_star(
                                    ud=ud,
                                    wcs=wcs,
                                    jitter_psf=jitter_psf,
                                    affine=affine,
                                    sb_optical_psf=sb_optical_psf,
                                    sbparams=sbparams)

        # Find the overlapping pixels between the science image
        # and the star stamp created above
        stamp_overlap = star_stamp.bounds & sci_img.bounds
        
        # Add the star stamp to the science image
        sci_img[stamp_overlap] += star_stamp[stamp_overlap]
        star_flux = np.sum(star_stamp.array)

        # Create a row to append to truth catalog
        obj_type = "star"
        truth_row = [i,
                     obj_type,
                     star_truth.x,
                     star_truth.y,
                     star_truth.ra,
                     star_truth.dec,
                     star_truth.g1,
                     star_truth.g2,
                     star_truth.mu,
                     star_truth.z,
                     star_flux]

        # Append row to the truth catalog
        truth_catalog.addRow(truth_row)
        logger.info("Star: {} created".format(i))

    return sci_img, truth_catalog