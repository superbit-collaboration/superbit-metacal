# Standard imports
import galsim
import numpy as np

# Modularized Galsim code related imports
from object_maker import make_cluster_galaxy

__all__ = ["inject_cluster_galaxies"]

def inject_cluster_galaxies(sci_img,
                            sbparams,
                            jitter_psf,
                            truth_catalog,
                            bandpass,
                            wcs,
                            affine,
                            cluster_cat,
                            sb_optical_psf,
                            radius,
                            logger):
    """ Inject cluster member galaxies to the science image

    Args:
        sci_img ([type]): [description]
        sbparams ([type]): [description]
        jitter_psf ([type]): [description]
        truth_catalog ([type]): [description]
        bandpass ([type]): [description]
        wcs ([type]): [description]
        affine ([type]): [description]
        cluster_cat ([type]): [description]
        sb_optical_psf ([type]): [description]
        radius ([type]): [description]
        logger ([type]): [description]

    Returns:
        sci_img: Science image with the cluster member galaxies injected.
        truth_catalog: Truth catalog for each cluster member
    """
    
    center_coords = galsim.CelestialCoord(sbparams.center_ra,
                                          sbparams.center_dec)
    centerpix = wcs.toImage(center_coords)
    
    print('')

    for i in range(sbparams.n_cluster_gal):
        ud = galsim.UniformDeviate(sbparams.cluster_seed + i + 1)

        cluster_gal_stamp, cluster_gal_truth = make_cluster_galaxy(
                                               ud=ud,
                                               wcs=wcs,
                                               jitter_psf=jitter_psf,
                                               affine=affine,
                                               centerpix=centerpix,
                                               cluster_cat=cluster_cat,
                                               sb_optical_psf=sb_optical_psf,
                                               bandpass=bandpass,
                                               sbparams=sbparams,
                                               radius=radius,
                                               logger=logger)

        # Find the overlapping pixels between the science image
        # and the background galaxy stamp created above
        stamp_overlap = cluster_gal_stamp.bounds & sci_img.bounds

        # Add the bkg galaxy stamp to the science image
        sci_img[stamp_overlap] += cluster_gal_stamp[stamp_overlap]
        cluster_gal_flux = np.sum(cluster_gal_stamp.array)

        # Create a row to append to truth catalog
        obj_type = "cluster_galaxy"
        truth_row = [i,
                     obj_type,
                     cluster_gal_truth.x,
                     cluster_gal_truth.y,
                     cluster_gal_truth.ra,
                     cluster_gal_truth.dec,
                     cluster_gal_truth.g1,
                     cluster_gal_truth.g2,
                     cluster_gal_truth.mu,
                     cluster_gal_truth.z,
                     cluster_gal_flux]

        # Append row to the truth catalog
        truth_catalog.addRow(truth_row)

        logger.info("Cluster galaxy: {} created".format(i))

    return sci_img, truth_catalog