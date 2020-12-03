# Standard imports
import sys
import os
import logging
import glob
from astropy.table import Table
import galsim
import galsim.des
import numpy as np

# Modularized Galsim code related imports
import sbparameters 
import optical_psf
from tools import file_naming
from tools import truth
from tools import wcs_tools
from object_injector import inject_bkg_galaxies
from object_injector import inject_cluster_galaxies
from object_injector import inject_stars

def main(argv):
    """
    Make SuperBIT simulations by injecting background galaxies,
    cluster galaxies, and stars using:
    1) Estimate of the PSF of SuperBIT with an optical component
       (estimated using Zemax) and a pointing jitter component estimated 
       from the 2019 flight data.
    2) Noise model estimated using the 2019 flight data
    """

    # For logging
    global logger
    logging.basicConfig(format="%(message)s", 
                        level=logging.INFO, 
                        stream=sys.stdout)
    logger = logging.getLogger("mock_superbit_data")
    
    # Extract SuperBIT parameters defined in the config file
    sbparams = sbparameters.SuperBITParameters(argv=argv)

    # Set up the NFWHalo of the galaxy cluster to be simulated
    nfw_halo = galsim.NFWHalo(mass=sbparams.mass, 
                              conc=sbparams.nfw_conc, 
                              redshift=sbparams.nfw_z_cluster,
                              omega_m=sbparams.omega_m, 
                              omega_lam=sbparams.omega_lam) 
                    
    # Read in galaxy catalog, as well as catalog containing
    # information from COSMOS fits like redshifts, hlr, etc.   
    cosmos_cat = galsim.COSMOSCatalog(sbparams.cat_file_name, 
                                      dir=sbparams.cosmosdir)
    cluster_cat = galsim.COSMOSCatalog(sbparams.cluster_cat_name)

    fits_cat_data = Table.read(os.path.join(sbparams.cosmosdir, 
                                     sbparams.fit_file_name))
    
    # Load optical PSF
    sb_optical_psf = optical_psf.load_optical_psf(sbparams, band="lum")

    # Load SuperBIT LUM bandpass
    bandpass = galsim.Bandpass(sbparams.bp_file, 
                               wave_type='nm', 
                               blue_limit=310, 
                               red_limit=1100)
    
    # Make SuperBIT simulated images for n different exposures 
    # for m different psf (optical + jitter psf) models. The jitter PSF 
    # currenly used is the 121 s kernel. Why the 121 s kernel instead of other
    # exposure times?

    all_jitter_psfs = glob.glob(sbparams.jitter_psf_path + "*121*.psf")

    for jitter_psf_file in all_jitter_psfs:
        for i in np.arange(1, sbparams.n_exp + 1):
            logger.info('\nBeginning exposure: {}'.format(i))

            rng = galsim.BaseDeviate(sbparams.noise_seed+i)
            
            # Construct FITS file name for the simulated image
            fits_file_name = file_naming.fits_filename(
                                           sbparams=sbparams, 
                                           band='lum',
                                           exposure_index=i)

            # Construct Truth catalog file name
            truth_file_name = file_naming.truth_filename(
                                           sbparams=sbparams,
                                           band='lum',
                                           exposure_index=i)
            
            # Set up a truth catalog
            truth_catalog = truth.truth_catalog_setup()

            # Set up the science image on the hypothetical CCD
            sci_img = galsim.ImageF(sbparams.image_xsize, 
                                    sbparams.image_ysize)
            sci_img.fill(0)
            sci_img.setOrigin(0, 0)
            
            # Sky background noise level in ADU
            sky_level = sbparams.exp_time * sbparams.sky_bkg

            # If you wanted to make a non-trivial WCS system, 
            # could set theta to a non-zero number
            theta = 0. # in degrees
            affine, wcs = wcs_tools.get_affine_wcs(
                                            theta=theta,
                                            sbparams=sbparams,
                                            image_center=sci_img.true_center)
            
            sci_img.wcs = wcs

            # Now, read in the PSFEx jitter PSF model (.psf file). The .psf 
            # file is read directly into an Interpolated GSObject, so can be 
            # manipulated if needed. The DES class is used to read the 
            # .psf file.
            jitter_psf = galsim.des.DES_PSFEx(
                                    file_name=jitter_psf_file,
                                    wcs=wcs)
            
            # For a given jitter + optical psf model, simulate and 
            # inject background galaxy stamps into the science image.
            sci_img_bkg, truth_catalog = inject_bkg_galaxies(
                                                sci_img=sci_img,
                                                sbparams=sbparams,
                                                wcs=wcs,
                                                jitter_psf=jitter_psf,
                                                affine=affine,
                                                fits_cat_data=fits_cat_data,
                                                cosmos_cat=cosmos_cat,
                                                nfw_halo=nfw_halo,
                                                sb_optical_psf=sb_optical_psf,
                                                bandpass=bandpass,
                                                truth_catalog=truth_catalog,
                                                logger=logger)

            # For a given jitter + optical psf model, simulate and 
            # inject cluster galaxy stamps into the science image.
            sci_img_clusters, truth_catalog = inject_cluster_galaxies(
                                                sci_img=sci_img,
                                                sbparams=sbparams,
                                                jitter_psf=jitter_psf,
                                                truth_catalog=truth_catalog,
                                                bandpass=bandpass,
                                                wcs=wcs,
                                                affine=affine,
                                                cluster_cat=cluster_cat,
                                                sb_optical_psf=sb_optical_psf,
                                                radius=2000,
                                                logger=logger)

            # For a given jitter + optical psf model, simulate and 
            # inject star stamps into the science image.
            sci_img_stars, truth_catalog = inject_stars(
                                                sci_img=sci_img,
                                                sbparams=sbparams,
                                                wcs=wcs,
                                                jitter_psf=jitter_psf,
                                                truth_catalog=truth_catalog,
                                                sb_optical_psf=sb_optical_psf,
                                                affine=affine,
                                                logger=logger)

            # Add the bkg, cluster galaxies, and stars to one image
            sci_img = sci_img_stars + sci_img_bkg + sci_img_clusters
    
            # Add dark current to the science image
            dark_noise = sbparams.dark_current * sbparams.exp_time
            sci_img += dark_noise

            # Add sky and read noise to the science image
            sky_read_noise = galsim.CCDNoise(rng=rng,
                                             sky_level=sky_level,
                                             gain=1/sbparams.gain,
                                             read_noise=sbparams.read_noise)
            sci_img.addNoise(sky_read_noise)
            
            if not os.path.exists(os.path.dirname(fits_file_name)):
                os.makedirs(os.path.dirname(fits_file_name))
            sci_img.write(fits_file_name)

            # Write truth catalog to file. 
            truth_catalog.write(truth_file_name)
            logger.info('\nFITS image written to: {}'.format(fits_file_name))
            logger.info('Truth catalog written to: {}'.format(truth_file_name))
            logger.info('\nProgram complete for exposure: {}'.format(i)) 
    
if __name__ == "__main__":
    main(sys.argv)