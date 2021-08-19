# Copyright (c) 2012-2019 by the GalSim developers team on GitHub
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
# https://github.com/GalSim-developers/GalSim
#
# GalSim is free software: redistribution and use in source and binary forms,
# with or without modification, are permitted provided that the following
# conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions, and the disclaimer given in the accompanying LICENSE
#    file.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions, and the disclaimer given in the documentation
#    and/or other materials provided with the distribution.
#

import sys
import os
import math
import numpy
import logging
import time
import galsim
import galsim.des
import pdb
import glob
import scipy
import astropy

class truth():
    
    def __init__(self):
        '''
        class to store attributes of a mock galaxy or star
        :x/y: object position in full image
        :ra/dec: object position in WCS --> may need revision?
        :g1/g2: NFW shear moments
        :mu: NFW magnification
        :z: galaxy redshift
        :variance: of stamp pixel noise
        '''
    
        self.x = None 
        self.y = None
        self.ra = None
        self.dec = None
        self.g1 = 0.0
        self.g2 = 0.0
        self.mu = 1.0
        self.z = 0.0
        self.variance=0.0

def nfw_lensing(nfw_halo, pos, nfw_z_source):
    """ 
    - For some much-needed tidiness in main(), place the function that shears each galaxy here
    - Usage is borrowed from demo9.py
    - nfw_halo is galsim.NFW() object created in main()
    - pos is position of galaxy in image
    - nfw_z_source is background galaxy redshift
    """
    try:
        g1,g2 = nfw_halo.getShear( pos , nfw_z_source )
        nfw_shear = galsim.Shear(g1=g1,g2=g2)
    except:
        # This shouldn't happen, since we exclude the inner 10 arcsec, but it's a
        # good idea to use the try/except block here anyway.
        import warnings
        warnings.warn("Warning: NFWHalo shear is invalid -- probably strong lensing!  " +
                          "Using shear = 0.")
        nfw_shear = galsim.Shear(g1=0,g2=0)
        
    nfw_mu = nfw_halo.getMagnification( pos , nfw_z_source )
    gtot=numpy.sqrt(g1**2 +g2**2)
    if gtot > 0.10:
        import warnings
        warnings.warn("Warning: gtot>0.10 means strong lensing! Setting g1=g2=0; mu=1.")      
        g1=0.0
        g2=0.0
        mu=1.0
        nfw_shear = galsim.Shear(g1=g1,g2=g2)
        
    elif nfw_mu > 1.6:
        import warnings
        warnings.warn("Warning: mu > 25 means strong lensing! Setting g1=g2=0 and mu=1.")
        g1=0.0
        g2=0.0
        mu=1.0
        nfw_shear = galsim.Shear(g1=g1,g2=g2)

    return nfw_shear, nfw_mu

def get_wcs_info(psfname):

    imdir='/Users/jemcclea/Research/SuperBIT_2019/A2218/Clean/'
    imroot=psfname.split('psfex_output/')[1].replace('_cat.psf','.fits')
    imagen=os.path.join(imdir,imroot)
    imw=galsim.wcs.readFromFitsHeader(astropy.io.fits.getheader(imagen))
    
    return imw[0]

def make_a_galaxy(ud,this_im_wcs,psf,affine):
    """
    Method to make a single galaxy object and return stamp for 
    injecting into larger GalSim image
    """
    # Choose a random RA, Dec around the sky_center.
    # Note that for this to come out close to a square shape, we need to account for the
    # cos(dec) part of the metric: ds^2 = dr^2 + r^2 d(dec)^2 + r^2 cos^2(dec) d(ra)^2
    # So need to calculate dec first.
    center_dec=this_im_wcs.center.dec
    center_ra=this_im_wcs.center.ra 
    dec = center_dec + (ud()-0.3) * image_ysize_arcsec * galsim.arcsec
    ra = center_ra + (ud()-0.3) * image_xsize_arcsec / numpy.cos(dec) * galsim.arcsec
    
    world_pos = galsim.CelestialCoord(ra,dec)
    
    # We will need the image position as well, so use the wcs to get that
    image_pos = this_im_wcs.posToImage(world_pos)
   
    # We also need this in the tangent plane, which we call "world coordinates" here,
    # since the PowerSpectrum class is really defined on that plane, not in (ra,dec).
    # This is still an x/y corrdinate 
    uv_pos=affine.toWorld(image_pos)
    
    # Draw the redshift from a power law distribution: N(f) ~ f^-2
    # TAKEN FROM DEMO9.PY
    redshift_dist = galsim.DistDeviate(ud, function = lambda x:x**-2,
                                           x_min = 0.3,
                                           x_max = 1.0)
    gal_z = redshift_dist()
    try:
        # Get the reduced shears and magnification at this point
        nfw_shear, mu = nfw_lensing(nfw, uv_pos, gal_z)
        g1=nfw_shear.g1; g2=nfw_shear.g2
        
        binom = galsim.BinomialDeviate(ud, N=1, p=0.5)
        real = binom()
        
        if real:
            # For real galaxies, we will want to whiten the noise in the image (below).
            gal = cosmos_cat.makeGalaxy(gal_type='real', rng=ud, noise_pad_size=20)
        else:
            gal = cosmos_cat.makeGalaxy(gal_type='parametric', rng=ud)
            
        # Apply a random rotation
        theta = ud()*2.0*numpy.pi*galsim.radians
        gal = gal.rotate(theta)
        
        
        # Rescale the flux to match the observed A2218 flux. The flux of gal can't be edited directly,
        # so we resort to the following kludge. 
        # This automatically scales up the noise variance by flux_scaling**2.
        
        gal_flux_dist = galsim.DistDeviate(ud,function='/Users/jemcclea/Research/GalSim/examples/output/empirical_psfs/v3/gal_flux300_prob.txt')
        gal_flux=gal_flux_dist()
        
        flux_scaling=gal_flux/gal.flux*(exp_time/300.)
        gal *= flux_scaling
        
        # Apply the cosmological (reduced) shear and magnification at this position using a single
        # GSObject method.
        try:
            gal = gal.lens(g1, g2, mu)
        except:
            print("could not lens galaxy, setting default values...")
            g1 = 0.0; g2 = 0.0
            mu = 1.0
        
        
        
        # Generate PSF at location of galaxy
        # Convolve galaxy image with the PSF.
        this_psf = psf.getPSF(image_pos)
        #final_psf=galsim.Convolve(this_psf,optics)
        gsp=galsim.GSParams(maximum_fft_size=16384)
        final = galsim.Convolve([this_psf, gal],gsparams=gsp)
        
        # Account for the fractional part of the position
        # cf. demo9.py for an explanation of this nominal position stuff.
        x_nominal = image_pos.x + 0.5
        y_nominal = image_pos.y + 0.5
        ix_nominal = int(math.floor(x_nominal+0.5))
        iy_nominal = int(math.floor(y_nominal+0.5))
        dx = x_nominal - ix_nominal
        dy = y_nominal - iy_nominal
        offset = galsim.PositionD(dx,dy)
        position=[ix_nominal,iy_nominal,ra.deg,dec.deg]
    except:
        pdb.set_trace()
    # We use method='no_pixel' here because the SDSS PSF image that we are using includes the
    # pixel response already.
    stamp = final.drawImage(wcs=this_im_wcs.local(image_pos), offset=offset, method='no_pixel')
    
    # If desired, one can also draw the PSF and output its moments too, as:
    #  psf_stamp = psf.drawImage(scale=0.206, offset=offset, method='no_pixel')
    
    # Recenter the stamp at the desired position:
    stamp.setCenter(ix_nominal,iy_nominal)
    
    new_variance=0.0
    
    if real:
        if True:
            # We use the symmetrizing option here.
            new_variance = stamp.symmetrizeNoise(final.noise, 8)
        else:
            # Here is how you would do it if you wanted to fully whiten the image.
            new_variance = stamp.whitenNoise(final.noise)
    print("noise variance is %f" % new_variance)
    galaxy_truth=truth()
    galaxy_truth.ra=ra.deg; galaxy_truth.dec=dec.deg
    galaxy_truth.x=ix_nominal; galaxy_truth.y=iy_nominal
    galaxy_truth.g1=g1; galaxy_truth.g2=g2
    galaxy_truth.mu = mu; galaxy_truth.z = gal_z
    galaxy_truth.flux = stamp.added_flux
    galaxy_truth.variance=new_variance
    
    return stamp, galaxy_truth

def make_a_star(ud,this_im_wcs,psf,affine):
    # Choose a random RA, Dec around the sky_center.
    center_dec=this_im_wcs.center.dec
    center_ra=this_im_wcs.center.ra 
    dec = center_dec + (ud()-0.5) * image_ysize_arcsec * galsim.arcsec
    ra = center_ra + (ud()-0.5) * image_xsize_arcsec / numpy.cos(dec) * galsim.arcsec
    
    world_pos = galsim.CelestialCoord(ra,dec)
    
    # We will need the image position as well, so use the wcs to get that
    image_pos = this_im_wcs.posToImage(world_pos)
   
    # We also need this in the tangent plane, which we call "world coordinates" here,
    # since the PowerSpectrum class is really defined on that plane, not in (ra,dec).
    # This is still an x/y corrdinate 
    uv_pos=affine.toWorld(image_pos)
    
    # Draw star flux at random; based on distribution of star fluxes in real images
    # Generate PSF at location of star, convolve simple Airy with the PSF to make a star
    
    flux_dist = galsim.DistDeviate(ud,function='/Users/jemcclea/Research/GalSim/examples/output/empirical_psfs/v2/stars_flux300_prob.txt')#,interpolant='floor')
    star_flux = (flux_dist())*(exp_time/300.)
    shining_star = galsim.Airy(lam=lam, obscuration=0.3840, diam=tel_diam, scale_unit=galsim.arcsec,flux=star_flux)
    this_psf = psf.getPSF(image_pos)
    star=galsim.Convolve([shining_star,this_psf])
    
    # Account for the fractional part of the position
    # cf. demo9.py for an explanation of this nominal position stuff.
    x_nominal = image_pos.x + 0.5
    y_nominal = image_pos.y + 0.5
    ix_nominal = int(math.floor(x_nominal+0.5))
    iy_nominal = int(math.floor(y_nominal+0.5))
    dx = x_nominal - ix_nominal
    dy = y_nominal - iy_nominal
    offset = galsim.PositionD(dx,dy)
    star_stamp = star.drawImage(wcs=this_im_wcs.local(image_pos), offset=offset, method='no_pixel')
    
    # Recenter the stamp at the desired position:
    star_stamp.setCenter(ix_nominal,iy_nominal)
    
    star_truth=truth()
    star_truth.ra = ra.deg; star_truth.dec = dec.deg
    star_truth.x = ix_nominal; star_truth.y = iy_nominal

    return star_stamp, star_truth

def main(argv):
    """
    Make images using model PSFs and galaxy cluster shear:
      - The galaxies come from COSMOSCatalog, which can produce either RealGalaxy profiles
        (like in demo10) and parametric fits to those profiles.  We choose 40% of the galaxies
        to use the images, and the other 60% to use the parametric fits
      - The real galaxy images include some initial correlated noise from the original HST
        observation.  However, we whiten the noise of the final image so the final image has
        stationary Gaussian noise, rather than correlated noise.
    """
    logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
    logger = logging.getLogger("mock_superbit_data")

    # Define some parameters we'll use below.
    # Normally these would be read in from some parameter file.
    global pixel_scale
    pixel_scale = 0.206               # arcsec/pixel
    global image_xsize
    image_xsize = 6665               # size of image in pixels
    global image_ysize
    image_ysize = 4453                # size of image in pixels
    global image_xsize_arcsec
    image_xsize_arcsec = image_xsize*pixel_scale # size of big image in each dimension (arcsec)
    global image_ysize_arcsec
    image_ysize_arcsec = image_ysize*pixel_scale # size of big image in each dimension (arcsec)
    global center_ra
    #center_ra = 19.3*galsim.hours     # The RA, Dec of the center of the image on the sky
    global center_dec
    #center_dec = -33.1*galsim.degrees
    global nobj
    nobj = 1700                   # number of galaxies in entire field -- an adjustment to ensure ~1100 detections
    global nstars
    nstars = 370                     # number of stars in the entire field
    #global flux_scaling               # Let's figure out the flux for a 0.5 m class telescope
    global tel_diam
    tel_diam = 0.5                    
    global lam
    lam = 587                         # Central wavelength
    global exp_time
    global noise_variance
    global sky_level
   
    psf_path = '/Users/jemcclea/Research/SuperBIT_2019/superbit-ngmix/scripts/outputs/psfex_output'
    global nfw                        # will store the NFWHalo information
    global cosmos_cat                 # will store the COSMOS catalog from which we draw objects
    
    # Set up the NFWHalo:
    mass=5E14              # Cluster mass (Msol/h)
    nfw_conc = 4           # Concentration parameter = virial radius / NFW scale radius
    nfw_z_halo = 0.3       # redshift of the halo
    nfw_z_source = 0.6     # redshift of the lensed sources
    omega_m = 0.3          # Omega matter for the background cosmology.
    omega_lam = 0.7        # Omega lambda for the background cosmology.
    
    nfw = galsim.NFWHalo(mass=mass, conc=nfw_conc, redshift=nfw_z_halo,
                             omega_m=omega_m, omega_lam=omega_lam)
    logger.info('Set up NFW halo for lensing')

    # Read in galaxy catalog
    if True:
        # The catalog we distribute with the GalSim code only has 100 galaxies.
        # The galaxies will typically be reused several times here.
        cat_file_name = 'real_galaxy_catalog_23.5_example.fits'
        dir = 'data'
        cosmos_cat = galsim.COSMOSCatalog(cat_file_name, dir=dir)
    else:
        # If you've run galsim_download_cosmos, you can leave out the cat_file_name and dir
        # to use the full COSMOS catalog with 56,000 galaxies in it.
        cosmos_cat = galsim.COSMOSCatalog()
    logger.info('Read in %d galaxies from catalog', cosmos_cat.nobjects)
    
    # The catalog returns objects that are appropriate for HST in 1 second exposures.  So for our
    # telescope we scale up by the relative area and exposure time. 
    hst_eff_area = 2.4**2 * (1.-0.33**2)
    sbit_eff_area = tel_diam**2 * (1.-0.3840**2) 
    #flux_scaling = (sbit_eff_area/hst_eff_area) * exp_time

  
    ###
    ### LOOP OVER PSFs TO MAKE GROUPS OF IMAGES
    ### WITHIN EACH PSF, ITERATE 5 TIMES TO MAKE 5 SEPARATE IMAGES
    ###
    all_psfs=glob.glob(psf_path+"/*150*.psf")
    logger.info('Beginning loop over jitter/optical psfs')
    # random_seed = 24783923
    random_seed = 247
    i=0
    for psf_filen in all_psfs:
        logger.info('Beginning PSF %s...'% psf_filen)
        
        rng = galsim.BaseDeviate(random_seed)

        # This is specific to empirical PSFs
        try:
            timescale=psf_filen.split('target_')[1].split('_WCS')[0]
        except:
            timescale=psf_filen.split('sci_')[1].split('_WCS')[0]
            
        outname=''.join(['mockSuperbit_empiricalPSF_',timescale,'_',str(i),'.fits'])
        truth_file_name=''.join(['./output/truth_empiricalPSF_',timescale,'_',str(i),'.dat'])
        file_name = os.path.join('output',outname)

        # Set up the image:
        if timescale=='150':
            print("Automatically detecting a 150s exposure image, setting flux scale and noise accordingly")
            noise_variance = 1.8e3           # ADU^2  (Just use simple Gaussian noise here.) -->150s
            sky_level = 51                   # ADU / arcsec^2 -->150s
            exp_time=150.

        else:
            print("Automatically detecting a 300s exposure image, setting flux scale and noise accordingly")
            noise_variance = 2.55e3           # ADU^2  (Just use simple Gaussian noise here.) -->300s
            sky_level = 106                   # ADU / arcsec^2 -->300s
            exp_time=300.
 
        # Setting up a truth catalog
        names = [ 'gal_num', 'x_image', 'y_image',
                      'ra', 'dec', 'g1_meas', 'g2_meas', 
                      'nfw_g1', 'nfw_g2', 'nfw_mu', 'redshift','flux', 'var']
        types = [ int, float, float, float,
                      float, float, float, float,
                      float, float, float, float, float]
        truth_catalog = galsim.OutputCatalog(names, types)

        # Set up the image:           
        full_image = galsim.ImageF(image_xsize, image_ysize)
        full_image.fill(sky_level)
        full_image.setOrigin(0,0)
            
    
        # We keep track of how much noise is already in the image from the RealGalaxies.
        noise_image = galsim.ImageF(image_xsize, image_ysize)
        noise_image.setOrigin(0,0)

        # Make a slightly non-trivial WCS.  We'll use a slightly rotated coordinate system
        # and center it at the image center.
        """
        theta = 0.17 * galsim.degrees
        dudx = numpy.cos(theta) * pixel_scale
        dudy = -numpy.sin(theta) * pixel_scale
        dvdx = numpy.sin(theta) * pixel_scale
        dvdy = numpy.cos(theta) * pixel_scale
        
        image_center = full_image.true_center
        affine = galsim.AffineTransform(dudx, dudy, dvdx, dvdy, origin=full_image.true_center)
        sky_center = galsim.CelestialCoord(ra=center_ra, dec=center_dec)
        
        wcs = galsim.TanWCS(affine, sky_center, units=galsim.arcsec)
        """

        this_im_wcs=get_wcs_info(psf_filen)
        affine=this_im_wcs.affine(full_image.true_center) 
        full_image.wcs = this_im_wcs
        
        
        # Now let's read in the PSFEx PSF model.  We read the image directly into an
        # InterpolatedImage GSObject, so we can manipulate it as needed 
        psf_wcs=this_im_wcs
        psf_file = os.path.join(psf_path,psf_filen)
        psf = galsim.des.DES_PSFEx(psf_file,wcs=psf_wcs)
        logger.info('Constructed PSF object from PSFEx file')

    
        # Loop over galaxy objects:
        for k in range(nobj):
            time1 = time.time()
                
            # The usual random number generator using a different seed for each galaxy.
            ud = galsim.UniformDeviate(random_seed+k+1)

            try: 
                # make single galaxy object
                stamp,truth = make_a_galaxy(ud=ud,this_im_wcs=this_im_wcs,psf=psf,affine=affine)                
                # Find the overlapping bounds:
                bounds = stamp.bounds & full_image.bounds
                    
                # We need to keep track of how much variance we have currently in the image, so when
                # we add more noise, we can omit what is already there.
    
                noise_image[bounds] += truth.variance
            
                # Finally, add the stamp to the full image.
                
                full_image[bounds] += stamp[bounds]
                time2 = time.time()
                tot_time = time2-time1
                logger.info('Galaxy %d positioned relative to center t=%f s',
                                k, tot_time)
                g1_real=stamp.FindAdaptiveMom().observed_shape.g1 
                g2_real=stamp.FindAdaptiveMom().observed_shape.g2
                #g1_real=-9999.
                #g2_real=-9999.
                row = [ k,truth.x, truth.y, truth.ra, truth.dec, 
                            g1_real, g2_real, truth.g1, truth.g2, truth.mu,
                            truth.z, truth.flux, truth.variance]
                truth_catalog.addRow(row)
            except:
                logger.info('Galaxy %d has failed, skipping...',k)
                #pdb.set_trace()
                pass


        ####
        ### Now repeat process for stars!
        ####
    
        random_seed_stars=3221987
        
        for k in range(nstars):
            time1 = time.time()
            ud = galsim.UniformDeviate(random_seed_stars+k+1)
        
            star_stamp,truth=make_a_star(ud=ud,this_im_wcs=this_im_wcs,psf=psf,affine=affine)
            bounds = star_stamp.bounds & full_image.bounds
            
            # Add the stamp to the full image.
            try: 
                full_image[bounds] += star_stamp[bounds]
                
                time2 = time.time()
                tot_time = time2-time1
                
                logger.info('Star %d: positioned relative to center, t=%f s',
                            k,  tot_time)
                
                g1_real=star_stamp.FindAdaptiveMom().observed_shape.g1
                g2_real=star_stamp.FindAdaptiveMom().observed_shape.g2
                #g1_real = -9999.
                #g2_real = -9999.
                this_var = -9999.
                this_flux=numpy.sum(star_stamp.array)
                row = [ k,truth.x, truth.y, truth.ra, truth.dec, 
                            g1_real, g2_real, truth.g1, truth.g2, truth.mu,
                            truth.z, this_flux,this_var]
                truth_catalog.addRow(row)
                
            except:
                logger.info('Star %d has failed, skipping...',k)
                pass
                    
            
        # We already have some noise in the image, but it isn't uniform.  So the first thing to do is
        # to make the Gaussian noise uniform across the whole image.  
        max_current_variance = numpy.max(noise_image.array)
        noise_image = max_current_variance - noise_image
        vn = galsim.VariableGaussianNoise(rng, noise_image)
        full_image.addNoise(vn)
        
        # Now max_current_variance is the noise level across the full image.  We don't want to add that
        # twice, so subtract off this much from the intended noise that we want to end up in the image.
        noise_variance -= max_current_variance

        # Now add Gaussian noise with this variance to the final image.
        
        try:
            noise = galsim.GaussianNoise(rng, sigma=math.sqrt(noise_variance))
        except:
            noise = galsim.GaussianNoise(rng, sigma=math.sqrt(1800))

        full_image.addNoise(noise)
        logger.info('Added noise to final large image')
        
        # Now write the image to disk.  It is automatically compressed with Rice compression,
        # since the filename we provide ends in .fz.
        full_image.write(file_name)
        logger.info('Wrote image to %r',file_name)
        
        # Write truth catalog to file. 
        truth_catalog.write(truth_file_name)
        
        # Compute some sky positions of some of the pixels to compare with the values of RA, Dec
        # that ds9 reports.  ds9 always uses (1,1) for the lower left pixel, so the pixel coordinates
        # of these pixels are different by 1, but you can check that the RA and Dec values are
        # the same as what GalSim calculates.

        i=i+1
        logger.info(' ')
        logger.info('completed run %d for psf %s',i,psf_filen)
    logger.info('completed all images')

if __name__ == "__main__":
    main(sys.argv)
