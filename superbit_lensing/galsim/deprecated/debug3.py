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
        self.g1_nopsf = -9999.
        self.g2_nopsf = -9999.
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
    """
    try:
        g1,g2 = nfw_halo.getShear( pos , nfw_z_source )
        nfw_shear = galsim.Shear(g1=g1,g2=g2)
    except:
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

    """
    g1,g2 = nfw_halo.getShear( pos , nfw_z_source )
    nfw_shear = galsim.Shear(g1=g1,g2=g2)
    nfw_mu = nfw_halo.getMagnification( pos , nfw_z_source )

    if nfw_mu < 0:
        import warnings
        warnings.warn("Warning: mu < 0 means strong lensing!  Using mu=25.")
        nfw_mu = 25
    elif nfw_mu > 25:
        import warnings
        warnings.warn("Warning: mu > 25 means strong lensing!  Using mu=25.")
        nfw_mu = 25        
    
    
    return nfw_shear, nfw_mu

def get_wcs_info(psfname):

    imdir='/Users/jemcclea/Research/SuperBIT_2019/A2218/Clean/'
    imroot=psfname.split('psfex_output/')[1].replace('_cat.psf','.fits')
    imagen=os.path.join(imdir,imroot)
    imw=galsim.wcs.readFromFitsHeader(astropy.io.fits.getheader(imagen))
    
    return imw[0]


def make_a_galaxy(ud,wcs,affine):
    """
    Method to make a single galaxy object and return stamp for 
    injecting into larger GalSim image
    """

    #gsp=galsim.GSParams(maximum_fft_size=32308)
    gsp=galsim.GSParams(maximum_fft_size=16154)

    # Choose a random RA, Dec around the sky_center.
    # Note that for this to come out close to a square shape, we need to account for the
    # cos(dec) part of the metric: ds^2 = dr^2 + r^2 d(dec)^2 + r^2 cos^2(dec) d(ra)^2
    # So need to calculate dec first.
    dec = center_dec + (ud()-0.5) * image_ysize_arcsec * galsim.arcsec
    ra = center_ra + (ud()-0.5) * image_xsize_arcsec / numpy.cos(dec) * galsim.arcsec
    world_pos = galsim.CelestialCoord(ra,dec)
    
    # We will need the image position as well, so use the wcs to get that
    image_pos = wcs.toImage(world_pos)
   
    # We also need this in the tangent plane, which we call "world coordinates" here
    # This is still an x/y corrdinate 
    uv_pos = affine.toWorld(image_pos)
  
    # Draw the redshift from a power law distribution: N(f) ~ f^-2
    redshift_dist = galsim.DistDeviate(ud, function = lambda x:x**-2,
                                           x_min = 0.5,
                                           x_max = 1.5)
    gal_z = redshift_dist()
    
    
    # Get the reduced shears and magnification at this point
    nfw_shear, mu = nfw_lensing(nfw, uv_pos, gal_z)
    g1=nfw_shear.g1; g2=nfw_shear.g2
    
    gal = cosmos_cat.makeGalaxy(gal_type='parametric', rng=ud,gsparams=gsp)
    
    
    # Apply a random rotation
    theta = ud()*2.0*numpy.pi*galsim.radians
    gal = gal.rotate(theta)
    
    
    # Rescale the flux to match the observed A2218 flux. The flux of gal can't be edited directly,
    # so we resort to the following kludge. 
    # This automatically scales up the noise variance by flux_scaling**2.
    gal *= flux_scaling
    logger.debug('rescaled galaxy flux to %f' % flux_scaling)
    
    
    # Apply the cosmological (reduced) shear and magnification at this position using a single
    # GSObject method.

    try:
        """
        gal = gal.magnify(nfw_mu)
        gal = gal.shear(total_shear)
        """
        gal = gal.lens(g1, g2, mu)
    except:
        print("could not lens galaxy, setting default values...")
        g1 = 0.0; g2 = 0.0
        mu = 1.0
    
    # Generate PSF at location of galaxy
    # Convolve galaxy image with the PSF.
    #this_psf = psf.getPSF(image_pos)
    #logger.debug("obtained PSF at image position")

    this_psf = galsim.Gaussian(flux=1., fwhm=psf_fwhm)
    logger.debug("created Gaussian PSF at image position")
    
    final = galsim.Convolve([this_psf, gal],gsparams=gsp)
    logger.debug("Convolved galaxy and PSF at image position")

    
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

    # We use method='no_pixel' for "final" because the PSFEx image that we are using includes the
    # pixel response already.
    """
    semifinal=gal.drawImage(wcs=wcs.local(image_pos), offset=offset, method='no_pixel') 
    stamp = final.drawImage(wcs=wcs.local(image_pos), offset=offset, method='no_pixel')
    """
    try:
        semifinal=gal.drawImage(wcs=wcs.local(image_pos), offset=offset) 
        stamp = final.drawImage(wcs=wcs.local(image_pos), offset=offset)
        logger.debug("Images drawn")

        # Recenter the stamp at the desired position:
        semifinal.setCenter(ix_nominal,iy_nominal)
        stamp.setCenter(ix_nominal,iy_nominal)
        logger.debug("Created stamp and set center!")
        new_variance=0.0 
    except:
        logger.debug("semifinal stamp failed -- probably too big")
        pass
        
    # If desired, one can also draw the PSF and output its moments too, as:
    # psf_stamp = galsim.ImageF(stamp.bounds)
    # psf_im=this_psf.drawImage(psf_stamp,wcs=wcs.local(image_pos), offset=offset,method='no_pixel')
     
    try:
        new_variance = stamp.whitenNoise(final.noise)
        logger.debug("whitened stamp noise")
    except:
        logger.debug("no noise to whiten")

    logger.debug("noise variance is %f" % new_variance)
    galaxy_truth=truth()
    galaxy_truth.ra=ra.deg; galaxy_truth.dec=dec.deg
    galaxy_truth.x=ix_nominal; galaxy_truth.y=iy_nominal
    galaxy_truth.g1=g1; galaxy_truth.g2=g2
    galaxy_truth.mu = mu; galaxy_truth.z = gal_z
    galaxy_truth.flux = stamp.added_flux
    galaxy_truth.variance=new_variance
    try:
        galaxy_truth.fwhm=final.calculateFWHM()
        galaxy_truth.final_sigmaSize=stamp.FindAdaptiveMom().moments_sigma
        galaxy_truth.nopsf_sigmaSize=semifinal.FindAdaptiveMom().moments_sigma
    except:
        
        logger.debug('failed at FWHM/mom_size stage')        
        galaxy_truth.fwhm=-9999.0
        galaxy_truth.mom_size=stamp.FindAdaptiveMom().moments_sigma
        galaxy_truth.final_sigmaSize=-9999.0
        galaxy_truth.nopsf_sigmaSize=-9999.0
      
    try:
        galaxy_truth.g1_nopsf=semifinal.FindAdaptiveMom().observed_shape.g1
        galaxy_truth.g2_nopsf=semifinal.FindAdaptiveMom().observed_shape.g2
    except:
        logger.debug('Could not obtain no_psf galaxy adaptive moments, setting to zero')
        galaxy_truth.g2_nopsf=0
        galaxy_truth.g1_nopsf=0
        #pdb.set_trace()

    return stamp, galaxy_truth


def make_a_star(ud,wcs=None,affine=None):
    #star_flux = 1.e5    # total counts on the image
    #star_sigma = 2.     # arcsec
    
    logger.debug('entered make a star method...')

    # Choose a random RA, Dec around the sky_center.
    dec = center_dec + (ud()-0.5) * image_ysize_arcsec * galsim.arcsec
    ra = center_ra + (ud()-0.5) * image_xsize_arcsec / numpy.cos(dec) * galsim.arcsec
    world_pos = galsim.CelestialCoord(ra,dec)
    
    # We will need the image position as well, so use the wcs to get that
    image_pos = wcs.toImage(world_pos)
    
    # We also need this in the tangent plane, which we call "world coordinates" here,
    # This is still an x/y corrdinate 
    uv_pos = affine.toWorld(image_pos)
    logger.debug('made it through WCS calculations...')

    
    # Draw star flux at random; based on distribution of star fluxes in real images
    # Generate PSF at location of star, convolve simple Airy with the PSF to make a star
    if (exp_time==300):
        flux_dist = galsim.DistDeviate(ud, function = lambda x:x**-0.9, x_min = 799.2114, x_max = 890493.9)
    else:
        flux_dist = galsim.DistDeviate(ud, function = lambda x:x**-1., x_min = 1226.2965, x_max = 968964.0) 
    star_flux = flux_dist()
    
    star_flux = flux_dist()
    #shining_star = galsim.Airy(lam=lam, obscuration=0.3840, diam=tel_diam, scale_unit=galsim.arcsec,flux=star_flux)
    shining_star = galsim.Airy(lam=lam, obscuration=0.1, diam=tel_diam, scale_unit=galsim.arcsec,flux=star_flux)
    logger.debug('created star object with flux')

    #this_psf = psf.getPSF(image_pos)
    this_psf = galsim.Gaussian(flux=1., fwhm=psf_fwhm)

    star=galsim.Convolve([shining_star,this_psf])
    # Final profile is the convolution of these
    # Can include any number of things in the list, all of which are convolved
    # together to make the final flux profile.
    logger.debug('convolved star & psf')
  
    # Account for the fractional part of the position
    # cf. demo9.py for an explanation of this nominal position stuff.
    x_nominal = image_pos.x + 0.5
    y_nominal = image_pos.y + 0.5
    ix_nominal = int(math.floor(x_nominal+0.5))
    iy_nominal = int(math.floor(y_nominal+0.5))
    dx = x_nominal - ix_nominal
    dy = y_nominal - iy_nominal
    offset = galsim.PositionD(dx,dy)
    #star_stamp = star.drawImage(wcs=wcs.local(image_pos),offset=offset, method='no_pixel')
    star_stamp = star.drawImage(wcs=wcs.local(image_pos),offset=offset)
    logger.debug('made a star_stamp')
    
    # Recenter the stamp at the desired position:
    star_stamp.setCenter(ix_nominal,iy_nominal)
    
    star_truth=truth()
    star_truth.ra = ra.deg; star_truth.dec = dec.deg
    star_truth.x = ix_nominal; star_truth.y = iy_nominal
    star_truth.flux=star_stamp.added_flux
    logger.debug('made it through star recentering and flux adding')
    
    try:
        star_truth.fwhm=final.calculateFWHM()
        #star_truth.mom_size=star_stamp.FindAdaptiveMom().moments_sigma
        star_truth.final_sigmaSize=star_stamp.FindAdaptiveMom().moments_sigma
        star_truth.nopsf_sigmaSize=-9999.

    except:
        #pdb.set_trace()
        star_truth.fwhm=-9999.0
        star_truth.final_sigmaSize=-9999.
        star_truth.nopsf_sigmaSize=-9999.

    """
    
    results = star_stamp.FindAdaptiveMom()
    logger.debug('HSM reports that the image has observed shape and size:')
    logger.debug('    e1 = %.3f, e2 = %.3f, sigma = %.3f (pixels)', results.observed_shape.e1,
                results.observed_shape.e2, results.moments_sigma)
    
    logger.debug('Expected values in the limit that pixel response and noise are negligible:')
    logger.debug('    e1 = %.3f, e2 = %.3f, sigma = %.3f', 0.0, 0.0, math.sqrt(star_sigma**2 + psf_sigma**2)/pixel_scale)
    """
    return star_stamp, star_truth

def main(argv):
    """
    Make images using model PSFs and galaxy cluster shear:
      - The galaxies come from COSMOSCatalog, which can produce either RealGalaxy profiles
        (like in demo10) and parametric fits to those profiles.  
      - The real galaxy images include some initial correlated noise from the original HST
        observation.  However, we whiten the noise of the final image so the final image has
        stationary Gaussian noise, rather than correlated noise.
    """
    logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
    global logger
    logger = logging.getLogger("mock_superbit_data")

    # Define some parameters we'll use below.
    # Normally these would be read in from some parameter file.
    global pixel_scale
    pixel_scale = 0.206                   # arcsec/pixel
    global image_xsize 
    image_xsize = 6665                    # size of image in pixels
    global image_ysize
    image_ysize = 4453                    # size of image in pixels
    global image_xsize_arcsec
    image_xsize_arcsec = image_xsize*pixel_scale # size of big image in each dimension (arcsec)
    global image_ysize_arcsec
    image_ysize_arcsec = image_ysize*pixel_scale # size of big image in each dimension (arcsec)
    global center_ra
    center_ra = 19.3*galsim.hours         # The RA, Dec of the center of the image on the sky
    global center_dec
    center_dec = -33.1*galsim.degrees
    global nobj
    nobj = 2400                        # number of galaxies in entire field; this number matches empirical
    global nstars
    nstars = 550                         # number of stars in the entire field
    global flux_scaling                  # Let's figure out the flux for a 0.5 m class telescope
    global tel_diam
    tel_diam = 0.5
    global psf_fwhm
    psf_fwhm = 0.4
    global lam
    lam = 587                            # Central wavelength for an airy disk
    global exp_time
    exp_time = 300
    global noise_variance
    global sky_level
   
    psf_path = '/Users/jemcclea/Research/SuperBIT/superbit-ngmix/scripts/outputs/psfex_output'
    global nfw                        # will store the NFWHalo information
    global cosmos_cat                 # will store the COSMOS catalog from which we draw objects
    
    # Set up the NFWHalo:
    mass=5E15              # Cluster mass (Msol/h)
    nfw_conc = 4           # Concentration parameter = virial radius / NFW scale radius
    nfw_z_halo = 0.17     # redshift of the halo --> correct!
    #nfw_z_halo = 0.3       # incorrect placeholder redshift 
    nfw_z_source = 0.6     # redshift of the lensed sources
    omega_m = 0.3          # Omega matter for the background cosmology.
    omega_lam = 0.7        # Omega lambda for the background cosmology.
    
    nfw = galsim.NFWHalo(mass=mass, conc=nfw_conc, redshift=nfw_z_halo,
                             omega_m=omega_m, omega_lam=omega_lam)
    logger.info('Set up NFW halo for lensing')

    # Read in galaxy catalog
    
    cat_file_name = 'real_galaxy_catalog_23.5.fits'
    dir = 'data/COSMOS_23.5_training_sample'
    #cat_file_name = 'real_galaxy_catalog_23.5_example.fits'
    #dir = 'data'
    cosmos_cat = galsim.COSMOSCatalog()#cat_file_name, dir=dir)
    logger.info('Read in %d galaxies from catalog', cosmos_cat.nobjects)
    
    # The catalog returns objects that are appropriate for HST in 1 second exposures.  So for our
    # telescope we scale up by the relative area and exposure time.
    # Will also multiply by the gain and relative pixel scales...
    hst_eff_area = 2.4**2 * (1.-0.33**2)
    sbit_eff_area = tel_diam**2 * (1.-0.1**2) 
    
  
    ###
    ### LOOP OVER PSFs TO MAKE GROUPS OF IMAGES
    ### WITHIN EACH PSF, ITERATE 5 TIMES TO MAKE 5 SEPARATE IMAGES
    ###
    #all_psfs=glob.glob(psf_path+"/*.psf")
    #all_psfs=glob.glob(psf_path+"/*300*.psf")

    random_seed = 34509376814
    
    i=0
    for psf_filen in range(5):
        
        logger.info('Beginning PSF %s...'% psf_filen)
        rng = galsim.BaseDeviate(random_seed)

        timescale=str(exp_time)
       
        outname=''.join(['mockSuperbit_shear_',timescale,'_',str(i),'.fits'])
        truth_file_name=''.join(['./output-debug/truth_shear_',timescale,'_',str(i),'.dat'])
        file_name = os.path.join('output-debug',outname)

        # Set up the image:
        if timescale=='150':
            print("Automatically detecting a 150s exposure image, setting flux scale and noise accordingly")
            #noise_variance=570               # ADU^2  (Just use simple Gaussian noise here.)
            noise_variance=235               # ADU^2  (Just use simple Gaussian noise here.)
            sky_level = 51                   # ADU 
            exp_time=150.
            
        else:
            print("Automatically detecting a 300s exposure image, setting flux scale and noise accordingly")
            #noise_variance=800              # ADU^2  (Just use simple Gaussian noise here.)
            noise_variance=400
            sky_level = 106                 # ADU  
            exp_time=300.
            
        flux_scaling = (sbit_eff_area/hst_eff_area) * exp_time * 3.33 * (.206/.05)
                
        # Setting up a truth catalog
        names = [ 'gal_num', 'x_image', 'y_image',
                      'ra', 'dec', 'g1_nopsf', 'g2_nopsf','g1_meas', 'g2_meas', 'fwhm','final_sigmaSize',
                      'nopsf_sigmaSize','nfw_g1', 'nfw_g2', 'nfw_mu', 'redshift','flux', 'stamp_sum', 'noisevar']
        types = [ int, float, float, float, float, float,
                      float, float, float, float, float, float,
                      float, float,float, float, float,float, float]
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
        theta = 0.17 * galsim.degrees
        dudx = numpy.cos(theta) * pixel_scale
        dudy = -numpy.sin(theta) * pixel_scale
        dvdx = numpy.sin(theta) * pixel_scale
        dvdy = numpy.cos(theta) * pixel_scale
        
        image_center = full_image.true_center
        affine = galsim.AffineTransform(dudx, dudy, dvdx, dvdy, origin=full_image.true_center)
        sky_center = galsim.CelestialCoord(ra=center_ra, dec=center_dec)
        
        wcs = galsim.TanWCS(affine, sky_center, units=galsim.arcsec)
        full_image.wcs = wcs

        
        # Now let's read in the PSFEx PSF model.  We read the image directly into an
        # InterpolatedImage GSObject, so we can manipulate it as needed
        """
        psf_wcs=wcs
        psf_file = os.path.join(psf_path,psf_filen)
        psf = galsim.des.DES_PSFEx(psf_file,wcs=psf_wcs)
        logger.info('Constructed PSF object from PSFEx file')
        """
        # Loop over galaxy objects:

        for k in range(nobj):
            time1 = time.time()
                
            # The usual random number generator using a different seed for each galaxy.
            ud = galsim.UniformDeviate(random_seed+k+1)

            try: 
                # make single galaxy object
                logger.debug("about to make stamp %d...",k)
                stamp,truth = make_a_galaxy(ud=ud,wcs=wcs,affine=affine)
                logger.debug("stamp %d is made",k)
                # Find the overlapping bounds:
                bounds = stamp.bounds & full_image.bounds
                    
                # We need to keep track of how much variance we have currently in the image, so when
                # we add more noise, we can omit what is already there.
                noise_image[bounds] += truth.variance
            
                # Finally, add the stamp to the full image.
                full_image[bounds] += stamp[bounds]
                logger.debug("stamp %d added to full image",k)
                time2 = time.time()
                tot_time = time2-time1
                logger.info('Galaxy %d positioned relative to center t=%f s',
                                k, tot_time)
                try:
                    g1_real=stamp.FindAdaptiveMom().observed_shape.g1 
                    g2_real=stamp.FindAdaptiveMom().observed_shape.g2
                except:
                    g1_real=-9999.
                    g2_real=-9999.
                logger.debug("Galaxy %d made it past g1/g2_real stage",k)
                sum_flux=numpy.sum(stamp.array)
                row = [ k,truth.x, truth.y, truth.ra, truth.dec, truth.g1_nopsf, truth.g2_nopsf, g1_real, g2_real, truth.fwhm, truth.final_sigmaSize, truth.nopsf_sigmaSize,truth.g1,truth.g2, truth.mu, truth.z, truth.flux, sum_flux, truth.variance]
                truth_catalog.addRow(row)
                logger.debug("row for galaxy %d added to truth catalog\n\n",k)
                
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
            try:

                star_stamp,truth=make_a_star(ud=ud,wcs=wcs,affine=affine)
                bounds = star_stamp.bounds & full_image.bounds
                logger.debug("star stamp & truth catalog made for star %d" %k)
                # Add the stamp to the full image.
                full_image[bounds] += star_stamp[bounds]
            
                time2 = time.time()
                tot_time = time2-time1
            
                logger.info('Star %d: positioned relative to center, t=%f s',
                                k,  tot_time)

                try:
                    g1_real=star_stamp.FindAdaptiveMom().observed_shape.g1
                    g2_real=star_stamp.FindAdaptiveMom().observed_shape.g2
                except:
                    g1_real = -9999.
                    g2_real = -9999.
                this_var = -9999.
                sum_flux=numpy.sum(star_stamp.array)
                row = [ k,truth.x, truth.y, truth.ra, truth.dec, 
                           truth.g1_nopsf, truth.g2_nopsf, g1_real, g2_real, truth.fwhm, truth.final_sigmaSize, truth.nopsf_sigmaSize, truth.g1,
                            truth.g2, truth.mu, truth.z, truth.flux, sum_flux, truth.variance]
                truth_catalog.addRow(row)
                            
            except:
                logger.info('Star %d has failed, skipping...',k)
                pdb.set_trace()
                pass
                    
            
        # We already have some noise in the image, but it isn't uniform.  So the first thing to do is
        # to make the Gaussian noise uniform across the whole image.
        
        #max_current_variance = numpy.max(noise_image.array)
        #noise_image = max_current_variance - noise_image
       
        vn = galsim.VariableGaussianNoise(rng, noise_image)
        full_image.addNoise(vn)

        
        # Now max_current_variance is the noise level across the full image.  We don't want to add that
        # twice, so subtract off this much from the intended noise that we want to end up in the image.
        #noise_variance -= max_current_variance

        # Now add Gaussian noise with this variance to the final image.
        noise = galsim.GaussianNoise(rng, sigma=math.sqrt(noise_variance))
        full_image.addNoise(noise)
        logger.info('Added noise to final output image')

        
        # Now write the image to disk.  
        full_image.write(file_name)

        
        # Add a FLUXSCL keyword for later stacking
        this_hdu=astropy.io.fits.open(file_name)
        this_hdu[0].header['FLXSCALE'] = 300.0/exp_time
        this_hdu.writeto(file_name,overwrite='True')
        logger.info('Wrote image to %r',file_name)

        
        # Write truth catalog to file. 
        truth_catalog.write(truth_file_name)
        
        i=i+1
        logger.info('completed run %d for psf %s',i,psf_filen)
        
    logger.info('completed all images')

if __name__ == "__main__":
    main(sys.argv)
