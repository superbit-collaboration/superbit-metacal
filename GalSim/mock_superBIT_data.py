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
from astropy.table import Table

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

    if nfw_mu < 0.00:
        import warnings
        warnings.warn("Warning: gtot>0.50 means strong lensing! Setting g1=g2=0; mu=1.")      
        g1=0.0
        g2=0.0
        mu=1.0
        nfw_shear = galsim.Shear(g1=g1,g2=g2)
        
    elif nfw_mu > 25:
        import warnings
        warnings.warn("Warning: mu > 25 means strong lensing! Setting g1=g2=0 and mu=1.")
        g1=0.0
        g2=0.0
        mu=1.0
        nfw_shear = galsim.Shear(g1=g1,g2=g2)

    return nfw_shear, nfw_mu

def make_a_galaxy(ud,wcs,psf,affine,fitcat):
    """
    Method to make a single galaxy object and return stamp for 
    injecting into larger GalSim image
    """
    # Choose a random RA, Dec around the sky_center.
    # Note that for this to come out close to a square shape, we need to account for the
    # cos(dec) part of the metric: ds^2 = dr^2 + r^2 d(dec)^2 + r^2 cos^2(dec) d(ra)^2
    # So need to calculate dec first.
    dec = center_dec + (ud()-0.5) * image_ysize_arcsec * galsim.arcsec
    ra = center_ra + (ud()-0.5) * image_xsize_arcsec / numpy.cos(dec) * galsim.arcsec
    world_pos = galsim.CelestialCoord(ra,dec)
    # We will need the image position as well, so use the wcs to get that
    image_pos = wcs.toImage(world_pos)
   
    # We also need this in the tangent plane, which we call "world coordinates" here.
    # This is still an x/y corrdinate 
    uv_pos = affine.toWorld(image_pos)
    
    # Create chromatic galaxy
    bp_dir = '/Users/jemcclea/Research/GalSim/examples/data'
    #bp_dir = '/users/jmcclear/data/superbit/galsim/data'
    bp_file=os.path.join(bp_dir,'lum_throughput.csv')
    bandpass = galsim.Bandpass(bp_file,wave_type='nm',blue_limit = 310,red_limit=1100)
    gal = cosmos_cat.makeGalaxy(gal_type='parametric', rng=ud,chromatic=True)
    logger.debug('created chromatic galaxy')

    # Obtain galaxy redshift from the COSMOS profile fit catalog
    gal_z=fitcat['zphot'][gal.index]

    # Apply a random rotation
    theta = ud()*2.0*numpy.pi*galsim.radians
    gal = gal.rotate(theta)
 
    # Get the reduced shears and magnification at this point
    nfw_shear, mu = nfw_lensing(nfw, uv_pos, gal_z)
    g1=nfw_shear.g1; g2=nfw_shear.g2

    # Apply the cluster (reduced) shear and magnification at this position using
    # a single GSObject method.
    try:
        gal = gal.lens(g1, g2, mu)
        logger.debug('sheared galaxy')
    except:
        print("could not lens galaxy, setting default values...")
        g1 = 0.0; g2 = 0.0
        mu = 1.0

    # This automatically scales up the noise variance by flux_scaling**2.
    gal *= flux_scaling
    logger.debug('rescaled galaxy with scaling factor %f' % flux_scaling)

        
    # Generate PSF at location of galaxy
    # Convolve galaxy image with the PSF.    
    this_psf = psf.getPSF(image_pos)
    gsp=galsim.GSParams(maximum_fft_size=16384)
    final = galsim.Convolve([this_psf, optics,gal],gsparams=gsp)
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
    
    # We use method='no_pixel' here because the SDSS PSF image that we are using includes the
    # pixel response already.
    this_stamp_image = galsim.Image(64, 64,wcs=wcs.local(image_pos))
    stamp = final.drawImage(bandpass,image=this_stamp_image, offset=offset, method='no_pixel')
    stamp.setCenter(ix_nominal,iy_nominal)
    logger.debug('drew & centered galaxy!')    
    
    galaxy_truth=truth()
    galaxy_truth.ra=ra.deg; galaxy_truth.dec=dec.deg
    galaxy_truth.x=ix_nominal; galaxy_truth.y=iy_nominal
    galaxy_truth.g1=g1; galaxy_truth.g2=g2
    galaxy_truth.mu = mu; galaxy_truth.z = gal_z
    galaxy_truth.flux = stamp.added_flux
    logger.debug('created truth values')
    
    try:
        galaxy_truth.fwhm=final.calculateFWHM()
        galaxy_truth.mom_size=stamp.FindAdaptiveMom().moments_sigma
    except:
        logger.debug('fwhm or sigma calculation failed')
        galaxy_truth.fwhm=-9999.0
        galaxy_truth.mom_size=-9999.
    
    return stamp, galaxy_truth

def make_cluster_galaxy(ud,wcs,psf,affine,centerpix,cluster_cat):
    """
    Method to make a single galaxy object and return stamp for 
    injecting into larger GalSim image

    Galaxies defined here are not lensed, and are magnified to
    look more "cluster-like." 
    """
    
    # Choose a random position within 200 pixels of the sky_center
    radius = 200
    max_rsq = radius**2
    while True:  # (This is essentially a do..while loop.)
        x = (2.*ud()-1) * radius 
        y = (2.*ud()-1) * radius 
        rsq = x**2 + y**2
        
        if rsq <= max_rsq: break

    # We will need the image position as well, so use the wcs to get that,
    # plus a small gaussian jitter so cluster doesn't look too box-like
    image_pos = galsim.PositionD(x+centerpix.x+(ud()-0.5)*10,y+centerpix.y+(ud()-0.5)*10)
    world_pos = wcs.toWorld(image_pos)
    ra=world_pos.ra; dec = world_pos.dec
   
    # We also need this in the tangent plane, which we call "world coordinates" here,
    # This is still an x/y corrdinate 
    uv_pos = affine.toWorld(image_pos)

    # Fixed redshift for cluster galaxies
    gal_z = 0.17
    
    # Create chromatic galaxy    
    bp_dir = '/Users/jemcclea/Research/GalSim/examples/data'
    bp_file=os.path.join(bp_dir,'lum_throughput.csv')
    bandpass = galsim.Bandpass(bp_file,wave_type='nm',blue_limit = 310,red_limit=1100)
    gal = cluster_cat.makeGalaxy(gal_type='parametric', rng=ud,chromatic=True)
    logger.debug('created cluster galaxy')

    # Apply a random rotation
    theta = ud()*2.0*numpy.pi*galsim.radians
    gal = gal.rotate(theta)
    
    # This automatically scales up the noise variance by flux_scaling**2.
    # The "magnify" is just for drama
    gal *= flux_scaling
    gal.magnify(10)
    logger.debug('rescaled galaxy with scaling factor %f' % flux_scaling)

        
    # Generate PSF at location of galaxy
    # Convolve galaxy image with the PSF.    
    this_psf = psf.getPSF(image_pos)
    gsp=galsim.GSParams(maximum_fft_size=16384)
    final = galsim.Convolve([this_psf,gal,optics],gsparams=gsp)
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
    
    # Draw galaxy image
    this_stamp_image = galsim.Image(128, 128,wcs=wcs.local(image_pos))
    cluster_stamp = final.drawImage(bandpass,image=this_stamp_image, offset=offset,method='no_pixel')
    cluster_stamp.setCenter(ix_nominal,iy_nominal)
    logger.debug('drew & centered galaxy!')    

    cluster_galaxy_truth=truth()
    cluster_galaxy_truth.ra=ra.deg; cluster_galaxy_truth.dec=dec.deg
    cluster_galaxy_truth.x=ix_nominal; cluster_galaxy_truth.y=iy_nominal
    cluster_galaxy_truth.g1=g1; cluster_galaxy_truth.g2=g2
    cluster_galaxy_truth.mu = mu; cluster_galaxy_truth.z = gal_z
    cluster_galaxy_truth.flux = cluster_stamp.added_flux
    logger.debug('created truth values')
    
    try:
        cluster_galaxy_truth.fwhm=final.calculateFWHM()
        cluster_galaxy_truth.mom_size=cluster_stamp.FindAdaptiveMom().moments_sigma
    except:
        logger.debug('fwhm or sigma calculation failed')
        cluster_galaxy_truth.fwhm=-9999.0
        cluster_galaxy_truth.mom_size=-9999.
    
    return cluster_stamp, cluster_galaxy_truth


def make_a_star(ud,wcs,psf,affine):
    """
    makes a star-like object for injection into larger image.
    """
    
    # Choose a random RA, Dec around the sky_center.
    dec = center_dec + (ud()-0.5) * image_ysize_arcsec * galsim.arcsec
    ra = center_ra + (ud()-0.5) * image_xsize_arcsec / numpy.cos(dec) * galsim.arcsec
    world_pos = galsim.CelestialCoord(ra,dec)
    
    # We will need the image position as well, so use the wcs to get that
    image_pos = wcs.toImage(world_pos)
    
    # We also need this in the tangent plane, which we call "world coordinates" here,
    # This is still an x/y corrdinate 
    uv_pos = affine.toWorld(image_pos)

    # Draw star flux at random; based on distribution of star fluxes in real images  
    flux_dist = galsim.DistDeviate(ud, function = lambda x:x**-1.5, x_min = 799.2114, x_max = 890493.9)
    star_flux = flux_dist()
    
    # Generate PSF at location of star, convolve with optical model to make a star
    deltastar = galsim.DeltaFunction(flux=star_flux)  
    this_psf = psf.getPSF(image_pos)
    star=galsim.Convolve([optics, this_psf,deltastar])
        
    # Account for the fractional part of the position
    # cf. demo9.py for an explanation of this nominal position stuff.
    x_nominal = image_pos.x + 0.5
    y_nominal = image_pos.y + 0.5
    ix_nominal = int(math.floor(x_nominal+0.5))
    iy_nominal = int(math.floor(y_nominal+0.5))
    dx = x_nominal - ix_nominal
    dy = y_nominal - iy_nominal
    offset = galsim.PositionD(dx,dy)
    star_stamp = star.drawImage(wcs=wcs.local(image_pos), offset=offset, method='no_pixel')

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
        (like in demo10) and parametric fits to those profiles. We chose parametric fits since
        these are required for chromatic galaxies (ones with filter response included)
      - The real galaxy images include some initial correlated noise from the original HST
        observation, which would need to be whitened. But we are using parametric galaxies, 
        so this isn't a concern.
    """
    
    global logger
    logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
    logger = logging.getLogger("mock_superbit_data")


    # Define some parameters we'll use below.
    # Defining globals is bad practice, but convenient
    # Normally these would be read in from some parameter file.
    global pixel_scale
    pixel_scale = 0.206          # arcsec/pixel
    global sky_bkg               # mean sky background from AG's paper
    sky_bkg = 0.32               # ADU / s / pix
    global sky_sigma             # standard deviation of sky background   
    sky_sigma = 0.0957           # ADU / s / pix
    gain = 3.33
    global image_xsize
    image_xsize = 6665           # size of image in pixels
    global image_ysize
    image_ysize = 4453           # size of image in pixels
    global image_xsize_arcsec
    image_xsize_arcsec = image_xsize*pixel_scale # size of big image in each dimension (arcsec)
    global image_ysize_arcsec
    image_ysize_arcsec = image_ysize*pixel_scale # size of big image in each dimension (arcsec)
    global center_ra
    center_ra = 19.3*galsim.hours     # The RA, Dec of the center of the image on the sky
    global center_dec
    center_dec = -33.1*galsim.degrees
    global center_coords
    center_coords = galsim.CelestialCoord(center_ra,center_dec)
    
    global n                     # number of exposures to create per PSF model
    n = 9
    global exp_time
    exp_time = 300
    global nobj
    nobj = 16250                # number of galaxies in entire field, based on COSMOS 25.2 depth
    global nstars
    nstars = 350                # number of stars in the entire field
    global flux_scaling           
    global tel_diam
    tel_diam = 0.5                    
    global lam                   # Fiducial wavelength for which optical aberrations are initially defined
    lam =625                     # nanometers  
    global optics                # will store the Zernicke component of the PSF
    global cosmos_cat            # will store the COSMOS catalog from which we draw objects
    
    psf_path = '/Users/jemcclea/Research/GalSim/examples/data/flight_jitter_only_oversampled_1x'
    #psf_path = '/users/jmcclear/data/superbit/galsim/data/flight_jitter_only_oversampled_1x'
    
    # Set up the NFWHalo:
    global nfw                   # will store the NFWHalo information
    mass=1E15                    # Cluster mass (Msol/h)
    nfw_conc = 4                 # Concentration parameter = virial radius / NFW scale radius
    nfw_z_halo = 0.17            # redshift of the halo
    omega_m = 0.3                # Omega matter for the background cosmology.
    omega_lam = 0.7              # Omega lambda for the background cosmology.
    
    nfw = galsim.NFWHalo(mass=mass, conc=nfw_conc, redshift=nfw_z_halo,
                             omega_m=omega_m, omega_lam=omega_lam)
    logger.info('Set up NFW halo for lensing')

    # Read in galaxy catalog, as well as catalog containing
    # information from COSMOS fits like redshifts, hlr, etc.   
    dir = 'data/COSMOS_25.2_training_sample/'
    cat_file_name = 'real_galaxy_catalog_25.2.fits'
    fit_file_name = 'real_galaxy_catalog_25.2_fits.fits'
    cosmos_cat = galsim.COSMOSCatalog(cat_file_name, dir=dir)
    fitcat = Table.read(os.path.join(dir,fit_file_name))
    logger.info('Read in %d galaxies from catalog and associated fit info', cosmos_cat.nobjects)
        
    cluster_cat = galsim.COSMOSCatalog('data/real_galaxy_catalog_23.5_example.fits')
    logger.info('Read in %d cluster galaxies from catalog', cosmos_cat.nobjects)

    
    
    # The catalog returns objects that are appropriate for HST in 1 second exposures.  So for our
    # telescope we scale up by the relative area, exposure time, pixel scale and detector gain   
    hst_eff_area = 2.4**2 * (1.-0.33**2)
    sbit_eff_area = tel_diam**2 * (1.-0.380**2) 
    flux_scaling = (sbit_eff_area/hst_eff_area) * exp_time *gain*(pixel_scale/.05)**2 


    ### Now create PSF. First, define Zernicke polynomial component
    ### note: aberrations were definined for lam = 550, and close to the
    ### center of the camera. The PSF degrades at the edge of the FOV
    lam_over_diam = lam * 1.e-9 / tel_diam    # radians
    lam_over_diam *= 206265  # arcsec
    aberrations = numpy.zeros(38)             # Set the initial size.
    aberrations[0] = 0.                       # First entry must be zero
    aberrations[1] = -0.00305127
    aberrations[4] = -0.02474205              # Noll index 4 = Defocus
    aberrations[11] = -0.01544329             # Noll index 11 = Spherical
    aberrations[22] = 0.00199235
    aberrations[26] = 0.00000017
    aberrations[37] = 0.00000004
    logger.info('Calculated lambda over diam = %f arcsec', lam_over_diam)

    # Define strut parameters. BIT has four orthogonal struts that
    # are ~12mm wide, and the exit pupil diameter is 137.4549 mm (Zemax)
    nstruts = 4                           
    strut_thick = 0.087                  # as a fraction of pupil diameter
    strut_angle = 90 * galsim.degrees    # angle between the vertical and the strut closest to it
    
    optics = galsim.OpticalPSF(lam=lam,diam=tel_diam, obscuration = 0.380, nstruts=nstruts,
                                   strut_angle=strut_angle, strut_thick=strut_thick, aberrations = aberrations)
    logger.info('Made telescope PSF profile')
    
  
    ###
    ### LOOP OVER PSFs TO MAKE GROUPS OF IMAGES
    ### WITHIN EACH PSF, ITERATE n TIMES TO MAKE n SEPARATE IMAGES
    ###
    
    all_psfs=glob.glob(psf_path+"/*121*.psf")
    logger.info('Beginning loop over jitter/optical psfs')
  
    for psf_filen in all_psfs:
        logger.info('Beginning PSF %s...'% psf_filen)
        
        for i in numpy.arange(1,n+1):          
            logger.info('Beginning loop %d'% i)

            random_seed = 23058923781
            rng = galsim.BaseDeviate(random_seed)

            try:
                root=psf_filen.split('data/')[1].split('/')[0]
                #timescale=psf_filen.split('_1x/')[1].split('.')[0]
                timescale=str(exp_time)
                outname=''.join(['mock_superbit_',root,timescale,str(i).zfill(3),'.fits'])
                truth_file_name=''.join(['./output-jitter/truth_',root,timescale,str(i).zfill(3),'.dat'])
                file_name = os.path.join('output-jitter',outname)

            except:
                print("naming failed, check path")
                pdb.set_trace()

                
            # Setting up a truth catalog
            names = [ 'gal_num', 'x_image', 'y_image',
                        'ra', 'dec', 'g1_meas', 'g2_meas', 'nfw_mu', 'redshift','flux' ]
            types = [ int, float, float, float,
                        float, float, float, float, float, float]
            truth_catalog = galsim.OutputCatalog(names, types)

            
            # Set up the image:
            full_image = galsim.ImageF(image_xsize, image_ysize)
            sky_level = exp_time * sky_bkg
            full_image.fill(sky_level)
            full_image.setOrigin(0,0)
            
    
            # We keep track of how much noise is already in the image from the RealGalaxies.
            noise_image = galsim.ImageF(image_xsize, image_ysize)
            noise_image.setOrigin(0,0)

            
            # If you wanted to make a non-trivial WCS system, could set theta to a non-zero number
            theta = 0.0 * galsim.degrees
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
            psf_wcs=wcs
            psf_file = os.path.join(psf_path,psf_filen)
            psf = galsim.des.DES_PSFEx(psf_file,wcs=psf_wcs)
            logger.info('Constructed PSF object from PSFEx file')

            #####
            ## Loop over galaxy objects:
            #####
            
            for k in range(nobj):
                time1 = time.time()
                
                # The usual random number generator using a different seed for each galaxy.
                ud = galsim.UniformDeviate(random_seed+k+1)

                try: 
                    # make single galaxy object
                    stamp,truth = make_a_galaxy(ud=ud,wcs=wcs,psf=psf,affine=affine,fitcat=fitcat)                
                    # Find the overlapping bounds:
                    bounds = stamp.bounds & full_image.bounds
                    
                    # We need to keep track of how much variance we have currently in the image, so when
                    # we add more noise, we can omit what is already there.
    
                    # noise_image[bounds] += truth.variance
            
                    # Finally, add the stamp to the full image.
                
                    full_image[bounds] += stamp[bounds]
                    time2 = time.time()
                    tot_time = time2-time1
                    logger.info('Galaxy %d positioned relative to center t=%f s',
                                k, tot_time)
                    this_flux=numpy.sum(stamp.array)
                    row = [ k,truth.x, truth.y, truth.ra, truth.dec, truth.g1, truth.g2, truth.mu,truth.z, this_flux]
                    truth_catalog.addRow(row)
                except:
                    logger.info('Galaxy %d has failed, skipping...',k)
                    

            #####
            ### Inject cluster galaxy objects:
            #####


            random_seed=892375351

            center_coords = galsim.CelestialCoord(center_ra,center_dec)
            centerpix = wcs.toImage(center_coords)
            
            for k in range(30):
                time1 = time.time()
            
                # The usual random number generator using a different seed for each galaxy.
                ud = galsim.UniformDeviate(random_seed+k+1)
                
                try: 
                    # make single galaxy object
                    cluster_stamp,truth = make_cluster_galaxy(ud=ud,wcs=wcs,affine=affine,psf=psf,
                                                                  centerpix=centerpix,cluster_cat=cluster_cat)                
                    # Find the overlapping bounds:
                    bounds = cluster_stamp.bounds & full_image.bounds
                    
                    # We need to keep track of how much variance we have currently in the image, so when
                    # we add more noise, we can omit what is already there. This is more relevant to
                    # "real" galaxy images, not parametric like we have
            
                    #noise_image[bounds] += truth.variance
            
                    # Finally, add the stamp to the full image.
                    
                    full_image[bounds] += cluster_stamp[bounds]
                    time2 = time.time()
                    tot_time = time2-time1
                    logger.info('Cluster galaxy %d positioned relative to center t=%f s',
                                    k, tot_time)
                    this_flux=numpy.sum(stamp.array)
                    row = [ k,truth.x, truth.y, truth.ra, truth.dec, truth.g1, truth.g2, truth.mu,truth.z, this_flux]
                    truth_catalog.addRow(row)
                except:
                    logger.info('Cluster galaxy %d has failed, skipping...',k)
                    pdb.set_trace()
                
            #####
            ### Now repeat process for stars!
            #####
    
            random_seed_stars=2308173501873

            for k in range(nstars):
                time1 = time.time()
                ud = galsim.UniformDeviate(random_seed_stars+k+1)

                star_stamp,truth=make_a_star(ud=ud,wcs=wcs,psf=psf,affine=affine)
                bounds = star_stamp.bounds & full_image.bounds
               
                # Add the stamp to the full image.
                try: 
                    full_image[bounds] += star_stamp[bounds]
            
                    time2 = time.time()
                    tot_time = time2-time1
                    
                    logger.info('Star %d: positioned relative to center, t=%f s',
                                k,  tot_time)
                    this_flux=numpy.sum(star_stamp.array)
                    row = [ k,truth.x, truth.y, truth.ra, truth.dec, truth.g1, truth.g2, truth.mu,
                                truth.z, this_flux]
                    truth_catalog.addRow(row)
                    
                except:
                    logger.info('Star %d has failed, skipping...',k)

            
            # If real-type COSMOS galaxies are used, the noise across the image won't be uniform. Since this code is
            # using parametric-type galaxies, the following section can be commented out.
            #
            # The first thing to do is to make the Gaussian noise uniform across the whole image.
                  
            max_current_variance = numpy.max(noise_image.array)
            noise_image = max_current_variance - noise_image
            
            vn = galsim.VariableGaussianNoise(rng, noise_image)
            full_image.addNoise(vn)
        
            # Now max_current_variance is the noise level across the full image.  We don't want to add that
            # twice, so subtract off this much from the intended noise that we want to end up in the image.
            
            this_sky_sigma = sky_sigma*exp_time
            this_sky_sigma -= numpy.sqrt(max_current_variance)
            
     
            # Regardless of galaxy type, add Gaussian noise with this variance to the final image.
            
            noise = galsim.GaussianNoise(rng, sigma=this_sky_sigma)
            full_image.addNoise(noise)
        
            logger.debug('Added noise to final output image')
            full_image.write(file_name)

 
            # Write truth catalog to file. 
            truth_catalog.write(truth_file_name)
            logger.info('Wrote image to %r',file_name)

            logger.info(' ')
            logger.info('completed run %d for psf %s',i,psf_filen)
            i=i+1
            logger.info(' ')
            
        logger.info(' ')
        logger.info('completed all images')
        logger.info(' ')

if __name__ == "__main__":
    main(sys.argv)
