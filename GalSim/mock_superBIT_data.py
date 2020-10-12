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
import galsim.convolve
import pdb
import glob
import scipy
import yaml
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
        logger.debug("NFWHalo shear is invalid")
        
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

def make_a_galaxy(ud,wcs,psf,affine,fitcat,cosmos_cat,nfw,optics,bandpass):
    """
    Method to make a single galaxy object and return stamp for 
    injecting into larger GalSim image
    """
    # Choose a random RA, Dec around the sky_center.
    # Note that for this to come out close to a square shape, we need to account for the
    # cos(dec) part of the metric: ds^2 = dr^2 + r^2 d(dec)^2 + r^2 cos^2(dec) d(ra)^2
    # So need to calculate dec first.
    dec = sbparams.center_dec + (ud()-0.5) * sbparams.image_ysize_arcsec * galsim.arcsec
    ra = sbparams.center_ra + (ud()-0.5) * sbparams.image_xsize_arcsec / numpy.cos(dec) * galsim.arcsec
    world_pos = galsim.CelestialCoord(ra,dec)
    # We will need the image position as well, so use the wcs to get that
    image_pos = wcs.toImage(world_pos)
   
    # We also need this in the tangent plane, which we call "world coordinates" here.
    # This is still an x/y corrdinate 
    uv_pos = affine.toWorld(image_pos)
    
    # Create chromatic galaxy
    gal = cosmos_cat.makeGalaxy(gal_type='parametric', rng=ud, chromatic=True)
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
    gal *= sbparams.flux_scaling
    logger.debug('rescaled galaxy with scaling factor %f' % sbparams.flux_scaling)

        
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

def make_cluster_galaxy(ud, wcs, psf, affine, centerpix, cluster_cat, optics, bandpass):
    """
    Method to make a single galaxy object and return stamp for 
    injecting into larger GalSim image

    Galaxies created here are not lensed, and are magnified to
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
    # FIXME: This appears to be missing and should be fixed????
    g1 = 0.0; g2 = 0.0
    mu = 1.0
    
    # Create chromatic galaxy    
    gal = cluster_cat.makeGalaxy(gal_type='parametric', rng=ud,chromatic=True)
    logger.debug('created cluster galaxy')

    # Apply a random rotation
    theta = ud()*2.0*numpy.pi*galsim.radians
    gal = gal.rotate(theta)
    
    # This automatically scales up the noise variance by flux_scaling**2.
    # The "magnify" is just for drama
    gal *= sbparams.flux_scaling
    gal.magnify(10)
    logger.debug('rescaled galaxy with scaling factor %f' % sbparams.flux_scaling)

        
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


def make_a_star(ud,wcs,psf,affine,optics):
    """
    makes a star-like object for injection into larger image.
    """
    
    # Choose a random RA, Dec around the sky_center.
    dec = sbparams.center_dec + (ud()-0.5) * sbparams.image_ysize_arcsec * galsim.arcsec
    ra = sbparams.center_ra + (ud()-0.5) * sbparams.image_xsize_arcsec / numpy.cos(dec) * galsim.arcsec
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

class SuperBITParameters:
        def __init__(self, config_file=None, argv=None):
            """
            Initialize default params and overwirte with config_file params and / or commmand line
            parameters.
            """
            # Define some default default parameters below.
            # These are used in the absence of a .yaml config_file or command line args.
            self.pixel_scale= 0.206     # Pixel scale                           [arcsec/px]
            self.sky_bkg    = 0.32      # mean sky background from AG's paper   [ADU / s / px]
            self.sky_sigma  = 0.0957    # standard deviation of sky background  [ADU / s / px]  
            self.gain       = 3.33      # Camera gain                           [ADU / e-]
            self.image_xsize= 6665      # Horizontal image size                 [px]
            self.image_ysize= 4453      # Vertical image size                   [px]
            self.cra        = 19.3      # Central Right Ascension               [hrs]
            self.cdec       = -33.1     # Central Declination                   [deg]
            self.nexp       = 9         # Number of exposures per PSF model     []
            self.exp_time   = 300       # Exposure time per image               [s]
            self.nobj       = 3000      # Number of galaxies (COSMOS 25.2 depth)[]
            self.nstars     = 350       # Number of stars in the field          []
            self.tel_diam   = 0.5       # Telescope aperture diameter           [m]
            self.nclustergals = 30      # Number of cluster galaxies (arbitrary)[]

            self.lam        = 625       # Fiducial wavelength for abberations   [nm]
            self.mass       = 1E15      # Cluster mass                          [Msol / h]
            self.nfw_conc   = 4         # Concentration parameter = virial radius / NFW scale radius
            self.nfw_z_halo = 0.17      # redshift of the halo                  []
            self.omega_m    = 0.3       # Omega matter for the background cosmology.
            self.omega_lam  = 0.7       # Omega lambda for the background cosmology.

            # Define strut parameters. BIT has four orthogonal struts that
            # are ~12mm wide, and the exit pupil diameter is 137.4549 mm (Zemax)
            self.nstruts    = 4         # Number of M2 struts                   []
            self.strut_thick= 0.087     # Fraction of diameter strut thickness  [m/m]
            self.strut_theta= 90        # Angle between vertical and nearest    [deg]
            self.obscuration= 0.380     # Fraction of aperture obscured by M2   []

            # Define some paths and filenames
            self.psf_path = '/Users/jemcclea/Research/GalSim/examples/data/flight_jitter_only_oversampled_1x'
            self.cosmosdir  = 'data/COSMOS_25.2_training_sample/' # Path to COSMOS data directory 
            self.cat_file_name = 'real_galaxy_catalog_25.2.fits' # catalog file name for COSMOS
            self.fit_file_name = 'real_galaxy_catalog_25.2_fits.fits' # fit file name for COSMOS
            self.cluster_cat_name = 'data/real_galaxy_catalog_23.5_example.fits' # path to cluster catalog
            self.bp_file = 'data/lum_throughput.csv' # file with bandpass data
            self.outdir = './output-jitter/' # directory where output images and truth catalogs are saved

            # Define RNG seeds
            self.noise_seed = 23058923781 
            self.galobj_seed = 23058923781
            self.cluster_seed = 892375351
            self.stars_seed = 2308173501873

            # Check for config_file params to overwrite defaults
            if config_file is not None:
                self._load_config_file(config_file)

            # Check for command line args to overwrite config_file and / or defaults
            if argv is not None:
                self._load_command_line(argv)

            # Process parameters
            self._process_params()

        def _process_params(self):
            """
            Derive the parameters from the base parameters
            """
            self.center_ra = self.cra * galsim.hours
            self.center_dec = self.cdec * galsim.degrees
            self.image_xsize_arcsec = self.image_xsize * self.pixel_scale 
            self.image_ysize_arcsec = self.image_ysize * self.pixel_scale 
            self.center_coords = galsim.CelestialCoord(self.center_ra,self.center_dec)
            self.strut_angle = self.strut_theta * galsim.degrees

            # The catalog returns objects that are appropriate for HST in 1 second exposures.  So for our
            # telescope we scale up by the relative area, exposure time, pixel scale and detector gain   
            hst_eff_area = 2.4**2 * (1.-0.33**2)
            sbit_eff_area = self.tel_diam**2 * (1.-0.380**2) 
            self.flux_scaling = (sbit_eff_area/hst_eff_area) * self.exp_time * self.gain*(self.pixel_scale/.05)**2 
        def _load_config_file(self, config_file):
            """
            Load parameters from configuration file. Only parameters that exist in the config_file
            will be overwritten.
            """
            logger.info('Loading parameters from %s' % (config_file))
            with open(config_file) as fsettings:
                config = yaml.load(fsettings, Loader=yaml.FullLoader)
            self._load_dict(config)
            self._process_params()
        def _args_to_dict(self, argv):
            """
            Converts a command line argument array to a dictionary.
            """
            d = {}
            for arg in argv[1:]:
                try:
                    (option, value) = arg.split("=", 1)
                except:
                    (option, value) = (arg, None)
                d[option] = value
            return d

        def _load_command_line(self, argv):
            """
            Load parameters from the command line argumentts. Only parameters that are provided in
            the command line will be overwritten.
            """
            logger.info('Processing command line args')
            # Parse arguments here
            self._load_dict(self._args_to_dict(argv))
            self._process_params()

        def _load_dict(self, d):
            """
            Load parameters from a dictionary.
            """
            for (option, value) in d.items():
                if option == "pixel_scale":     
                    self.pixel_scale = float(value)
                elif option == "sky_bkg":        
                    self.sky_bkg = float(value) 
                elif option == "sky_sigma":     
                    self.sky_sigma = float(value)
                elif option == "gain":          
                    self. gain = float(value)   
                elif option == "image_xsize":   
                    self.image_xsize = int(value)    
                elif option == "image_ysize":   
                    self.image_ysize = int(value)    
                elif option == "center_ra":     
                    self.center_ra = float(value)
                elif option == "center_dec":
                    self.center_dec = float(value)
                elif option == "nexp":      
                    self.nexp = int(value)          
                elif option == "exp_time":   
                    self.exp_time = float(value) 
                elif option == "nobj":     
                    self.nobj = int(value)     
                elif option == "nclustergal":     
                    self.nclustergal = int(value)     
                elif option == "nstars": 
                    self.nstars = int(value)    
                elif option == "tel_diam": 
                    self.tel_diam = float(value)
                elif option == "lam":     
                    self.lam = float(value)      
                elif option == "psf_path": 
                    self.psf_path = str(value) 
                elif option == "cluster_mass": 
                    self.mass = int(value)         
                elif option == "nfw_conc":   
                    self.nfw_conc = float(value) 
                elif option == "nfw_z_halo": 
                    self.nfw_z_halo = float(value)
                elif option == "omega_m":   
                    self.omega_m = float(value)  
                elif option == "omega_lam":
                    self.omega_lam = float(value)
                elif option == "config_file":
                    self._load_config_file(str(value))
                elif option == "cosmosdir":
                    self.cosmosdir = str(value)
                elif option == "cosmosdir":
                    self.cosmosdir = str(value)
                elif option == "cat_file_name":
                    self.cat_file_name = str(value)
                elif option == "fit_file_name":
                    self.fit_file_name = str(value)
                elif option == "cluster_cat_name":
                    self.cluster_cat_name = str(value)
                elif option == "bp_file":
                    self.bp_file = str(value)
                elif option == "outdir":
                    self.outdir = str(value)
                elif option == "noise_seed":     
                    self.noise_seed = int(value)     
                elif option == "galobj_seed":     
                    self.galobj_seed = int(value)     
                elif option == "cluster_seed":     
                    self.cluster_seed = int(value)     
                elif option == "stars_seed":     
                    self.stars_seed = int(value)     

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
    logging.basicConfig(format="%(message)s", level=logging.DEBUG, stream=sys.stdout)
    logger = logging.getLogger("mock_superbit_data")

    # Define some parameters we'll use below.
    global sbparams
    sbparams = SuperBITParameters(argv=argv)
    
    # Set up the NFWHalo:
    nfw = galsim.NFWHalo(mass=sbparams.mass, conc=sbparams.nfw_conc, redshift=sbparams.nfw_z_halo,
                     omega_m=sbparams.omega_m, omega_lam=sbparams.omega_lam)
    logger.info('Set up NFW halo for lensing')

    # Read in galaxy catalog, as well as catalog containing
    # information from COSMOS fits like redshifts, hlr, etc.   
    cosmos_cat = galsim.COSMOSCatalog(sbparams.cat_file_name, dir=sbparams.cosmosdir)
    fitcat = Table.read(os.path.join(sbparams.cosmosdir, sbparams.fit_file_name))
    logger.info('Read in %d galaxies from catalog and associated fit info', cosmos_cat.nobjects)

    cluster_cat = galsim.COSMOSCatalog(sbparams.cluster_cat_name)
    logger.info('Read in %d cluster galaxies from catalog', cosmos_cat.nobjects)
    

    ### Now create PSF. First, define Zernicke polynomial component
    ### note: aberrations were definined for lam = 550, and close to the
    ### center of the camera. The PSF degrades at the edge of the FOV
    lam_over_diam = sbparams.lam * 1.e-9 / sbparams.tel_diam    # radians
    lam_over_diam *= 206265.
    aberrations = numpy.zeros(38)             # Set the initial size.
    aberrations[0] = 0.                       # First entry must be zero
    aberrations[1] = -0.00305127
    aberrations[4] = -0.02474205              # Noll index 4 = Defocus
    aberrations[11] = -0.01544329             # Noll index 11 = Spherical
    aberrations[22] = 0.00199235
    aberrations[26] = 0.00000017
    aberrations[37] = 0.00000004
    logger.info('Calculated lambda over diam = %f arcsec', lam_over_diam)

    # will store the Zernicke component of the PSF
    optics = galsim.OpticalPSF(lam=sbparams.lam,diam=sbparams.tel_diam, 
                        obscuration=sbparams.obscuration, nstruts=sbparams.nstruts, 
                        strut_angle=sbparams.strut_angle, strut_thick=sbparams.strut_thick,
                        aberrations=aberrations)
    logger.info('Made telescope PSF profile')
    
    # load SuperBIT bandpass
    bandpass = galsim.Bandpass(sbparams.bp_file, wave_type='nm', blue_limit=310, red_limit=1100)

    ###
    ### LOOP OVER PSFs TO MAKE GROUPS OF IMAGES
    ### WITHIN EACH PSF, ITERATE n TIMES TO MAKE n SEPARATE IMAGES
    ###
    
    all_psfs=glob.glob(sbparams.psf_path+"/*121*.psf")
    logger.info('Beginning loop over jitter/optical psfs')
  
    for psf_filen in all_psfs:
        logger.info('Beginning PSF %s...'% psf_filen)
        
        for i in numpy.arange(1,sbparams.nexp+1):          
            logger.info('Beginning loop %d'% i)

            rng = galsim.BaseDeviate(sbparams.noise_seed)

            try:
                root=psf_filen.split('data/')[1].split('/')[0]
                timescale=str(sbparams.exp_time)
                outname=''.join(['mock_superbit_',root,timescale,str(i).zfill(3),'.fits'])
                truth_file_name=''.join([sbparams.outdir, 'truth_', root, timescale, str(i).zfill(3), '.dat'])
                file_name = os.path.join(sbparams.outdir, outname)

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
            full_image = galsim.ImageF(sbparams.image_xsize, sbparams.image_ysize)
            sky_level = sbparams.exp_time * sbparams.sky_bkg
            full_image.fill(sky_level)
            full_image.setOrigin(0,0)
            
    
            # We keep track of how much noise is already in the image from the RealGalaxies.
            noise_image = galsim.ImageF(sbparams.image_xsize, sbparams.image_ysize)
            noise_image.setOrigin(0,0)

            
            # If you wanted to make a non-trivial WCS system, could set theta to a non-zero number
            theta = 0.0 * galsim.degrees
            dudx = numpy.cos(theta) * sbparams.pixel_scale
            dudy = -numpy.sin(theta) * sbparams.pixel_scale
            dvdx = numpy.sin(theta) * sbparams.pixel_scale
            dvdy = numpy.cos(theta) * sbparams.pixel_scale
            image_center = full_image.true_center
            affine = galsim.AffineTransform(dudx, dudy, dvdx, dvdy, origin=full_image.true_center)
            sky_center = galsim.CelestialCoord(ra=sbparams.center_ra, dec=sbparams.center_dec)
        
            wcs = galsim.TanWCS(affine, sky_center, units=galsim.arcsec)
            full_image.wcs = wcs

            
            # Now let's read in the PSFEx PSF model. We read the model directly into an
            # GSObject, so we can manipulate it as needed 
            psf_wcs=wcs
            psf = galsim.des.DES_PSFEx(psf_filen,wcs=psf_wcs)
            logger.info('Constructed PSF object from PSFEx file')

            #####
            ## Loop over galaxy objects:
            #####
            
            for k in range(sbparams.nobj):
                time1 = time.time()
                
                # The usual random number generator using a different seed for each galaxy.
                ud = galsim.UniformDeviate(sbparams.galobj_seed+k+1)

                try: 
                    # make single galaxy object
                    stamp,truth = make_a_galaxy(ud=ud,wcs=wcs,psf=psf,affine=affine,fitcat=fitcat,
                            cosmos_cat=cosmos_cat,optics=optics,nfw=nfw,bandpass=bandpass)                
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
            ### - Note that "cluster" is just for aesthetics
            ### - So, 'n_cluster_gals' is arbitrary
            ### - You could concievably create a method to base the number of galaxies injected
            ###   using some scaling relation between (NFW) mass and richness to set n_cluster_gals
            ###   to something based in reality. 
            #####

            center_coords = galsim.CelestialCoord(sbparams.center_ra,sbparams.center_dec)
            centerpix = wcs.toImage(center_coords)
            
            for k in range(sbparams.nclustergal):
                time1 = time.time()
            
                # The usual random number generator using a different seed for each galaxy.
                ud = galsim.UniformDeviate(sbparams.cluster_seed+k+1)
                
                try: 
                    # make single galaxy object
                    cluster_stamp,truth = make_cluster_galaxy(ud=ud,wcs=wcs,affine=affine,psf=psf,
                                                                  centerpix=centerpix,
                                                                  cluster_cat=cluster_cat,
                                                                  optics=optics,
                                                                  bandpass=bandpass)                
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
    
            for k in range(sbparams.nstars):
                time1 = time.time()
                ud = galsim.UniformDeviate(sbparams.stars_seed+k+1)

                star_stamp,truth = make_a_star(ud=ud,wcs=wcs,psf=psf,affine=affine,optics=optics)
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
            
            this_sky_sigma = sbparams.sky_sigma*sbparams.exp_time
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
