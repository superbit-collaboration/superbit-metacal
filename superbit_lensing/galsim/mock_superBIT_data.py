# GalSim is opyright (c) 2012-2019 by the GalSim developers team on GitHub
# https://github.com/GalSim-developers
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
##
## TO DO: find a way to write n_obj, exptime, filter, and other useful info to a FITS header

import sys
import os
import math
import numpy
import logging
import time
import galsim
import galsim.des
import galsim.convolve
import pdb, pudb
import glob
import scipy
import yaml
import numpy as np
from functools import reduce
from astropy.table import Table
from mpi_helper import MPIHelper

class truth():
    
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
        self.fwhm = 0.0
        self.mom_size = 0.0
        self.n = 0.0
        self.hlr = 0.0
        self.scale_h_over_r = 0.0

def nfw_lensing(nfw_halo, pos, nfw_z_source):
    """ 
    - For some much-needed tidiness in main(), place the function that shears each galaxy here
    - Usage is borrowed from demo9.py
    - nfw_halo is galsim.NFW() object created in main()
    - pos is position of galaxy in image
    - nfw_z_source is background galaxy redshift
    """

    g1,g2 = nfw_halo.getShear( pos , nfw_z_source )
    nfw_shear = galsim.Shear(g1=g1,g2=g2)
    nfw_mu = nfw_halo.getMagnification( pos , nfw_z_source )

    if nfw_mu < 0:
        """                                                                                                                                                                               
        This doesn't seem to play well with MPI/batch scripting...                                                                                                                                      import warnings
        warnings.warn("Warning: mu < 0 means strong lensing!  Using mu=25.")
        """
        print("Warning: mu < 0 means strong lensing!  Using mu=25.")
        nfw_mu = 25
    elif nfw_mu > 25:
        print("Warning: mu > 25 means strong lensing!  Using mu=25.")
        nfw_mu = 25        
    
    return nfw_shear, nfw_mu

def make_a_galaxy(ud,wcs,affine,cosmos_cat,nfw,optics,sbparams):
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
    logger.debug('created galaxy position')
    
    ## Draw a Galaxy from scratch
    index = int(np.floor(ud()*len(cosmos_cat))) # This is a kludge to obain a repeatable index
    gal_z = cosmos_cat[index]['ZPDF']                    
    gal_flux = cosmos_cat[index][sbparams.bandpass]*sbparams.exp_time
    inclination = cosmos_cat[index]['phi_cosmos10']*galsim.radians 
    q = cosmos_cat[index]['q_cosmos10']
    # Cosmos HLR is in units of HST pix, convert to arcsec.
    half_light_radius=cosmos_cat[index]['hlr_cosmos10']*0.03*np.sqrt(q) 
    n = cosmos_cat[index]['n_sersic_cosmos10']
    logger.debug('galaxy z=%f flux=%f hlr=%f sersic_index=%f'%(gal_z,gal_flux,half_light_radius,n))

    ## InclinedSersic requires 0.3 < n < 6;
    ## set galaxy's n to another value if it falls outside this range
    if n<0.3:
        n=0.3
    elif n>6:
        n=4
    else:
        pass

    ## Very large HLRs will also make GalSim fail
    ## Set to a default, ~large but physical value.
    if half_light_radius > 2:
            half_light_radius = 2
    else:
        pass


    gal = galsim.InclinedSersic(n=n,
                                flux=gal_flux,
                                half_light_radius=half_light_radius,
                                inclination=inclination,
                                scale_h_over_r=q
                                )

   
    logger.debug('created galaxy')
            
    ## Apply a random rotation
    theta = ud()*2.0*numpy.pi*galsim.radians
    gal = gal.rotate(theta)    
    
    ## Get the reduced shears and magnification at this point
    try:
        nfw_shear, mu = nfw_lensing(nfw, uv_pos, gal_z)
        g1=nfw_shear.g1; g2=nfw_shear.g2
        gal = gal.lens(g1, g2, mu)
        
    except:
        print("could not lens galaxy at z = %f, setting default values..." % gal_z)
        g1 = 0.0; g2 = 0.0
        mu = 1.0

    jitter_psf = galsim.Gaussian(flux=1,fwhm=sbparams.jitter_fwhm)
    final=galsim.Convolve([jitter_psf,gal,optics])
    
    logger.debug("Convolved star and PSF at galaxy position")
    
    stamp = final.drawImage(wcs=wcs.local(image_pos))
    stamp.setCenter(image_pos.x,image_pos.y)
    logger.debug('drew & centered galaxy!')    
    galaxy_truth=truth()
    galaxy_truth.ra=ra.deg; galaxy_truth.dec=dec.deg
    galaxy_truth.x=image_pos.x; galaxy_truth.y=image_pos.y
    galaxy_truth.g1=g1; galaxy_truth.g2=g2
    galaxy_truth.mu = mu; galaxy_truth.z = gal_z
    galaxy_truth.flux = stamp.added_flux
    galaxy_truth.n = n; galaxy_truth.hlr = half_light_radius
    #galaxy_truth.inclination = inclination.deg # storing in degrees for human readability
    galaxy_truth.scale_h_over_r = q

    logger.debug('created truth values')

    try:
        galaxy_truth.fwhm=final.calculateFWHM()
    except galsim.errors.GalSimError:
        logger.debug('fwhm calculation failed')
        galaxy_truth.fwhm=-9999.0

    try:
        galaxy_truth.mom_size=stamp.FindAdaptiveMom().moments_sigma
    except galsim.errors.GalSimError:
        logger.debug('sigma calculation failed')
        galaxy_truth.mom_size=-9999.
        
    logger.debug('stamp made, moving to next galaxy')
    return stamp, galaxy_truth

def make_cluster_galaxy(ud, wcs,affine, centerpix, cluster_cat, optics, sbparams):
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
    image_pos = galsim.PositionD(x+centerpix.x+(ud()-0.5)*50,y+centerpix.y+(ud()-0.5)*50)
    world_pos = wcs.toWorld(image_pos)
    ra=world_pos.ra; dec = world_pos.dec
   
    # We also need this in the tangent plane, which we call "world coordinates" here,
    # This is still an x/y corrdinate 
    uv_pos = affine.toWorld(image_pos)
   
    # Fixed redshift for cluster galaxies
    gal_z = sbparams.nfw_z_halo
    g1 = 0.0; g2 = 0.0
    mu = 1.0
    
    # Create galaxy    
    gal = cluster_cat.makeGalaxy(gal_type='parametric', rng=ud)
    logger.debug('created cluster galaxy')

    # Apply a random rotation
    theta = ud()*2.0*numpy.pi*galsim.radians
    gal = gal.rotate(theta)
    
    # The "magnify" is just for drama; factor of 1.2207 turns us into e-
    gal *= (sbparams.flux_scaling*1.2207)
    gal.magnify(4)
    logger.debug('rescaled galaxy with scaling factor %f' % sbparams.flux_scaling)

    jitter_psf = galsim.Gaussian(flux=1,fwhm=sbparams.jitter_fwhm)
    final=galsim.Convolve([jitter_psf,gal,optics])

    logger.debug("Convolved star and PSF at galaxy position")

    
    # Draw galaxy image
    this_stamp_image = galsim.Image(128, 128,wcs=wcs.local(image_pos))
    #cluster_stamp = final.drawImage(bandpass,image=this_stamp_image)
    cluster_stamp = final.drawImage(image=this_stamp_image)

    #cluster_stamp.setCenter(ix_nominal,iy_nominal)
    cluster_stamp.setCenter(image_pos.x,image_pos.y)

    logger.debug('drew & centered galaxy!')    

    cluster_galaxy_truth=truth()
    cluster_galaxy_truth.ra=ra.deg; cluster_galaxy_truth.dec=dec.deg
    #cluster_galaxy_truth.x=ix_nominal; cluster_galaxy_truth.y=iy_nominal
    cluster_galaxy_truth.x=image_pos.x; cluster_galaxy_truth.y=image_pos.y
    cluster_galaxy_truth.g1=g1; cluster_galaxy_truth.g2=g2
    cluster_galaxy_truth.mu = mu; cluster_galaxy_truth.z = gal_z
    cluster_galaxy_truth.flux = cluster_stamp.added_flux
    logger.debug('created truth values')
    
    try:
        cluster_galaxy_truth.fwhm=final.calculateFWHM()
    except galsim.errors.GalSimError:
        logger.debug('fwhm calculation failed')
        cluster_galaxy_truth.fwhm=-9999.0

    try:
        cluster_galaxy_truth.mom_size=cluster_stamp.FindAdaptiveMom().moments_sigma
    except:
        logger.debug('sigma calculation failed')
        cluster_galaxy_truth.mom_size=-9999.
    
    return cluster_stamp, cluster_galaxy_truth


def make_a_star(ud, wcs, affine, optics, sbparams):
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
    #flux_dist = galsim.DistDeviate(ud, function = lambda x:x**-1.5, x_min = 799.2114, x_max = 890493.9)
    flux_dist = galsim.DistDeviate(ud, function = lambda x:x**-1.5, x_min = 533, x_max = 59362)
    star_flux = flux_dist()*1.2207
    
    # Generate PSF at location of star, convolve with optical model to make a star
    deltastar = galsim.DeltaFunction(flux=star_flux)  
    jitter_psf = galsim.Gaussian(flux=1,fwhm=sbparams.jitter_fwhm)
    star=galsim.Convolve([jitter_psf,deltastar,optics])

    star_stamp = star.drawImage(wcs=wcs.local(image_pos)) # before it was scale = 0.206, and that was bad!
    star_stamp.setCenter(image_pos.x,image_pos.y)
    
    star_truth=truth()
    star_truth.ra = ra.deg; star_truth.dec = dec.deg
    star_truth.x = image_pos.x; star_truth.y =image_pos.y

    try:
        star_truth.fwhm=star.calculateFWHM()
    except galsim.errors.GalSimError:
        logger.debug('fwhm calculation failed')
        star_truth.fwhm=-9999.0

    try:
        star_truth.mom_size=star_stamp.FindAdaptiveMom().moments_sigma
    except galsim.errors.GalSimError:
        logger.debug('sigma calculation failed')
        star_truth.mom_size=-9999.

    return star_stamp, star_truth

class SuperBITParameters:
        def __init__(self, config_file=None, argv=None):
            """
            Initialize default params and overwirte with config_file params and / or commmand line
            parameters.
            """
            # Check for config_file params to overwrite defaults
            if config_file is not None:
                logger.info('Loading parameters from %s' % (config_file))
                self._load_config_file(config_file)
            else:
                # Define some default default parameters below.
                # These are used in the absence of a .yaml config_file or command line args.
                logger.info('Loading default config file...')
                self._load_config_file('config_files/superbit_parameters_forecast.yaml')

            # Check for command line args to overwrite config_file and / or defaults
            if argv is not None:
                self._load_command_line(argv)

            return

        def _load_config_file(self, config_file):
            """
            Load parameters from configuration file. Only parameters that exist in the config_file
            will be overwritten.
            """
            with open(config_file) as fsettings:
                config = yaml.load(fsettings, Loader=yaml.FullLoader)
            self._load_dict(config)

            return

        def _args_to_dict(self, argv):
            """
            Converts a command line argument array to a dictionary.
            """
            d = {}
            for arg in argv[1:]:
                optval = arg.split("=", 1)
                option = optval[0]
                value = optval[1] if len(optval) > 1 else None
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

        def _load_dict(self, d):
            """
            Load parameters from a dictionary.
            """
            for (option, value) in d.items():
                if option == "config_file":
                    logger.info('Overriding default config!\nLoading parameters from %s' % (value))
                    self._load_config_file(str(value))
                elif option == "pixel_scale":     
                    self.pixel_scale = float(value)
                elif option == "sky_bkg":        
                    self.sky_bkg = float(value) 
                elif option == "sky_sigma":     
                    self.sky_sigma = float(value)
                elif option == "gain":          
                    self.gain = float(value)   
                elif option == "read_noise":
                    self.read_noise = float(value)
                elif option == "dark_current":
                    self.dark_current = float(value)
                elif option == "dark_current_std":
                    self. dark_current_std = float(value)
                elif option == "image_xsize":   
                    self.image_xsize = int(value)    
                elif option == "image_ysize":   
                    self.image_ysize = int(value)    
                elif option == "center_ra":     
                    self.center_ra = float(value) * galsim.hours
                elif option == "center_dec": 
                    self.center_dec = float(value) * galsim.degrees
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
                elif option == "mass": 
                    self.mass = float(value)         
                elif option == "nfw_conc":   
                    self.nfw_conc = float(value) 
                elif option == "nfw_z_halo": 
                    self.nfw_z_halo = float(value)
                elif option == "omega_m":   
                    self.omega_m = float(value)  
                elif option == "omega_lam":
                    self.omega_lam = float(value)
                elif option == "cosmosdir":
                    self.cosmosdir = str(value)
                elif option == "datadir":
                    self.datadir = str(value)
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
                elif option == "nstruts":     
                    self.nstruts = int(value)     
                elif option == "nstruts":     
                    self.nstruts = int(value)     
                elif option == "strut_thick":     
                    self.strut_thick = float(value)     
                elif option == "strut_theta":  
                    self.strut_theta = float(value)        
                elif option == "obscuration":  
                    self.obscuration = float(0.380)
                elif option == "bandpass":
                    self.bandpass=str(value)
                elif option == "jitter_fwhm":
                    self.jitter_fwhm=float(value)
                else:
                    raise ValueError("Invalid parameter \"%s\" with value \"%s\"" % (option, value))

            # Derive image parameters from the base parameters
            self.image_xsize_arcsec = self.image_xsize * self.pixel_scale 
            self.image_ysize_arcsec = self.image_ysize * self.pixel_scale 
            self.center_coords = galsim.CelestialCoord(self.center_ra,self.center_dec)
            self.strut_angle = self.strut_theta * galsim.degrees
            
            # OUR NEW CATALOG IS ALREADY SCALED TO SUPERBIT 0.5 m MIRROR.  
            # Scaling used for cluster galaxies, which are drawn from default GalSim-COSMOS catalog   
            hst_eff_area = 2.4**2 #* (1.-0.33**2)
            sbit_eff_area = self.tel_diam**2 #* (1.-0.380**2) 
            self.flux_scaling = (sbit_eff_area/hst_eff_area) * self.exp_time * self.gain 
            if not hasattr(self,'jitter_fwhm'):
                self.jitter_fwhm = 0.1

# function to help with reducing MPI results from each process to single result
def combine_images(im1, im2):
    """Combine two galsim.Image objects into one."""
    # easy since they support +. Try using in-place operation to reduce memory
    im1 += im2
    return im1

def combine_catalogs(t1, t2):
    """Combine two galsim.OutputCatalog objects into one"""
    # as far as I can tell, they expose no way of doing this aside from messing
    # with the internal lists directly.
    t1.rows.extend(t2.rows)
    t1.sort_keys.extend(t2.sort_keys)
    return t1

def main(argv):
    """
    Make images using model PSFs and galaxy cluster shear:
      - The galaxies come from a processed COSMOS 2015 Catalog, scaled to match
        anticipated SuperBIT 2021 observations
      - The galaxy shape parameters are assigned in a probabilistic way through matching
        galaxy fluxes and redshifts to similar GalSim-COSMOS galaxies (see A. Gill+ 2021)
    """
    
    global logger
    logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
    logger = logging.getLogger("mock_superbit_data")

    M = MPIHelper()

    # Define some parameters we'll use below.
    sbparams = SuperBITParameters(argv=argv)
    
    # Set up the NFWHalo:
    nfw = galsim.NFWHalo(mass=sbparams.mass, conc=sbparams.nfw_conc, redshift=sbparams.nfw_z_halo,
                     omega_m=sbparams.omega_m, omega_lam=sbparams.omega_lam)

    logger.info('Set up NFW halo for lensing')

    # Read in galaxy catalog, as well as catalog containing
    # information from COSMOS fits like redshifts, hlr, etc.   
    # cosmos_cat = galsim.COSMOSCatalog(sbparams.cat_file_name, dir=sbparams.datadir)
    # fitcat = Table.read(os.path.join(sbparams.cosmosdir, sbparams.fit_file_name))

    cosmos_cat = Table.read(os.path.join(sbparams.datadir,sbparams.cat_file_name))
    logger.info('Read in %d galaxies from catalog and associated fit info', len(cosmos_cat))
    
    try:
        cluster_cat = galsim.COSMOSCatalog(sbparams.cluster_cat_name, dir=sbparams.datadir)
    except:
        cluster_cat = galsim.COSMOSCatalog(sbparams.cluster_cat_name)
    #logger.debug('Read in %d cluster galaxies from catalog' % cosmos_cat.nobjects)
    

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
        
    ###
    ### MAKE SIMULATED OBSERVATIONS 
    ### ITERATE n TIMES TO MAKE n SEPARATE IMAGES
    ###

        
    for i in numpy.arange(1,sbparams.nexp+1):          
        # get MPI processes in sync at start of each image
        M.barrier()
        
        #rng = galsim.BaseDeviate(sbparams.noise_seed+i)

        try:
            timescale=str(sbparams.exp_time)
            outname=''.join(['superbit_gaussJitter_',str(i).zfill(3),'.fits'])
            truth_file_name=''.join([sbparams.outdir, '/truth_gaussJitter_', str(i).zfill(3), '.dat'])
            file_name = os.path.join(sbparams.outdir, outname)

        except galsim.errors.GalSimError:
            print("naming failed, check path")
            pdb.set_trace()

            
        # Setting up a truth catalog
        names = [ 'gal_num', 'x_image', 'y_image',
                    'ra', 'dec', 'nfw_g1', 'nfw_g2', 'nfw_mu', 'redshift','flux','truth_fwhm','truth_mom',
                      'n','hlr','scale_h_over_r']
        types = [ int, float, float, float,float,float,
                    float, float, float, float, float, float,
                      float, float, float]
        truth_catalog = galsim.OutputCatalog(names, types)

        
        # Set up the image:
        full_image = galsim.ImageF(sbparams.image_xsize, sbparams.image_ysize)
        sky_level = sbparams.exp_time * sbparams.sky_bkg
        full_image.fill(sky_level)
        full_image.setOrigin(0,0)
        
        
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

        
        ## Now let's read in the PSFEx PSF model, if using.
        ## We read the image directly into an InterpolatedImage GSObject,
        ## so we can manipulate it as needed 
        #psf_wcs=wcs
        #psf = galsim.des.DES_PSFEx(psf_filen,wcs=psf_wcs)
        #logger.info('Constructed PSF object from PSFEx file')

        #####
        ## Loop over galaxy objects:
        #####
        
        # get local range to iterate over in this process
        local_start, local_end = M.mpi_local_range(sbparams.nobj)
        for k in range(local_start, local_end):
            time1 = time.time()
            
            # The usual random number generator using a different seed for each galaxy.
            ud = galsim.UniformDeviate(sbparams.galobj_seed+k+1)

            try: 
                # make single galaxy object
                stamp,truth = make_a_galaxy(ud=ud,wcs=wcs,affine=affine,
                        cosmos_cat=cosmos_cat,optics=optics,nfw=nfw,
                        sbparams=sbparams)                
                # Find the overlapping bounds:
                bounds = stamp.bounds & full_image.bounds
                
                # We need to keep track of how much variance we have currently in the image, so when
                # we add more noise, we can omit what is already there.

                # noise_image[bounds] += truth.variance
        
                # Finally, add the stamp to the full image.
            
                full_image[bounds] += stamp[bounds]
                time2 = time.time()
                tot_time = time2-time1
                logger.info('Galaxy %d positioned relative to center t=%f s\n',
                            k, tot_time)
                this_flux=numpy.sum(stamp.array)
                row = [ k,truth.x, truth.y, truth.ra, truth.dec, truth.g1, truth.g2, truth.mu,truth.z,
                            this_flux,truth.fwhm, truth.mom_size,
                            truth.n, truth.hlr, truth.scale_h_over_r]
                truth_catalog.addRow(row)
            except galsim.errors.GalSimError:
                logger.info('Galaxy %d has failed, skipping...',k)

        #####
        ### Inject cluster galaxy objects:
        #####
     
        center_coords = galsim.CelestialCoord(sbparams.center_ra,sbparams.center_dec)
        centerpix = wcs.toImage(center_coords)
        
        # get local range to iterate over in this process
        local_start, local_end = M.mpi_local_range(sbparams.nclustergal)
        for k in range(local_start, local_end):

            time1 = time.time()
        
            # The usual random number generator using a different seed for each galaxy.
            ud = galsim.UniformDeviate(sbparams.cluster_seed+k+1)
            
            try: 
                # make single galaxy object
                cluster_stamp,truth = make_cluster_galaxy(ud=ud,wcs=wcs,affine=affine,
                                                              centerpix=centerpix,
                                                              cluster_cat=cluster_cat,
                                                              optics=optics,
                                                              sbparams=sbparams)                
                # Find the overlapping bounds:
                bounds = cluster_stamp.bounds & full_image.bounds
                
                # We need to keep track of how much variance we have currently in the image, so when
                # we add more noise, we can omit what is already there.
        
                #noise_image[bounds] += truth.variance
        
                # Finally, add the stamp to the full image.
                
                full_image[bounds] += cluster_stamp[bounds]
                time2 = time.time()
                tot_time = time2-time1
                logger.info('Cluster galaxy %d positioned relative to center t=%f s\n',
                                k, tot_time)
                this_flux=numpy.sum(cluster_stamp.array)
                row = [ k,truth.x, truth.y, truth.ra, truth.dec, truth.g1, truth.g2, truth.mu,truth.z,
                            this_flux,truth.fwhm,truth.mom_size,
                            truth.n, truth.hlr,truth.scale_h_over_r]
                truth_catalog.addRow(row)
            except galsim.errors.GalSimError:
                logger.info('Cluster galaxy %d has failed, skipping...',k)
                
        
            
        #####
        ### Now repeat process for stars!
        #####
        
        # get local range to iterate over in this process
        local_start, local_end = M.mpi_local_range(sbparams.nstars)
        for k in range(local_start, local_end):
            time1 = time.time()
            ud = galsim.UniformDeviate(sbparams.stars_seed+k+1)

            star_stamp,truth = make_a_star(ud=ud, wcs=wcs, affine=affine, 
                    optics=optics, sbparams=sbparams)
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
                            truth.z, this_flux,truth.fwhm,truth.mom_size,
                            truth.n, truth.hlr,truth.scale_h_over_r]
                truth_catalog.addRow(row)
                
            except galsim.errors.GalSimError:
                logger.info('Star %d has failed, skipping...',k)

        # Gather results from MPI processes, reduce to single result on root
        # Using same names on left and right sides is hiding lots of MPI magic
        full_image = M.gather(full_image)
        truth_catalog = M.gather(truth_catalog)
        if M.is_mpi_root():
            full_image = reduce(combine_images, full_image)
            truth_catalog = reduce(combine_catalogs, truth_catalog)
        else:
            # do the adding of noise and writing to disk entirely on root
            # root and the rest meet again at barrier at start of loop
            continue
        

        # The first thing to do is to make the Gaussian noise uniform across the whole image.
        
        # Add dark current
        logger.info('Adding Dark current')
        dark_noise = sbparams.dark_current * sbparams.exp_time
        full_image += dark_noise
        
        # Add ccd noise
        logger.info('Adding CCD noise')
        noise = galsim.CCDNoise(
            sky_level=0, gain=sbparams.gain,
            read_noise=sbparams.read_noise)
        full_image.addNoise(noise)
        
        logger.debug('Added noise to final output image')
        if not os.path.exists(os.path.dirname(file_name)):
            os.makedirs(os.path.dirname(file_name))
        full_image.write(file_name)

     
        # Write truth catalog to file. 
        truth_catalog.write(truth_file_name)
        logger.info('Wrote image to %r',file_name)

            
    logger.info(' ')
    logger.info('completed all images')
    logger.info(' ')

if __name__ == "__main__":
    main(sys.argv)
