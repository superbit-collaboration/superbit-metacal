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
import logging
import time
import galsim
import galsim.des
import galsim.convolve
import pdb
from glob import glob
import pickle
import scipy
import yaml
import numpy as np
import fitsio
from astropy.io import fits
from numpy.random import SeedSequence, default_rng
from functools import reduce
from astropy.table import Table
from argparse import ArgumentParser
from mpi_helper import MPIHelper
from multiprocessing import Pool

import superbit_lensing.utils as utils

def parse_args():
    parser = ArgumentParser()

    parser.add_argument('config_file', action='store', type=str,
                        help='Configuration file for mock sims')
    parser.add_argument('-run_name', action='store', type=str, default='',
                        help='Name of mock simulation run')
    parser.add_argument('-outdir', action='store', type=str,
                        help='Output directory of simulated files')
    parser.add_argument('-ncores', action='store', type=int, default=1,
                        help='Number of cores to use for multiproessing')
    parser.add_argument('--mpi', action='store_true', default=False,
                        help='Use to turn on mpi')
    parser.add_argument('--clobber', action='store_true', default=False,
                        help='Turn on to overwrite existing files')
    parser.add_argument('--vb', action='store_true', default=False,
                        help='Turn on for verbose prints')

    return parser.parse_args()

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

        self.cosmos_index = -1
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
        print("Warning: mu < 0 means strong lensing!  Using mu=25.")
        nfw_mu = 25
    elif nfw_mu > 25:
        print("Warning: mu > 25 means strong lensing!  Using mu=25.")
        nfw_mu = 25

    return nfw_shear, nfw_mu

def make_obj_runner(batch_indices, *args, **kwargs):
    '''
    Handles the batch running of make_obj() over multiple cores
    '''

    res = []
    for i in batch_indices:
        res.append(make_obj(i, *args, **kwargs))

    return res

def make_obj(i, obj_type, *args, **kwargs):
    '''
    Runs the approrpriate "make_a_{obj}" function given object type.
    Particularly useful for multiprocessing wrappers
    '''

    logprint = args[-1]

    func = None

    func_map = {
        'gal': make_a_galaxy,
        'cluster_gal': make_cluster_galaxy,
        'star': make_a_star
    }

    obj_types = func_map.keys()
    if obj_type not in obj_types:
        raise ValueError(f'Object type must be one of {obj_types}!')

    func = func_map[obj_type]


    try:
        obj_index = int(i)
        logprint(f'Starting {obj_type} {i}')
        stamp, truth = func(*args, **kwargs,obj_index=i)
        logprint(f'{obj_type} {i} completed succesfully')
    except galsim.errors.GalSimError:
        logprint(f'{obj_type} {i} has failed, skipping...')
        return i, None, None

    return i, stamp, truth

def combine_objs(make_obj_outputs, full_image, truth_catalog, exp_num):
    '''
    (i, stamps, truths) are the output of make_obj
    exp_num is the exposure number. Only add to truth table if == 1
    '''

    # flatten outputs into 1 list
    make_obj_outputs = [item for sublist in make_obj_outputs
                        for item in sublist]

    for i, stamp, truth in make_obj_outputs:

        if (stamp is None) or (truth is None):
            continue

        # Find the overlapping bounds:
        bounds = stamp.bounds & full_image.bounds

        # Finally, add the stamp to the full image.
        try:
            full_image[bounds] += stamp[bounds]
        except galsim.errors.GalSimBoundsError as e:
            print(e)

        this_flux = np.sum(stamp.array)

        if exp_num == 1:
            row = [i, truth.cosmos_index, truth.x, truth.y,
                   truth.ra, truth.dec,
                   truth.g1, truth.g2,
                   truth.mu,truth.z,
                   this_flux, truth.fwhm, truth.mom_size,
                   truth.n, truth.hlr, truth.scale_h_over_r,
                   truth.obj_class
                   ]
            truth_catalog.addRow(row)

    return full_image, truth_catalog

def make_a_galaxy(ud, wcs, affine, cosmos_cat, nfw, psf, sbparams, logprint, obj_index=None):
    """
    Method to make a single galaxy object and return stamp for
    injecting into larger GalSim image
    """

    # Choose a random RA, Dec around the sky_center.
    # Note that for this to come out close to a square shape, we need to account for the
    # cos(dec) part of the metric: ds^2 = dr^2 + r^2 d(dec)^2 + r^2 cos^2(dec) d(ra)^2
    # So need to calculate dec first.
    dec = sbparams.center_dec + (ud()-0.5) * sbparams.image_ysize_arcsec * galsim.arcsec
    ra = sbparams.center_ra + (ud()-0.5) * sbparams.image_xsize_arcsec / np.cos(dec) * galsim.arcsec
    world_pos = galsim.CelestialCoord(ra,dec)

    # We will need the image position as well, so use the wcs to get that
    image_pos = wcs.toImage(world_pos)

    # We also need this in the tangent plane, which we call "world coordinates" here.
    # This is still an x/y corrdinate
    uv_pos = affine.toWorld(image_pos)
    logprint.debug('created galaxy position')

    ## Draw a Galaxy from scratch
    ## Note units of sbparams.gain is assumed to be be e-/ADU.
    index = int(np.floor(ud()*len(cosmos_cat)))
    gal_z = cosmos_cat[index]['ZPDF']
    gal_flux = cosmos_cat[index][sbparams.bandpass] * sbparams.exp_time / sbparams.gain
    phi = cosmos_cat[index]['c10_sersic_fit_phi'] * galsim.radians
    q = cosmos_cat[index]['c10_sersic_fit_q']
    # Cosmos HLR is in units of HST pix, convert to arcsec.
    half_light_radius=cosmos_cat[index]['c10_sersic_fit_hlr']*0.03*np.sqrt(q)
    n = cosmos_cat[index]['c10_sersic_fit_n']
    logprint.debug(f'galaxy z={gal_z} flux={gal_flux} hlr={half_light_radius} ' + \
                   f'sersic_index={n}')

    # Sersic class requires index n >= 0.3
    if (n < 0.3):
        n = 0.3

    gal = galsim.Sersic(n = n,
                        flux = gal_flux,
                        half_light_radius = half_light_radius)

    gal = gal.shear(q = q, beta = phi)
    logprint.debug('created galaxy')

    ## Apply a random rotation
    theta = ud()*2.0*np.pi*galsim.radians
    gal = gal.rotate(theta)

    ## Apply a random rotation
    theta = ud()*2.0*np.pi*galsim.radians
    gal = gal.rotate(theta)

    ## Get the reduced shears and magnification at this point
    try:
        nfw_shear, mu = nfw_lensing(nfw, uv_pos, gal_z)
        g1=nfw_shear.g1; g2=nfw_shear.g2
        gal = gal.lens(g1, g2, mu)
    except galsim.errors.GalSimError:
        logprint(f'could not lens galaxy at z = {gal_z}, setting default values...')
        g1 = 0.0; g2 = 0.0
        mu = 1.0

    final = galsim.Convolve([psf, gal])

    logprint.debug('Convolved star and PSF at galaxy position')

    stamp = final.drawImage(wcs=wcs.local(image_pos))
    stamp.setCenter(image_pos.x,image_pos.y)
    logprint.debug('drew & centered galaxy!')
    galaxy_truth=truth()
    galaxy_truth.cosmos_index = index
    galaxy_truth.ra=ra.deg; galaxy_truth.dec=dec.deg
    galaxy_truth.x=image_pos.x; galaxy_truth.y=image_pos.y
    galaxy_truth.g1=g1; galaxy_truth.g2=g2
    galaxy_truth.mu = mu; galaxy_truth.z = gal_z
    galaxy_truth.flux = stamp.added_flux
    galaxy_truth.n = n; galaxy_truth.hlr = half_light_radius
    #galaxy_truth.inclination = inclination.deg # storing in degrees for human readability
    galaxy_truth.scale_h_over_r = q
    galaxy_truth.obj_class = 'gal'

    logprint.debug('created truth values')

    try:
        galaxy_truth.fwhm=final.calculateFWHM()
    except galsim.errors.GalSimError:
        logprint.debug('fwhm calculation failed')
        galaxy_truth.fwhm=-9999.0

    try:
        galaxy_truth.mom_size=stamp.FindAdaptiveMom().moments_sigma
    except galsim.errors.GalSimError:
        logprint.debug('sigma calculation failed')
        galaxy_truth.mom_size=-9999.

    logprint.debug('stamp made, moving to next galaxy')
    return stamp, galaxy_truth

def make_cluster_galaxy(ud, wcs,affine, centerpix, cluster_cat, psf, sbparams, logprint, obj_index=None):
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
    image_pos = galsim.PositionD(x+centerpix.x+(ud()-0.5)*100,y+centerpix.y+(ud()-0.5)*100)
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
    logprint.debug('created cluster galaxy')

    # Apply a random rotation
    theta = ud()*2.0*np.pi*galsim.radians
    gal = gal.rotate(theta)

    # The "magnify" is just for drama
    gal *= sbparams.flux_scaling
    gal.magnify(2)
    logprint.debug(f'rescaled galaxy with scaling factor {sbparams.flux_scaling}')

    final = galsim.Convolve([psf, gal])

    logprint.debug('Convolved star and PSF at galaxy position')


    # Draw galaxy image
    this_stamp_image = galsim.Image(128, 128,wcs=wcs.local(image_pos))
    #cluster_stamp = final.drawImage(bandpass,image=this_stamp_image)
    cluster_stamp = final.drawImage(image=this_stamp_image)

    #cluster_stamp.setCenter(ix_nominal,iy_nominal)
    cluster_stamp.setCenter(image_pos.x,image_pos.y)

    logprint.debug('drew & centered galaxy!')

    cluster_galaxy_truth=truth()
    cluster_galaxy_truth.ra=ra.deg; cluster_galaxy_truth.dec=dec.deg
    #cluster_galaxy_truth.x=ix_nominal; cluster_galaxy_truth.y=iy_nominal
    cluster_galaxy_truth.x=image_pos.x; cluster_galaxy_truth.y=image_pos.y
    cluster_galaxy_truth.g1=g1; cluster_galaxy_truth.g2=g2
    cluster_galaxy_truth.mu = mu; cluster_galaxy_truth.z = gal_z
    cluster_galaxy_truth.flux = cluster_stamp.added_flux
    cluster_galaxy_truth.obj_class = 'cluster_gal'
    logprint.debug('created truth values')

    try:
        cluster_galaxy_truth.fwhm=final.calculateFWHM()
    except galsim.errors.GalSimError:
        logprint.debug('fwhm calculation failed')
        cluster_galaxy_truth.fwhm=-9999.0

    try:
        cluster_galaxy_truth.mom_size=cluster_stamp.FindAdaptiveMom().moments_sigma
    except:
        logprint.debug('sigma calculation failed')
        cluster_galaxy_truth.mom_size=-9999.

    return cluster_stamp, cluster_galaxy_truth


def make_a_star(ud, pud, k, wcs, affine, psf, sbparams, logprint, obj_index=None):
    """
    makes a star-like object for injection into larger image.
    """

    # Choose a random RA, Dec around the sky_center.
    dec = sbparams.center_dec + (ud()-0.5) * sbparams.image_ysize_arcsec * galsim.arcsec
    ra = sbparams.center_ra + (ud()-0.5) * sbparams.image_xsize_arcsec / np.cos(dec) * galsim.arcsec
    world_pos = galsim.CelestialCoord(ra,dec)

    # We will need the image position as well, so use the wcs to get that
    image_pos = wcs.toImage(world_pos)

    # We also need this in the tangent plane, which we call "world coordinates" here,
    # This is still an x/y corrdinate
    uv_pos = affine.toWorld(image_pos)

    # Draw star flux at random; based on either a semi-analytic distribution or GAIA stars
    # Default to blue stars, which are plenty bright

    index = obj_index - 1

    if sbparams.star_cat is not None:
        if sbparams.bandpass in ['crates_lum', 'crates_shape']:
            star_flux = sbparams.star_cat['bitflux_electrons_lum'][index]

        elif sbparams.bandpass=='crates_b':
            star_flux = sbparams.star_cat['bitflux_electrons_b'][index]

        else:
            raise NotImplementedError('Star catalog sampling only implemented ' +\
                                      'for crates_shape and crates_b!')

        star_flux *= sbparams.exp_time / sbparams.gain

    else:
        pud = np.random.default_rng()
        p = pud.power(0.4)
        flux_p = (10/p) - 10.
        star_flux = flux_p

        if sbparams.bandpass=='crates_b':
            star_flux *= 0.8271672
        else:
            raise NotImplementedError('Star power law only implemented for crates_b!')

    # Generate PSF at location of star, convolve with optical model to make a star
    deltastar = galsim.DeltaFunction(flux=star_flux)
    star = galsim.Convolve([psf, deltastar])

    star_stamp = star.drawImage(wcs=wcs.local(image_pos)) # before it was scale = 0.206, and that was bad!
    star_stamp.setCenter(image_pos.x, image_pos.y)

    star_truth = truth()
    star_truth.ra = ra.deg; star_truth.dec = dec.deg
    star_truth.x = image_pos.x; star_truth.y = image_pos.y
    star_truth.obj_class = 'star'

    try:
        star_truth.fwhm = star.calculateFWHM()
    except galsim.errors.GalSimError:
        logprint.debug('fwhm calculation failed')
        star_truth.fwhm =- 9999.0

    try:
        star_truth.mom_size=star_stamp.FindAdaptiveMom().moments_sigma
    except galsim.errors.GalSimError:
        logprint.debug('sigma calculation failed')
        star_truth.mom_size=-9999.

    return star_stamp, star_truth

class SuperBITParameters:
    __req_params = ['run_name', 'outdir']
    __req_defaults = ['', None]

    def __init__(self, config_file, logprint, args=None):
        """
        Initialize default params and overwirte with config_file params and / or commmand line
        parameters.
        """

        self.logprint = logprint

        self.logprint(f'Loading parameters from {config_file}')
        self._load_config_file(config_file)

        # Check for command line args to overwrite config_file and / or defaults
        if args is not None:
            self._load_command_line(args)

        # Check that certain params are set either on command line or in config
        utils.check_req_params(self, self.__req_params, self.__req_defaults)

        self._set_seeds()

        # Set up stellar injection
        self._setup_stars()

        # Set up dark frame
        self._setup_darks()

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

    def _args_to_dict(self, args):
        """
        Converts argparse command line arguments to a dictionary.
        """
        return vars(args)
        # d = {}
        # for arg in args[1:]:
        #     optval = arg.split("=", 1)
        #     option = optval[0]
        #     value = optval[1] if len(optval) > 1 else None
        #     d[option] = value

        # return d

    def _load_command_line(self, args):
        """
        Load parameters from the command line argumentts. Only parameters that are provided in
        the command line will be overwritten.
        """
        self.logprint('Processing command line args')
        # Parse arguments here
        self._load_dict(self._args_to_dict(args))

        return

    def _load_dict(self, d):
        """
        Load parameters from a dictionary.
        """
        ignore = ['config_file', 'vb']
        for (option, value) in d.items():
            if option in ignore:
                # config file already processed above
                continue
            # This would be much simpler:
            # else:
                # setattr(self, option, value)
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
                self.dark_current_std = float(value)
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
                self.nstars = value
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
            elif option == "star_cat_name":
                self.star_cat_name = str(value)
            elif option == "bp_file":
                self.bp_file = str(value)
            elif option == "outdir":
                self.outdir = str(value)
            elif option == "master_seed":
                self.master_seed = int(value)
            elif option == "noise_seed":
                self.noise_seed = int(value)
            elif option == "dithering_seed":
                self.dithering_seed = int(value)
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
                self.obscuration = float(value)
            elif option == "bandpass":
                self.bandpass=str(value)
            elif option == "jitter_fwhm":
                self.jitter_fwhm=float(value)
            elif option == "run_name":
                self.run_name=str(value)
            elif option == "clobber":
                self.clobber=bool(value)
            elif option == "mpi":
                self.mpi = bool(value)
            elif option == "ncores":
                self.ncores = int(value)
            elif option == "use_optics":
                self.use_optics = bool(value)
            elif option == "sample_gaia_cats":
                self.sample_gaia_cats = bool(value)
            elif option == "gaia_dir":
                self.gaia_dir = str(value)
            elif option == "sample_darks":
                self.sample_darks = bool(value)
            elif option == "dark_dir":
                self.dark_dir = str(value)
            elif option == "dark_image_name":
                self.dark_image_name = str(value)
            elif option == "noise_seed":
                try:
                    self.noise_seed = int(value)
                except:
                    self.noise_seed = None
            elif option == "galobj_seed":
                try:
                    self.galobj_seed = int(value)
                except:
                    self.galobj_seed = None
            elif option == "cluster_seed":
                try:
                    self.cluster_seed = int(value)
                except:
                    self.cluster_seed = None
            elif option == "stars_seed":
                try:
                    self.stars_seed = int(value)
                except:
                    self.stars_seed = None
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
        self.flux_scaling = (sbit_eff_area/hst_eff_area) * self.exp_time
        if not hasattr(self,'jitter_fwhm'):
            self.jitter_fwhm = 0.1

        return

    def _setup_stars(self):
        valid_args = ['nstars', 'star_cat_name', 'sample_gaia_cats', 'gaia_dir']

        for arg in valid_args:
            if not hasattr(self, arg):
                setattr(self, arg, None)

        assert (self.nstars is not None) or \
               (self.star_cat_name is not None) or \
               (self.sample_gaia_cats is not None)

        if (self.star_cat_name is not None) and (self.sample_gaia_cats is not None):
            raise AttributeError('Cannot set both `star_cat_name` and ' +\
                                 '`sample_gaia_cats`!')

        # if sampling from a GAIA cat, do that first
        if self.sample_gaia_cats is True:
            if self.gaia_dir is None:
                raise AttributeError('Must set `gaia_dir` if sampling from gaia cats!')

            gaia_cats = glob(f'{self.gaia_dir}/GAIA*.csv')
            sample_gaia_rng = np.random.default_rng(self.stars_seed)
            self.star_cat_name = sample_gaia_rng.choice(gaia_cats)

        if self.star_cat_name is not None:
            star_fname = os.path.join(self.datadir, self.star_cat_name)
            self.star_cat = Table.read(star_fname)
        else:
            self.star_cat = None

        if self.nstars is None:
            self.nstars = len(self.star_cat)

        return

    def _setup_darks(self):
        '''
        Grab darks
        '''
        valid_args = ['dark_image_name', 'sample_darks', 'dark_dir']

        for arg in valid_args:
            if not hasattr(self, arg):
                setattr(self, arg, None)

        assert (self.dark_image_name is not None) or \
               (self.sample_darks is not None) or \
               (self.dark_dir is not None)

        if (self.dark_image_name is not None) and (self.sample_darks is not None):
            raise AttributeError('Cannot set both `dark_image_name` and ' +\
                                 '`sample_darks`!')

        # if using a specific dark image, try that first
        if self.dark_image_name is not None:
            print(f'Using dark {self.dark_image_name}')

        elif self.sample_darks is True:
            if self.dark_dir is None:
                raise AttributeError('Must set `dark_dir` if sampling dark images!')

            # Note that this is assuming a file naming structure like
            # D{exptime}_*.fits, will fail otherwise

        # Guess we aren't using a dark!
        else:
            self.dark_image_name = None

        return

    def _set_seeds(self):
        '''
        Handle the setting of various seeds
        '''

        seed_types = ['galobj_seed', 'cluster_seed', 'stars_seed', 'noise_seed',
                      'dithering_seed']
        Nseeds = len(seed_types)
        needed_seeds = Nseeds

        master_seed = None
        seeds = dict(zip(seed_types, Nseeds*[None]))

        if hasattr(self, 'master_seed'):
            # can't pass separate seeds if a master seed is passed
            for seed in seed_types:
                if hasattr(self, seed):
                    raise AttributeError(f'Cannot set {seed} if a ' +\
                                         'master_seed is set!')
            master_seed = self.master_seed

        else:
            for seed_name in seeds.keys():
                if hasattr(self, seed_name):
                    seeds[seed_name] = getattr(self, seed_name)
                    needed_seeds -= 1

        assert needed_seeds >= 0
        if needed_seeds > 0:
            # Create safe, independent obj seeds given a master seed
            new_seeds = utils.generate_seeds(
                needed_seeds, master_seed=master_seed
                )

            k = 0
            for seed_name, val in seeds.items():
                if val is None:
                    val = new_seeds.pop()
                    seeds[seed_name] = val
                    setattr(self, seed_name, val)
                    k += 1

            assert k == needed_seeds
            assert len(new_seeds) == 0
            assert not (None in dict(seeds).values())

        for seed_name, val in seeds.items():
            print(seed_name, val)

        return

    # TODO: This should be updated to be sensible. see issue #10
    def make_mask_files(self, logprint, clobber):
        mask_dir = os.path.join(self.outdir, 'mask_files')
        mask_file = 'forecast_mask.fits'
        mask_outfile = os.path.join(mask_dir, mask_file)

        if not os.path.exists(mask_dir):
            os.mkdir(mask_dir)

        if os.path.exists(mask_outfile):
            self.logprint('Removing old mask file...')
            os.remove(mask_outfile)

        Nx = self.image_xsize
        Ny = self.image_ysize

        # x and y are flipped in fits convention vs. np array
        mask = np.zeros((Ny, Nx), dtype='i4')

        mask_fits = fitsio.FITS(mask_outfile, 'rw')

        for ext in range(self.nexp):
            try:
                mask_fits.write(mask, ext=ext, clobber=clobber)
                logprint(f'Wrote mask to {mask_outfile}')
            except OSError as e:
                logprint(f'OSError: {e}')
                raise e

        return

    # TODO: This should be updated to be sensible. see issue #10
    def make_weight_files(self, logprint, clobber):
        weight_dir = os.path.join(self.outdir, 'weight_files')
        weight_file = 'forecast_weight.fits'
        weight_outfile = os.path.join(weight_dir, weight_file)

        if not os.path.exists(weight_dir):
            os.mkdir(weight_dir)

        if os.path.exists(weight_outfile):
            self.logprint('Removing old weight file...')
            os.remove(weight_outfile)

        Nx = self.image_xsize
        Ny = self.image_ysize

        # x and y are flipped in fits convention vs. np array
        weight = np.ones((Ny, Nx), dtype='f8')

        weight_fits = fitsio.FITS(weight_outfile, 'rw')

        for ext in range(self.nexp):
            try:
                weight_fits.write(weight, ext=ext, clobber=clobber)
                logprint(f'Wrote weight to {weight_outfile}')
            except OSError as e:
                logprint(f'OSError: {e}')
                raise e

        return

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

def main(args):
    """
    Make images using model PSFs and galaxy cluster shear:
      - The galaxies come from a processed COSMOS 2015 Catalog, scaled to match
        anticipated SuperBIT 2023 observations
      - The galaxy shape parameters are assigned in a probabilistic way through matching
        galaxy fluxes and redshifts to similar GalSim-COSMOS galaxies (see A. Gill+ 2023)
    """

    config_file = args.config_file
    run_name = args.run_name
    mpi = args.mpi
    ncores = args.ncores
    clobber = args.clobber
    vb = args.vb

    start_time = time.time()

    # If outdir is None, will need to move it later after it is set
    if args.outdir is None:
        temp_log = True
    else:
        temp_log = False

    logfile = f'generate_mocks.log'
    log = utils.setup_logger(logfile, logdir=args.outdir)
    logprint = utils.LogPrint(log, vb)

    if mpi is True:
        M = MPIHelper()

    # Define some parameters we'll use below.
    sbparams = SuperBITParameters(config_file, logprint, args=args)

    # if galsim config run_name does not agree with passed arg,
    # it should be overridden (to match pipeline run_name)
    assert run_name == sbparams.run_name

    # Set up the NFWHalo:
    nfw = galsim.NFWHalo(mass=sbparams.mass, conc=sbparams.nfw_conc, redshift=sbparams.nfw_z_halo,
                     omega_m=sbparams.omega_m, omega_lam=sbparams.omega_lam)

    logprint('Set up NFW halo for lensing')

    # Read in galaxy catalog, as well as catalog containing
    # information from COSMOS fits like redshifts, hlr, etc.
    # cosmos_cat = galsim.COSMOSCatalog(sbparams.cat_file_name, dir=sbparams.datadir)
    # fitcat = Table.read(os.path.join(sbparams.cosmosdir, sbparams.fit_file_name))

    cosmos_cat = Table.read(os.path.join(sbparams.datadir,
                                         sbparams.cat_file_name))
    logprint(f'Read in {len(cosmos_cat)} galaxies from catalog and associated fit info')

    size_wg = (cosmos_cat['FLUX_RADIUS'] > 0) & (cosmos_cat['c10_sersic_fit_hlr'] < 50)
    cosmos_cat = cosmos_cat[size_wg]

    try:
        cluster_cat = galsim.COSMOSCatalog(sbparams.cluster_cat_name,
                                           dir=sbparams.cosmosdir)
    except:
        cluster_cat = galsim.COSMOSCatalog(sbparams.cluster_cat_name)

    ### Now create PSF. First, define Zernicke polynomial component
    ### note: aberrations were definined for lam = 550, and close to the
    ### center of the camera. The PSF degrades at the edge of the FOV
    lam_over_diam = sbparams.lam * 1.e-9 / sbparams.tel_diam    # radians
    lam_over_diam *= 206265.

    aberrations = np.zeros(38)             # Set the initial size.
    aberrations[0] = 0.                       # First entry must be zero
    aberrations[1] = -0.00305127
    aberrations[4] = -0.02474205              # Noll index 4 = Defocus
    aberrations[11] = -0.01544329             # Noll index 11 = Spherical
    aberrations[22] = 0.00199235
    aberrations[26] = 0.00000017
    aberrations[37] = 0.00000004
    logprint(f'Calculated lambda over diam = {lam_over_diam} arcsec')

    # gaussian jitter component from gondola instabilities
    jitter_psf = galsim.Gaussian(flux=1, fwhm=sbparams.jitter_fwhm)

    # due to how the config is structured...
    if hasattr(sbparams, 'use_optics'):
        use_optics = sbparams.use_optics
    else:
        use_optics = True

    if use_optics is False:
        optics = None
        psf = jitter_psf
        logprint('\nuse_optics is False; using jitter-only PSF\n')

    elif use_optics is True:
        # will store the Zernicke component of the PSF
        optics = galsim.OpticalPSF(lam=sbparams.lam,diam=sbparams.tel_diam,
                        obscuration=sbparams.obscuration, nstruts=sbparams.nstruts,
                        strut_angle=sbparams.strut_angle, strut_thick=sbparams.strut_thick,
                        aberrations=aberrations)

        psf = galsim.Convolve([jitter_psf, optics])

        logprint('\n Use_optics is True; convolving telescope optics PSF profile\n')

    ###
    ### Make generic WCS
    ###

    # If you wanted to make a non-trivial WCS system, could set theta to a non-zero number
    fiducial_full_image = galsim.Image(sbparams.image_xsize,
                                        sbparams.image_ysize,
                                        dtype=np.uint16
                                        )

    theta = 0.0 * galsim.degrees
    dudx = np.cos(theta) * sbparams.pixel_scale
    dudy = -np.sin(theta) * sbparams.pixel_scale
    dvdx = np.sin(theta) * sbparams.pixel_scale
    dvdy = np.cos(theta) * sbparams.pixel_scale

    affine = galsim.AffineTransform(dudx, dudy, dvdx, dvdy, origin=fiducial_full_image.true_center)
    sky_center = galsim.CelestialCoord(ra=sbparams.center_ra, dec=sbparams.center_dec)
    wcs = galsim.TanWCS(affine, sky_center, units=galsim.arcsec)

    ##
    ## Define RNG for dither offsets
    ##
    rng = np.random.default_rng(sbparams.dithering_seed)

    ##
    ## If we are sampling, grab all the darks we need
    ## There is a little trickery here: if sample_darks is true, then
    ## self.dark_image_name is an array!
    if sbparams.sample_darks is True:
        dark_im_basename = f'{sbparams.dark_dir}/D{int(sbparams.exp_time)}_*.fits'
        all_dark_images = glob(dark_im_basename)

        if len(all_dark_images) == 0:
            err = f'Found no dark images like {dark_im_basename}'
            logprint(err)
            raise OSError(err)

        else:
            dark_rng = np.random.default_rng(sbparams.noise_seed+1)
            dark_images = dark_rng.choice(all_dark_images,
                                    replace=False, size=sbparams.nexp)
            logprint(f'Using darks {dark_images}')
    ###
    ### MAKE SIMULATED OBSERVATIONS
    ### ITERATE n TIMES TO MAKE n SEPARATE IMAGES
    ###

    for i in np.arange(1, sbparams.nexp+1):
        if mpi is True:
            # get MPI processes in sync at start of each image
            M.barrier()

        outnum = str(i).zfill(3)
        outname = f'{run_name}_{outnum}.fits'
        file_name = os.path.join(sbparams.outdir, outname)

        # Set up a truth catalog during first image generation
        if i == 1:
            truth_file_name = os.path.join(sbparams.outdir,
                                           f'{run_name}_truth.fits')
            names = ['gal_num', 'cosmos_index','x_image', 'y_image',
                     'ra', 'dec', 'nfw_g1', 'nfw_g2',
                     'nfw_mu', 'redshift', 'flux',
                     'truth_fwhm','truth_mom', 'n',
                     'hlr', 'scale_h_over_r', 'obj_class']
            types = [int, int, float, float, float, float, float,
                     float, float, float, float, float, float,
                     float, float, float, str]
            truth_catalog = galsim.OutputCatalog(names, types)

        # Set up the image:
        full_image = galsim.Image(sbparams.image_xsize,
                                    sbparams.image_ysize,
                                    dtype=np.uint16
                                    )

        sky_level = sbparams.exp_time * sbparams.sky_bkg / sbparams.gain
        full_image.fill(sky_level)

        ## Define X & Y dither offsets
        dither_offsets = rng.integers(-100, 100, size=2)
        logprint(f'dithers are {dither_offsets}')
        full_image.setOrigin(dither_offsets[0], dither_offsets[1])
        full_image.wcs = wcs

        #####
        ## Loop over galaxy objects:
        #####
        print('Starting galaxy injections')

        if mpi is False:
            start = time.time()
            with Pool(ncores) as pool:
                # Create batches
                batch_indices = utils.setup_batches(sbparams.nobj, ncores)

                full_image, truth_catalog = combine_objs(
                    pool.starmap(
                        make_obj_runner,
                        ([
                          batch_indices[k],
                          'gal',
                          galsim.UniformDeviate(sbparams.galobj_seed+k+1),
                          wcs,
                          affine,
                          cosmos_cat,
                          nfw,
                          psf,
                          sbparams,
                          logprint
                          ] for k in range(ncores))
                        ),
                    full_image,
                    truth_catalog,
                    i
                    )

            dt = time.time() - start
            logprint(f'Total time for galaxy injections: {dt:.1f}s')

        else:
            # get local range to iterate over in this process
            local_start, local_end = M.mpi_local_range(sbparams.nobj)
            for k in range(local_start, local_end):
                time1 = time.time()

                # The usual random number generator using a different seed for each galaxy.
                ud = galsim.UniformDeviate(sbparams.galobj_seed+k+1)

                try:
                    # make single galaxy object
                    stamp, truth = make_a_galaxy(ud=ud,
                                                wcs=wcs,
                                                affine=affine,
                                                cosmos_cat=cosmos_cat,
                                                psf=psf,
                                                nfw=nfw,
                                                sbparams=sbparams,
                                                logprint=logprint
                                                )
                    # Find the overlapping bounds:
                    bounds = stamp.bounds & full_image.bounds

                    # Finally, add the stamp to the full image.

                    full_image[bounds] += stamp[bounds]
                    time2 = time.time()
                    tot_time = time2-time1
                    logprint(f'Galaxy {k} positioned relative to center t={tot_time} s')
                    this_flux=np.sum(stamp.array)

                    if i == 1:
                        row = [ k, truth.cosmos_index, truth.x, truth.y, truth.ra, truth.dec, truth.g1,
                                truth.g2, truth.mu,truth.z,
                                this_flux, truth.fwhm, truth.mom_size,
                                truth.n, truth.hlr, truth.scale_h_over_r, truth.obj_class]
                        truth_catalog.addRow(row)
                except galsim.errors.GalSimError:
                    logprint(f'Galaxy {k} has failed, skipping...')

        #####
        ### Inject cluster galaxy objects:
        #####
        print('Starting cluster galaxy injections')

        center_coords = galsim.CelestialCoord(sbparams.center_ra,sbparams.center_dec)
        centerpix = wcs.toImage(center_coords)

        if mpi is False:
            start = time.time()
            with Pool(ncores) as pool:
                batch_indices = utils.setup_batches(sbparams.nclustergal, ncores)

                full_image, truth_catalog = combine_objs(
                    pool.starmap(
                        make_obj_runner,
                        ([
                          batch_indices[k],
                          'cluster_gal',
                          galsim.UniformDeviate(sbparams.cluster_seed+k+1),
                          wcs,
                          affine,
                          centerpix,
                          cluster_cat,
                          psf,
                          sbparams,
                          logprint
                          ] for k in range(ncores))
                        ),
                    full_image,
                    truth_catalog,
                    i
                    )

            dt = time.time() - start
            logprint(f'Total time for cluster galaxy injections: {dt:.1f}s')

        else:
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
                                                            psf=psf,
                                                            sbparams=sbparams,
                                                            logprint=logprint)
                    # Find the overlapping bounds:
                    bounds = cluster_stamp.bounds & full_image.bounds

                    # Finally, add the stamp to the full image.

                    full_image[bounds] += cluster_stamp[bounds]
                    time2 = time.time()
                    tot_time = time2-time1
                    logprint(f'Cluster galaxy {k} positioned relative to center t={tot_time} s')
                    this_flux=np.sum(cluster_stamp.array)

                    if i == 1:
                        row = [ k, truth.cosmos_index, truth.x, truth.y, truth.ra, truth.dec,
                                truth.g1, truth.g2, truth.mu, truth.z,
                                this_flux, truth.fwhm, truth.mom_size,
                                truth.n, truth.hlr, truth.scale_h_over_r, truth.obj_class]
                        truth_catalog.addRow(row)
                except galsim.errors.GalSimError:
                    logprint(f'Cluster galaxy {k} has failed, skipping...')

        #####
        ### Now repeat process for stars!
        #####
        print('Starting star injections')

        if mpi is False:
            start = time.time()
            pud = np.random.default_rng(sbparams.stars_seed)

            with Pool(ncores) as pool:
                batch_indices = utils.setup_batches(sbparams.nstars, ncores)

                full_image, truth_catalog = combine_objs(
                    pool.starmap(
                        make_obj_runner,
                        ([
                          batch_indices[k],
                          'star',
                          galsim.UniformDeviate(sbparams.stars_seed+k+1),
                          pud,
                          batch_indices[k],
                          wcs,
                          affine,
                          psf,
                          sbparams,
                          logprint
                          ] for k in range(sbparams.ncores))
                        ),
                    full_image,
                    truth_catalog,
                    i
                    )

            dt = time.time() - start
            logprint(f'Total time for star injections: {dt:.1f}s')


        else:
            # get local range to iterate over in this process
            local_start, local_end = M.mpi_local_range(sbparams.nstars)
            for k in range(local_start, local_end):
                time1 = time.time()
                ud = galsim.UniformDeviate(sbparams.stars_seed+k+1)
                pud = np.random.default_rng(sbparams.stars_seed)
                star_stamp,truth = make_a_star(ud=ud,pud=pud,
                                            index=k,
                                            wcs=wcs,
                                            affine=affine,
                                            psf=psf,
                                            sbparams=sbparams,
                                            logprint=logprint
                                            )
                bounds = star_stamp.bounds & full_image.bounds

                # Add the stamp to the full image.
                try:
                    full_image[bounds] += star_stamp[bounds]

                    time2 = time.time()
                    tot_time = time2-time1

                    logprint(f'Star {k}: positioned relative to center, t={tot_time} s')
                    this_flux=np.sum(star_stamp.array)

                    if i == 1:
                        row = [ k, truth.cosmos_index, truth.x, truth.y, truth.ra, truth.dec,
                                truth.g1, truth.g2, truth.mu,
                                truth.z, this_flux, truth.fwhm,truth.mom_size,
                                truth.n, truth.hlr, truth.scale_h_over_r, truth.obj_class]
                        truth_catalog.addRow(row)

                except galsim.errors.GalSimError:
                    logprint(f'Star {k} has failed, skipping...')

        # If not using MPI, then this is already done
        if mpi is True:
            # Gather results from MPI processes, reduce to single result on root
            # Using same names on left and right sides is hiding lots of MPI magic
            full_image = M.gather(full_image)
            truth_catalog = M.gather(truth_catalog)
            if M.is_mpi_root():
                full_image = reduce(combine_images, full_image)

                if i == 1:
                    truth_catalog = reduce(combine_catalogs, truth_catalog)
                else:
                    # do the adding of noise and writing to disk entirely on root
                    # root and the rest meet again at barrier at start of loop
                    continue

        if (mpi is False) or (M.is_mpi_root()):

            if sbparams.sample_darks is True:
                sbparams.dark_image_name = dark_images[i-1]

            if sbparams.dark_image_name is not None:
                sbparams.read_noise = 0

            # Add ccd noise
            logprint('Adding CCD noise')
            noise = galsim.CCDNoise(
                sky_level=0,
                gain=sbparams.gain,
                read_noise=sbparams.read_noise,
                rng=galsim.BaseDeviate(sbparams.noise_seed)
                )
            full_image.addNoise(noise)
            logprint.debug('Added noise to final output image')

            # Add dark current
            if sbparams.dark_image_name is not None:
                logprint(f'Adding dark current frame {sbparams.dark_image_name}')
                dark_image = fitsio.read(sbparams.dark_image_name)
                full_image += dark_image

            else:
                # Add dark current
                logprint('Adding Dark current')
                dark_noise = sbparams.dark_current * sbparams.exp_time
                full_image += dark_noise

            if not os.path.exists(os.path.dirname(file_name)):
                os.makedirs(os.path.dirname(file_name))

            try:
                full_image.write(file_name, clobber=clobber)
                logprint(f'Wrote image to {file_name}')
            except OSError as e:
                logprint(f'OSError: {e}')
                raise e

            # Write truth catalog to file.
            if i == 1:
                try:
                    truth_catalog.write(truth_file_name)
                    logprint(f'Wrote truth to {truth_file_name}')

                    # It can be useful to load the true PSF into memory for
                    # later tests. So we pickle it now and save the filename
                    # into the truth catalog header
                    psf_outfile = os.path.join(
                        sbparams.outdir, 'true_psf.pkl'
                        )
                    with open(psf_outfile, 'wb') as psf_pfile:
                        pickle.dump(psf, psf_pfile)

                    with fits.open(truth_file_name, mode='update') as handle:
                        handle[0].header['psf_pkl'] = psf_outfile

                except OSError as e:
                    logprint(f'OSError: {e}')
                    raise e

    logprint('\nCompleted all images\n')

    if (mpi is False) or (M.is_mpi_root()):
        logprint('Creating masks')
        logprint.warning('For now, we just write a simple mask file with all 1s')
        sbparams.make_mask_files(logprint, clobber)

        logprint('Creating weights')
        logprint.warning('For now, we just write a simple weight file with all 1s')
        sbparams.make_weight_files(logprint, clobber)

    # Log file was created before outdir is setup in some cases
    # If so, move from temp location to there
    if temp_log is True:
        oldfile = os.path.join(logdir, logfile)
        newfile = os.path.join(sbparams.outdir, logfile)
        os.replace(oldfile, newfile)

    if (mpi is False) or (M.is_mpi_root()):
        logprint('\nDone!\n')

    end_time = time.time()
    logprint('\n\ngalsim execution time = {end_time - start_time}\n\n')

    return

if __name__ == "__main__":
    args = parse_args()
    main(args)

