# Custom modules
#from requests import get
import instrument as inst
import photometry as phot

from stars import make_a_star
from gals import make_a_galaxy
from cluster import make_a_cluster_galaxy, sample_a_cluster, des2superbit

# Packages
from argparse import ArgumentParser
from pathlib import Path
from multiprocessing import Pool
import pandas as pd
import numpy as np
from numpy.lib import recfunctions as rf
import fitsio
import galsim
import sys
import os
import glob
from astropy import units as u
from astroquery.vizier import Vizier
import astropy.coordinates as coord
from astropy.table import Table, join
from astropy.io import fits
import yaml
import math
import datetime
import time
import random

from superbit_lensing import utils
from superbit_lensing.oba import oba_io

import ipdb

OBA_SIM_DATA_DIR = Path(__file__).parent / 'data/'

def parse_args():

    parser = ArgumentParser()

    parser.add_argument('config_file', type=str,
                        help='The filepath of the simulation config file')

    return parser.parse_args()

def get_zernike(band, strehl_ratio=None):
    if strehl_ratio is None:
        strehl_ratio = 100
    sr = str(int(strehl_ratio))
    fname = OBA_SIM_DATA_DIR / f'psf/{band}_{sr}.csv'
    return np.genfromtxt(
        fname,
        delimiter=','
        )[:, 1][1:]

def setup_seeds(config):

    if 'master_seed' in config:
        master_seed = config['master_seed']
    else:
        master_seed = None

    seeds = {
        'gals': None,
        'stars': None,
        'cluster': None,
        'noise': None,
        'dithering': None,
        'roll': None,
    }

    Nseeds = len(seeds)
    seeds_to_set, master_seed = utils.generate_seeds(
        Nseeds, master_seed=master_seed, return_master=True
    )

    for key in seeds.keys():
        seeds[key] = seeds_to_set.pop()

    seeds['master'] = master_seed

    return seeds

def set_config_defaults(config):

    if 'run_name' not in config:
        config['run_name'] = 'quicktest'

    if 'overwrite' not in config:
        config['overwrite'] = False

    if 'bands' not in config:
        config['bands'] = ['u', 'b', 'g', 'r', 'lum', 'nir']

    if 'vb' not in config:
        config['vb'] = False

    if 'fresh' not in config:
        config['fresh'] = False

    if 'max_fft_size' not in config:
        config['max_fft_size'] = 2**18

    if 'ncores' not in config:
        config['ncores'] = 1

    if 'starting_roll' not in config:
        config['starting_roll'] = 0

    if 'rot_rate' not in config:
        config['rot_rate'] = 0.25 # deg / exp

    if 'star_stamp_size' not in config:
        config['star_stamp_size'] = 1000

    if 'strehl_ratio' not in config:
        # NOTE: needed to change this from 100 to handle simultaneous
        # output formats between Ajay & Spencer
        config['strehl_ratio'] = None

    return config

def compute_im_bounding_box(ra, dec, im_xsize, im_ysize, theta):
    '''
    ra, dec: image center in deg
    im_xsize, im_ysize: image len in arcsec
    theta: roll angle as galsim.Angle

    Use the roll angle theta to determine the box inscribing the rotated
    rectangular SCI image

    returns: ra_bounds, dec_bounds
        The new (conservative) ra/dec bounds, in deg
    '''

    if np.pi/2 < theta.rad < 3*np.pi/2:
        csign = -1.0
    else:
        csign = 1.0
    if np.pi < theta.rad < 2*np.pi:
        ssign = -1.0
    else:
        ssign = 1.0

    # original box lengths
    Lx = im_xsize.to(u.deg).value
    Ly = im_ysize.to(u.deg).value

    # compute the new, larger box
    new_Lx = Ly * (ssign*np.sin(theta.rad)) + Lx * (csign*np.cos(theta.rad))
    new_Ly = Lx * (ssign*np.sin(theta.rad)) + Ly * (csign*np.cos(theta.rad))

    ra_min = ra - new_Lx/2.
    ra_max = ra + new_Lx/2.

    dec_min = dec - new_Ly/2.
    dec_max = dec + new_Ly/2.

    ra_bounds = [ra_min, ra_max]
    dec_bounds = [dec_min, dec_max]

    return ra_bounds, dec_bounds

def make_obj_runner(batch_indices, obj_type, obj_cat, *args, **kwargs):
    '''
    Handles the batch running of make_obj() over multiple cores
    '''

    res = []
    for i in batch_indices:
        # replace obj catalog with single obj to reduce mem passing
        obj = obj_cat[i]
        res.append(make_obj(i, obj_type, obj, *args, **kwargs))

    return res

def make_obj(i, obj_type, obj, *args, **kwargs):
    '''
    Runs the approrpriate "make_a_{obj}" function given object type.
    Particularly useful for multiprocessing wrappers
    '''

    logprint = args[-1]

    func = None

    func_map = {
        'gal': make_a_galaxy,
        'cluster_gal': make_a_cluster_galaxy,
        'star': make_a_star
    }

    obj_types = func_map.keys()
    if obj_type not in obj_types:
        raise ValueError(f'Object type must be one of {obj_types}!')

    func = func_map[obj_type]

    try:
        obj_index = int(i)
        if i % 100 == 0:
            logprint(f'Starting {obj_type} {i}')
        stamp, truth = func(obj_index, obj, *args, **kwargs)
        # logprint(f'{obj_type} {i} completed succesfully')

    except galsim.errors.GalSimError:
        logprint(f'{obj_type} {i} has failed, skipping...')
        return i, None, None

    return i, stamp, truth

def combine_objs(make_obj_outputs, full_image, truth_catalog, exp_num,
                 logprint):
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
            pass
            # logprint(f'obj {i} out of bounds. Skipping.')

        if exp_num == 0:
            if len(truth_catalog) == 0:
                truth_catalog = Table(truth)
            else:
                truth_catalog.add_row(truth)
        else:
            # for objects that were not in the first exposure
            if truth['id'] not in truth_catalog['id']:
                truth_catalog.add_row(truth)

    return full_image, truth_catalog

def setup_stars(ra, dec, width_deg, height_deg):

    Vizier.ROW_LIMIT = -1

    coordinates = coord.SkyCoord(
        ra=ra * u.deg,
        dec=dec * u.deg,
        frame='icrs'
        )

    # we add some buffer to account for dithering & rolls
    # TODO: understand why it is so sensitive to the prefactor...
    qsize = 1.3 * np.max([height_deg.value, width_deg.value])
    gaia_cat = Vizier.query_region(
        coordinates=coordinates,
        height=qsize*u.deg,
        width=qsize*u.deg,
        catalog='I/345/gaia2'
        )

    gaia_cat = gaia_cat[0].filled()

    # Convert Table to pandas dataframe
    df_stars = pd.DataFrame()

    df_stars['RA_ICRS'] = gaia_cat['RA_ICRS']
    df_stars['DE_ICRS'] = gaia_cat['DE_ICRS']

    df_stars['FG'] = gaia_cat['FG']
    df_stars['FBP'] = gaia_cat['FBP']

    df_stars['BPmag'] = (-2.5 * np.log10(gaia_cat['FBP'])) + 25.3861560855
    df_stars['Gmag'] = (-2.5 * np.log10(gaia_cat['FG'])) + 25.7915509947

    df_stars.dropna()

    df_stars = df_stars[df_stars['Gmag'] >= -5]
    df_stars = df_stars[df_stars['BPmag'] >= -5]

    df_stars = df_stars.reset_index(drop=True)

    star_cat = Table(df_stars.to_records())

    return star_cat

def setup_gals(n_gal_total_img, gal_rng,high_snr=True):

    cosmos_dir = OBA_SIM_DATA_DIR / 'cosmos_catalog/'
    cosmos_file = cosmos_dir / 'cosmos15_superbit2023_phot_shapes.csv'
    cat_df = pd.read_csv(str(cosmos_file))

    # Filter the data for FLUX_RADIUS > than 0 and HLR < 50
    cat_df = cat_df[cat_df['FLUX_RADIUS'] >= 0]
    cat_df = cat_df[cat_df['c10_sersic_fit_hlr'] < 50]
    if high_snr:
        cat_df = cat_df[cat_df['mag_lum'] < 23.5]

    # Randomly sample from galaxy catalog
    gal_cat = Table(cat_df.sample(
        n=n_gal_total_img, random_state=gal_rng
        ).to_records())

    # TODO: understand why this happens...
    gal_cat.remove_column('Unnamed: 0')

    return gal_cat

def setup_cluster(logm, z, ra, dec, telescope, camera, bandpass, exp_time, rng=None):
    '''
    Pick a redmapper cluster at similar logm & z. Then add SB ADU fluxes
    '''

    # NOTE: for now, we'll be hacky
    cluster_dir = OBA_SIM_DATA_DIR / 'redmapper/'
    clusters_file = cluster_dir / 'y3_gold_2.2.1_wide_sofcol_run2_redmapper_v6.4.22+2_lgt20_vl50_catalog.fit'
    clusters = Table.read(clusters_file)

    # NOTE: This needs to be the associated redmapper members catalog *matched*
    # to DES GOLD, as we need morphological information in addition to photometry
    members_file = cluster_dir / 'redmapper_members_gold_match.fits'
    members = Table.read(members_file)

    # pick 1 cluster
    cluster = sample_a_cluster(clusters, logm, z, rng=rng)

    # grab the member galaxies of that cluster
    cluster_gals = join(
        cluster, members, join_type='left', keys=['MEM_MATCH_ID'],
        table_names=['cluster', 'member']
        )

    # only grab those with a greater than 50% membership probability
    cluster_gals = cluster_gals[cluster_gals['P'] > 0.5]

    # makes best guess at SB band fluxes given DES band fluxes
    cluster_gals = des2superbit(cluster_gals, telescope, camera, bandpass, exp_time)

    # keep member positions relative to cluster center, but offset to new target
    cluster_gals['ra_sim'] = (cluster_gals['RA_member'] - cluster_gals['RA']) + ra
    cluster_gals['dec_sim'] = (cluster_gals['DEC_member'] - cluster_gals['DEC']) + dec

    return cluster_gals

def main(args):

    config_file = args.config_file
    config = utils.read_yaml(config_file)

    config = set_config_defaults(config)

    target_name = config['target_name']
    run_name = config['run_name']
    bands = config['bands']
    starting_roll = config['starting_roll'] * galsim.degrees
    rot_rate = config['rot_rate']
    max_fft_size = config['max_fft_size']
    strehl_ratio = config['strehl_ratio']
    ncores = config['ncores']
    fresh = config['fresh']
    overwrite = config['overwrite']
    vb = config['vb']

    bkg_height= config['bkg_loc']
    
    if 'calibrated' in config.keys():
        calibrated = config['calibrated'] #should be True/False
    else:
        calibrated =  False
        
    run_dir = Path(utils.TEST_DIR, f'euclid/{run_name}/{target_name}/')

    # WARNING: cleans all existing files in run_dir!
    if fresh is True:
        try:
            utils.rm_tree(run_dir)
        except OSError:
            pass

    # setup logger
    logdir = run_dir
    logfile = str(logdir / f'{run_name}_{target_name}_sim.log')

    log = utils.setup_logger(logfile, logdir=logdir)
    logprint = utils.LogPrint(log, vb=vb)

    # dict of independent seeds for given types
    seeds = setup_seeds(config)
    master_seed = seeds['master']
    logprint(f'Using master_seed of {master_seed}')

    # guaranteed to need noise seed
    noise_rng = np.random.default_rng(seeds['noise'])

    gs_params = galsim.GSParams(maximum_fft_size=max_fft_size)

    # Dict for pivot wavelengths
    piv_dict = {
        'u': 395.35082727585194,
        'b': 476.22025867791064,
        'g': 596.79880208687230,
        'r': 640.32425638312820,
        'nir': 814.02475812251110,
        'lum': 522.73829660009810,
        'vis': 710.343 	#from http://svo2.cab.inta-csic.es/svo/theory/fps3/index.php?id=Euclid/VIS.vis&&mode=browse&gname=Euclid&gname2=VIS
    }

    # Initialize the camera, telescope, bandpass
    #camera = inst.Camera('imx455')
    #telescope = inst.Telescope('superbit')
    camera = inst.Camera('vis')
    telescope = inst.Telescope('euclid')

    bandpass = inst.Bandpass('bandpass')
    bandpass.wavelengths = camera.wavelengths
    bandpass.plate_scale = (
        ((206265 * camera.pixel_size.to(u.micron).value)
         / (1000 * telescope.focal_length.to(u.mm))).value
        * (u.arcsec/u.pixel)
        )

    pix_scale = bandpass.plate_scale.value

    exp_time = config['exp_time'] # seconds
    n_exp = config['n_exp']

    target_list = Path(__file__).parent / 'target_list_full_focus.csv'
    # df = pd.read_csv(str(target_list))
    targets = Table.read(str(target_list))

    # Number of background galaxies to add
    sci_img_size_x_arcsec = camera.npix_H * bandpass.plate_scale
    sci_img_size_y_arcsec = camera.npix_V * bandpass.plate_scale

    height_deg = ((sci_img_size_y_arcsec.value) * u.arcsec).to(u.deg)
    width_deg = ((sci_img_size_x_arcsec.value) * u.arcsec).to(u.deg)

    sci_img_area = (sci_img_size_x_arcsec * sci_img_size_y_arcsec).to(
        u.arcmin**2
        )

    # the *sampled* area has to be bigger than this, to allow for rolls
    im_buffer = 30 # arcsec
    img_radius = (im_buffer + np.sqrt(
        sci_img_size_x_arcsec.value**2 + sci_img_size_y_arcsec.value**2
        )) * u.arcsec
    sampled_img_area = ((2*img_radius)**2).to(u.arcmin**2)

    n_gal_sqarcmin = 97.55 * (u.arcmin**-2)

    if 'ngals' in config:
        n_gal_total_img = config['ngals']
    else:
        # use fiducial density
        n_gal_total_img = round(
            (sampled_img_area * n_gal_sqarcmin).value
            )

    cosmos_plate_scale = 0.03 # arcsec/pix

    dither_pix = 100
    dither_deg = pix_scale * dither_pix / 3600 # dither_deg
    dither_rng = np.random.default_rng(seeds['dithering'])

    if strehl_ratio is None:
        sr_tag = ''
    else:
        sr_tag = f'sr_{strehl_ratio}_'

    if 'add_galaxies' in config:
        add_galaxies = config['add_galaxies']
    else:
        add_galaxies = True

    if 'add_stars' in config:
        add_stars = config['add_stars']
    else:
        add_stars = True

    if 'add_cluster' in config:
        add_cluster = config['add_cluster']
    else:
        add_cluster = True

    # setup galaxies
    if add_galaxies is True:
        gal_rng = np.random.default_rng(seeds['gals'])

        gal_cat = setup_gals(n_gal_total_img, gal_rng)
        Ngals = len(gal_cat)

        # will populate w/ a truth cat for each band
        truth_gal_cat = None

    if add_stars is True:
        # will populate w/ a truth cat for each band
        truth_star_cat = None

    if add_cluster is True:
        cluster_rng = np.random.default_rng(seeds['cluster'])

        # will populate w/ a truth cat for each band
        truth_cluster_cat = None

    # Start main process
    row = np.where(targets['target'] == target_name)

    target = targets[row][0]

    ra = target['ra'] * u.deg
    dec = target['dec'] * u.deg
    cluster_z = target['z']
    logm = target['mass'] # log10 solmass
    mass = 10**(logm) # solmass
    conc = target['c']

    # setup stars (query only for now)
    if add_stars is True:
        logprint('Setting up stars...')

        star_cat = setup_stars(
            ra.value, dec.value, height_deg, width_deg
            )

        Nstars = len(star_cat)

        # make initial truth cat (*all* sources that could possibly be drawn,
        # but w/o all cols)
        outfile = run_dir / f'initial_{sr_tag}truth_stars_{target_name}.fits'
        # fitsio.write(outfile, star_cat, overwrite=overwrite)
        star_cat.write(outfile, overwrite=overwrite)

    # setup gals (sampling positions only for now)
    if add_galaxies is True:
        logprint('setting up galaxies...')

        # we use the larger box to handle roll angles
        # first, use the diagonal which is ~20% larger. Then add buffer
        # ra_sample_len = img_1.3*(1.2*img_max_len).to(u.deg).value
        # dec_sample_len = 1.3*(1.2*img_max_len / 2.).to(u.deg).value

        ra_radius = img_radius.to(u.deg).value
        dec_radius = img_radius.to(u.deg).value

        sampled_ra = gal_rng.uniform(
            ra.value - ra_radius,
            ra.value + ra_radius,
            size=Ngals
            )

        sampled_dec = sample_uniform_dec(
            dec.value - dec_radius,
            dec.value + dec_radius,
            N=Ngals,
            rng=gal_rng
            )

        gal_cat['ra'] = sampled_ra
        gal_cat['dec'] = sampled_dec

        # gal_cat = rf.append_fields(
        #     l_cat,
        #     ['ra', 'dec'],
        #     [sampled_ra, sampled_dec],
        #     dtypes=['float32'],
        #     usemask=False
        # )

        # make initial truth cat (*all* sources that could possibly be drawn,
        # but w/o all cols)
        outfile = run_dir / f'initial_{sr_tag}truth_gals_{target_name}.fits'
        # fitsio.write(outfile, gal_cat, overwrite=overwrite)
        gal_cat.write(outfile, overwrite=overwrite)

    # setup cluster (sample redmapper for similar M & z, then at SB fluxes)
    if add_cluster is True:
        logprint('Setting up cluster...')
        cluster_cat = setup_cluster(
            logm, cluster_z, ra.value, dec.value, telescope, camera, bandpass,
            exp_time, rng=cluster_rng
            )

        # make initial truth cat (*all* sources that could possibly be drawn,
        # but w/o all cols)
        outfile = run_dir / f'initial_{sr_tag}truth_cluster_{target_name}.fits'
        cluster_cat.write(outfile, overwrite=overwrite)
        # fitsio.write(outfile, cluster_cat, overwrite=overwrite)

    # We want to rotate the sky by (5 min + 1 min overhead) each
    # new target. Each band inherits the last roll of the previous band
    theta = starting_roll # already a galsim.Angle
    # rot_rate = 0.25 # deg / min

    for band in bands:
        # pivot wavelength
        piv_wave = piv_dict[band]

        bandpass.transmission = phot.get_transmission(band=band)

        if band == 'vis':
            psf_file = 'sed_true_26892756.os.fits'
            psf_image = galsim.fits.read(psf_file, hdu=1)
            psf_sim = galsim.InterpolatedImage(psf_image, scale=0.1/5)
            #psf_sim=galsim.Image(psf_image)

        else:

            aberrations = get_zernike(band=band, strehl_ratio=strehl_ratio)

            optical_zernike_psf = galsim.OpticalPSF(
                lam=piv_wave,
                diam=telescope.diameter.value,
                aberrations=aberrations,
                obscuration=telescope.obscuration,
                nstruts=telescope.nstruts,
                flux=1
                )

            jitter_psf = galsim.Gaussian(sigma=0.05, flux=1)

            psf_sim = galsim.Convolve([optical_zernike_psf, jitter_psf])

        # setup truth cats for this band
        truth_gals = Table()
        truth_stars = Table()
        truth_cluster = Table()

        # setup stellar fluxes
        if add_stars is True:
            if piv_wave > 600:
                gaia_mag = 'Gmag'
            else:
                gaia_mag = 'BPmag'

            # Find counts to add for the star
            mean_fnu_star_mag = phot.abmag_to_mean_fnu(
                abmag=star_cat[gaia_mag]
                )

            mean_flambda = phot.mean_flambda_from_mean_fnu(
                mean_fnu=mean_fnu_star_mag,
                bandpass_transmission=bandpass.transmission,
                bandpass_wavelengths=bandpass.wavelengths
                )

            crate_electrons_pix = phot.crate_from_mean_flambda(
                mean_flambda=mean_flambda,
                illum_area=telescope.illum_area.value,
                bandpass_transmission=bandpass.transmission,
                bandpass_wavelengths=bandpass.wavelengths
                )

            crate_adu_pix = crate_electrons_pix / camera.gain.value

            # what is actually used below
            star_flux_adu = crate_adu_pix * exp_time * 1

            star_cat[f'flux_adu_{band}'] = star_flux_adu
            # star_cat = rf.append_fields(
            #     star_cat,
            #     f'flux_adu_{band}',
            #     star_flux_adu,
            #     dtypes='float32',
            #     usemask=False
            # )

        for exp_num in range(n_exp):

            logprint(f'Image simulation starting for {target_name}, {band}; ' +
                    f'{exp_num+1} of {n_exp}')

            if run_name is None:
                rn = ''
            else:
                rn = f'{run_name}/'

            outdir = os.path.join(
                utils.TEST_DIR, f'euclid/{rn}{target_name}/{band}/{sr_tag}'
                )
            utils.make_dir(outdir)

            ra_sim = dither_rng.uniform(
                ra.value - dither_deg, ra.value + dither_deg
                )
            dec_sim = dither_rng.uniform(
                dec.value - dither_deg, dec.value + dither_deg
                )

            # in pixels
            dither_ra = (ra.value - ra_sim) * 3600 / pix_scale
            dither_dec = (dec.value - dec_sim) * 3600 / pix_scale

            # Step 1: Construct the science image
            sci_img = galsim.Image(
                ncol=camera.npix_H.value,
                nrow=camera.npix_V.value,
                # dtype=np.uint16
                )

            # Step 3: Fill the science image with mean sky bkg (Gill et al. 2020)
            crate_sky_electron_pix = phot.crate_bkg(
                illum_area=telescope.illum_area,
                bkg_height=bkg_height, #add
                bandpass=bandpass,
                bkg_type='raw',
                strength='ave'
                )

            crate_sky_adu_pix = crate_sky_electron_pix / camera.gain.value
            sky_adu_pix = crate_sky_adu_pix * exp_time * 1  # ADU

            sci_img.fill(sky_adu_pix)
            sci_img.setOrigin(0, 0)

            # Step 4: WCS setup
            # TODO: generalize
            if exp_num > 0:
                # 5min + 1 minute fudge
                theta += (rot_rate * 6) * galsim.degrees

            dudx = np.cos(theta.rad) * pix_scale
            dudy = -np.sin(theta.rad) * pix_scale
            dvdx = np.sin(theta.rad) * pix_scale
            dvdy = np.cos(theta.rad) * pix_scale

            image_center = sci_img.true_center

            affine = galsim.AffineTransform(
                dudx, dudy, dvdx, dvdy, origin=image_center
                )

            sky_center = galsim.CelestialCoord(
                ra=ra_sim * galsim.degrees,
                dec=dec_sim * galsim.degrees
                )

            wcs = galsim.TanWCS(affine, sky_center, units=galsim.arcsec)
            sci_img.wcs = wcs

            sci_img_bounds = sci_img.bounds

            # Update the PSF to account for roll angle
            psf_roll = psf_sim.rotate(theta)

            # Step 5: Setup image bounds & cluster halo
            # NOTE: these define the minimum bounding box of the (possibly
            # rotated) image. This will allow us to skip galaxies
            # off of the current image more efficiently
            # ra_bounds, dec_bounds = compute_im_bounding_box(
            #     ra_sim,
            #     dec_sim,
            #     sci_img_size_x_arcsec,
            #     sci_img_size_y_arcsec,
            #     theta
            #     )

            # TODO: use cornish for actual overlap. For now, use a
            # sufficiently large buffer
            # ra_buff = .2
            # ra_bounds[0] -= ra_buff
            # ra_bounds[1] += ra_buff
            # dec_buff = ra_buff
            # dec_bounds[0] -= dec_buff
            # dec_bounds[1] += dec_buff

            # ra_bounds *= u.deg
            # dec_bounds *= u.deg

            # setup NFW halo at the target position
            halo_pos = galsim.CelestialCoord(
                ra=ra.value*galsim.degrees,
                dec=dec.value*galsim.degrees
                )

            halo_pos_im = wcs.toImage(halo_pos) * pix_scale

            nfw_halo = galsim.NFWHalo(
                mass=mass,
                conc=conc,
                redshift=cluster_z,
                omega_m=0.3,
                halo_pos=halo_pos_im,
                omega_lam=0.7
                )

            # Step 6: Add stars (setup done earlier)
            if add_stars is True:

                start = time.time()

                logprint(f'Adding {Nstars} stars')

                star_stamp_size = config['star_stamp_size']

                # NOTE: in progress...
                with Pool(ncores) as pool:
                    batch_indices = utils.setup_batches(Nstars, ncores)

                    sci_img, truth_stars = combine_objs(
                        pool.starmap(
                            make_obj_runner,
                            ([
                                batch_indices[k],
                                'star',
                                star_cat,
                                band,
                                wcs,
                                psf_roll,
                                camera,
                                exp_time,
                                pix_scale,
                                # ra_bounds,
                                # dec_bounds,
                                gs_params,
                                star_stamp_size,
                                logprint,
                            ] for k in range(ncores))
                        ),
                        sci_img,
                        truth_stars,
                        exp_num,
                        logprint
                    )

                logprint('Done adding stars.')

                star_time = time.time() - start
                logprint(f'Stars took {star_time:.1f} s')

            # step 6
            if add_galaxies is True:

                logprint(f'Adding {n_gal_total_img} background galaxies')
                logprint(f'Parallelizing across {ncores} cores')

                start = time.time()

                with Pool(ncores) as pool:
                    batch_indices = utils.setup_batches(Ngals, ncores)

                    sci_img, truth_gals = combine_objs(
                        pool.starmap(
                            make_obj_runner,
                            ([
                                batch_indices[k],
                                'gal',
                                gal_cat,
                                band,
                                wcs,
                                psf_roll,
                                nfw_halo,
                                camera,
                                exp_time,
                                pix_scale,
                                cosmos_plate_scale,
                                # ra_bounds,
                                # dec_bounds,
                                gs_params,
                                logprint
                            ] for k in range(ncores))
                        ),
                        sci_img,
                        truth_gals,
                        exp_num,
                        logprint
                    )

                gal_time = time.time() - start
                logprint('Done with galaxies')
                logprint(f'Galaxies took {gal_time:.1f} s')

            if add_cluster is True:

                Ncluster_gals = len(cluster_cat)
                logprint(f'Adding {Ncluster_gals} cluster galaxies')
                logprint(f'Parallelizing across {ncores} cores')

                start = time.time()

                with Pool(ncores) as pool:
                    batch_indices = utils.setup_batches(Ncluster_gals, ncores)

                    sci_img, truth_cluster = combine_objs(
                        pool.starmap(
                            make_obj_runner,
                            ([
                                batch_indices[k],
                                'cluster_gal',
                                cluster_cat,
                                band,
                                wcs,
                                psf_roll,
                                camera,
                                exp_time,
                                pix_scale,
                                gs_params,
                                logprint,
                            ] for k in range(ncores))
                        ),
                        sci_img,
                        truth_cluster,
                        exp_num,
                        logprint
                    )

                gal_time = time.time() - start
                logprint('Done with cluster galaxies')
                logprint(f'Galaxies took {gal_time:.1f} s')

            # add shot noise on sky + sources
            # TODO: sort out why this turns sci_img to ints...
            noise_dev = galsim.BaseDeviate(seeds['noise'])
            noise = galsim.PoissonNoise(sky_level=0.0, rng=noise_dev)
            sci_img.addNoise(noise)
            sci_img = sci_img.array

            # Step 8: Add a dark frame

            #dark_dir = OBA_SIM_DATA_DIR / 'darks/minus10/'
            #darks = np.sort(glob.glob(str(dark_dir / '*.fits')))
            #dark_fname = np.random.choice(darks)

            # we upcast for now, cast back later
            #dark = fits.getdata(dark_fname).astype('float32')
            #sci_img += dark

            # limit the flux
            sci_img[sci_img >= (2**16)] = 2**16 - 1
            sci_img[sci_img < 0] = 0

            if calibrated is False:
                # *now* cast to int16
                sci_img = sci_img.astype('uint16')
            else:
                sci_img = sci_img * camera.gain.value / exp_time
                wgt_img = np.ones_like(sci_img)

            # HEADERS
            hdr = fits.Header()
            hdr['TARGET'] = target_name
            hdr['EXPTIME'] = int(exp_time)
            hdr['band'] = band
            hdr['strehl'] = strehl_ratio
            hdr['stars'] = int(add_stars)
            hdr['galaxies'] = int(add_galaxies)
            hdr['TRG_RA'] = ra.value # in deg
            hdr['TRG_DEC'] = dec.value # in deg
            hdr['dither_ra'] = dither_ra # in pixels
            hdr['dither_dec'] = dither_dec # in pixels
            hdr['roll_theta'] = theta.deg # in deg
            #hdr['dark'] = Path(dark_fname).name

            # TODO/QUESTION: For an unknown reason, BZERO is getting
            # set to 2^15 for an unknown reason unless we do this...
            # hdr['BZERO'] = 0

            # add WCS info to header
            wcs.writeToFitsHeader(hdr, bounds=sci_img_bounds)

            dt_now = datetime.datetime.now()
            unix_time = int(time.mktime(dt_now.timetuple()))

            band_int = oba_io.band2index(band)

            # Path checks
            Path(outdir).mkdir(parents=True, exist_ok=True)
            cal_dir = os.path.join(outdir, 'cal')
            os.makedirs(cal_dir, exist_ok=True)

            #output_fname = f'{outdir}/{target_name}_{band_int}_{exp_time}_{unix_time}_cal.fits'


            if calibrated is True:
                output_fname = f'{cal_dir}/{target_name}_{band_int}_{exp_time}_{unix_time}_cal.fits'
                img_hdulist = fits.HDUList([fits.PrimaryHDU(data=sci_img,header=hdr),fits.ImageHDU(data=wgt_img)])
            else:
                output_fname = f'{cal_dir}/{target_name}_{band_int}_{exp_time}_{unix_time}.fits'
                img_hdulist = fits.HDUList([fits.PrimaryHDU(data=sci_img,header=hdr)])
            img_hdulist.writeto(output_fname,overwrite=overwrite)
            
            #fits.writeto(
            #    filename=output_fname,
            #    data=sci_img,
            #    header=hdr,
            #    overwrite=overwrite
            #    )

            logprint(f'Image simulation complete for {target_name}, {band}\n')

        if add_stars is True:
            if truth_star_cat is None:
                truth_star_cat = truth_stars.copy()
            else:
                truth_star_cat = join(
                    truth_star_cat,
                    truth_stars,
                    join_type='left',
                    )

        if add_galaxies is True:
            if truth_gal_cat is None:
                truth_gal_cat = truth_gals.copy()
            else:
                truth_gal_cat = join(
                    truth_gal_cat,
                    truth_gals,
                    join_type='left',
                    )

        if add_cluster is True:
            if truth_cluster_cat is None:
                truth_cluster_cat = truth_cluster.copy()
            else:
                truth_cluster_cat = join(
                    truth_cluster_cat,
                    truth_cluster,
                    join_type='left',
                    )

    # merge truth cats & save
    if add_stars is True:
        outfile = f'{run_dir}/{sr_tag}truth_stars_{target_name}.fits'
        truth_star_cat.write(outfile, overwrite=overwrite)

    if add_galaxies is True:
        outfile = f'{run_dir}/{sr_tag}truth_gals_{target_name}.fits'
        truth_gal_cat.write(outfile, overwrite=overwrite)

    if add_cluster is True:
        outfile = f'{run_dir}/{sr_tag}truth_cluster_{target_name}.fits'
        truth_cluster_cat.write(outfile, overwrite=overwrite)

    return 0

def sample_uniform_dec(d1, d2, N=1, rng=None):
    '''
    Sample N random DEC values from d1 to d2, accounting for curvature of sky.

    d1 & d2 must be in deg
    '''

    if rng is None:
        rng = np.random.default_rng()

    d1, d2 = np.deg2rad(d1), np.deg2rad(d2)

    # Uniform sampling from 0 to 1
    P = rng.random(N)

    # Can't use `sample_uniform()` as dec needs angular weighting
    delta = np.arcsin(P * (np.sin(d2) - np.sin(d1)) +np.sin(d1))

    return np.rad2deg(delta)

if __name__ == '__main__':
    args = parse_args()
    rc = main(args)

    if rc == 0:
        print('sim_full_focus.py completed succesfully!')
    else:
        print(f'sim_full_focus.py failed w/ return code {rc}')
