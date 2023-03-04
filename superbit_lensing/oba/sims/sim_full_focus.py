# Custom modules
#from requests import get
import instrument as inst
import photometry as phot

from stars import make_a_star
from gals import make_a_galaxy

# Packages
from argparse import ArgumentParser
from pathlib import Path
from multiprocessing import Pool
import pandas as pd
import numpy as np
from numpy.lib import recfunctions as rf
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

def get_transmission(band):
    sim_dir = OBA_SIM_DATA_DIR

    return np.genfromtxt(
        sim_dir / f'instrument/bandpass/{band}_2023.csv',
        delimiter=','
        )[:, 2][1:]

def get_zernike(band, strehl_ratio):
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
    seeds_to_set = utils.generate_seeds(Nseeds, master_seed=master_seed)

    for key in seeds.keys():
        seeds[key] = seeds_to_set.pop()

    return seeds

def set_config_defaults(config):

    if 'run_name' not in config:
        config['run_name'] = 'quicktest'

    if 'overwrite' not in config:
        config['overwrite'] = False

    if 'bands' not in config:
        config['bands'] = ['b', 'lum', 'g', 'r', 'nir', 'u']

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

    if 'star_stamp_size' not in config:
        config['star_stamp_size'] = 1000

    if 'strehl_ratio' not in config:
        config['strehl_ratio'] = 100

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

    # NOTE: haven't made it robust to |theta| > 90
    assert abs(theta.deg) <= 90
    if theta.deg > 90:
        ipdb.set_trace()

    # original box lengths
    Lx = im_xsize.to(u.deg).value
    Ly = im_ysize.to(u.deg).value

    # compute the new, larger box
    new_Lx = Ly * np.sin(theta.rad) + Lx * np.cos(theta.rad)
    new_Ly = Lx * np.sin(theta.rad) + Ly * np.cos(theta.rad)

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
        # 'cluster_gal': make_cluster_galaxy,
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
    qsize = 1.1 * np.max([height_deg.value, width_deg.value])
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

    star_cat = df_stars.to_records()

    return star_cat

def setup_gals(n_gal_total_img, gal_rng):

    cosmos_dir = OBA_SIM_DATA_DIR / 'cosmos_catalog/'
    cosmos_file = cosmos_dir / 'cosmos15_superbit2023_phot_shapes.csv'
    cat_df = pd.read_csv(str(cosmos_file))

    # Filter the data for FLUX_RADIUS > than 0 and HLR < 50
    cat_df = cat_df[cat_df['FLUX_RADIUS'] >= 0]
    cat_df = cat_df[cat_df['c10_sersic_fit_hlr'] < 50]

    # Randomly sample from galaxy catalog
    gal_cat = cat_df.sample(
        n=n_gal_total_img, random_state=gal_rng
        ).to_records()

    return gal_cat

def main(args):

    config_file = args.config_file
    config = utils.read_yaml(config_file)

    config = set_config_defaults(config)

    target_name = config['target_name']
    run_name = config['run_name']
    bands = config['bands']
    starting_roll = config['starting_roll'] * galsim.degrees
    max_fft_size = config['max_fft_size']
    strehl_ratio = config['strehl_ratio']
    ncores = config['ncores']
    fresh = config['fresh']
    overwrite = config['overwrite']
    vb = config['vb']

    run_dir = Path(utils.TEST_DIR, f'ajay/{run_name}/{target_name}/')

    # WARNING: cleans all existing files in run_dir!
    if fresh is True:
        try:
            utils.rm_tree(run_dir)
        except OSError:
            pass

    # setup logger
    logdir = run_dir / target_name
    logfile = str(logdir / f'{run_name}_{target_name}_sim.log')

    log = utils.setup_logger(logfile, logdir=logdir)
    logprint = utils.LogPrint(log, vb=vb)

    # dict of independent seeds for given types
    seeds = setup_seeds(config)

    gs_params = galsim.GSParams(maximum_fft_size=max_fft_size)

    # Dict for pivot wavelengths
    piv_dict = {
        'u': 395.35082727585194,
        'b': 476.22025867791064,
        'g': 596.79880208687230,
        'r': 640.32425638312820,
        'nir': 814.02475812251110,
        'lum': 522.73829660009810
    }

    # Initialize the camera, telescope, bandpass
    camera = inst.Camera('imx455')
    telescope = inst.Telescope('superbit')

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
    img_max_len = np.max([
    # img_max_len = np.min([
        sci_img_size_x_arcsec.value, sci_img_size_y_arcsec.value
        ]) * u.arcsec
    sampled_img_area = (img_max_len**2).to(u.arcmin**2)

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

    if 'add_galaxies' in config:
        add_galaxies = config['add_galaxies']
    else:
        add_galaxies = True

    if 'add_stars' in config:
        add_stars = config['add_stars']
    else:
        add_stars = True

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

    # Start main process
    row = np.where(targets['target'] == target_name)

    target = targets[row][0]

    ra = target['ra'] * u.deg
    dec = target['dec'] * u.deg
    cluster_z = target['z']
    mass = 10**(target['mass']) # solmass
    conc = target['c']

    # setup stars (query only for now)
    if add_stars is True:
        logprint('Setting up stars...')

        star_cat = setup_stars(
            ra.value, dec.value, height_deg, width_deg
            )

        Nstars = len(star_cat)

    # setup gals (sampling positions only for now)
    if add_galaxies is True:

        # we use the larger box to handle roll angles
        sample_len = 1.01*(img_max_len / 2.).to(u.deg).value

        sampled_ra = gal_rng.uniform(
            ra.value - sample_len,
            ra.value + sample_len,
            size=Ngals
            )
        sampled_dec = np.random.uniform(
            dec.value - sample_len,
            dec.value + sample_len,
            size=Ngals
            )

        gal_cat = rf.append_fields(
            gal_cat,
            ['ra', 'dec'],
            [sampled_ra, sampled_dec],
            dtypes=['float32'],
            usemask=False
        )

    # We want to rotate the sky by (5 min + 1 min overhead) each
    # new target. Each band inherits the last roll of the previous band
    theta = starting_roll # already a galsim.Angle
    rot_rate = 0.25 # deg / min

    for band in bands:
        # pivot wavelength
        piv_wave = piv_dict[band]

        bandpass.transmission = get_transmission(band=band)

        aberrations = get_zernike(band=band, strehl_ratio=strehl_ratio)

        optical_zernike_psf = galsim.OpticalPSF(
            lam=piv_wave,
            diam=telescope.diameter.value,
            aberrations=aberrations,
            obscuration=0.38,
            nstruts=4,
            flux=1
            )

        jitter_psf = galsim.Gaussian(sigma=0.05, flux=1)

        psf_sim = galsim.Convolve([optical_zernike_psf, jitter_psf])

        # setup truth cats for this band
        truth_gals = Table()
        truth_stars = Table()

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

            star_cat = rf.append_fields(
                star_cat,
                f'flux_adu_{band}',
                star_flux_adu,
                dtypes='float32',
                usemask=False
            )

        for exp_num in range(n_exp):

            logprint(f'Image simulation starting for {target_name}, {band}; ' +
                    f'{exp_num+1} of {n_exp}')

            if run_name is None:
                rn = ''
            else:
                rn = f'{run_name}/'

            if strehl_ratio == 100:
                sr = ''
            else:
                sr = f'{strehl_ratio}/'

            outdir = os.path.join(
                utils.TEST_DIR, f'ajay/{rn}{target_name}/{band}/{sr}'
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
            ra_bounds, dec_bounds = compute_im_bounding_box(
                ra_sim,
                dec_sim,
                sci_img_size_x_arcsec,
                sci_img_size_y_arcsec,
                theta
                )

            # TODO: use cornish for actual overlap. For now, use a
            # sufficiently large buffer
            ra_buff = .015
            ra_bounds[0] -= ra_buff
            ra_bounds[1] += ra_buff
            dec_buff = ra_buff / 2.
            dec_bounds[0] -= dec_buff
            dec_bounds[1] += dec_buff

            ra_bounds *= u.deg
            dec_bounds *= u.deg

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
                                ra_bounds,
                                dec_bounds,
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
                                ra_bounds,
                                dec_bounds,
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

            # add shot noise on sky + sources
            # TODO: sort out why this turns sci_img to ints...
            noise = galsim.PoissonNoise(sky_level=0.0)
            sci_img.addNoise(noise)
            sci_img = sci_img.array

            # Step 8: Add a dark frame
            dark_dir = OBA_SIM_DATA_DIR / 'darks/minus10/'
            darks = np.sort(glob.glob(str(dark_dir / '*.fits')))
            dark_fname = np.random.choice(darks)

            # we upcast for now, cast back later
            dark = fits.getdata(dark_fname).astype('float32')
            sci_img += dark

            # limit the flux
            sci_img[sci_img >= (2**16)] = 2**16 - 1
            sci_img[sci_img < 0] = 0

            # *now* cast to int16
            sci_img = sci_img.astype('uint16')

            # HEADERS
            hdr = fits.Header()
            hdr['EXPTIME'] = int(exp_time)
            hdr['band'] = band
            hdr['strehl'] = strehl_ratio
            hdr['stars'] = int(add_stars)
            hdr['galaxies'] = int(add_galaxies)
            hdr['TARGET_RA'] = ra.value # in deg
            hdr['TARGET_DEC'] = dec.value # in deg
            hdr['dither_ra'] = dither_ra # in pixels
            hdr['dither_dec'] = dither_dec # in pixels
            hdr['roll_theta'] = theta.deg # in deg
            hdr['dark'] = Path(dark_fname).name

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

            output_fname = f'{outdir}/{target_name}_{exp_time}_{band_int}_{unix_time}.fits'

            fits.writeto(
                filename=output_fname,
                data=sci_img,
                header=hdr,
                overwrite=overwrite
                )

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

    # merge truth cats & save
    if add_stars is True:
        outfile = f'{run_dir}/truth_stars_{target_name}.fits'
        truth_star_cat.write(outfile, overwrite=overwrite)

    if add_galaxies is True:
        outfile = f'{run_dir}/truth_gals_{target_name}.fits'
        truth_gal_cat.write(outfile, overwrite=overwrite)

    return 0

if __name__ == '__main__':
    args = parse_args()
    rc = main(args)

    if rc == 0:
        print('sim_full_focus.py completed succesfully!')
    else:
        print(f'sim_full_focus.py failed w/ return code {rc}')
