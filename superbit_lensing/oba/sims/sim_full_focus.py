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

def compute_im_bounding_box(ra, dec, im_xsize, im_ysize, theta):
    '''
    ra, dec, theta: deg
    im_xsize, im_ysize: arcsec

    Use the roll angle theta to determine the box inscribing the rotated
    rectangular SCI image

    returns: ra_bounds, dec_bounds
        The new (conservative) ra/dec bounds, in deg
    '''

    # NOTE: haven't made it robust to |theta| > 90
    assert abs(theta) <= 90

    # original box lengths
    Lx = im_xsize.to(u.deg).value
    Ly = im_ysize.to(u.deg).value

    # to rad
    theta_rad = np.deg2rad(theta)

    # compute the new, larger box
    new_Lx = Ly * np.sin(theta_rad) + Lx * np.cos(theta_rad)
    new_Ly = Lx * np.sin(theta_rad) + Ly * np.cos(theta_rad)

    ra_min = ra - new_Lx/2.
    ra_max = ra + new_Lx/2.

    dec_min = dec - new_Ly/2.
    dec_max = dec + new_Ly/2.

    ra_bounds = [ra_min, ra_max] * u.deg
    dec_bounds = [dec_min, dec_max] *u.deg

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
        logprint(f'Starting {obj_type} {i}')
        stamp, truth = func(obj_index, obj, *args, **kwargs)
        logprint(f'{obj_type} {i} completed succesfully')

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
            logprint(f'{indx} out of bounds. Skipping.')

        if exp_num == 0:
            if len(truth_catalog) == 0:
                truth_catalog = Table(truth)
            else:
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

    # required
    target_name = config['target_name']

    if 'run_name' in config:
        run_name = config['run_name']
    else:
        run_name = 'quicktest'

    run_dir = Path(utils.TEST_DIR, f'ajay/{run_name}')

    # WARNING: cleans all existing files in run_dir!
    if 'fresh' in config:
        if config['fresh'] is True:
            try:
                utils.rm_tree(run_dir)
            except OSError:
                pass

    if 'overwrite' in config:
        overwrite = config['overwrite']
    else:
        overwrite = False

    if 'vb' in config:
        vb = config['vb']
    else:
        vb = False

    # setup logger
    logdir = run_dir
    logfile = str(logdir / f'{run_name}_sim.log')

    log = utils.setup_logger(logfile, logdir=logdir)
    logprint = utils.LogPrint(log, vb=vb)

    # dict of independent seeds for given types
    seeds = setup_seeds(config)

    if 'max_fft_size' not in config:
        max_fft_size = 2**18
    else:
        max_fft_size = config['max_fft_size']

    gs_params = galsim.GSParams(maximum_fft_size=max_fft_size)

    if 'ncores' in config:
        ncores = config['ncores']
    else:
        ncores = 1

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

    if 'bands' in config:
        bands = config['bands']
    else:
        bands = ['b', 'lum', 'g', 'r', 'nir', 'u']

    exp_time = config['exp_time'] # seconds
    n_exp = config['n_exp']

    target_list = Path(__file__).parent / 'target_list_full_focus.csv'
    # df = pd.read_csv(str(target_list))
    targets = Table.read(str(target_list))

    # Do 12 exposures at full focus (Strehl ratio of 1)
    strehl_ratios = (np.full(n_exp, 100))

    # Number of background galaxies to add
    sci_img_size_x_arcsec = camera.npix_H * bandpass.plate_scale
    sci_img_size_y_arcsec = camera.npix_V * bandpass.plate_scale

    height_deg = ((sci_img_size_y_arcsec.value) * u.arcsec).to(u.deg)
    width_deg = ((sci_img_size_x_arcsec.value) * u.arcsec).to(u.deg)

    sci_img_area = (sci_img_size_x_arcsec * sci_img_size_y_arcsec).to(
        u.arcmin**2
        )

    # the *sampled* area has to be bigger than this, to allow for rolls
    # TODO: Revert!!
    # img_max_len = np.max([
    img_max_len = np.min([
        sci_img_size_x_arcsec.value, sci_img_size_y_arcsec.value
        ]) * u.arcsec
    sampled_img_area = (img_max_len**2).to(u.arcmin**2)

    n_gal_sqarcmin = 97.55 * (u.arcmin**-2)

    # TODO: Revert!!
    # n_gal_total_img = round((sampled_img_area * n_gal_sqarcmin).value)
    n_gal_total_img = 300

    cosmos_plate_scale = 0.03 # arcsec/pix

    # TODO: Revert!!
    dither_pix = 1
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
    z = target['z']
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

        sample_len = (img_max_len / 2.).to(u.deg).value

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

    for band in bands:
        # We want to rotate the sky by (5 min + 1 min overhead) each
        # new target
        theta_master = 0 * u.radian
        rot_rate = 0.25 # deg / min

        # pivot wavelength
        piv_wave = piv_dict[band]

        bandpass.transmission = get_transmission(band=band)

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

        for exp_num, strehl in enumerate(strehl_ratios):

            logprint(f'Image simulation starting for {target_name}, {band}; ' +
                    f'{exp_num+1} of {n_exp}')

            if run_name is None:
                rn = ''
            else:
                rn = f'{run_name}/'

            outdir = os.path.join(
                utils.TEST_DIR, f'ajay/{rn}{target_name}/{band}/{strehl}/'
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

            # Step 1: Get the optical PSF, given the Strehl ratio
            aberrations = get_zernike(band=band, strehl_ratio=strehl)

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

            # Step 2: Construct the science image
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
            if strehl < 100:
                theta = 0 * galsim.degrees
            else:
                theta = theta_master.to(u.deg).value * galsim.degrees

            dudx = np.cos(theta) * pix_scale
            dudy = -np.sin(theta) * pix_scale
            dvdx = np.sin(theta) * pix_scale
            dvdy = np.cos(theta) * pix_scale

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

            # Step 5: Setup image bounds & cluster halo
            # NOTE: these define the minimum bounding box of the (possibly
            # rotated) image. This will allow us to skip galaxies
            # off of the current image more efficiently
            ra_bounds, dec_bounds = compute_im_bounding_box(
                ra_sim,
                dec_sim,
                sci_img_size_x_arcsec,
                sci_img_size_y_arcsec,
                theta.deg
                )

            ra_bounds *= u.deg
            dec_bounds *= u.deg

            # setup NFW halo at the center
            nfw_halo_pos_xasec = (camera.npix_H.value/2) * pix_scale
            nfw_halo_pos_yasec = (camera.npix_V.value/2) * pix_scale

            halo_pos = galsim.PositionD(
                x=nfw_halo_pos_xasec, y=nfw_halo_pos_yasec
                )

            nfw_halo = galsim.NFWHalo(
                mass=mass,
                conc=conc,
                redshift=z,
                omega_m=0.3,
                halo_pos=halo_pos,
                omega_lam=0.7
                )

            # Step 6: Add stars (setup done earlier)
            if add_stars is True:

                start = time.time()

                logprint(f'Adding {Nstars} stars')

                # NOTE: in progress...
                # with Pool(ncores) as pool:
                #     batch_indices = utils.setup_batches(Nobjs, ncores)

                #     full_image, truth_catalog = combine_objs(
                #         pool.starmap(
                #             make_obj_runner,
                #             ([
                #                 batch_indices,
                #                 'star',
                #                 wcs,
                #             ])
                #         )

                for idx in range(len(star_cat)):

                    if idx % 100 == 0:
                        logprint(f'{idx} of {len(star_cat)} completed')

                    this_flux_adu = star_flux_adu[idx]

                    # Assign real position to the star on the sky
                    star_ra_deg = star_cat['RA_ICRS'][idx] * galsim.degrees
                    star_dec_deg = star_cat['DE_ICRS'][idx] * galsim.degrees

                    world_pos = galsim.CelestialCoord(
                        star_ra_deg, star_dec_deg
                        )
                    image_pos = wcs.toImage(world_pos)

                    star = galsim.DeltaFunction(flux=this_flux_adu)

                    # Position fractional stuff
                    x_nominal = image_pos.x + 0.5
                    y_nominal = image_pos.y + 0.5

                    ix_nominal = int(math.floor(x_nominal+0.5))
                    iy_nominal = int(math.floor(y_nominal+0.5))

                    dx = x_nominal - ix_nominal
                    dy = y_nominal - iy_nominal

                    offset = galsim.PositionD(dx,dy)

                    # convolve star with the psf
                    convolution = galsim.Convolve([psf_sim, star])

                    # TODO: figure out stamp size issue...
                    star_image = convolution.drawImage(
                        # nx=1000,
                        # ny=1000,
                        wcs=wcs.local(image_pos),
                        offset=offset,
                        method='auto',
                        # dtype=np.uint16
                        )

                    star_image.setCenter(ix_nominal, iy_nominal)

                    stamp_overlap = star_image.bounds & sci_img.bounds

                    # Check to ensure star is not out of bounds on the image
                    try:
                        sci_img[stamp_overlap] += star_image[stamp_overlap]

                    except galsim.errors.GalSimBoundsError:
                        # logprint('Out of bounds star. Skipping.')
                        continue

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
                                psf_sim,
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
            hdr['strehl'] = strehl
            hdr['stars'] = int(add_stars)
            hdr['galaxies'] = int(add_galaxies)
            hdr['TARGET_RA'] = ra.value # in deg
            hdr['TARGET_DEC'] = dec.value # in deg
            hdr['dither_ra'] = dither_ra # in pixels
            hdr['dither_dec'] = dither_dec # in pixels
            hdr['roll_theta'] = theta.deg # in deg
            hdr['dark'] = Path(dark_fname).name

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

            # Update the roll angle
            if strehl == 100:
                theta_master += ((rot_rate * 6) * u.deg).to(u.radian)

            logprint(f'Image simulation complete for {target_name}, {band}, {strehl}.\n')

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
        outfile = f'{run_dir}/truth_stars.fits'
        truth_star_cat.write(outfile, overwrite=overwrite)
    if add_galaxies is True:
        outfile = f'{run_dir}/truth_gals.fits'
        truth_gal_cat.write(outfile, overwrite=overwrite)

    return 0

if __name__ == '__main__':
    args = parse_args()
    rc = main(args)

    if rc == 0:
        print('sim_full_focus.py completed succesfully!')
    else:
        print(f'sim_full_focus.py failed w/ return code {rc}')
