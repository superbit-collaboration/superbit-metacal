# Custom modules
#from requests import get
import instrument as inst
import photometry as phot

from stars import make_a_star

# Packages
from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import numpy as np
import galsim
import sys
import os
import glob
from astropy import units as u
from astroquery.vizier import Vizier
import astropy.coordinates as coord
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

        # if exp_num == 1:
            # TODO:
            # row = [i, truth.cosmos_index, truth.x, truth.y,
            #        truth.ra, truth.dec,
            #        truth.g1, truth.g2,
            #        truth.mu,truth.z,
            #        this_flux, truth.fwhm, truth.mom_size,
            #        truth.n, truth.hlr, truth.scale_h_over_r,
            #        truth.obj_class
            #        ]
            # truth_catalog.addRow(row)

    return full_image, truth_catalog

def setup_stars(ra, dec, width_deg, height_deg):

    Vizier.ROW_LIMIT = -1

    coordinates = coord.SkyCoord(
        ra=ra * u.deg,
        dec=dec * u.deg,
        frame='icrs'
        )

    # we add some buffer to account for dithering & rolls
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

    return df_stars

def main(args):

    config_file = args.config_file
    config = utils.read_yaml(config_file)

    if 'run_name' in config:
        run_name = config['run_name']
    else:
        run_name = 'quicktest'

    # WARNING: cleans all existing files in run_dir!
    if 'fresh' in config:
        if config['fresh'] is True:
            run_dir = Path(utils.TEST_DIR, f'ajay/{run_name}')
            try:
                utils.rm_tree(run_dir)
            except OSError:
                pass

    if 'overwrite' in config:
        overwrite = config['overwrite']
    else:
        overwrite = False

    # dict of independent seeds for given types
    seeds = setup_seeds(config)

    if 'max_fft_size' not in config:
        max_fft_size = 2**18
    else:
        max_fft_size = config['max_fft_size']

    big_fft = galsim.GSParams(maximum_fft_size=max_fft_size)

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

    cosmos_dir = OBA_SIM_DATA_DIR / 'cosmos_catalog/'
    cosmos_file = cosmos_dir / 'cosmos15_superbit2023_phot_shapes.csv'
    cat_df = pd.read_csv(str(cosmos_file))

    # Filter the data for FLUX_RADIUS > than 0 and HLR < 50
    cat_df = cat_df[cat_df['FLUX_RADIUS'] >= 0]
    cat_df = cat_df[cat_df['c10_sersic_fit_hlr'] < 50]

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

    bands = ['b', 'lum', 'g', 'r', 'nir', 'u']

    exp_time = config['exp_time'] # seconds
    n_exp = config['n_exp']

    target_list = Path(__file__).parent / 'target_list_full_focus.csv'
    df = pd.read_csv(str(target_list))

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

    n_gal_sqarcmin = 97.55 * (u.arcmin**-2)

    n_gal_total_img = round((sci_img_area * n_gal_sqarcmin).value)

    cosmos_plate_scale = 0.03 # arcsec/pix

    dither_pix = 100
    dither_deg = pix_scale * dither_pix / 3600 # dither_deg

    for idx, row in df.iterrows():
        target = row['target']
        ra = row['ra'] * u.deg
        dec = row['dec'] * u.deg
        z = row['z']
        mass = 10**(row['mass']) # solmass
        conc = row['c']

        # setup stars
        if config['add_stars'] is True:
            print('Setting up stars...')

            df_stars = setup_stars(
                ra.value, dec.value, height_deg, width_deg
                )

            Nstars = len(df_stars)

        for band in bands:
            # We want to rotate the sky by (5 min + 1 min overhead) each
            # new target
            theta_master = 0 * u.radian
            rot_rate = 0.25 # deg / min

            # pivot wavelength
            piv_wave = piv_dict[band]

            bandpass.transmission = get_transmission(band=band)

            # setup stellar fluxes
            if config['add_stars'] is True:
                if piv_wave > 600:
                    gaia_mag = 'Gmag'
                else:
                    gaia_mag = 'BPmag'

                # Find counts to add for the star
                mean_fnu_star_mag = phot.abmag_to_mean_fnu(
                    abmag=df_stars[gaia_mag]
                    )

                mean_flambda = phot.mean_flambda_from_mean_fnu(
                    mean_fnu=mean_fnu_star_mag.to_numpy(),
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

                print(f'Image simulation starting for {target}, {band}; ' +
                      f'{exp_num+1} of {n_exp}')

                if run_name is None:
                    rn = ''
                else:
                    rn = f'{run_name}/'

                outdir = os.path.join(
                    utils.TEST_DIR, f'ajay/{rn}{target}/{band}/{strehl}/'
                    )
                utils.make_dir(outdir)

                ra_sim = np.random.uniform(
                    ra.value - dither_deg, ra.value + dither_deg
                    )
                dec_sim = np.random.uniform(
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

                # Step 5: Add stars (setup done earlier)
                if config['add_stars'] is True:

                    start = time.time()

                    print(f'Adding {Nstars} stars')

                    # NOTE: in progres...
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

                    for idx in range(len(df_stars)):

                        if idx % 100 == 0:
                            print(f'{idx} of {len(df_stars)} completed')

                        this_flux_adu = star_flux_adu[idx]

                        # Assign real position to the star on the sky
                        star_ra_deg = df_stars['RA_ICRS'][idx] * galsim.degrees
                        star_dec_deg = df_stars['DE_ICRS'][idx] * galsim.degrees

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

                        star_image = convolution.drawImage(
                            nx=1000,
                            ny=1000,
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
                            # print('Out of bounds star. Skipping.')
                            continue

                    print('Done adding stars.')

                    star_time = time.time() - start
                    print(f'Stars took {star_time:.1f} s')

                # step 6
                if config['add_galaxies'] is True:

                    start = time.time()

                    sci_img_ra_min = (ra_sim * u.deg - ((sci_img_size_x_arcsec.to(u.arcsecond).to(u.deg)) / 2)).to(u.deg)
                    sci_img_ra_max = (ra_sim * u.deg + ((sci_img_size_x_arcsec.to(u.arcsecond).to(u.deg)) / 2)).to(u.deg)

                    sci_img_dec_min = (dec_sim * u.deg - ((sci_img_size_y_arcsec.to(u.arcsecond).to(u.deg)) / 2)).to(u.deg)
                    sci_img_dec_max = (dec_sim * u.deg + ((sci_img_size_y_arcsec.to(u.arcsecond).to(u.deg)) / 2)).to(u.deg)
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

                    # Only inject bkg galaxies
                    cat_df_filt = cat_df[cat_df['ZPDF'] >= z]

                    print(f'Adding {n_gal_total_img} background galaxies')

                    seed = np.arange(0, n_gal_total_img, 1)

                    for idx in range(n_gal_total_img):
                        if idx % 100 == 0:
                            print(f'{idx} of {n_gal_total_img} completed')

                        # Randomly sample a galaxy
                        gal_df = cat_df_filt.sample(n=1, random_state=seed[idx])
                        gal_z = gal_df.ZPDF.item()

                        crate_dict = {
                            'b': gal_df.crates_b.item(),
                            'u': gal_df.crates_u.item(),
                            'r': gal_df.crates_r.item(),
                            'nir': gal_df.crates_nir.item(),
                            'lum': gal_df.crates_lum.item(),
                            'g': gal_df.crates_g.item()
                            }

                        # Get its Sersic parameters
                        gal_n_sersic_cosmos10 = gal_df.c10_sersic_fit_n.item()

                        if gal_n_sersic_cosmos10 < 0.3:
                            gal_n_sersic_cosmos10 = 0.3

                        crate = crate_dict[band] # galaxy count rate e/s

                        flux_adu = int(
                            crate * exp_time * 1 / camera.gain.value
                            )

                        hlr_arcsec = (gal_df.c10_sersic_fit_hlr.item()
                                    * np.sqrt(gal_df.c10_sersic_fit_q.item())
                                    * cosmos_plate_scale)

                        if hlr_arcsec <= 0:
                            continue

                        hlr_pixels = hlr_arcsec / pix_scale

                        gal = galsim.Sersic(
                            n=gal_n_sersic_cosmos10,
                            half_light_radius=hlr_arcsec,
                            flux=flux_adu
                            )

                        gal = gal.shear(
                            q=gal_df.c10_sersic_fit_q.item(),
                            beta=gal_df.c10_sersic_fit_phi.item()*galsim.radians
                            )

                        # Set the seed, so that each observation have same galaxy locations
                        np.random.seed(seed[idx])

                        gal_ra = np.random.uniform(
                            sci_img_ra_min.value, sci_img_ra_max.value
                            )
                        gal_dec = np.random.uniform(
                            sci_img_dec_min.value, sci_img_dec_max.value
                            )

                        gal_ra *= galsim.degrees
                        gal_dec *= galsim.degrees

                        world_pos = galsim.CelestialCoord(gal_ra, gal_dec)
                        image_pos = wcs.toImage(world_pos)

                        # galaxy 2d position on sci img in arcsec
                        gal_pos_cent_xasec = image_pos.x * pix_scale
                        gal_pos_cent_yasec = image_pos.y * pix_scale

                        # get the expected shear from the halo at galaxy 2d position
                        g1, g2 = nfw_halo.getShear(
                            pos=galsim.PositionD(
                                x=gal_pos_cent_xasec,
                                y=gal_pos_cent_yasec
                                ),
                            z_s=gal_z,
                            units=galsim.arcsec,
                            reduced=True
                            )

                        abs_val = np.sqrt(g1**2 + g2**2)

                        if abs_val >= 1:  # strong lensing
                            g1 = 0
                            g2 = 0

                        nfw_mu = nfw_halo.getMagnification(
                            pos=galsim.PositionD(
                                x=gal_pos_cent_xasec,
                                y=gal_pos_cent_yasec
                                ),
                            z_s=gal_z,
                            units=galsim.arcsec
                            )

                        # strong lensing
                        if nfw_mu < 0:
                            print('Warning: mu < 0 means strong lensing! Using mu=25.')
                            nfw_mu = 25
                        elif nfw_mu > 25:
                            print('Warning: mu > 25 means strong lensing! Using mu=25.')
                            nfw_mu = 25

                        # lens the Sersic object galaxy
                        gal = gal.lens(g1=g1, g2=g2, mu=nfw_mu)

                        # Position fractional stuff
                        x_nominal = image_pos.x + 0.5
                        y_nominal = image_pos.y + 0.5

                        ix_nominal = int(math.floor(x_nominal+0.5))
                        iy_nominal = int(math.floor(y_nominal+0.5))

                        dx = x_nominal - ix_nominal
                        dy = y_nominal - iy_nominal

                        offset = galsim.PositionD(dx,dy)

                        # convolve galaxy GSObject with the psf (optical+jitter convolved already)
                        convolution = galsim.Convolve(
                            [psf_sim, gal], gsparams=big_fft
                            )
                        stamp_size = round(hlr_pixels + 500)

                        gal_image = convolution.drawImage(
                            nx=stamp_size,
                            ny=stamp_size,
                            wcs=wcs.local(image_pos),
                            offset=offset,
                            method='auto',
                            # dtype=np.uint16
                            )

                        gal_image.setCenter(ix_nominal, iy_nominal)

                        stamp_overlap = gal_image.bounds & sci_img.bounds

                        # Check to ensure galaxy is not out of bounds on the image
                        try:
                            sci_img[stamp_overlap] += gal_image[stamp_overlap]

                        except galsim.errors.GalSimBoundsError:
                            print('Out of bounds star. Skipping.')
                            continue

                    gal_time = time.time() - start
                    print('Done with galaxies')
                    print(f'Galaxies took {gal_time:.1f} s')


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
                hdr['stars'] = int(config['add_stars'])
                hdr['galaxies'] = int(config['add_galaxies'])
                hdr['TARGET_RA'] = ra.value # in deg
                hdr['TARGET_DEC'] = dec.value # in deg
                hdr['dither_ra'] = dither_ra # in pixels
                hdr['dither_dec'] = dither_dec # in pixels
                hdr['roll_theta'] = theta.deg # in deg
                hdr['dark'] = Path(dark_fname).name

                dt_now = datetime.datetime.now()
                unix_time = int(time.mktime(dt_now.timetuple()))

                band_int = oba_io.band2index(band)

                # Path checks
                Path(outdir).mkdir(parents=True, exist_ok=True)

                output_fname = f'{outdir}/{target}_{exp_time}_{band_int}_{unix_time}.fits'

                fits.writeto(
                    filename=output_fname,
                    data=sci_img,
                    header=hdr,
                    overwrite=overwrite
                    )

                # Update the roll angle
                if strehl == 100:
                    theta_master += ((rot_rate * 6) * u.deg).to(u.radian)

                print(f'Image simulation complete for {target}, {band}, {strehl}.\n')

if __name__ == '__main__':
    args = parse_args()
    rc = main(args)

    if rc == 0:
        print('sim_full_focus.py completed succesfully!')
    else:
        print(f'sim_full_focus.py failed w/ return code {rc}')
