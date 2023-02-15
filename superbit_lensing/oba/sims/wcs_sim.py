# Custom modules
#from requests import get
import instrument as inst
import photometry as phot

# Packages
import pandas as pd
import numpy as np
import galsim
import sys
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

def get_config_file(config_file):
    "Read config yaml file"
    with open(config_file, 'r') as stream:
        try:
            config_data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return config_data

def get_piv(band):
    if band == 'u':
        return 395.35082727585194
    elif band == 'b':
        return 476.22025867791064
    elif band == 'g':
        return 596.79880208687230
    elif band == 'r':
        return 640.32425638312820
    elif band == 'nir':
        return 814.02475812251110
    elif band == 'lum':
        return 522.73829660009810
    else:
        raise ValueError("Invalid band.")

def get_transmission(band):
    if band == 'u':
        return np.genfromtxt('data/instrument/bandpass/u_2023.csv', delimiter=',')[:, 2][1:]
    elif band == 'b':
        return np.genfromtxt('data/instrument/bandpass/b_2023.csv', delimiter=',')[:, 2][1:]
    elif band == 'g':
        return np.genfromtxt('data/instrument/bandpass/g_2023.csv', delimiter=',')[:, 2][1:]
    elif band == 'r':
        return np.genfromtxt('data/instrument/bandpass/r_2023.csv', delimiter=',')[:, 2][1:]
    elif band == 'nir':
        return np.genfromtxt('data/instrument/bandpass/nir_2023.csv', delimiter=',')[:, 2][1:]
    elif band == 'lum':
        return np.genfromtxt('data/instrument/bandpass/lum_2023.csv', delimiter=',')[:, 2][1:]
    else:
        raise ValueError("Invalid band.")

def get_zernike(band, strehl_ratio):
    return np.genfromtxt(fname="data/psf/" 
                         + band 
                         + "_" 
                         + str(int(strehl_ratio)) 
                         + ".csv", delimiter=',')[:, 1][1:]

def main():
    big_fft = galsim.GSParams(maximum_fft_size=300000)
    cf = get_config_file("config.yaml")
    img_check_file = str(sys.argv[1])
    config_df = pd.read_csv(img_check_file)

    cat_df = pd.read_csv(cf['catalog_file'])

    # Filter the data for FLUX_RADIUS > than 0 and HLR < 50
    cat_df = cat_df[cat_df['FLUX_RADIUS'] >= 0]
    cat_df = cat_df[cat_df['c10_sersic_fit_hlr'] < 50]

    # Initialize the camera, telescope, bandpass
    camera = inst.Camera(cf['camera'])
    telescope = inst.Telescope('superbit')
    bandpass = inst.Bandpass('bandpass')
    bandpass.wavelengths = camera.wavelengths

    for index, row in config_df.iterrows():
        print(index, row)
        bandpass.transmission = get_transmission(band=row['band'])
        bandpass.plate_scale = ( ((206265 * camera.pixel_size.to(u.micron).value)
                                    / (1000 * telescope.focal_length.to(u.mm))).value 
                                    * (u.arcsec/u.pixel) )

        # PSF
        aberrations = get_zernike(band=row['band'], strehl_ratio=row['strehl_ratio'])
        piv_wave = get_piv(band=row['band'])

        optical_zernike_psf = galsim.OpticalPSF(lam=piv_wave,
                                                diam=telescope.diameter.value,
                                                aberrations=aberrations,
                                                obscuration=0.38,
                                                nstruts=4,
                                                flux=1)

        optical_airy = galsim.Airy(lam=piv_wave, diam=telescope.diameter.value)
        jitter_psf = galsim.Gaussian(sigma=cf['pjit'], flux=1)

        if row['psf'] == 'total_zernike':
            psf_sim = galsim.Convolve([optical_zernike_psf, jitter_psf])
        elif row['psf'] == 'total_airy':
            psf_sim = galsim.Convolve([optical_airy, jitter_psf])
        elif row['psf'] == 'optical_zernike':
            psf_sim = optical_zernike_psf
        elif row['psf'] == 'optical_airy':
            psf_sim = optical_airy
        else:
            raise ValueError("Invalid PSF type specified.")
    
        # Construct the science image
        sci_img = galsim.Image(ncol=camera.npix_H.value,
                               nrow=camera.npix_V.value,
                               dtype=np.uint16)

        # Fill the science image with zeros
        sci_img.fill(0)
        sci_img.setOrigin(0, 0)
    
        # Coordinates/wcs
        cluster_ra_deg = cf['halo_ra'] * u.deg
        cluster_dec_deg = cf['halo_dec'] * u.deg

        # Number of background galaxies to add
        sci_img_size_x_arcsec = camera.npix_H * bandpass.plate_scale
        sci_img_size_y_arcsec = camera.npix_V * bandpass.plate_scale
        sci_img_area = (sci_img_size_x_arcsec * sci_img_size_y_arcsec).to(u.arcmin**2)

        n_gal_sqarcmin = 97.55 * (u.arcmin**-2)
        n_gal_total_img = round((sci_img_area * n_gal_sqarcmin).value)

        # Only inject bkg galaxies
        cat_df_filt = cat_df[cat_df['ZPDF'] >= cf['halo_z']] 

        # setup NFW halo
        nfw_halo_pos_xasec = (camera.npix_H.value/2) * bandpass.plate_scale.value
        nfw_halo_pos_yasec = (camera.npix_V.value/2) * bandpass.plate_scale.value

        nfw_halo = galsim.NFWHalo(mass=cf['halo_mass'],
                                  conc=cf['halo_conc'],
                                  redshift=cf['halo_z'],
                                  omega_m=0.3,
                                  halo_pos=galsim.PositionD(x=nfw_halo_pos_xasec, y=nfw_halo_pos_yasec),
                                  omega_lam=0.7)

        # Rotation 
        theta = 0 * galsim.degrees
        dudx = np.cos(theta) * bandpass.plate_scale.value
        dudy = -np.sin(theta) * bandpass.plate_scale.value
        dvdx = np.sin(theta) * bandpass.plate_scale.value
        dvdy = np.cos(theta) * bandpass.plate_scale.value
        image_center = sci_img.true_center
        affine = galsim.AffineTransform(dudx, dudy, dvdx, dvdy, origin=image_center) 

        sky_center = galsim.CelestialCoord(ra=cluster_ra_deg.value * galsim.degrees, 
                                        dec=cluster_dec_deg.value * galsim.degrees)
        wcs = galsim.TanWCS(affine, sky_center, units=galsim.arcsec)
        sci_img.wcs = wcs

        sci_img_size_x_arcsec = camera.npix_H.value * bandpass.plate_scale.value * u.arcsecond
        sci_img_size_y_arcsec = camera.npix_V.value * bandpass.plate_scale.value * u.arcsecond
        
        sci_img_ra_min = (cluster_ra_deg - ((sci_img_size_x_arcsec.to(u.arcsecond).to(u.deg)) / 2)).to(u.deg)
        sci_img_ra_max = (cluster_ra_deg + ((sci_img_size_x_arcsec.to(u.arcsecond).to(u.deg)) / 2)).to(u.deg)
        sci_img_dec_min = (cluster_dec_deg - ((sci_img_size_y_arcsec.to(u.arcsecond).to(u.deg)) / 2)).to(u.deg)
        sci_img_dec_max = (cluster_dec_deg + ((sci_img_size_y_arcsec.to(u.arcsecond).to(u.deg)) / 2)).to(u.deg)

        exp_time = cf['exp_time']
        n_exp = cf['n_exp']
    
        # if cf['add_read']:
        #     read_noise_std = camera.read_noise.value * np.sqrt(n_exp)
        #     sci_img += phot.get_read_noise(image_shape=sci_img.array.shape,
        #                                     read_noise_std=read_noise_std,
        #                                     gain=camera.gain.value)
        
        # add dark frame (from a randomly sampled 300 sec dark frame)
        
        if cf['add_dark']:
            print("Adding dark.")
            dark_dir = "/home/gill/sims/data/darks/minus10/*"
            dark_fname = random.choice(np.sort(glob.glob(dark_dir)))
            print("Dark filename is: {}".format(dark_fname))
            dark_data = fits.getdata(dark_fname)
            sci_img += dark_data

        # add sky noise
        if cf['add_sky']:
            print("Adding sky.")
            crate_sky_electron_pix = phot.crate_bkg(illum_area=telescope.illum_area,
                                                    bandpass=bandpass,
                                                    bkg_type='raw',
                                                    strength='ave')

            crate_sky_adu_pix = crate_sky_electron_pix / camera.gain.value
            sky_adu_pix = crate_sky_adu_pix * exp_time * n_exp  # ADU

            sci_img += phot.get_sky_bkg(image_shape=sci_img.array.shape,
                                        sky_adu_pix=sky_adu_pix)

        # STARS (optional)
        if cf['add_stars']:
            Vizier.ROW_LIMIT = -1
            coordinates = coord.SkyCoord(ra=cluster_ra_deg,
                                         dec=cluster_dec_deg,
                                         frame='icrs')

            height = ((sci_img_size_y_arcsec.value) * u.arcsec).to(u.deg)
            width = ((sci_img_size_x_arcsec.value) * u.arcsec).to(u.deg)

            result = Vizier.query_region(coordinates=coordinates,
                                         height=height,
                                         width=width,
                                         catalog='I/345/gaia2')
            result = result[0].filled()

            df_stars = pd.DataFrame()
            df_stars['RA_ICRS'] = result['RA_ICRS']
            df_stars['DE_ICRS'] = result['DE_ICRS']
            df_stars['FG'] = result['FG']
            df_stars['FBP'] = result['FBP']
            df_stars['BPmag'] = (-2.5 * np.log10(result['FBP'])) + cf['ZP_BP']
            df_stars['Gmag'] = (-2.5 * np.log10(result['FG'])) + cf['ZP_G']
            df_stars.dropna()
            df_stars = df_stars[df_stars['Gmag'] >= -5]
            df_stars = df_stars[df_stars['BPmag'] >= -5]
            df_stars = df_stars.reset_index(drop=True)
            
            print("Adding {} stars".format(len(df_stars)))

            for idx in range(len(df_stars)):
                if idx % 100 == 0:
                    print("Number of stars added: {} of {}".format(idx, len(df_stars)))

                if piv_wave > 600:
                    gaia_mag = 'Gmag'
                else:
                    gaia_mag = 'BPmag'

                mean_fnu_star_mag = phot.abmag_to_mean_fnu(abmag=df_stars[gaia_mag][idx])

                mean_flambda = phot.mean_flambda_from_mean_fnu(mean_fnu=mean_fnu_star_mag,
                                                               bandpass_transmission=bandpass.transmission,
                                                               bandpass_wavelengths=bandpass.wavelengths)
                crate_electrons_pix = phot.crate_from_mean_flambda(mean_flambda=mean_flambda,
                                                                   illum_area=telescope.illum_area.value,
                                                                   bandpass_transmission=bandpass.transmission,
                                                                   bandpass_wavelengths=bandpass.wavelengths)
                crate_adu_pix = crate_electrons_pix / camera.gain.value
                flux_adu = int(crate_adu_pix * exp_time * n_exp)

                if flux_adu > 2**16 - 1:
                    flux_adu = 2**16  - 1

                star_ra_deg = df_stars['RA_ICRS'][idx] * galsim.degrees
                star_dec_deg = df_stars['DE_ICRS'][idx] * galsim.degrees
                world_pos = galsim.CelestialCoord(star_ra_deg, star_dec_deg)

                image_pos = wcs.toImage(world_pos)  
                
                star = galsim.DeltaFunction(flux=int(flux_adu))

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
                star_image = convolution.drawImage(nx=1000, ny=1000,
                                                   wcs=wcs.local(image_pos),
                                                   offset=offset,
                                                   method='auto',
                                                   dtype=np.uint16)
                    
                star_image.setCenter(ix_nominal, iy_nominal)

                stamp_overlap = star_image.bounds & sci_img.bounds
                sci_img[stamp_overlap] += star_image[stamp_overlap]
            print("Done adding stars.")

        if cf['add_galaxies']:
            print("Adding {} background galaxies".format(n_gal_total_img))

            seed = np.arange(0, n_gal_total_img, 1)

            for idx in range(n_gal_total_img):
                if idx % 1000 == 0:
                    print("Galaxies added: {} of {}".format(idx, n_gal_total_img))

                # Randomly sample a galaxy
                gal_df = cat_df_filt.sample(n=1, random_state=seed[idx])
                gal_z = gal_df.ZPDF.item()

                # Get its Sersic parameters
                gal_n_sersic_cosmos10 = gal_df.c10_sersic_fit_n.item()

                if gal_n_sersic_cosmos10 < 0.3:
                    gal_n_sersic_cosmos10 = 0.3

                # crate
                if row['band'] == 'b':
                    crate = gal_df.crates_b.item()
                elif row['band'] == 'g':
                    crate = gal_df.crates_g.item()
                elif row['band'] == 'u':
                    crate = gal_df.crates_u.item()
                elif row['band'] == 'r':
                    crate = gal_df.crates_r.item()
                elif row['band'] == 'nir':
                    crate = gal_df.crates_nir.item()
                elif row['band'] == 'lum':
                    crate = gal_df.crates_lum.item()

                flux_adu = int(crate * cf['exp_time'] * cf['n_exp'] / camera.gain.value)
                hlr_arcsec = (gal_df.c10_sersic_fit_hlr.item()
                            * np.sqrt(gal_df.c10_sersic_fit_q.item())
                            * cf['cosmos_plate_scale'])

                if hlr_arcsec <= 0:
                    continue

                hlr_pixels = hlr_arcsec / bandpass.plate_scale.value
                gal = galsim.Sersic(n=gal_n_sersic_cosmos10,
                                    half_light_radius=hlr_arcsec,
                                    flux=flux_adu)

                gal = gal.shear(q=gal_df.c10_sersic_fit_q.item(),
                                beta=gal_df.c10_sersic_fit_phi.item()*galsim.radians)

                # Set the seed, so that each observation have same galaxy locations
                np.random.seed(seed[idx])

                gal_ra = np.random.uniform(sci_img_ra_min.value, sci_img_ra_max.value)
                gal_dec = np.random.uniform(sci_img_dec_min.value, sci_img_dec_max.value)

                gal_ra *= galsim.degrees
                gal_dec *= galsim.degrees
            
                world_pos = galsim.CelestialCoord(gal_ra, gal_dec)
                image_pos = wcs.toImage(world_pos)  

                # galaxy 2d position on sci img in arcsec
                gal_pos_cent_xasec = image_pos.x * bandpass.plate_scale.value
                gal_pos_cent_yasec = image_pos.y * bandpass.plate_scale.value

                # get the expected shear from the halo at galaxy 2d position
                g1, g2 = nfw_halo.getShear(pos=galsim.PositionD(x=gal_pos_cent_xasec,
                                                                y=gal_pos_cent_yasec),
                                                                z_s=gal_z,
                                                                units=galsim.arcsec,
                                                                reduced=True)
                abs_val = np.sqrt(g1**2 + g2**2)

                if abs_val >= 1:  # strong lensing
                    g1 = 0
                    g2 = 0

                nfw_mu = nfw_halo.getMagnification(pos=galsim.PositionD(x=gal_pos_cent_xasec,
                                                                        y=gal_pos_cent_yasec),
                                                                        z_s=gal_z,
                                                                        units=galsim.arcsec)
                # strong lensing
                if nfw_mu < 0:
                    print("Warning: mu < 0 means strong lensing! Using mu=25.")
                    nfw_mu = 25
                elif nfw_mu > 25:
                    print("Warning: mu > 25 means strong lensing! Using mu=25.")
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
                convolution = galsim.Convolve([psf_sim, gal], gsparams=big_fft)
                stamp_size = round(hlr_pixels + 500)
                gal_image = convolution.drawImage(nx=stamp_size, ny=stamp_size,
                                                wcs=wcs.local(image_pos),
                                                offset=offset,
                                                method='auto',
                                                dtype=np.uint16)
                    
                gal_image.setCenter(ix_nominal, iy_nominal)
                stamp_overlap = gal_image.bounds & sci_img.bounds
                sci_img[stamp_overlap] += gal_image[stamp_overlap]

        # HEADERS
        hdr = fits.Header()
        hdr['EXPTIME'] = int(cf['exp_time'])
        hdr['NEXP'] = int(cf['n_exp'])
        hdr['band'] = row['band']
        hdr['strehl'] = row['strehl_ratio']
        hdr['stars'] = int(cf['add_stars'])
        hdr['galaxies'] = int(cf['add_galaxies'])
        hdr['psf_type'] = row['psf']
        odir = "/home/gill/sims/data/sim_data/"
        band = row['band']
        psf = row['psf']
        strehl = row['strehl_ratio']
        dt_now = datetime.datetime.now()
        unix_time = int(time.mktime(dt_now.timetuple()))
        tar_name = "bullet"
        exp_time = int(cf['exp_time'])
        bands_dict = {'u':0, 'b':1, 'g':2, "dark":3, 'r':4, "nir":5, "lum":6}
        band_int = bands_dict[band]

        output_fname = f"{odir}{band}/{tar_name}_{exp_time}_{band_int}_{unix_time}_{strehl}.fits"
        
        fits.writeto(filename=output_fname,
                        data=sci_img.array,
                        overwrite=True)

        print(f"Image simulation complete for {band}, {psf}, {strehl}.\n")

if __name__ == "__main__":
    main()
