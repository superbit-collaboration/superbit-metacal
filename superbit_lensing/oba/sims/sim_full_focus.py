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

from pathlib import Path

def get_transmission(band):
    return np.genfromtxt(f'data/instrument/bandpass/{band}_2023.csv', delimiter=',')[:, 2][1:]

def get_zernike(band, strehl_ratio):
    return np.genfromtxt(fname="data/psf/" 
                         + band 
                         + "_" 
                         + str(int(strehl_ratio)) 
                         + ".csv", delimiter=',')[:, 1][1:]

def main():

    # Dict for pivot wavelengths
    piv_dict = {'u': 395.35082727585194, 
                'b': 476.22025867791064, 
                'g': 596.79880208687230,
                'r': 640.32425638312820, 
                'nir': 814.02475812251110, 
                'lum': 522.73829660009810}
    
    bands_dict = {'u':0, 'b':1, 'g':2, 
                  "dark":3, 'r':4, 
                  "nir":5, "lum":6}
    
    big_fft = galsim.GSParams(maximum_fft_size=300000)
    
    cat_df = pd.read_csv("/home/gill/sims/data/cosmos_catalog/cosmos15_superbit2023_phot_shapes.csv")

    # Filter the data for FLUX_RADIUS > than 0 and HLR < 50
    cat_df = cat_df[cat_df['FLUX_RADIUS'] >= 0]
    cat_df = cat_df[cat_df['c10_sersic_fit_hlr'] < 50]

    # Initialize the camera, telescope, bandpass
    camera = inst.Camera("imx455")
    telescope = inst.Telescope('superbit')

    bandpass = inst.Bandpass('bandpass')
    bandpass.wavelengths = camera.wavelengths
    bandpass.plate_scale = ( ((206265 * camera.pixel_size.to(u.micron).value)
                            / (1000 * telescope.focal_length.to(u.mm))).value 
                            * (u.arcsec/u.pixel) )

    bands = ['lum', 'b', 'g', 'r', 'nir', 'u']
    
    exp_time = 300 # seconds
    n_exp = 1 

    df = pd.read_csv('target_list_full_focus.csv')
   
    # Do 12 exposures at full focus (Strehl ratio of 1)
    strehl_ratios = (np.full(12, 100))

    # Number of background galaxies to add
    sci_img_size_x_arcsec = camera.npix_H * bandpass.plate_scale
    sci_img_size_y_arcsec = camera.npix_V * bandpass.plate_scale
    
    height_deg = ((sci_img_size_y_arcsec.value) * u.arcsec).to(u.deg)
    width_deg = ((sci_img_size_x_arcsec.value) * u.arcsec).to(u.deg)
    
    sci_img_area = (sci_img_size_x_arcsec * sci_img_size_y_arcsec).to(u.arcmin**2)

    n_gal_sqarcmin = 97.55 * (u.arcmin**-2)
    
    n_gal_total_img = round((sci_img_area * n_gal_sqarcmin).value)
    
    cosmos_plate_scale = 0.03 # arcsec/pix
    
    add_stars = True
    add_galaxies = False 
    
    dither_pix = 100
    dither_deg = bandpass.plate_scale.value * dither_pix / 3600 # dither_deg
    
    for idx, row in df.iterrows():
        target = row["target"]
        ra = row["ra"] * u.deg
        dec = row["dec"] * u.deg
        z = row['z']
        mass = 10**(row['mass']) # solmass
        conc = row['c']

        for band in bands:
            # We want to rotate the sky by (5 min + 1 min overhead) each new target
            theta_master = 0 * u.radian
            rot_rate = 0.25 # deg / min

            for strehl in strehl_ratios:
                
                print(f"Image simulation starting for {target}, {band}, {strehl}.\n")
                
                ra_sim = np.random.uniform(ra.value - dither_deg, ra.value + dither_deg)
                dec_sim = np.random.uniform(dec.value - dither_deg, dec.value + dither_deg)

                bandpass.transmission = get_transmission(band=band)

                # Step 1: Get the optical PSF, given the Strehl ratio
                aberrations = get_zernike(band=band, strehl_ratio=strehl)
                
                piv_wave = piv_dict[band]

                optical_zernike_psf = galsim.OpticalPSF(lam=piv_wave,
                                                        diam=telescope.diameter.value,
                                                        aberrations=aberrations,
                                                        obscuration=0.38,
                                                        nstruts=4,
                                                        flux=1)

                jitter_psf = galsim.Gaussian(sigma=0.05, flux=1)

                psf_sim = galsim.Convolve([optical_zernike_psf, jitter_psf])
        
                # Step 2: Construct the science image
                sci_img = galsim.Image(ncol=camera.npix_H.value,
                                    nrow=camera.npix_V.value,
                                    dtype=np.uint16)

                # Step 3: Fill the science image with zeros
                sci_img.fill(0)
                sci_img.setOrigin(0, 0)
        
                # Step 4: WCS setup 
                if strehl < 100:
                    theta = 0 * galsim.degrees
                else:
                    theta = theta_master.to(u.deg).value * galsim.degrees

                dudx = np.cos(theta) * bandpass.plate_scale.value
                dudy = -np.sin(theta) * bandpass.plate_scale.value
                dvdx = np.sin(theta) * bandpass.plate_scale.value
                dvdy = np.cos(theta) * bandpass.plate_scale.value

                image_center = sci_img.true_center

                affine = galsim.AffineTransform(dudx, dudy, dvdx, dvdy, origin=image_center) 

                sky_center = galsim.CelestialCoord(ra=ra_sim * galsim.degrees, 
                                                   dec=dec_sim * galsim.degrees)

                wcs = galsim.TanWCS(affine, sky_center, units=galsim.arcsec)
                sci_img.wcs = wcs

                # Step 5: Add a dark frame
                dark_dir = "/home/gill/sims/data/darks/minus10/*"
                dark_fname = random.choice(np.sort(glob.glob(dark_dir)))
                
                sci_img += fits.getdata(dark_fname)

                # Step 6: Add sky noise (Gill et al. 2020)
                crate_sky_electron_pix = phot.crate_bkg(illum_area=telescope.illum_area,
                                                        bandpass=bandpass,
                                                        bkg_type='raw',
                                                        strength='ave')

                crate_sky_adu_pix = crate_sky_electron_pix / camera.gain.value
                sky_adu_pix = crate_sky_adu_pix * exp_time * n_exp  # ADU

                sci_img += phot.get_sky_bkg(image_shape=sci_img.array.shape,
                                            sky_adu_pix=sky_adu_pix)

                # Step 7: Add stars
                if add_stars:

                    Vizier.ROW_LIMIT = -1
                    
                    coordinates = coord.SkyCoord(ra=ra_sim * u.deg,
                                                 dec=dec_sim * u.deg,
                                                 frame='icrs')

                    gaia_cat = Vizier.query_region(coordinates=coordinates,
                                                   height=height_deg,
                                                   width=width_deg,
                                                   catalog='I/345/gaia2')
                
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
            
                    print("Adding {} stars".format(len(df_stars)))

                    for idx in range(len(df_stars)):

                        if idx % 200 == 0:
                            print("Number of stars added: {} of {}".format(idx, len(df_stars)))

                        if piv_wave > 600:
                            gaia_mag = 'Gmag'
                        else:
                            gaia_mag = 'BPmag'

                        # Find counts to add for the star
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

                        # Limit the flux
                        if flux_adu > 2**16 - 1:
                            flux_adu = 2**16  - 1

                        # Assign real position to the star on the sky
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

                        # Check to ensure star is not out of bounds on the image
                        try:
                            sci_img[stamp_overlap] += star_image[stamp_overlap]
                        
                        except galsim.errors.GalSimBoundsError:
                            
                            print("Out of bounds star. Skipping.")
                            continue
                    
                print("Done adding stars.")

                if add_galaxies:
                    
                    sci_img_ra_min = (ra_sim * u.deg - ((sci_img_size_x_arcsec.to(u.arcsecond).to(u.deg)) / 2)).to(u.deg)
                    sci_img_ra_max = (ra_sim * u.deg + ((sci_img_size_x_arcsec.to(u.arcsecond).to(u.deg)) / 2)).to(u.deg)
                    
                    sci_img_dec_min = (dec_sim * u.deg - ((sci_img_size_y_arcsec.to(u.arcsecond).to(u.deg)) / 2)).to(u.deg)
                    sci_img_dec_max = (dec_sim * u.deg + ((sci_img_size_y_arcsec.to(u.arcsecond).to(u.deg)) / 2)).to(u.deg)
                        
                    # setup NFW halo at the center
                    nfw_halo_pos_xasec = (camera.npix_H.value/2) * bandpass.plate_scale.value
                    nfw_halo_pos_yasec = (camera.npix_V.value/2) * bandpass.plate_scale.value

                    nfw_halo = galsim.NFWHalo(mass=mass,
                                              conc=conc,
                                              redshift=z,
                                              omega_m=0.3,
                                              halo_pos=galsim.PositionD(x=nfw_halo_pos_xasec, y=nfw_halo_pos_yasec),
                                              omega_lam=0.7)

                    # Only inject bkg galaxies
                    cat_df_filt = cat_df[cat_df['ZPDF'] >= z] 

                    print("Adding {} background galaxies".format(n_gal_total_img))

                    seed = np.arange(0, n_gal_total_img, 1)

                    for idx in range(n_gal_total_img):
                        if idx % 2000 == 0:
                            print("Galaxies added: {} of {}".format(idx, n_gal_total_img))

                        # Randomly sample a galaxy
                        gal_df = cat_df_filt.sample(n=1, random_state=seed[idx])
                        gal_z = gal_df.ZPDF.item()

                        crate_dict = {'b': gal_df.crates_b.item(),
                                      'u': gal_df.crates_u.item(),
                                      'r': gal_df.crates_r.item(),
                                      'nir': gal_df.crates_nir.item(),
                                      'lum': gal_df.crates_lum.item(),
                                      'g': gal_df.crates_g.item()}

                        # Get its Sersic parameters
                        gal_n_sersic_cosmos10 = gal_df.c10_sersic_fit_n.item()

                        if gal_n_sersic_cosmos10 < 0.3:
                            gal_n_sersic_cosmos10 = 0.3

                        crate = crate_dict[band] # galaxy count rate e/s

                        flux_adu = int(crate * exp_time * n_exp / camera.gain.value)
                        
                        hlr_arcsec = (gal_df.c10_sersic_fit_hlr.item()
                                    * np.sqrt(gal_df.c10_sersic_fit_q.item())
                                    * cosmos_plate_scale)

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

                        # Check to ensure galaxy is not out of bounds on the image
                        try:
                            sci_img[stamp_overlap] += gal_image[stamp_overlap]
                        
                        except galsim.errors.GalSimBoundsError:
                            print("Out of bounds star. Skipping.")
                            continue

                # HEADERS
                hdr = fits.Header()
                hdr['EXPTIME'] = int(exp_time)
                hdr['NEXP'] = int(n_exp)
                hdr['band'] = band
                hdr['strehl'] = strehl
                hdr['stars'] = int(add_stars)
                hdr['galaxies'] = int(add_galaxies)
                hdr['TARGET_RA'] = ra.value
                hdr['TARGET_DEC'] = dec.value
                odir = f"/home/gill/sims/data/sims/{target}/{band}/{strehl}/"
                
                dt_now = datetime.datetime.now()
                unix_time = int(time.mktime(dt_now.timetuple()))
                
                band_int = bands_dict[band]

                # Path checks
                Path(odir).mkdir(parents=True, exist_ok=True)

                output_fname = f"{odir}/{target}_{exp_time}_{band_int}_{unix_time}.fits"
            
                fits.writeto(filename=output_fname,
                             data=sci_img.array,
                             header=hdr,
                             overwrite=True)

                # Update the roll angle
                if strehl == 100:
                    theta_master += ((rot_rate * 6) * u.deg).to(u.radian)

                print(f"Image simulation complete for {target}, {band}, {strehl}.\n")

if __name__ == "__main__":
    main()
