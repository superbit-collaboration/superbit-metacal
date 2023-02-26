import instrument as inst
import photometry as phot

import galsim

def make_a_star(idx, piv_wave, df_stars, wcs, telescope, camera, bandpass,
                exp_time):
    '''
    Make a single GAIA star given inputs. Setup for multiprocessing

    idx: int
        The batch index
    piv_wave: float
        The pivot wavelength in nm
    df_stars: pandas.DataFrame
        The stellar catalog
    wcs: galsim.WCS
        The image WCS
    telescope: instrument.Telescope
        An instance of the Telescope class
    camera: instrument.Camera
        An instance of the Camera class
    bandpass: instrument.Bandpass
        An instance of the Bandpass class
    exp_time: int, float
        The exposure time

    returns:
    '''

    if piv_wave > 600:
        gaia_mag = 'Gmag'
    else:
        gaia_mag = 'BPmag'

    # Find counts to add for the star
    mean_fnu_star_mag = phot.abmag_to_mean_fnu(
        abmag=df_stars[gaia_mag][idx]
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

    flux_adu = int(crate_adu_pix * exp_time * 1)

    # Limit the flux
    if flux_adu > 2**16 - 1:
        flux_adu = 2**16  - 1

    # Assign real position to the star on the sky
    star_ra_deg = df_stars['RA_ICRS'][idx] * galsim.degrees
    star_dec_deg = df_stars['DE_ICRS'][idx] * galsim.degrees

    world_pos = galsim.CelestialCoord(
        star_ra_deg, star_dec_deg
        )
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

    star_image = convolution.drawImage(
        nx=1000,
        ny=1000,
        wcs=wcs.local(image_pos),
        offset=offset,
        method='auto',
        dtype=np.uint16
        )

    star_image.setCenter(ix_nominal, iy_nominal)

    stamp_overlap = star_image.bounds & sci_img.bounds

    # Check to ensure star is not out of bounds on the image
    sci_img[stamp_overlap] += star_image[stamp_overlap]

    # gets caught above
    # except galsim.errors.GalSimBoundsError:
    #     print('Out of bounds star. Skipping.')

    return
