import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import bisect
import astropy
from astropy.io import fits
from astropy import units as u
from astropy.time import Time
import astropy.coordinates as coord
from astropy.constants import h, c, k_B
from astropy.convolution import AiryDisk2DKernel
from pathlib import Path

OBA_SIM_DATA_DIR = Path(__file__).parent / 'data/'

def get_transmission(band):
    sim_dir = OBA_SIM_DATA_DIR

    return np.genfromtxt(
        sim_dir / f'instrument/bandpass/{band}_2023.csv',
        delimiter=','
        )[:, 2][1:]

def t_given_nexp_SNR(source_count_rate,
                     read_noise,
                     sky_noise,
                     dark_current,
                     SNR=5,
                     n_pixels=1,
                     n_exp=1):

    a = source_count_rate**2. * n_exp**2
    b = -SNR**2 * ((source_count_rate * n_exp)
                   + (dark_current * n_pixels * n_exp)
                   + (sky_noise * n_pixels * n_exp))
    c = -SNR**2. * read_noise**2 * n_pixels * n_exp

    t_pos = (-b + np.sqrt(b**2 - (4*a*c))) / (2*a)
    t_neg = (-b - np.sqrt(b**2 - (4*a*c))) / (2*a)

    return t_pos, t_neg


def interpolator(x,
                 y,
                 x_min=300,
                 x_max=1101,
                 inc=1):
    x_int = np.arange(x_min, x_max, inc)
    f = interp1d(x, y)
    return x_int, f(x_int)


def wave_to_freq(wavelength):
    """
    Wavelengeth to frequency converter
    """
    return (c / wavelength).to(u.Hz).value


def mean_fnu_to_abmag(fd_freq):
    """
    Convert flux density [erg/s/cm^2/Hz] to ABmag
    """
    return -2.5*np.log10(fd_freq) - 48.60


def abmag_to_mean_fnu(abmag):
    """
    Convert ABmag to flux density [erg/s/cm^2/Hz] 
    """
    return 10**((abmag+48.6) / -2.5)


def SNR(source_count_rate,
        read_noise,
        sky_noise,
        dark_current,
        exp_time,
        n_pixels=1,
        n_exp=1):
    """
    Simple CCD equation. Given source count rate, exposure time, 
    read and sky noise, calculates the SNR.
    inputs:
    1) source_count_rate: (float) [electrons/second]
    2) read_noise: (float) [electrons/pixel]
    3) sky_noise: (float) [electrons/second/pixel]
    4) dark_current: (float) [electrons/second/pixel]
    5) exp_time: (float) exposure time of sensor [sec]
    6) n_pixels: (float) number of pixels contained in the 
        circular aperture of the source on the detector [] 
    7) n_exp: (int) number of exposures 
    returns:
    1) signal-to-noise ratio: (float)
    """
    signal = source_count_rate * exp_time * n_exp

    # invidivual noise terms
    noise_source = source_count_rate * exp_time * n_exp
    noise_sky = sky_noise * exp_time * n_pixels * n_exp
    noise_dark = dark_current * exp_time * n_pixels * n_exp
    noise_read = n_pixels * n_exp * read_noise**2

    noise = np.sqrt(noise_source
                    + noise_sky
                    + noise_dark
                    + noise_read)

    return signal / noise


def count_rate_given_snr(SNR,
                         read_noise,
                         sky_noise,
                         dark_current,
                         exp_time,
                         n_pixels=1,
                         n_exp=1):
    """Given SNR, calculate expected electrons per second

    Args:
        SNR ([type]): [description]
        read_noise ([type]): [description]
        sky_noise ([type]): [description]
        dark_current ([type]): [description]
        exp_time ([type]): [description]
        n_pixels (int, optional): [description]. Defaults to 1.
        n_exp (int, optional): [description]. Defaults to 1.
    """
    a = exp_time**2 * n_exp**2
    b = -SNR**2 * exp_time * n_exp
    const = ((sky_noise * exp_time * n_exp * n_pixels)
             + (dark_current * exp_time * n_exp * n_pixels)
             + (read_noise**2 * n_exp * n_pixels))
    c = -SNR**2 * const

    crate_pos = (-b + np.sqrt(b**2 - (4*a*c))) / (2*a)
    crate_neg = (-b - np.sqrt(b**2 - (4*a*c))) / (2*a)

    return crate_pos, crate_neg


def mean_fnu_from_countrate(count_rate,
                            bandpass,
                            illumination_area):
    count_rate /= u.s
    wave_int = np.array(bandpass.wavelengths) * u.nm
    illumination_area *= u.cm**2

    integrand = np.array(bandpass.transmission) * wave_int

    del_lam = ((np.max(wave_int) - np.min(wave_int)) / len(wave_int))

    integral = (np.trapz(y=integrand,
                         x=wave_int,
                         dx=del_lam).to(u.cm**2))

    flambda = ((count_rate*h*c / illumination_area)
               / integral).to(u.erg/u.s/u.cm**2/u.nm)

    mean_fnu = mean_fnu_from_flam_pivot(flambda=flambda.value,
                                        bandpass_transmission=np.array(
                                            bandpass.transmission),
                                        bandpass_wavelengths=np.array(bandpass.wavelengths))

    abmag = mean_fnu_to_abmag(mean_fnu)

    return mean_fnu, abmag


def mean_fnu_from_flambda(bandpass,
                          flambda,
                          wavelengths):
    wavelengths = wavelengths*u.nm
    flambda *= u.erg/u.s/u.cm**2/u.nm

    del_lam = ((np.max(wavelengths) - np.min(wavelengths))
               / len(wavelengths))

    numerator = np.trapz(y=bandpass.transmission*flambda,
                         x=wavelengths,
                         dx=del_lam)
    denominator = np.trapz(y=bandpass.transmission*c/wavelengths**2.,
                           x=wavelengths,
                           dx=del_lam)
    return (numerator / denominator).to(u.erg/u.s/u.cm**2/u.Hz).value


def mean_flambda_from_flambda(flambda,
                              bandpass_transmission,
                              bandpass_wavelengths):

    flambda *= u.erg/u.s/u.cm**2/u.nm

    bandpass_wavelengths *= u.nm
    del_lam = ((np.max(bandpass_wavelengths) - np.min(bandpass_wavelengths))
               / len(bandpass_wavelengths))

    numerator = np.trapz(y=bandpass_transmission*flambda*bandpass_wavelengths,
                         x=bandpass_wavelengths,
                         dx=del_lam)
    denominator = np.trapz(y=bandpass_transmission*bandpass_wavelengths,
                           x=bandpass_wavelengths,
                           dx=del_lam)
    mean_flambda = (numerator/denominator).to(u.erg/u.s/u.cm**2/u.nm).value
    return mean_flambda  # should be good


def crate_from_flambda(flambda,
                       illum_area,
                       bandpass_transmission,
                       bandpass_wavelengths):

    illum_area *= u.cm**2
    wave_int = bandpass_wavelengths * u.nm
    flambda *= u.erg/u.s/u.cm**2/u.nm

    integrand = (illum_area
                 * flambda
                 * bandpass_transmission
                 * wave_int) / (h * c)

    del_lam = ((np.max(wave_int) - np.min(wave_int)) / len(wave_int))

    count_rate = (np.trapz(y=integrand,
                           x=wave_int,
                           dx=del_lam).to(u.s**-1))  # electron/s
    return count_rate.value


def crate_bkg(illum_area,
              bandpass,
              bkg_height='stratosphere',
              bkg_type='raw',
              strength='ave'):

    illum_area *= u.cm**2.

    if bkg_height == 'stratosphere':
        # Zodi and airglow corrected  backgrounds
        if bkg_type == 'corrected' and strength == 'ave':
            fnu_data = pd.read_csv("data/bkg/bkg_ave_c.csv")
            wave = fnu_data['wavelength']
            fnu = fnu_data['fnu']

        elif bkg_type == 'corrected' and strength == 'low':
            fnu_data = pd.read_csv("data/bkg/bkg_low_c.csv")
            wave = fnu_data['wavelength']
            fnu = fnu_data['fnu']

        elif bkg_type == 'corrected' and strength == 'high':
            fnu_data = pd.read_csv("data/bkg/bkg_high_c.csv")
            wave = fnu_data['wavelength']
            fnu = fnu_data['fnu']

        # Raw measured backgrounds
        elif bkg_type == 'raw' and strength == 'ave':
            fnu_data = pd.read_csv("data/bkg/bkg_ave_r.csv")
            wave = fnu_data['wavelength']
            fnu = fnu_data['fnu']

        elif bkg_type == 'raw' and strength == 'low':
            fnu_data = pd.read_csv("data/bkg/bkg_low_r.csv")
            wave = fnu_data['wavelength']
            fnu = fnu_data['fnu']

        elif bkg_type == 'raw' and strength == 'high':
            fnu_data = pd.read_csv("data/bkg/bkg_high_r.csv")
            wave = fnu_data['wavelength']
            fnu = fnu_data['fnu']

        wave = np.array(wave) * u.nm
        fnu = np.array(fnu) * (u.erg/u.s/u.cm**2/u.Hz)

        flambda = ((c / wave**2) * fnu).to(u.erg/u.s/u.cm**2/u.nm).value

        bkg_crate_arcsec = crate_from_flambda(flambda=flambda,
                                              illum_area=illum_area.value,
                                              bandpass_transmission=np.array(bandpass.transmission),
                                              bandpass_wavelengths=np.array(bandpass.wavelengths))

        bkg_crate_e_pix = bkg_crate_arcsec * bandpass.plate_scale.value**2

    else:
        raise ValueError("Background height type invalid.")

    return bkg_crate_e_pix


def crate_from_mean_flambda(mean_flambda,
                            illum_area,
                            bandpass_transmission,
                            bandpass_wavelengths):
    '''
    NOTE: mean_flambda can be an array
    '''

    illum_area *= u.cm**2
    wave_int = bandpass_wavelengths * u.nm
    mean_flambda *= u.erg/u.s/u.cm**2/u.nm

    count_rates = np.zeros(mean_flambda.shape)

    for i, mf in enumerate(mean_flambda):
        integrand = (illum_area
                    * mf
                    * bandpass_transmission
                    * wave_int) / (h * c)

        del_lam = ((np.max(wave_int) - np.min(wave_int)) / len(wave_int))

        count_rates[i] = (np.trapz(y=integrand,
                                   x=wave_int,
                                   dx=del_lam).to(u.s**-1)).value  # electron/s


    return count_rates

def mean_fnu_from_flam_pivot(flambda,
                             bandpass_transmission,
                             bandpass_wavelengths):
    mean_flambda = mean_flambda_from_flambda(flambda=flambda,
                                             bandpass_transmission=bandpass_transmission,
                                             bandpass_wavelengths=bandpass_wavelengths)
    mean_flambda *= (u.erg/u.s/u.cm**2/u.nm)

    piv_wave = pivot_wavelength(bandpass_transmission=bandpass_transmission,
                                bandpass_wavelengths=bandpass_wavelengths)
    piv_wave *= u.nm

    mean_fnu = (piv_wave**2 / c) * mean_flambda
    return mean_fnu.to(u.erg/u.s/u.cm**2/u.Hz).value


def mean_flambda_from_mean_fnu(mean_fnu,
                               bandpass_transmission,
                               bandpass_wavelengths):
    mean_fnu *= (u.erg/u.s/u.cm**2/u.Hz)
    piv = pivot_wavelength(bandpass_transmission, bandpass_wavelengths) * u.nm
    mean_flambda = (c * mean_fnu / piv**2).to(u.erg/u.s/u.cm**2/u.nm)
    return mean_flambda.value


def mean_flambda_from_flambda(flambda,
                              bandpass_transmission,
                              bandpass_wavelengths):
    flambda *= u.erg/u.s/u.cm**2/u.nm

    wavelengths = bandpass_wavelengths * u.nm
    del_lam = ((np.max(wavelengths) - np.min(wavelengths))
               / len(wavelengths))

    numerator = np.trapz(y=bandpass_transmission*flambda*wavelengths,
                         x=wavelengths,
                         dx=del_lam)
    denominator = np.trapz(y=bandpass_transmission*wavelengths,
                           x=wavelengths,
                           dx=del_lam)
    mean_flambda = (numerator/denominator).to(u.erg/u.s/u.cm**2/u.nm).value
    return mean_flambda


def pivot_wavelength(bandpass_transmission,
                     bandpass_wavelengths):
    bandpass_wavelengths *= u.nm
    integrand_num = bandpass_transmission * bandpass_wavelengths
    integrand_den = bandpass_transmission / bandpass_wavelengths

    del_lam = ((np.max(bandpass_wavelengths) - np.min(bandpass_wavelengths))
               / len(bandpass_wavelengths))

    numerator = (np.trapz(y=integrand_num,
                          x=bandpass_wavelengths,
                          dx=del_lam))

    denominator = (np.trapz(y=integrand_den,
                            x=bandpass_wavelengths,
                            dx=del_lam))

    pivot_wave = np.sqrt(numerator/denominator)

    return pivot_wave.to(u.nm).value


def filt_trans(wave_min, wave_max):
    wavelengths = np.arange(300, 1101, 1)
    filter_transmission = np.zeros(len(wavelengths))

    for idx, wave in enumerate(wavelengths):
        if (wave >= wave_min
                and wave <= wave_max):
            filter_transmission[idx] = 1
    return filter_transmission


def get_dark_current(image_shape,
                     dark_current,
                     exposure_time,
                     gain=1.0,
                     hot_pixels=False,
                     hot_pixels_percentage=0.01):

    dark_adu = dark_current * exposure_time / gain
    dark_im = (np.random.poisson(
        lam=dark_adu, size=image_shape)).astype(np.uint32)

    if hot_pixels:
        y_max, x_max = dark_im.shape
        n_hot = int((hot_pixels_percentage/100) * x_max * y_max)

        rng = np.random.RandomState(100)
        hot_x = rng.randint(0, x_max, size=n_hot)
        hot_y = rng.randint(0, y_max, size=n_hot)

        hot_current = 1000 * dark_current

        for i in range(len(hot_x)):
            dark_im[hot_x[i], hot_y[i]] = (hot_current
                                           * exposure_time / gain)
    return dark_im


def get_read_noise(image_shape,
                   read_noise_std,
                   gain=1):
    return (np.random.normal(loc=0,
                             scale=read_noise_std/gain,
                             size=image_shape)).astype(np.int16)


def get_sky_bkg(image_shape,
                sky_adu_pix):
    return (np.random.poisson(lam=sky_adu_pix,
                              size=image_shape)).astype(np.uint16)


def diff_lim(array, D=1.35):
    dat1 = ((1.22 * array * u.nm)
            / ((D * u.m).to(u.nm))
            * u.rad).to(u.arcsec).value
    return [r"%.2f$^{\prime\prime}$" % z for z in dat1]


def get_airy_psf(lam, D=1.35):
    psf = ((1.22 * lam * u.nm)
           / ((D * u.m).to(u.nm))
           * u.rad).to(u.arcsec).value
    return psf


def pixel_illumnation_fraction(psf_fwhm, pixel_scale):
    """given a psf fwhm and a plate scale assume airy disk
    psf and return illumnation fraction of the pixel

    Parameters
    ----------
    psf_fwhm : float
        full width half max of the psf in millarcs
    pixel_scale : float
        pixel scale in millarcseconds

    Returns
    -------
    float
        the fraction of the psf that illuminates a single pixel
    """
    pixel = np.zeros((pixel_scale * 11, pixel_scale * 11))
    pixel[
        5 * pixel_scale: 6 * pixel_scale,
        5 * pixel_scale: 6 * pixel_scale,
    ] = 1

    return np.sum(
        AiryDisk2DKernel(
            psf_fwhm, x_size=pixel_scale * 11, y_size=pixel_scale *
            11).array * pixel)
