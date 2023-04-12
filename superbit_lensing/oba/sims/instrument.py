from xml.dom import ValidationErr
import numpy as np
from astropy import units as u
import pandas as pd

# instrument data path
path = 'data/instrument/'


class Telescope:
    """Class to store telescope properties
    """

    def __init__(self, name):
        self.name = name

        if self.name == 'superbit':
            self.efl = 5500 * u.mm
            self.transmission = 0.7
            self.diameter = 0.5 * u.m
            self.f_number = 11
            self.focal_length = self.diameter * self.f_number
            self.fov_x = .4 * u.degree
            self.fov_y = .3 * u.degree
            self.aperture = (np.pi*(self.diameter/2)**2.).to(u.cm**2)
            self.obscuration = 0.38
            self.illum_area = self.aperture * (1 - self.obscuration)

        elif self.name == 'gigabit':
            self.transmission = 0.7  # taken as monochromatic for simplicity
            self.obscuration = 0.38
            self.efl = 5500 * u.mm
            self.diam = 1300 * u.mm
            self.nstruts = 4

        elif self.name == 'euclid':
            self.diameter = 1.2 * u.m
            self.f_number = 20.42 #from https://www.mersen.com/sites/default/files/publications-media/gs-full-sic-euclid-s-very-large-telescope-icso-2018-mersen.pdf
            self.focal_length = self.diameter * self.f_number
            self.aperture = (np.pi*(self.diameter/2)**2.).to(u.cm**2)
            self.obscuration = 0.13 #from https://www.spiedigitallibrary.org/conference-proceedings-of-spie/9904/99040Q/VIS-the-visible-imager-for-Euclid/10.1117/12.2234739.full?webSyncID=0c89a0ce-46e9-e6ec-7a49-dab6a0cbad05&sessionGUID=9329ad88-3c9d-902b-c999-3ced268bead4&SSO=1
            self.illum_area = self.aperture * (1 - self.obscuration)

        else:
            raise ValueError("Invalid name of telescope.")


class Camera:
    """Class to store camera properties
    """

    def __init__(self, name):
        self.name = name

        if self.name == 'imx455':
            data = np.genfromtxt(path+'camera/imx455.csv', delimiter=',')
            self.wavelengths = data[:, 0][1:]
            self.transmission = data[:, 1][1:]
            self.transmission_err = data[:, 2][1:]
            self.pixel_size = 3.76 * u.micron
            self.read_noise = 2.08 * u.electron / u.pixel
            self.dark_current = 0.0035 * (u.electron/u.s/u.pixel)
            self.npix_H = 9600 * u.pixel
            self.npix_V = 6422 * u.pixel
            self.adc_bits = 16
            self.gain = 0.343 * (u.electron)  # e/ADU

        elif self.name == 'vis':
            data = np.genfromtxt(path+'camera/Euclid_vis.csv', delimiter=',')
            self.wavelengths = data[:, 0][1:]
            self.transmission = data[:, 1][1:]
            self.transmission_err = data[:, 2][1:]
            self.pixel_size = 12 * u.micron
            self.npix_H = 4096 * u.pixel
            self.npix_V = 4132 * u.pixel
            self.gain = 3.1 * (u.electron)  # e/ADU from Niemi et al 2015
            
        else:
            raise ValueError("Invalid name of camera.")


class Bandpass:
    """Class to store a bandpass.
    """

    def __init__(self, name):
        self.name = name
