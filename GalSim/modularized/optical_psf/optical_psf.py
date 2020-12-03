import numpy as np
import galsim

__all__ = ["load_optical_psf"]

def load_optical_psf(sbparams, band="lum"):
    """Extract the optical PSF using abberations estimated by Zemax
       Note: This is only valid at one wavelength at the moment (550 nm).
    Args:
        sbparams (class object): SuperBIT parameters
        band (str): band
    Return:
        optical_psf (galsim object)
    """

    aberrations = np.zeros(38)           

    if band=='lum':
        lam_over_diam = sbparams.lam * 1.e-9 / sbparams.tel_diam    # radians
        lam_over_diam *= 206265.

        aberrations[0] = 0.                    # First entry must be zero
        aberrations[1] = -0.00305127
        aberrations[4] = -0.02474205           # Noll index 4 = Defocus
        aberrations[11] = -0.01544329          # Noll index 11 = Spherical
        aberrations[22] = 0.00199235
        aberrations[26] = 0.00000017
        aberrations[37] = 0.00000004

    optical_psf = galsim.OpticalPSF(lam=sbparams.lam,
                                    diam=sbparams.tel_diam, 
                                    obscuration=sbparams.obscuration, 
                                    nstruts=sbparams.nstruts, 
                                    strut_angle=sbparams.strut_angle, 
                                    strut_thick=sbparams.strut_thick,
                                    aberrations=aberrations)
    return optical_psf



