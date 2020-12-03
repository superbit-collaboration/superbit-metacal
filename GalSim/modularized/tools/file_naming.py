import os

__all__ = ["fits_filename", "truth_filename"]

def fits_filename(sbparams, 
                  band,
                  exposure_index):
    """Generates the filename for the FITS file for the SuperBIT simulation
    for a given PSF model.

    Args:
        sbparams (class object): SuperBIT parameters
        jitter_psf_file (.psf file from PSFEX): SuperBIT jitter PSF model
        exposure_index (int): Exposure index in the loop
    Returns: fits_file_name (str)
    """
    fits_file_name = ''.join([sbparams.outdir,
                             'mock_superbit_',
                              band,
                             '_',
                              str(sbparams.exp_time),
                              's_',
                              str(exposure_index).zfill(3),
                              '.fits'])
    return fits_file_name

def truth_filename(sbparams, 
                   band,
                   exposure_index):
    """Generates the filename for the truth catalog

    Args:
        sbparams (class object): SuperBIT parameters
        jitter_psf_file (.psf file from PSFEX): SuperBIT jitter PSF model
        exposure_index (int): Exposure index in the loop
    Returns: truth_file_name (str)
    """

    truth_file_name = ''.join([sbparams.outdir,
                              'truth_',
                              band,
                              '_',
                              str(sbparams.exp_time),  
                              's_', 
                              str(exposure_index).zfill(3),
                              '.dat'])
    return truth_file_name