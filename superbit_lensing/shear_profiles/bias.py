import numpy as np
from astropy.table import Table

def compute_shear_bias(profile_tab):
    '''
    Function to compute the max. likelihood estimator for the bias of a shear profile
    relative to the input NFW profile and the uncertainty on the bias.

    Saves the shear bias estimator ("alpha") and the uncertainty on the bias ("asigma")
    within the meta of the input profile_tab

    The following columns are assumed present in profile_tab:

    :gtan:      tangential shear of data
    :err_gtan:  RMS uncertainty for tangential shear data
    :nfw_gtan:  reference (NFW) tangential shear

    '''

    assert isinstance(profile_tab, Table)

    try:
        T = profile_tab['mean_nfw_gtan']
        D = profile_tab['mean_gtan']
        errbar = profile_tab['err_gtan']

    except KeyError as kerr:
        print('Shear bias calculation:')
        print('required columns not found; check input names?')
        raise kerr

    # C = covariance, alpha = shear bias maximum likelihood estimator
    C = np.diag(errbar**2)
    numer = T.T.dot(np.linalg.inv(C)).dot(D)
    denom = T.T.dot(np.linalg.inv(C)).dot(T)
    alpha = numer/denom

    # sigalpha: Cramer-Rao bound uncertainty on Ahat
    sig_alpha = 1. / np.sqrt((T.T.dot(np.linalg.inv(C)).dot(T)))

    print('# ')
    print(f'# shear bias is {alpha:.4f} +/- {sig_alpha:.3f}')
    print('# ')

    # add this information to profile_tab metadata
    profile_tab.meta.update({'alpha': alpha, 'sig_alpha': sig_alpha})

    return
