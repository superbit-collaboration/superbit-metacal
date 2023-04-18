import galsim
import numpy as np

def pujol_sims(gal_model,psf_model,step=0.02):
    gal1p = gal.shear(step,0)
    gal2p = gal.shear(0,step)

    #noshear
    obsns = galsim.Convolve([gal,psf_model])
    #1p
    obs1p = galsim.Convolve([gal1p,psf_model])
    #2p
    obs2p = galsim.Convolve([gal2p,psf_model])

    pujobsdict = {
        'noshear':obsns,
        '1p':obs1p,
        '2p':obs2p,  
    }  
    return  pujobsdict


def get_pujol_response(resdict):

    #noshear
    g0s = np.array(resdict['noshear'])

    #shear
    g1p = np.array(resdict['1p'])
    g2p = np.array(resdict['2p'])   

    R11 = (g1p[0]-g0s[0])/(2*step)
    R21 = (g1p[1]-g0s[1])/(2*step) 
    R12 = (g2p[0]-g0s[0])/(2*step)
    R22 = (g2p[1]-g0s[1])/(2*step)

    R = np.array(
    [[R11,R12],
     [R21,R22]])

    ellip_dict = {
        'noshear':g0s,
        '1p':g1p,
        '1m':g1m,
        '2p':g2p,
        '2m':g2m,  
    } 

    return ellip_dict, R