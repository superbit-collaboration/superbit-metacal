import numpy as np
import galsim

import ipdb

def measure_ellipticities(obsdict,method):
    resdict = {}
    for key in obsdict.keys():
        resdict += {key: method(obsdict[key])}
    return resdict

def get_metacal_images(deconv_gal,step,type_name):
    if type_name == "noshear":
        return deconv_gal.shear(g1=0,g2=0)
    elif type_name == "1p":
        return deconv_gal.shear(g1=step,g2=0) 
    elif type_name == "2p":
        return deconv_gal.shear(g1=0,g2=step)
    elif type_name == "1m":
        return deconv_gal.shear(g1=-step,g2=0)
    elif type_name == "2m":
        return deconv_gal.shear(g1=0,g2=-step)
  
def get_fixnoise(noise_image,step,type_name):
    sheared_noise = get_metacal_type(noise_image,step,type_name)
    galsimrot90 = galsim.Angle(90,galsim.AngleUnit(pi/180.))
    rotated_sheared_noise = sheared_noise.rotate(galsimrot90)
    return rotated_sheared_noise

def get_all_metacal(
    gals,
    psfs,
    reconv_psfs,
    noise_images=None,
    types=['noshear','1p','1m','2p','2m'],
    step=0.01,
    fixnoise = True,
):
   
    obsdict = {}
    for t in types:
        reconv_gals = []
        for i in range(len(gals)):
            #iterate exposures
            gal = gals[i]
            psf = psfs[i]
            inv_psf = galsim.Deconvolve(psf)
            deconv_gal = galsim.Convolve([gal,inv_psf])
            sheared_gal = get_metacal_images(deconv_gal,step,type_name=t)
            reconv_galaxy = galsim.Convolve([sheared_gal,reconv_psfs[i]])
            reconv_gals += [reconv_galaxy]
        obsdict[t] = {'obs' : reconv_gals,'psf' : psfs}

    if fixnoise == True:
        #TODO finish this
        for t in types:
            fix_noises = []
            #set number of exp by gals, so it breaks if number is different
            for i in range(len(gals)):
                fixnoise_image = get_fixnoise(noise_image[i],step,type_name)
                fix_noises += [fixnoise_image]
            
            obsdict[t] += np.array(fix_noises)
    
    return obsdict

def get_metacal_response(resdict,shear_type = 'g',step=0.01):
    '''gets the shear response for ngmix-like results'''

    #noshear
    g0s = np.array(resdict['noshear'][shear_type])

    #shear
    g1p = np.array(resdict['1p'][shear_type])
    g1m = np.array(resdict['1m'][shear_type])
    g2p = np.array(resdict['2p'][shear_type])
    g2m = np.array(resdict['2m'][shear_type])    

    R11 = (g1p[0]-g1m[0])/(2*step)
    R21 = (g1p[1]-g1m[1])/(2*step) 
    R12 = (g2p[0]-g2m[0])/(2*step)
    R22 = (g2p[1]-g2m[1])/(2*step)

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
