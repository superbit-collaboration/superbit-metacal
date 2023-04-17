import numpy as np
import galsim
from scipy.optimize import minimize


def generate_gal(i):
    #cat
    cat = galsim.COSMOSCatalog(sample="25.2")
    gal = cat.makeGalaxy(i, gal_type='parametric')
    gal_mag = cat.param_cat['mag_auto'][cat.orig_index[i]]  
    ##cfis photometry
    mag_zp = 32
    gal_flux = 10**(-(gal_mag-mag_zp)/2.5)
    gal = gal.withFlux(gal_flux)
    return gal

def galsimator(im_array):
    image = galsim.Image(im_array,scale=0.1)
    galsim_img = galsim.InterpolatedImage(image,x_interpolant='lanczos14')
    return galsim_img  

def noise_image(size=51,sky_level=400):
    rng = np.random.RandomState(1)
    noise=rng.normal(loc=sky_level,scale=np.sqrt(sky_level),size=[size,size])
    return noise 


def model(flux,hlr,g1,g2,psfs,boxsize,scale):
    """
        Creates a model to fit on n exposures, 
        number of exposures is defined by how many psfs
    """
    model_image_list = []
    for psf in psfs:

        try:

            gal = galsim.Gaussian(flux=flux,
            half_light_radius = hlr
            ).shear(
                g1=g1,
                g2=g2
            )
        except:
            gal = galsim.Gaussian(flux=flux,
            half_light_radius = hlr
            ).shear(
                g1=0.,
                g2=0.
            )

    model = galsim.Convolve([gal,psf])

    model_array = model.drawImage(
        nx=boxsize,
        ny=boxsize,
        scale=scale
    ).array
    model_image_list += [model_array]

  
    return model_image_list

def create_loss(model,images,psfs,boxsize,scale):
    """creates a loss function that depends only on fitable parameters"""
    return lambda theta :  sum((model(theta[0],theta[1],theta[2],theta[3],psfs,boxsize,scale) - images)**2)


def get_metacal_type(deconv_gal,step,type_name):
    if type_name == "noshear":
        return deconv_gal.shear(0,0)
    elif type_name == "1p":
        return deconv_gal.shear(step,0) 
    elif type_name == "2p":
        return deconv_gal.shear(0,step)
    elif type_name == "1m":
        return deconv_gal.shear(-step,0)
    elif type_name == "2m":
        return deconv_gal.shear(0,-step)
  
def get_fixnoise(noise_image,step,type_name):
    sheared_noise = get_metacal_type(noise_image,step,type_name)
    galsimrot90 = galsim.Angle(90,galsim.AngleUnit(pi/180.))
    rotated_sheared_noise = sheared_noise.rotate(galsimrot90)
    return rotated_sheared_noise

def get_all_metacal(
    gal,
    psf,
    reconv_psf,
    noise_image=None,
    types=['noshear','1p','1m','2p','2m'],
    step=0.01
):
  
    inv_psf = galsim.Deconvolve(psf)
    deconv_gal = galsim.Convolve([gal,inv_psf])
    obsdict = {}
    for t in types:
        sheared_gal = get_metacal_type(deconv_gal,step,type_name=t)
        reconv_galaxy = galsim.Convolve([sheared_gal,reconv_psf])
        obsdict +={t: }#TODO

    if fixnoise== True:
        for t in types:
            sheared_gal = get_metacal_type(deconv_gal,step,type_name=t)
            reconv_galaxy = galsim.Convolve([sheared_gal,reconv_psf])
            obsdict +={t: }
    
    return obsdict

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

def measure_ellipticities(obsdict,method):
    resdict = {}
    for key in obsdict.keys():
        resdict += {key: method(pujobsdict[key])}
    return resdict

def get_metacal_response(resdict,shear_type = 'g'):
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