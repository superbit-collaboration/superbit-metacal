import numpy as np
import galsim

def generate_gal(i,exposure_time,mag_zp = 25):
    """Generates galaxies from the COSMOS catalog"""
    #cat
    cat = galsim.COSMOSCatalog(sample="25.2")
    gal = cat.makeGalaxy(i, gal_type='parametric')
    gal_mag = cat.param_cat['mag_auto'][cat.orig_index[i]]  
    ##photometry
    gal_flux = 10**(-(gal_mag-mag_zp)/2.5)*exposure_time
    gal = gal.withFlux(gal_flux)
    return gal

def make_noise(boxsize=51,sky_level=1.,noise_level=1.):
    """Generate a simple noise image with the same values"""
    rng = np.random.RandomState(1)
    noise=rng.normal(loc=sky_level,scale=noise_level,size=[boxsize,boxsize])
    return noise 

def galsimator(im_array,pixel_scale=0.1):
    """Transforms arrays into galsim image objects"""
    image = galsim.Image(im_array,scale=pixel_scale)
    galsim_img = galsim.InterpolatedImage(image,x_interpolant='lanczos14')
    return galsim_img  

def gaussian_model(flux,half_light_radius,g1,g2,sky_levels,psf_objs,boxsize ,pixel_scale):
    """
    Creates a gaussian model to fit on multiple exposures, 
    number of exposures is defined by how many psfs are given

    flux: float - galaxy flux
    half_light_radius: float - galaxy half light radius 
    g1,g2: galaxy ellipticity
    sky_level: background level of each image
    boxsize: stamp size in pixels
    pixel_scale: pixel scale (arcsec/pix)
    """
    model_image_list = []
    for i, psf in enumerate(psf_objs):
        gal = galsim.Gaussian(
            flux=flux,
            half_light_radius = half_light_radius,
        ).shear(
            g1=g1,
            g2=g2
        )

        model = galsim.Convolve([gal,psf]) + galsimator(sky_levels[i]*np.ones([boxsize,boxsize]))

        model_array = model.drawImage(
            nx=boxsize,
            ny=boxsize,
            scale=pixel_scale
        ).array
        model_image_list += [model_array]

    return model_image_list

def create_loss(model,images,sky_levels,psf_objs,boxsize,pixel_scale=.1):
    """creates a loss function that depends only on fitable parameters"""
    def loss(theta):
        """
        theta: flux, flr, g1, g2
        """
        l = np.sum(abs(model(theta[0],theta[1],theta[2],theta[3],sky_levels = sky_levels,psf_objs=psf_objs,boxsize=boxsize,pixel_scale=pixel_scale) - images))

        return l

    return loss