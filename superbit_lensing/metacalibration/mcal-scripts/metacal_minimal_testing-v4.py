import numpy as np
import galsim
import ngmix
import matplotlib.pyplot as plt
import ipdb

mcal_shear = 0.01
#true_shear = -0.02
true_shear = 0.05


# Set up realistic observation parameters
random_seed = 6222019; ud = galsim.UniformDeviate(random_seed+1)
`

gal_ideal = galsim.InclinedExponential(80*galsim.degrees,half_light_radius=1.,flux=7000).rotate(20*galsim.degrees)
psf = galsim.Gaussian(fwhm=.5)
gal_ideal_observed = galsim.Convolve([gal_ideal,psf])
gal_ideal_image = gal_ideal_observed.drawImage(scale=0.206)

sky_image = galsim.ImageF(gal_ideal_image)
sky_level = 106
sky_image.fill(sky_level)
gal_ideal_image+= sky_image # add a flat sky noise to image

psf_image = psf.drawImage(scale=0.206)
psf_weight_image = np.ones_like(psf_image.array)*1e9
weight_image = np.ones_like(gal_ideal_image.array)*1e9

# Set up for metacalibration.
jj_im = ngmix.jacobian.DiagonalJacobian(scale=0.206,x=gal_ideal_image.center.x,y=gal_ideal_image.center.y)
jj_psf = ngmix.jacobian.DiagonalJacobian(scale=0.206,x=psf_image.center.x,y=psf_image.center.y)
psf_obs = ngmix.Observation(psf_image.array,weight=psf_weight_image,jacobian=jj_psf)
gal_obs = ngmix.Observation(gal_ideal_image.array,weight=weight_image,jacobian=jj_im,psf=psf_obs)

# Make the metacal observations.
mcal_obs = ngmix.metacal.get_all_metacal(gal_obs)

# Make a dictionary to hold the results.
result = {}

# Do the fit.
lm_pars = {'maxfev':2000, 'xtol':5.0e-5, 'ftol':5.0e-5}
max_pars = {'method':'lm','lm_pars':lm_pars}

for ikey in mcal_obs.keys():
    boot = ngmix.Bootstrapper(mcal_obs[ikey])
    boot.fit_psfs('gauss',1.)
    boot.fit_max('exp',max_pars)
    res = boot.get_fitter().get_result()
    result.update({ikey:res['g']})
    #fit,ax = plt.subplots(nrows=1,ncols=2,figsize=(14,7))

R1 = (result['1p'][0] - result['1m'][0])/(2*mcal_shear)
R2 = (result['2p'][1] - result['2m'][1])/(2*mcal_shear)
print(f"R1: {R1:.3} \nR2:{R2:.3} ")

# To just make a checkplot:

ikey = '1p'
boot = ngmix.Bootstrapper(mcal_obs[ikey])
boot.fit_psfs('gauss',1.)
boot.fit_max('exp',max_pars)
gm_1p = boot.get_fitter().get_convolved_gmix()
im_1p=gm_1p.make_image(gal_ideal_image.array.shape, jacobian=jac)
plt.imshow(im_1p)

gal_sheared = gal_ideal.shear(galsim.Shear(g1=true_shear,g2=0.0))
gal_sheared_observed = galsim.Convolve([gal_sheared,psf_metacal])
gal_sheared_image = gal_sheared_observed.drawImage(scale=0.206)
gal_sheared_weight = np.ones_like(gal_sheared_image.array)*1e9

psf_metacal_image = psf_metacal.drawImage(scale=0.206)
psf_weight_image = np.ones_like(psf_metacal_image.array)*1e9
jj_im = ngmix.jacobian.DiagonalJacobian(scale=0.206,x=gal_sheared_image.center.x,y=gal_sheared_image.center.y)
jj_psf = ngmix.jacobian.DiagonalJacobian(scale=0.206,x=psf_metacal_image.center.x,y=psf_metacal_image.center.y)
psf_metacal_obs = ngmix.Observation(psf_metacal_image.array,weight=psf_weight_image,jacobian=jj_psf)
gal_sheared_obs = ngmix.Observation(gal_sheared_image.array,weight=gal_sheared_weight,jacobian=jj_im,psf=psf_metacal_obs)



boot = ngmix.Bootstrapper(gal_obs)
boot.fit_psfs('gauss',1.)
boot.fit_max('exp',max_pars)
res = boot.get_fitter().get_result()


# Show the fits:
fig,(ax1,ax2,ax3) = plt.subplots(nrows=1,ncols=3)
ax1.imshow(gal_ideal_image.array,origin='lower')#),vmin=-20,vmax=350)

# model image
gm=boot.get_fitter().get_convolved_gmix()
model_im = gm.make_image(gal_ideal_image.array.shape, jacobian=jj_im) 
ax2.imshow(model_im,origin='lower')#,vmin=-20,vmax=350)

#diff
ax3.imshow((gal_ideal_image.array - model_im),origin='lower')#,vmin=-20,vmax=350)

