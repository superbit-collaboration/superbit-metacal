import numpy as np
import galsim
import ngmix
import matplotlib.pyplot as plt
import ipdb

mcal_shear = 0.01
#true_shear = -0.02
true_shear = 0.05

gal_ideal = galsim.InclinedExponential(80*galsim.degrees,half_light_radius=1.).rotate(20*galsim.degrees)
psf = galsim.Gaussian(fwhm=.5)
psf_metacal = galsim.Gaussian(fwhm=0.5*(1+2*mcal_shear))
gal_ideal_observed = galsim.Convolve([gal_ideal,psf])
gal_ideal_image = gal_ideal_observed.drawImage(scale=0.206)
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


boot.get_fitter().get_result() 
boot = ngmix.Bootstrapper(gal_sheared_obs)
boot.fit_psfs('gauss',1.)
boot.fit_max('exp',max_pars)
res = boot.get_fitter().get_result()

# Shapes without metacal:


# Predicted e1 shape:
e1_pred = result['noshear'][0] + true_shear * R1
print(f"Measured e1:\n  {res['g'][0]:.4f}")
print(f"Metacal predicted e1:\n  {e1_pred:.4f}")

# Show the fits:
fig,(ax1,ax2,ax3) = plt.subplots(nrows=1,ncols=3)
ax1.imshow(gal_sheared_obs.image,origin='lower')
