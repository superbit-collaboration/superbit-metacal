import meds
import ngmix
import mof
import numpy as np
import pdb
import astropy.table as Table



class SuperBITNgmixFitter():
    """
    class to process a set of observations from a MEDS file.
    
    meds_info: A dict or list of dicts telling us how to process the one or meds  
               files provided. 
    
    """
    def __init__(self, meds_info = None):

        # The meds file groups data together by source.
        # To run, we need to make an ObservationList object for each source.
        # Each ObservationList has a single Observation frome each epoch.
        # Each Observation object needs an image, a psf, a weightmap, and a wcs, expressed as a Jacobian object.
        #   The MEDS object has methods to get all of these.

        # Once that's done, we create a Bootstrapper, populate it with initial guesses
        #   and priors, and run it.
        # The results get compiled into a catalog (numpy structured array with named fields),
        #   which is either written to a .fits table or returned in memory.
        #
        # self.metcal will hold result of metacalibration
        # self.gal_results is the result of a simple fit to the galaxy shape within metcal bootstrapper
        
        self.medsObj = meds.MEDS(meds_info['meds_file'])
        self.catalog = self.medsObj.get_cat()
        self.metacal = None 
        self.gal_fit = None 

        
    def _generate_initial_guess(self,observation):
        # Generate a guess from the pixel scale.
        fwhm_guess= 4*observation.jacobian.get_scale()
        gmom = ngmix.gaussmom.GaussMom(observation,fwhm_guess)
        gmom.go()
        return gmom.result
    
    def _get_priors(self):

        # prior on ellipticity.  The details don't matter, as long
        # as it regularizes the fit.  This one is from Bernstein & Armstrong 2014
        
        g_sigma = 0.3
        g_prior = ngmix.priors.GPriorBA(g_sigma)
        
        # 2-d gaussian prior on the center
        # row and column center (relative to the center of the jacobian, which would be zero)
        # and the sigma of the gaussians
        
        # units same as jacobian, probably arcsec
        row, col = 0.0, 0.0
        row_sigma, col_sigma = 0.2, 0.2 # a bit smaller than pix size of SuperBIT
        cen_prior = ngmix.priors.CenPrior(row, col, row_sigma, col_sigma)
        
        # T prior.  This one is flat, but another uninformative you might
        # try is the two-sided error function (TwoSidedErf)
        
        Tminval = -1.0 # arcsec squared
        Tmaxval = 1000
        T_prior = ngmix.priors.FlatPrior(Tminval, Tmaxval)
        
        # similar for flux.  Make sure the bounds make sense for
        # your images
        
        Fminval = -1.e1
        Fmaxval = 1.e5
        F_prior = ngmix.priors.FlatPrior(Fminval, Fmaxval)
        
        # now make a joint prior.  This one takes priors
        # for each parameter separately
        priors = ngmix.joint_prior.PriorSimpleSep(
        cen_prior,
        g_prior,
        T_prior,
        F_prior)
    
        return priors

    def _get_jacobians(source_id = None):
        jlist = medsObj.get_jacobian_list(source_id)
        jac = [ngmix.Jacobian(row=jj['row0'],col=jj['col0'],dvdrow = jj['dvdrow'],dvdcol=jj['dvdcol'],dudrow=jj['dudrow'],dudcol=jj['dudcol']) for jj in jlist]
        return jac

    def make_psf():
        jitter_fwhm = 0.3
        jitter = galsim.Gaussian(flux=1., fwhm=jitter_fwhm)
        
        lam_over_diam = 0.257831 # units of arcsec
        aberrations = np.zeros(38)             # Set the initial size.
        aberrations[0] = 0.                       # First entry must be zero                       
        aberrations[1] = -0.00305127
        aberrations[4] = -0.02474205              # Noll index 4 = Defocus
        aberrations[11] = -0.01544329             # Noll index 11 = Spherical
        aberrations[22] = 0.00199235
        aberrations[26] = 0.00000017
        aberrations[37] = 0.00000004
        
        optics = galsim.OpticalPSF(lam=625,diam=0.5,
                                       obscuration=0.38, nstruts=4,
                                       strut_angle=(90*galsim.degrees), strut_thick=0.087,
                                       aberrations=aberrations)
        
        psf = galsim.Convolve([jitter,optics])
        return psf
        

    def _get_source_observations(self,source_id = None,psf_noise = 1e-6):
        source_id = index
        jlist = medsObj.get_jacobian_list(source_id)
        jaclist = [ngmix.Jacobian(row=jj['row0'],col=jj['col0'],dvdrow = jj['dvdrow'],dvdcol=jj['dvdcol'],\
                                      dudrow=jj['dudrow'],dudcol=jj['dudcol']) for jj in jlist]
        psf_cutouts = medsObj.get_cutout_list(source_id, type='psf')
        weight_cutouts = medsObj.get_cutout_list(source_id, type='weight')
        image_cutouts = medsObj.get_cutout_list(source_id, type='image')
        
        image_obslist = ngmix.observation.ObsList()

        psf = make_psf()
        psf_pixel_scale = 0.01
        gpsf_cutout = psf.drawImage(scale=psf_pixel_scale,nx=256,ny=256)
        
        for i in range(len(image_cutouts)):
            
            # Apparently it likes to add noise to the psf.        
            #this_psf = psf_cutouts[i] + psf_noise*np.random.randn(psf_cutouts[i].shape[0],psf_cutouts[i].shape[1])
            this_psf = gpsf_cutout.array
            this_psf_weight = np.zeros_like(this_psf) + 1./psf_noise**2
            
            this_image = image_cutouts[i]
            
            sky_sigma = (0.057*300)**2    # exp_time = 300, sky_var = 0.0957 ADU/pix/s
            this_weight = np.zeros_like(this_image)+ 1./sky_sigma
            #this_weight = weight_cutouts[i]

            # for some reason, PSF obs doesn't appear to like the Jacobian jacobian...
            jj_psf = ngmix.jacobian.DiagonalJacobian(scale=psf_pixel_scale,x=this_psf.shape[0]/2,y=this_psf.shape[1]/2)
            #jj_im = ngmix.jacobian.DiagonalJacobian(scale=0.206,x=this_image.shape[0]/2,y=this_image.shape[1]/2)
            jj =jaclist[i]
            
            psfObs = ngmix.observation.Observation(this_psf,weight = this_psf_weight, jacobian = jj_psf)
            imageObs = ngmix.observation.Observation(this_image,weight = this_weight,jacobian = jj, psf = psfObs)
            #imageObs.psf_nopix = imageObs.psf
            
            image_obslist.append(imageObs)
            
        return image_obslist
 

    
    def _fit_one(self,source_id,pars = None):
        """ 
        Workhorse method to perfom home-brewed metacalibration on an object. Returns the 
        unsheared ellipticities of each galaxy, as well as a label indicating whether the 
        galaxy passed the selection cuts for each shear step (i.e. no_shear,1p,1m,2p,2m). 

        """
        
        jaclist = _get_jacobians(source_id)
        obslist = self._get_source_observations(source_id)
        obslist = image_obslist
        
        mcal_shear = 0.01
        
        lm_pars = {'maxfev':2000, 'xtol':5.0e-5, 'ftol':5.0e-5}
        max_pars = {'method':'lm','lm_pars':lm_pars,'find_center':True}
        prior=_get_priors()
        psf_model = 'em5' # should come up with diagnostics for PSF quality
        gal_model = 'gauss'

        mcal_obs = ngmix.metacal.get_all_metacal(obslist)
        
        # Make a dictionary to hold the results
        result = {}
        mcal_obs_keys=['noshear', '1p', '1m', '2p', '2m']
        # Run the actual metacalibration fits on the observed galaxies with Bootstrapper()
        # and flag 
        #for ikey in mcal_obs_keys:
        ikey = 'noshear'
        boot = ngmix.Bootstrapper(mcal_obs[ikey])
        boot.fit_psfs(psf_model,1.)
        boot.fit_max(gal_model,max_pars,prior=prior)
        res = boot.get_fitter().get_result()
        #result.update({ikey:res['g']})
        try:
            res['Tpsf']=boot.psf_fitter._gm.get_T()
        except:
            res['Tpsf']=boot.psf_fitter._result['T']
            result.update({ikey:res})

        t_arr = []
        for p in boot.mb_obs_list[0]: 
            print(p.psf.get_gmix().get_e1e2T()) 
            t_arr.append(p.psf.get_gmix().get_e1e2T()[2]) 

        np.mean(t_arr)
        """ Wanna see the PSF model get rendered?? """
        #psf_cutout = psf_cutouts[3]
        #psf_cutout = gpsf_cutout.array
        psf_gm = boot.psf_fitter.get_gmix()
        psf_model_im = psf_gm.make_image(psf_cutout.shape,jacobian=jj_psf)
        fig,(ax1,ax2,ax3) = plt.subplots(nrows=1,ncols=3,figsize=(21,7))
        ax1.imshow(psf_cutout)#,vmin=0,vmax=(psf_cutout.max()*.90))
        ax2.imshow(psf_model_im)#,vmin=0,vmax=(psf_cutout.max()*.90))
        ax3.imshow(psf_cutout-psf_model_im)

        """ Wanna see the galaxy model get rendered? """ 

        image_cutout=image_cutouts[i] 
        gal_gm = boot.get_fitter().get_convolved_gmix() 
        gal_model_im = gal_gm.make_image(image_cutout.shape,jacobian=jj) 
        fig,(ax1,ax2,ax3) = plt.subplots(nrows=1,ncols=3,figsize=(21,7))
        ax1.imshow(junk_avg)#,vmin=0,vmax=(psf_cutout.max()*.90))
        ax2.imshow(gal_model_im)#,vmin=0,vmax=(psf_cutout.max()*.90))
        ax3.imshow(junk_avg-gal_model_im)

        R1 = (result['1p']['g'][0] - result['1m']['g'][0])/(2*mcal_shear)
        R2 = (result['2p']['g'][1] - result['2m']['g'][1])/(2*mcal_shear)
        print(f"R1: {R1:.3} \nR2:{R2:.3} ")

        return boot.get_fitter()

    
    def get_mcalib_shears(self,index):
        """
        Based on algorithm in Huff+ 2017 
        Assumes a shear step size of 0.01 (so 2*gamma = 0.02)

        Propagate fit results from at each of the 
        noshear/1p/1m/2p/2m stages of metacalibration. The values will be
        used later to compute both shear and selection responsivity

        """
        
        mcr = self.metacal
        try:

            r11=(mcr['1p']['g'][0] - mcr['1m']['g'][0])/0.02
            r12=(mcr['2p']['g'][0] - mcr['2m']['g'][0])/0.02
            r22=(mcr['2p']['g'][1] - mcr['2m']['g'][1])/0.02
            r21=(mcr['1p']['g'][1] - mcr['1m']['g'][1])/0.02

                
            R = [[r11,r12],[r21,r22]]
            Rinv = np.linalg.inv(R)
            ecorr=np.dot(Rinv,mcr['noshear']['g'])

 
            fit_result =[r11,r22,mcr['noshear']['g'][0],mcr['noshear']['g'][1],
                            ecorr[0],ecorr[1], mcr['noshear']['flux'], mcr['noshear']['T'], mcr['noshear']['Tpsf'],
                             mcr['noshear']['g_cov'][0,0], mcr['noshear']['g_cov'][1,1], mcr['noshear']['chi2per'],
                             mcr['1p']['T'], mcr['1p']['Tpsf'],mcr['1p']['g_cov'][0,0], mcr['1p']['g_cov'][1,1], mcr['1p']['chi2per'],
                             mcr['1m']['T'], mcr['1m']['Tpsf'], mcr['1m']['g_cov'][0,0], mcr['1m']['g_cov'][1,1], mcr['1m']['chi2per'],
                             mcr['2p']['T'], mcr['2p']['Tpsf'],mcr['2p']['g_cov'][0,0], mcr['2p']['g_cov'][1,1], mcr['2p']['chi2per'],
                             mcr['2m']['T'], mcr['2m']['Tpsf'], mcr['2m']['g_cov'][0,0], mcr['2m']['g_cov'][1,1], mcr['2m']['chi2per'],
                             mcr['noshear']['s2n'], mcr['1p']['s2n'], mcr['1m']['s2n'], mcr['2p']['s2n'], mcr['2m']['s2n'],
                             mcr['1p']['g'][0],mcr['1p']['g'][1],mcr['1m']['g'][0],mcr['1m']['g'][1],
                             mcr['2p']['g'][0],mcr['2p']['g'][1],mcr['2m']['g'][0],mcr['2m']['g'][1]
                             ]
        except:

            fit_result = [-9999.,-9999.,-9999.,-9999.,-9999.,
                              -9999.,-9999.,-9999.,-9999.,-9999.,
                              -9999.,-9999.,-9999.,-9999.,-9999.,
                              -9999.,-9999.,-9999.,-9999.,-9999.,
                              -9999.,-9999.,-9999.,-9999.,-9999.,
                              -9999.,-9999.,-9999.,-9999.,-9999.,
                              -9999.,-9999.,-9999.,-9999.,-9999.,
                              -9999.,-9999.,-9999.,-9999.,-9999.,
                              -9999.,-9999.,-9999.,-9999.,-9999.]
                
            
        return fit_result



def make_output_table(outfilename, gmix, mcal,identifying):
    """
    Create an output table and add in galaxy fit parameters

    :gmix:  array containing output parameters from the max_likelihood gmix fit
    :mcal: array containing output parameters from the metacalibration fit
    :identifying: contains ra, dec and ID for cross-matching later if needed


    """
    t = Table.Table()
    gmix=(np.array(gmix)); mcal=(np.array(mcal))
    identifying = np.array(identifying)
    try:
        t['id'] = identifying[:,0]
        t['ra'] = identifying[:,1]; t['dec'] = identifying[:,2]
    except:
        t['id'] = -9999.
        t['ra'] = -9999.; t['dec'] = -9999.

    try:
        t['g1_noshear'] = mcal[:,2]  # Ellipticity moments from the metacal fit, 
        t['g2_noshear'] = mcal[:,3]  # de/reconvolved with PSF 
        t['r11'] = mcal[:,0]         # finite difference response moments
        t['r22'] = mcal[:,1]
        t['g1_MC'] = mcal[:,4]       # Ellipticity moments from the metacal fit, 
        t['g2_MC'] = mcal[:,5]       # projected into the response matrix
        t['flux'] = mcal[:,6]        # flux from no_shear fit
        
        t['T_noshear'] = mcal[:,7]          # "noshear" fit parameters, for 
        t['Tpsf_noshear'] = mcal[:,8]       #  making quality cuts
        t['g1cov_noshear'] = mcal[:,9]
        t['g2cov_noshear'] = mcal[:,10]
        t['chi2per_noshear'] = mcal[:,11]

        t['g1_1p'] = mcal[:,37]       # "1p" fit parameters, for 
        t['g2_1p'] = mcal[:,38]       #  making quality cuts
        t['T_1p'] = mcal[:,12]          
        t['Tpsf_1p'] = mcal[:,13]       
        t['g1cov_1p'] = mcal[:,14]
        t['g2cov_1p'] = mcal[:,15]
        t['chi2per_1p'] = mcal[:,16]
       
        t['g1_1m'] = mcal[:,39]       # "1m" fit parameters, for 
        t['g2_1m'] = mcal[:,40]       #  making quality cuts
        t['T_1m'] = mcal[:,17]           
        t['Tpsf_1m'] = mcal[:,18]       
        t['g1cov_1m'] = mcal[:,19]
        t['g2cov_1m'] = mcal[:,20]
        t['chi2per_1m'] = mcal[:,21]

        t['g1_2p'] = mcal[:,41]       # "2p" fit parameters,for 
        t['g2_2p'] = mcal[:,42]       #  making quality cuts
        t['T_2p'] = mcal[:,22]          
        t['Tpsf_2p'] = mcal[:,23]       
        t['g1cov_2p'] = mcal[:,24]
        t['g2cov_2p'] = mcal[:,25]
        t['chi2per_2p'] = mcal[:,26]

        t['g1_2m'] = mcal[:,43]       # "2m" fit parameters,for 
        t['g2_2m'] = mcal[:,44]       #  making quality cuts
        t['T_2m'] = mcal[:,27]          
        t['Tpsf_2m'] = mcal[:,28]       
        t['g1cov_2m'] = mcal[:,29]
        t['g2cov_2m'] = mcal[:,30]
        t['chi2per_2m'] = mcal[:,31]
        
        t['s2n_noshear'] = mcal[:,32]
        t['s2n_1p'] = mcal[:,33]
        t['s2n_1m'] = mcal[:,34]
        t['s2n_2p'] = mcal[:,35]
        t['s2n_2m'] = mcal[:,36]

    except:
        
        t['g1_noshear'] = -9999.  # Ellipticity moments from the metacal fit, 
        t['g2_noshear'] = -9999.  # de/reconvolved with PSF
        t['r11'] = -9999.        # finite difference response moments
        t['r22'] = -9999.
        t['g1_MC'] = -9999.       # Ellipticity moments from the metacal fit, 
        t['g2_MC'] = -9999.       # projected into the response matrix        
        t['flux'] = -9999.
        
        t['T_noshear'] = -9999.        # "noshear" fit parameters, for 
        t['Tpsf_noshear'] = -9999.     #  making quality cuts
        t['g1cov_noshear'] = -9999.
        t['g2cov_noshear'] = -9999.
        t['chi2per_noshear'] = -9999.

        t['g1_1p'] = -9999.       # "1p" fit parameters, for 
        t['g2_1p'] = -9999.       #  making quality cuts
        t['T_1p'] = -9999.          
        t['Tpsf_1p'] = -9999.  
        t['g1cov_1p'] =-9999.
        t['g2cov_1p'] = -9999.
        t['chi2per_1p'] = -9999.

        t['g1_1m'] = -9999.       # "1p" fit parameters, for 
        t['g2_1m'] = -9999.       #  making quality cuts
        t['T_1m'] = -9999.       
        t['Tpsf_1m'] = -9999.    
        t['g1cov_1m'] =-9999.
        t['g2cov_1m'] = -9999.
        t['chi2per_1m'] = -9999.

        t['g1_2p'] = -9999.       # "2p" fit parameters, for 
        t['g2_2p'] = -9999.       #  making quality cuts
        t['T_2p'] = -9999.         
        t['Tpsf_2p'] = -9999.     
        t['g1cov_2p'] = -9999.
        t['g2cov_2p'] = -9999.
        t['chi2per_2p'] = -9999.

        t['g1_2m'] = -9999.       # "2m" fit parameters, for 
        t['g2_2m'] = -9999.       #  making quality cuts
        t['T_2m'] = -9999.        
        t['Tpsf_2m'] = -9999.   
        t['g1cov_2m'] = -9999.
        t['g2cov_2m'] = -9999.
        t['chi2per_2m'] = -9999.

        t['s2n_noshear'] = -9999.
        t['s2n_1p'] = -9999.
        t['s2n_1m'] = -9999.
        t['s2n_2p'] = -9999.
        t['s2n_2m'] = -9999.
    
    t.write(outfilename,format='ascii',overwrite=True)
    

def main(args):
    import matplotlib.pyplot as plt
    if ( ( len(args) < 4) or (args == '-h') or (args == '--help') ):
        print("\n### \n### ngmix_fit_testing is a routine which takes a medsfile as its input \
        and outputs shear-calibrated object shapes to a table\n###\n\n  python ngmix_fit_testing.py \
        medsfile start_index end_index outfilename\n \n")
    else:
        pass
    
    medsfile = args[1]
    index_start  = np.int(args[2])
    index_end = np.int(args[3])
    outfilename = args[4]
    meds_info = {'meds_file':args[1]}
    BITfitter = SuperBITNgmixFitter(meds_info) # MEDS file
    bootfit=[]                              # Array to hold parameters from Bootstrapper fit to object
    mcal=[]                                 # Array to hold Metacalibration fit parameters
    identifying=[]                          # Array to hold identifying information for object
    
    for i in range(index_start, index_end):

        identifying.append([BITfitter.medsObj['id'][i],BITfitter.medsObj['ra'][i],BITfitter.medsObj['dec'][i]])
        try:
            
            # metacal_fit is shear calibrated, gmix_fit is just Bootstrapper 
            metcal_fit = BITfitter._fit_one(i)           
            mcal_fit_pars = BITfitter.get_mcalib_shears(i)

            mcal.append(mcal_fit_pars)

            # Now for some plotting!
            image_cutout = BITfitter.medsObj.get_cutout(i,0)
            #image_cutout[image_cutout <=0] = 1E-2
            jj = ngmix.jacobian.DiagonalJacobian(scale=0.206,x=(image_cutout.shape[0])/2,y=(image_cutout.shape[1])/2)
            
            try:
                gmix = metcal_fit.get_convolved_gmix()
                model_image = gmix.make_image(image_cutout.shape,jacobian=jj)
                
                fig,(ax1,ax2,ax3) = plt.subplots(nrows=1,ncols=3,figsize=(21,7))
                ax1.imshow(image_cutout,vmin=0,vmax=(image_cutout.max()*.90))
                ax2.imshow(model_image,vmin=0,vmax=(image_cutout.max()*.90))
                x,y=gmix.get_cen()
                ax2.plot((x+image_cutout.shape[0]/2),(y+image_cutout.shape[1]/2),'Xr')
                ax3.imshow(image_cutout - model_image)
                fig.savefig('diagnostics_plots/diagnostics-ngmix2'+str(i)+'.png')
                plt.close(fig)
                
            except:
                # EM probably failed
                print("Bad gmix model, no image made :(")
                
            

        
        except:
            #pdb.set_trace()
            print("object %d failed, skipping..." % i)
            
            
    make_output_table(outfilename, bootfit, mcal, identifying)
    


if __name__ == '__main__':

    import pdb, traceback, sys
    try:
        main(sys.argv)
    except:
        thingtype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)    
        
