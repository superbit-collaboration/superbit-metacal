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
        
        Tminval = 0.0 # arcsec squared
        Tmaxval = 2000
        T_prior = ngmix.priors.FlatPrior(Tminval, Tmaxval)
        
        # similar for flux.  Make sure the bounds make sense for
        # your images
        
        Fminval = -1.e2
        Fmaxval = 1.e4
        F_prior = ngmix.priors.FlatPrior(Fminval, Fmaxval)
        
        # now make a joint prior.  This one takes priors
        # for each parameter separately
        priors = ngmix.joint_prior.PriorSimpleSep(
        cen_prior,
        g_prior,
        T_prior,
        F_prior)
    
        return priors
    
    def _get_jacobians(self,source_id = None):
        """
        jlist is a dict, jac is a list of ngmix Jacobians
        """
        jlist = self.medsObj.get_jacobian_list(source_id)
        jac = [ngmix.Jacobian(row=jj['row0'],col=jj['col0'],dvdrow = jj['dvdrow'],dvdcol=jj['dvdcol'],dudrow=jj['dudrow'],dudcol=jj['dudcol']) for jj in jlist]
        return jac
    
    def _get_source_observations(self,source_id = None,psf_noise = 1e-6):
        jaclist = self._get_jacobians(source_id)
        psf_cutouts = self.medsObj.get_cutout_list(source_id, type='psf')
        weight_cutouts = self.medsObj.get_cutout_list(source_id, type='weight')
        image_cutouts = self.medsObj.get_cutout_list(source_id, type='image')
        
        image_obslist = ngmix.observation.ObsList()


        for i in range(len(image_cutouts)):
            
            # Apparently it likes to add noise to the psf.        
            this_psf = psf_cutouts[i] + psf_noise*np.random.randn(psf_cutouts[i].shape[0],psf_cutouts[i].shape[1])
            this_psf_weight = np.zeros_like(this_psf) + 1./psf_noise**2
            
            # zap image pixels that are not finite.
            this_image = image_cutouts[i]
            this_image[~np.isfinite(this_image)] = 0.
            this_image[this_image <=0] = 1E-2
            sky_sigma = (0.0957*300)**2    # exp_time = 300, sky_var = 0.0957 ADU/pix/s
            this_weight = np.zeros_like(this_image)+ 1./sky_sigma
            
            jj_im = ngmix.jacobian.DiagonalJacobian(scale=0.206,x=(this_image.shape[0])/2,y=(this_image.shape[1])/2)
            jj_psf = ngmix.jacobian.DiagonalJacobian(scale=0.206,x=this_psf.shape[0]/2,y=this_psf.shape[1]/2)
        
            
            psfObs = ngmix.observation.Observation(this_psf,weight = this_psf_weight, jacobian = jj_psf)
            imageObs = ngmix.observation.Observation(this_image,weight = this_weight, jacobian = jj_im, psf = psfObs)
            #imageObs.psf_nopix = imageObs.psf 
            image_obslist.append(imageObs)
            
        return image_obslist
 
    def _fit_one(self,source_id,pars = None):
        """ 
        workhorse method to perfom metacalibration on an object
        """
        
        if pars is None:
            lm_pars = {'maxfev':2000, 'xtol':5.0e-5, 'ftol':5.0e-5}
            max_pars = {'method':'lm','lm_pars':lm_pars}
        # Get the catalog entry. This is a list of an ngmix.observation.Observation
        # instance for each snapshot (i.e., observation!) of the object
        # at index "source_id"
        #

        obslist = self._get_source_observations(source_id)
        # Choose priors
        prior = self._get_priors()
        
        # Construct an initial guess.
        psf_model='em3'        
        gal_model='exp'
        ntry=3
        #Tguess=4*obslist[0].jacobian.get_scale()**2
        Tguess=0.17
        mcal_shear = 0.01
        metacal_pars={'step': mcal_shear,'find_center':True}
        
        try:
            # #
            # BEGINNING METACAL FITTING STAGE
            # #
            # # star with a boostrapper.
            mcb=ngmix.bootstrap.MaxMetacalBootstrapper(obslist) 
            mcb.fit_metacal(psf_model, gal_model, max_pars, Tguess, prior=prior,  ntry=ntry, metacal_pars=metacal_pars)
            
            # # # Was fit successful? If not, set metacal result object to None
            try:
                mcr = mcb.get_metacal_result()
                self.metacal = mcr
                R1_gamma = (mcr['1p']['g'][0] - mcr['1m']['g'][0])/(2*mcal_shear)
                R2_gamma = (mcr['2p']['g'][1] - mcr['2m']['g'][1])/(2*mcal_shear)
                print(f"R1_gamma: {R1_gamma:.3} \nR2_gamma:{R2_gamma:.3} ")

            except:
                print("get_metacal_result() for object %d failed, skipping metacal step..." % source_id)
                mcr=None
                
            # # # Also get regular galaxy shape EM fit. If this fails, something got messed up
            mcb.fit_psfs(psf_model,Tguess)
            mcb.fit_max(gal_model=gal_model,pars=max_pars)
            gal_fit=mcb.get_fitter()
            self.gal_fit = gal_fit
            
            

        except:
            print("Creation of MaxMetacalBootstrapper failed, skipping object...")
            pdb.set_trace()
            gal_fit = None
            mcr = None

        return mcr, gal_fit

  
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
            #print("for index %d r11=%f r22=%f r12=%f r21=%f" % (index, r11, r22, r12, r21))

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
                             mcr['noshear']['s2n_r'], mcr['1p']['s2n_r'], mcr['1m']['s2n_r'], mcr['2p']['s2n_r'], mcr['2m']['s2n_r']
                             ]
        except:
            fit_result = [-9999.,-9999.,-9999.,-9999.,-9999.,-9999.,-9999.,-9999.,
                              -9999.,-9999.,-9999.,-9999.,-9999.,-9999.,-9999.,-9999.,
                              -9999.,-9999.,-9999.,-9999.,-9999.,-9999.,-9999.,-9999.,
                              -9999.,-9999.,-9999.,-9999.,-9999.,-9999.,-9999.,-9999.]
                
            print("No metacal parameters found for this object!!!")
            
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
    
    t['id'] = identifying[:,0]
    t['ra'] = identifying[:,1]; t['dec'] = identifying[:,2]
    try:
        t['g1_boot'] = gmix[:,2]       # Ellipticity moments from the basic Bootstrapper fit
        t['g2_boot'] = gmix[:,3]
        t['T_boot'] = gmix[:,4]
        t['flux_boot'] = gmix[:,5]
    except:
        t['g1_boot'] = -9999.
        t['g2_boot'] = -9999.
        t['T_boot']  = -9999.
        t['flux_boot'] = -9999.

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

        t['T_1p'] = mcal[:,12]          # "1p" fit parameters, for 
        t['Tpsf_1p'] = mcal[:,13]       #  making quality cuts
        t['g1cov_1p'] = mcal[:,14]
        t['g2cov_1p'] = mcal[:,15]
        t['chi2per_1p'] = mcal[:,16]
       
        t['T_1m'] = mcal[:,17]           # "1m" fit parameters, for 
        t['Tpsf_1m'] = mcal[:,18]       #  making quality cuts
        t['g1cov_1m'] = mcal[:,19]
        t['g2cov_1m'] = mcal[:,20]
        t['chi2per_1m'] = mcal[:,21]

        t['T_2p'] = mcal[:,22]          # "2p" fit parameters,for 
        t['Tpsf_2p'] = mcal[:,23]       #  making quality cuts
        t['g1cov_2p'] = mcal[:,24]
        t['g2cov_2p'] = mcal[:,25]
        t['chi2per_2p'] = mcal[:,26]

        t['T_2m'] = mcal[:,27]          # "2m" fit parameters,for 
        t['Tpsf_2m'] = mcal[:,28]       #  making quality cuts
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
        t['Tpsf_noshear'] = -9999.      #  making quality cuts
        t['g1cov_noshear'] = -9999.
        t['g2cov_noshear'] = -9999.
        t['chi2per_noshear'] = -9999.

        t['T_1p'] = -9999.          # "1p" fit parameters, for 
        t['Tpsf_1p'] = -9999.  #  making quality cuts
        t['g1cov_1p'] =-9999.
        t['g2cov_1p'] = -9999.
        t['chi2per_1p'] = -9999.
       
        t['T_1m'] = -9999.       # "1m" fit parameters, for 
        t['Tpsf_1m'] = -9999.    #  making quality cuts
        t['g1cov_1m'] =-9999.
        t['g2cov_1m'] = -9999.
        t['chi2per_1m'] = -9999.

        t['T_2p'] = -9999.         # "2p" fit parameters,for 
        t['Tpsf_2p'] = -9999.     #  making quality cuts
        t['g1cov_2p'] = -9999.
        t['g2cov_2p'] = -9999.
        t['chi2per_2p'] = -9999.

        t['T_2m'] = -9999.        # "2m" fit parameters,for 
        t['Tpsf_2m'] = -9999.   #  making quality cuts
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
        print("\n### \n### ngmix_fit_testing is a routine which takes a medsfile as\
        its input and outputs shear-calibrated object shapes to a table\n###\n\n  \
        python ngmix_fit_testing.py medsfile start_index end_index outfilename\n \n")
    else:
        pass
    
    medsfile = args[1]
    index_start  = np.int(args[2])
    index_end = np.int(args[3])
    outfilename = args[4]
    meds_info = {'meds_file':args[1]}
    BITfitter = SuperBITNgmixFitter(meds_info) # MEDS file
    bootfit=[]                                 # Array to hold parameters from ExpectMax fit to object
    mcal=[]                                 # Array to hold Metacal fit parameters
    identifying=[]                          # Array to hold identifying information for object
    
    for i in range(index_start, index_end):
        try:
            metacal_fit,gmix_fit = BITfitter._fit_one(i)
            mcal_fit_pars = BITfitter.get_mcalib_shears(i)
            mcal.append(mcal_fit_pars)
            try:
                bootfit.append(gmix_fit.get_result()['pars'])
            except:
                print("failed to append holding values")
                bootfit.append(np.array([-99999, -99999, -99999, -99999, -99999, -99999]))
                  
            identifying.append([BITfitter.medsObj['id'][i],BITfitter.medsObj['ra'][i],BITfitter.medsObj['dec'][i]])
            
            
            # Now for some plotting!
            image = BITfitter.medsObj.get_cutout(i,0)
            jac = BITfitter.medsObj.get_jacobian(i,0)
            jj = ngmix.Jacobian(row=jac['row0'],col=jac['col0'],dvdrow = jac['dvdrow'],
                                    dvdcol=jac['dvdcol'],dudrow=jac['dudrow'],dudcol=jac['dudcol'])
            try:
                gmix_model = gmix_fit.get_convolved_gmix()
                model_image = gmix_model.make_image(image.shape,jacobian=jj)
                fig,(ax1,ax2,ax3) = plt.subplots(nrows=1,ncols=3,figsize=(21,7))
                ax1.imshow(image)
                ax2.imshow(model_image)
                ax3.imshow(image - model_image)
                fig.savefig('diagnostics_plots/diagnostics'+str(i)+'.png')
                plt.close(fig)
            except:
                # EM probably failed
                print("Bad gmix model, no image made :(")

        
            
        except:
            ("object %d failed, skipping..." % i)
             
    make_output_table(outfilename, bootfit, mcal, identifying)
    


if __name__ == '__main__':

    import pdb, traceback, sys
    try:
        main(sys.argv)
    except:
        thingtype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)    
        
