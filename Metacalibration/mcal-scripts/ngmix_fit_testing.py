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
        self.pixel_scale = 0.206
        self.psf_scale = 0.206

        
    def _generate_initial_guess(self,observation):
        # Generate a guess from the pixel scale.
        fwhm_guess= 4*observation.jacobian.get_scale()     
        #fwhm_guess= 4*self.pixel_scale
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
        
        Tminval = -10.0 # arcsec squared
        Tmaxval = 1.e6
        T_prior = ngmix.priors.FlatPrior(Tminval, Tmaxval)
        
        # similar for flux.  Make sure the bounds make sense for
        # your images
        
        Fminval = -1.e4
        Fmaxval = 1.e9
        F_prior = ngmix.priors.FlatPrior(Fminval, Fmaxval)
        
        # now make a joint prior.  This one takes priors
        # for each parameter separately
        priors = ngmix.joint_prior.PriorSimpleSep(
        cen_prior,
        g_prior,
        T_prior,
        F_prior,
        )
    
        return priors
    
    def _get_jacobians(self,source_id = None):
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

            # make a dummy PSF wcs for this object
            this_jac = jaclist[i]
            this_psf_jac = ngmix.Jacobian(row=this_jac.row0,col=this_jac.col0,dvdrow = this_jac.dvdrow,dvdcol=0,dudrow=0,dudcol=this_jac.dudcol)
            # Apparently it likes to add noise to the psf.
            this_psf = psf_cutouts[i] + psf_noise*np.random.randn(psf_cutouts[i].shape[0],psf_cutouts[i].shape[1])
            this_psf_weight = np.zeros_like(this_psf) + 1./psf_noise**2
            
            # zap image pixels that are not finite.
            this_image = image_cutouts[i]
            this_image[~np.isfinite(this_image)] = 0.
            psfObs = ngmix.observation.Observation(this_psf,weight = this_psf_weight, jacobian = this_psf_jac)
            
            imageObs = ngmix.observation.Observation(image_cutouts[i],weight = weight_cutouts[i], jacobian = jaclist[i], psf = psfObs)
            image_obslist.append(imageObs)
            """

            this_image = image_cutouts[i]
            this_image[~np.isfinite(this_image)] = 0.
            psfObs = ngmix.observation.Observation(this_psf,weight = this_psf_weight,jacobian = this_psf_jac)
            imageObs = ngmix.observation.Observation(image_cutouts[i],weight = weight_cutouts[i], psf = psfObs)
            image_obslist.append(imageObs)
            """
        return image_obslist
 
    def _fit_one(self,source_id,pars = None):
        """ 
        workhorse method to perfom metacalibration on an object
        """
        
        if pars is None:
            lm_pars = {'maxfev':2000, 'xtol':5.0e-5, 'ftol':5.0e-5}
            max_pars = {'method':'lm','lm_pars':lm_pars}
        # Get the catalog entry.
        obslist = self._get_source_observations(source_id)
        # Choose priors
        prior = self._get_priors()
        # Construct an initial guess.
        psf_model='em5'
        gal_model='gauss'
        ntry=3
        
        Tguess=4*obslist[0].jacobian.get_scale()**2
        
        #Tguess=4*self.pixel_scale
        #Tguess=0.169744
        metacal_pars={'step': 0.01}
        
        try:
            # #
            # BEGINNING METACAL FITTING STAGE
            # #
            # # star with a boostrapper.
            mcb=ngmix.bootstrap.MaxMetacalBootstrapper(obslist) 
            mcb.fit_metacal(psf_model, 
                                gal_model, 
                                max_pars, 
                                Tguess, 
                                prior=prior,  
                                ntry=ntry, 
                                metacal_pars=metacal_pars)
            
            # # # Was fit successful? If not, set metacal result object to None
            try:
                mcr = mcb.get_metacal_result()
                self.metacal = mcr
            except:
                print("get_metacal_result() for object %d failed, skipping metacal step..." % source_id)
                mcr=None
                
            # # # Also get regular galaxy shape EM fit, allowing for exception
            """ 
            WHY MIXING EM AND MOF? IF EM FAILS, LEAVE AS FAILURE
            """
            try:
                mcb.fit_psfs(psf_model,Tguess)
                mcb.fit_max(gal_model='gauss',pars=max_pars)
                gal_fit=mcb.get_fitter()
                self.gal_fit = gal_fit
            except:
                print("EM fitting for object %d failed, skipping EM fitting..." % source_id)
                gal_fit = None

        except:

            """
            LEAVE OUT MOF FIT -- NOT ACTUALLY DOING WHAT I THOUGHT
            IT'S ~A DEBLENDER/MULTI-GALAXY FIITER!
            """
            print("Creation of MaxMetacalBootstrapper failed, skipping object...")
            gal_fit = None
            mcr = None

        return mcr, gal_fit

    
    def do_mof(self,obslist=None,prior=None):
        
        guess = self._generate_initial_guess(obslist[0])
        guess = prior.sample(n=1)[0]
        fitter = mof.MOF(obslist,'gauss',1,prior=prior)
        fitter.go(guess)
        #pdb.set_trace()
        gal_fit= fitter.get_result()
        self.gal_fit = gal_fit
        
        return
   
   
    def get_mcalib_shears(self,index):
        """
        Based on algorithm in Huff+ 2017 
        Assumes a shear step size of 0.01 (so 2*gamma = 0.02)
        """
        
        mcr = self.metacal
        try:

            """"
            ONLY DO THIS STEP FOR GALAXIES WHERE ALL OF 1P/2P/1M/2M ARE DEFINED
            Also make sure that the "noshear" measurement is defined

            Note: make histogram of these values (1m, etc.) 
            look for sentinel values

            """
            
            print("for index %d noshear=[%.3f,%.3f] 1p=[%.3f,%.3f] 1m=[%.3f,%.3f] 2p=[%.3f,%.3f] 2m=[%.3f,%.3f]"
                  % (index, mcr['noshear']['g'][0], mcr['noshear']['g'][1],mcr['1p']['g'][0], 
                     mcr['1p']['g'][1],mcr['1m']['g'][0], mcr['1m']['g'][1],
                     mcr['2p']['g'][0], mcr['2p']['g'][1], mcr['2m']['g'][0], mcr['2m']['g'][1]))

            r11=(mcr['1p']['g'][0] - mcr['1m']['g'][0])/0.02
            r12=(mcr['2p']['g'][0] - mcr['2m']['g'][0])/0.02
            r22=(mcr['2p']['g'][1] - mcr['2m']['g'][1])/0.02
            r21=(mcr['1p']['g'][1] - mcr['1m']['g'][1])/0.02
            #print("for index %d r11=%f r22=%f r12=%f r21=%f" % (index, r11, r22, r12, r21))

            R = [[r11,r12],[r21,r22]]
            Rinv = np.linalg.inv(R)
            ecorr=np.dot(Rinv,mcr['noshear']['g'])
 
            fit_result =[r11,r22,mcr['noshear']['g'][0],mcr['noshear']['g'][1],
                            ecorr[0],ecorr[1],mcr['noshear']['T'],mcr['noshear']['flux']]
        except:
            #pdb.set_trace()
            fit_result = [-9999.,-9999.,-9999.,-9999.,-9999.,-9999.,-9999.,-9999.]
            print("No metacal parameters found for this object!!!")
            
        return fit_result



def make_output_table(outfilename, holding, mcal,identifying):
    """
    Create an output table and add in galaxy fit parameters, with exceptions for galaxies where
    fit failed. Also, make a "fiatfile" format catalog, matched against full SExtractor catalog, 
    

    :holding:  array containing output parameters from the max_likelihood gmix fit
    :mcal: array containing output parameters from the metacalibration fit

    """
    
    t = Table.Table()
    hT=(np.array(holding)); mcal=(np.array(mcal))
    identifying = np.array(identifying)
    
    t['id'] = identifying[:,0]
    t['ra'] = identifying[:,1]; t['dec'] = identifying[:,2]

    try:
        t['g1_gmix'] = hT[:,2]       # Ellipticity moments from the plain gmix fit
        t['g2_gmix'] = hT[:,3]
        t['T_gmix'] = hT[:,4]
        t['noshear'] = hT[:,5]
    except:
        t['g1_gmix'] = -9999.
        t['g2_gmix'] = -9999.
        t['T_gmix']  = -9999.
        t['noshear'] = -9999.

    """ check: are these indexed correctly?"""
    try:
        t['g1_noshear'] = mcal[:,2]  # Ellipticity moments from the metacal fit, 
        t['g2_noshear'] = mcal[:,3]  # de/reconvolved with PSF 
        t['r11'] = mcal[:,0]         # finite difference response moments
        t['r22'] = mcal[:,1]
        t['g1_MC'] = mcal[:,4]       # Ellipticity moments from the metacal fit, 
        t['g2_MC'] = mcal[:,5]       # projected into the response matrix
        t['T'] = mcal[:,6]
        t['flux'] = mcal[:,7]
    except:
        pdb.set_trace()
        t['g1_noshear'] = -9999.  # Ellipticity moments from the metacal fit, 
        t['g2_noshear'] = -9999.  # de/reconvolved with PSF
        t['r11'] = -9999.        # finite difference response moments
        t['r22'] = -9999.
        t['g1_MC'] = -9999.       # Ellipticity moments from the metacal fit, 
        t['g2_MC'] = -9999.       # projected into the response matrix
        t['T'] = -9999.
        t['flux'] = -9999.

    
    t.write(outfilename,format='ascii',overwrite=True)


 
def main(args):
    import matplotlib.pyplot as plt
    if ( ( len(args) < 4) or (args == '-h') or (args == '--help') ):
        print("\n### \n### ngmix_fit_superbit is a routine which takes a medsfile as its input and outputs shear-calibrated object shapes to a table\n###\n\n  python ngmix_fit_superbit.py medsfile start_index end_index outfilename \n \n")
    else:
        pass
    
    medsfile = args[1]
    index_start  = np.int(args[2])
    index_end = np.int(args[3])
    outfilename = args[4]
    meds_info = {'meds_file':args[1]}
    bitfitter = SuperBITNgmixFitter(meds_info) # MEDS file
    holding=[]                              # Array to hold parameters from ExpectMax fit to object
    mcal=[]                                 # Array to hold Metacal fit parameters
    identifying=[]                          # Array to hold identifying information for object
    
    for i in range(index_start, index_end):
        try:
            print("i get here 1")
            #pdb.set_trace()
            metcal_fit,no_metacal_fit = bitfitter._fit_one(i)
            mcal_fit_pars = bitfitter.get_mcalib_shears(i)
            mcal.append(mcal_fit_pars)
            
            try:
                print("i get here 2")
                holding.append(no_metacal_fit.get_result()['pars'])
            except:
                print("i get here 3")
                print("failed to append holding values")
                holding.append(np.array([-99999, -99999, -99999, -99999, -99999, -99999]))
                  
            identifying.append([bitfitter.medsObj['id'][i],bitfitter.medsObj['ra'][i],bitfitter.medsObj['dec'][i]])
        
        
            # Now for some plotting!
            image = bitfitter.medsObj.get_cutout(i,0)
            jac = bitfitter.medsObj.get_jacobian(i,0)
            jj = ngmix.Jacobian(row=jac['row0'],col=jac['col0'],dvdrow = jac['dvdrow'],dvdcol=jac['dvdcol'],dudrow=jac['dudrow'],dudcol=jac['dudcol'])
        
            try:
                
                gmix = no_metacal_fit.get_convolved_gmix()
                model_image = gmix.make_image(image.shape,jacobian=jj)
                fig,(ax1,ax2,ax3) = plt.subplots(nrows=1,ncols=3,figsize=(21,7))
                ax1.imshow(image)
                ax2.imshow(model_image)
                ax3.imshow(image - model_image)
                #pdb.set_trace()
                fig.savefig('diagnostics_plots/diagnostics-'+str(i)+'.png')
               
                plt.close(fig)
            except:
                # EM probably failed
                print("Bad gmix model, no image made :(")
                pass
            make_output_table(outfilename, holding, mcal, identifying)
            
        except:
            ("object %d failed, skipping..." % i)
      
    


if __name__ == '__main__':

    import pdb, traceback, sys
    try:
        main(sys.argv)
    except:
        thingtype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)    
        
