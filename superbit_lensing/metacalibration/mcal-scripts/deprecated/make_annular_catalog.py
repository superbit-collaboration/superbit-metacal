import numpy as np
import pdb
from astropy.table import Table,vstack
import glob
import sys, os
from astropy.io import fits
from esutil import htm

class McalCats():

    """
    contains all metacal catalogs
    
    This class will concatenate all metacal catalogs, and calculate shear-calibrated
    ellipticities based on metacal procedure. This will then match against the parent
    SExtractor catalog (set in option in main), and output a "fiat-format" ascii file
    for shear profile computation
    """

    
    def __init__(self,cat_info = None):

        """
        :sexcat:   master sextractor catalog that was the basis for medsfile
        :outcat:   name of output "fiat"-format catalog
        :mcals:    list of input metacal catalogs
        :mcCat:    full metacal catalog
        :shearcat: catalog for shear

        """
        
        self.sexcat = cat_info['sexcat']
        self.outcat = cat_info['outcat']
        self.mcals = cat_info['mcals']
        self.mcCat = None
        self.shearcat = None 

    def _get_catalogs(self):
        """
        method to concatenate input mcal tables
        """
        
        holding={}
        try:
            for i in np.arange(len(self.mcals)):
                tab=Table.read(self.mcals[i],format='fits',hdu=2)
                holding["tab{0}".format(i)] = tab
        except:
            for i in np.arange(len(self.mcals)):
                tab=Table.read(self.mcals[i],format='ascii')
                holding["tab{0}".format(i)] = tab
       
        all_catalogs=vstack([holding[val] for val in holding.keys()])
        
        return all_catalogs

    def _make_table(self):
        """
        - Match master metacal catalog to source extractor catalog
        - Trim catalog on g_cov, T/T_psf, etc. 
        - Correct g1/g2_noshear for the Rinv quantity (see Huff & Mandelbaum 2017)
        - Make an output table containing (corrected) ellipticities
        - Convert table to a "fiat" format ascii file 
        """
        


        # This step both applies selection cuts and returns shear-calibrated
        # tangential ellipticity moments
        self._compute_metacal_quantities()

        # Now match trimmed & shear-calibrated catalog against sextractor
        gals=Table.read(self.sexcat,format='fits',hdu=2) 
        master_matcher = htm.Matcher(16,ra=self.mcCat['ra'],dec=self.mcCat['dec'])
        gals_ind,master_ind,dist=master_matcher.match(ra=gals['ALPHAWIN_J2000'],dec=gals['DELTAWIN_J2000'],
                                                          radius=5E-4,maxmatch=1)
        match_master=self.mcCat[master_ind]; gals=gals[gals_ind]

        # Write to file
        newtab=Table()
        newtab.add_columns([match_master['id'],match_master['ra'],match_master['dec']],names=['id','ra','dec'])
        newtab.add_columns([gals['X_IMAGE'],gals['Y_IMAGE']])
        newtab.add_columns([match_master['g1_boot'],match_master['g2_boot']],names=['g1_boot','g2_boot'])
        
        newtab.add_columns([match_master['g1_MC'],match_master['g2_MC']],names=['g1_MC','g2_MC'])
        newtab.add_columns([match_master['g1_Rinv'],match_master['g2_Rinv']],names=['g1_Rinv','g2_Rinv'])
        try:
            newtab.add_columns([match_master['T_gmix'],match_master['flux_gmix']],names=['T_noshear','flux'])
        except:
            newtab.add_columns([match_master['T_noshear'],match_master['flux']],names=['T_noshear','flux'])
        
        newtab.write(self.outcat,format='csv',overwrite=True) # file gets replaced in the run_sdsscsv2fiat step
        self._run_sdsscsv2fiat()

        return newtab

    
    def _compute_metacal_quantities(self):
        """ 
        - Cut sources on S/N and minimum size (adapted from DES cuts). 
        - compute mean r11 and r22 for galaxies: responsivity & selection
        - divide "no shear" g1/g2 by r11 and r22, and return

        """
        
        noshear_selection = self.mcCat[(self.mcCat['T_noshear']>=1.2*self.mcCat['Tpsf_noshear'])\
                                           & (self.mcCat['s2n_noshear']<1000)\
                                           & (self.mcCat['s2n_noshear']>10)\
                                           & (self.mcCat['T_noshear']>5)]
                                             
        selection_1p = self.mcCat[(self.mcCat['T_1p']>=1.2*self.mcCat['Tpsf_1p'])\
                                      & (self.mcCat['s2n_1p']<1000)\
                                      & (self.mcCat['s2n_1p']>10)\
                                      & (self.mcCat['T_1p']>5)]

        selection_1m = self.mcCat[(self.mcCat['T_1m']>=1.2*self.mcCat['Tpsf_1m'])\
                                      & (self.mcCat['s2n_1m']<1000)\
                                      & (self.mcCat['s2n_1m']>10)\
                                      & (self.mcCat['T_1m']>5)]

        selection_2p = self.mcCat[(self.mcCat['T_2p']>=1.2*self.mcCat['Tpsf_2p'])\
                                      & (self.mcCat['s2n_2p']<1000)\
                                      & (self.mcCat['s2n_2p']>10)\
                                      & (self.mcCat['T_2p']>5)]

        selection_2m = self.mcCat[(self.mcCat['T_2m']>=1.2*self.mcCat['Tpsf_2m'])\
                                      & (self.mcCat['s2n_2m']<1000)\
                                      & (self.mcCat['s2n_2m']>10)\
                                      & (self.mcCat['T_2m']>5)]

        
        r11_gamma=np.mean(noshear_selection['r11'])
        r22_gamma=np.mean(noshear_selection['r22'])

        # assuming delta_shear in ngmix_fit_superbit is 0.01
        r11_S = (np.mean(selection_1p['g1_noshear'])-np.mean(selection_1m['g1_noshear']))/0.02
        r22_S = (np.mean(selection_2p['g2_noshear'])-np.mean(selection_2m['g2_noshear']))/0.02

        print("# mean values <r11_gamma> = %f <r22_gamma> = %f" % (r11_gamma,r22_gamma))
        print("# mean values <r11_S> = %f <r22_S> = %f" % (r11_S,r22_S))


        # Write current mcCat to file for safekeeping!
        # Then replace it with the "noshear"-selected catalog
        self.mcCat.write('full_metacal_cat.csv',format='ascii.csv',overwrite=True)
        
        self.mcCat=noshear_selection
        self.mcCat['g1_Rinv']= self.mcCat['g1_noshear']/(r11_gamma + r11_S)
        self.mcCat['g2_Rinv']= self.mcCat['g2_noshear']/(r11_gamma + r11_S)


        return 0

    
    def _run_sdsscsv2fiat(self):
        """
        Utility function to run sdsscsv2fiat to convert a 
        text file to a "fiat" format output file

        Tests for success of sdsscsv2fiat command before deleting tmpfile
        """

        name_arg = self.outcat
        #fiat_name_arg = str(name_arg.split('.')[0]+'.fiat')        
        fiat_name_arg = name_arg.replace('csv','fiat')
        cmd = ' '.join(['sdsscsv2fiat',name_arg, ' > ',fiat_name_arg])
        os.system(cmd)

        return

    
    def compute_shear_profiles(self, r_inner=200, r_outer=1500, nbins=5, xcen=3486, ycen=2359):
        """ 
        placeholder function to create shear proiles for ellipticity moments in master catalog with
        annular.c
        
        see annular.c documentation of that code for more information

        TO DO: perhaps make the arguments a parameter dict or something
        
        annular -c"X_IMAGE Y_IMAGE g1 g2" -f "g1>-2" -s 200 -e 1500 -n 5 fitvd-mock.fiat 3486 2359 > fitvd-mock.annular
        """
        pass
      
        return
      
      
    def run(self):

        # get master metacal catalog
        self.mcCat = self._get_catalogs()
        
        # match master metacal catalog to source extractor cat, write to file
        self.shearcat = self._make_table()

        # make shear profiles
        self.compute_shear_profiles()
            
        return

    
def main(args):

    
    if (len(args)<4):
        print("arguments missing; call is:\n")
        print("     python make_annular_catalog.py sexcat outcatname.asc mcalcat1 [mcalcat2 mcalcat3...]\n\n")
    else:
        pass
    
    
    sexcat = args[1]        
    outcatalog = args[2]
    in_cats = args[3:]
    
    try:
        all_mcal_cats = glob.glob(in_cats)
    except:
        # in this case, in_cats is already a list!
        all_mcal_cats = in_cats
       
    cat_info={'sexcat':sexcat, 'outcat':outcatalog, 'mcals':all_mcal_cats}
    metaCats = McalCats(cat_info)

    # run everything
    metaCats.run()
    
    
if __name__ == '__main__':

    import pdb, traceback, sys
    try:
        main(sys.argv)
    except:
        thingtype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)    
        
