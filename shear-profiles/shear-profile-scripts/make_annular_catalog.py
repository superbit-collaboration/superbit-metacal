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
        - Correct g1/g2_noshear for the Rinv quantity (see Huff & Mandelbaum 2017)
        - Make an output table containing (corrected) ellipticities
        - Convert table to a "fiat" format ascii file 
        """
        
        gals=Table.read(self.sexcat,format='fits',hdu=2) #contains only analysis objects.
        master_matcher = htm.Matcher(16,ra=self.mcCat['ra'],dec=self.mcCat['dec'])
        gals_ind,master_ind,dist=master_matcher.match(ra=gals['ALPHAWIN_J2000'],dec=gals['DELTAWIN_J2000'],radius=5E-4,maxmatch=1)
        match_master=self.mcCat[master_ind]; gals=gals[gals_ind]

        match_master=self._compute_metacal_quantities(match_master)
        
        newtab=Table()
        newtab.add_columns([match_master['id'],match_master['ra'],match_master['dec']],names=['id','ra','dec'])
        newtab.add_columns([gals['X_IMAGE'],gals['Y_IMAGE']])
        newtab.add_columns([match_master['g1_gmix'],match_master['g2_gmix']],names=['g1_gmix','g2_gmix'])
        newtab.add_columns([match_master['T_gmix'],match_master['noshear']],names=['T_gmix','flux_gmix'])
        newtab.add_columns([match_master['g1_MC'],match_master['g2_MC']],names=['g1_MC','g2_MC'])
        newtab.add_columns([match_master['g1_Rinv'],match_master['g2_Rinv']],names=['g1_Rinv','g2_Rinv'])
        newtab.add_columns([match_master['T'],match_master['flux']],names=['T_mcal','flux_mcal'])
        
        newtab.write(self.outcat,format='csv',overwrite=True) # file gets replaced in the run_sdsscsv2fiat step
        self._run_sdsscsv2fiat()

        return newtab

    
    def _compute_metacal_quantities(self,match_master):
        """ probably doesn't need to be a separate function, 
        but potentially useful as a placeholder for other metacal calculations"""

        r11=np.mean(match_master['r11'])
        r22=np.mean(match_master['r22'])
        print("# mean values <r11> = %f <r22> = %f" % (r11,r22))
        match_master['g1_Rinv']= match_master['g1_noshear']/r11
        match_master['g2_Rinv']= match_master['g2_noshear']/r22
        
        return match_master

    
    def _run_sdsscsv2fiat(self):
        """
        Utility function to run sdsscsv2fiat to convert a 
        text file to a "fiat" format output file

        Tests for success of sdsscsv2fiat command before deleting tmpfile
        """

        name_arg = self.outcat
        fiat_name_arg = str(name_arg.split('.')[0]+'.fiat')        
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
        
