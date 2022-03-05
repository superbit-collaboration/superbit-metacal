import numpy as np
import pdb
from astropy.table import Table,vstack,hstack
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
        self.selCat = None

    def _get_catalogs(self,overwrite=False):
        """
        method to concatenate input mcal tables, if given, and
        match against supplied SExtractor (background) galaxy catalog
        """

        holding={}
        try:
            for i in np.arange(len(self.mcals)):
                tab=Table.read(self.mcals[i],format='fits',hdu=1)
                holding["tab{0}".format(i)] = tab
        except:
            for i in np.arange(len(self.mcals)):
                tab=Table.read(self.mcals[i],format='ascii')
                holding["tab{0}".format(i)] = tab

        all_catalogs=vstack([holding[val] for val in holding.keys()])


        # now do matching against sexcat
        try:
            gals=Table.read(self.sexcat,format='fits',hdu=2)
        except:
            gals=Table.read(self.sexcat,format='fits',hdu=1)

        master_matcher = htm.Matcher(16,ra=all_catalogs['ra'],dec=all_catalogs['dec'])
        gals_ind,master_ind,dist=master_matcher.match(ra=gals['ALPHAWIN_J2000'],dec=gals['DELTAWIN_J2000'],
                                                          radius=1.5E-4,maxmatch=1)
        print("%d/%d mcal objects matched to reference sexcat" %(len(gals_ind),len(all_catalogs)))

        match_master = all_catalogs[master_ind]; gals=gals[gals_ind]

        # Remove duplicate column names, giving preference to SExtractor catalog values
        duplicate_cols = np.intersect1d(match_master.colnames,gals.colnames)
        match_master.remove_columns(duplicate_cols)

        # To avoid confusion with SEXtractor 'X/Y_IMAGE'(uppercase),
        # remove GalSim truth catalog 'x/y_image' (lowercase!)
        #
        if ('x_image' or 'y_image') in gals.colnames:
            gals.remove_column('x_image')
            gals.remove_column('y_image')

        mcCat_bgGals = hstack([match_master,gals])


        # save full catalog to file
        try:
            mcCat_bgGals.write('full_metacal_bgCat.fits',format='fits',overwrite=overwrite)
        except OSError as err:
            print("{0}\nOverwrite set to False".format(err))

        return mcCat_bgGals

    def _make_table(self):
        """
        - Select from catalog on g_cov, T/T_psf, etc.
        - Correct g1/g2_noshear for the Rinv quantity (see Huff & Mandelbaum 2017)
        - Save shear-response corrected ellipticities to an output table
        """


        # Match full metacal catalog catalog against sextractor catalog

        #match_master=self.mcCat; gals=self.mcCat

        # This step both applies selection cuts and generates shear-calibrated
        # tangential ellipticity moments

        qualcuts = self._compute_metacal_quantities()

        # I would love to be able to save qualcuts as comments in FITS header
        # but can't figure it out rn
        # self.selCat.meta={qualcuts}

        #self.selCat.write('selected_metacal_bgCat.fits',overwrite=True)
        self.selCat.write(self.outcat, format='fits', overwrite=True)

        return


    def _compute_metacal_quantities(self):
        """
        - Cut sources on S/N and minimum size (adapted from DES cuts).
        - compute mean r11 and r22 for galaxies: responsivity & selection
        - divide "no shear" g1/g2 by r11 and r22, and return

        TO DO: make the list of cuts an external config file

        """



        min_Tpsf = 1.2 # orig 1.15
        max_sn = 1000
        min_sn = 10 # orig 8 for ensemble
        min_T = 0.03 # orig 0.05
        max_T = 10 # orig inf
        covcut=7e-3 # orig 1 for ensemble

        qualcuts = {'min_Tpsf':min_Tpsf, 'max_sn':max_sn, 'min_sn':min_sn,
                    'min_T':min_T, 'max_T':max_T, 'covcut':covcut}

        print('#\n# cuts applied: Tpsf_ratio>%.2f SN>%.1f T>%.2f covcut=%.1e\n#\n' \
                         % (min_Tpsf,min_sn,min_T,covcut))

        noshear_selection = self.mcCat[(self.mcCat['T_noshear']>=min_Tpsf*self.mcCat['Tpsf_noshear'])\
                                        & (self.mcCat['T_noshear']<max_T)\
                                        & (self.mcCat['T_noshear']>=min_T)\
                                        & (self.mcCat['s2n_r_noshear']>min_sn)\
                                        & (self.mcCat['s2n_r_noshear']<max_sn)\
                                        & (np.array(self.mcCat['pars_cov0_noshear'].tolist())[:,0,0]<covcut)\
                                        & (np.array(self.mcCat['pars_cov0_noshear'].tolist())[:,1,1]<covcut)
                                           ]

        selection_1p = self.mcCat[(self.mcCat['T_1p']>=min_Tpsf*self.mcCat['Tpsf_1p'])\
                                      & (self.mcCat['T_1p']<=max_T)\
                                      & (self.mcCat['T_1p']>=min_T)\
                                      & (self.mcCat['s2n_r_1p']>min_sn)\
                                      & (self.mcCat['s2n_r_1p']<max_sn)\
                                      & (np.array(self.mcCat['pars_cov0_1p'].tolist())[:,0,0]<covcut)\
                                      & (np.array(self.mcCat['pars_cov0_1p'].tolist())[:,1,1]<covcut)
                                       ]

        selection_1m = self.mcCat[(self.mcCat['T_1m']>=min_Tpsf*self.mcCat['Tpsf_1m'])\
                                      & (self.mcCat['T_1m']<=max_T)\
                                      & (self.mcCat['T_1m']>=min_T)\
                                      & (self.mcCat['s2n_r_1m']>min_sn)\
                                      & (self.mcCat['s2n_r_1m']<max_sn)\
                                      & (np.array(self.mcCat['pars_cov0_1m'].tolist())[:,0,0]<covcut)\
                                      & (np.array(self.mcCat['pars_cov0_1m'].tolist())[:,1,1]<covcut)
                                     ]

        selection_2p = self.mcCat[(self.mcCat['T_2p']>=min_Tpsf*self.mcCat['Tpsf_2p'])\
                                      & (self.mcCat['T_2p']<=max_T)\
                                      & (self.mcCat['T_2p']>=min_T)\
                                      & (self.mcCat['s2n_r_2p']>min_sn)\
                                      & (self.mcCat['s2n_r_2p']<max_sn)\
                                      & (np.array(self.mcCat['pars_cov0_2p'].tolist())[:,0,0]<covcut)\
                                      & (np.array(self.mcCat['pars_cov0_2p'].tolist())[:,1,1]<covcut)
                                      ]

        selection_2m = self.mcCat[(self.mcCat['T_2m']>=min_Tpsf*self.mcCat['Tpsf_2m'])\
                                      & (self.mcCat['T_2m']<=max_T)\
                                      & (self.mcCat['T_2m']>=min_T)\
                                      & (self.mcCat['s2n_r_2m']>min_sn)\
                                      & (self.mcCat['s2n_2m']<max_sn)\
                                      & (np.array(self.mcCat['pars_cov0_2m'].tolist())[:,0,0]<covcut)\
                                      & (np.array(self.mcCat['pars_cov0_2m'].tolist())[:,1,1]<covcut)
                                    ]


        r11_gamma=(np.mean(noshear_selection['g_1p'][:,0]) -np.mean(noshear_selection['g_1m'][:,0]))/0.02
        r22_gamma=(np.mean(noshear_selection['g_2p'][:,1]) -np.mean(noshear_selection['g_2m'][:,1]))/0.02


        # assuming delta_shear in ngmix_fit_superbit is 0.01
        r11_S = (np.mean(selection_1p['g_noshear'][:,0])-np.mean(selection_1m['g_noshear'][:,0]))/0.02
        r22_S = (np.mean(selection_2p['g_noshear'][:,1])-np.mean(selection_2m['g_noshear'][:,1]))/0.02

        print("# mean values <r11_gamma> = %f <r22_gamma> = %f" % (r11_gamma,r22_gamma))
        print("# mean values <r11_S> = %f <r22_S> = %f" % (r11_S,r22_S))
        print("%d objects passed selection criteria" % len(noshear_selection))

        # Populate the selCat attribute with "noshear"-selected catalog
        self.selCat = noshear_selection

        # compute noise; not entirely sure whether there needs to be a factor of 0.5 on tot_covar...
        # seems like not if I'm applying it just to tangential ellip, yes if it's being applied to each
        #shape_noise = np.std(np.sqrt(self.mcCat['g_noshear'][:,0]**2 + self.mcCat['g_noshear'][:,1]**2))
        shape_noise=0.1
        tot_covar = shape_noise + np.array(self.selCat['pars_cov_noshear'].tolist())[:,0,0] + np.array(self.selCat['pars_cov_noshear'].tolist())[:,1,1]
        weight = 1/tot_covar

        r11=( noshear_selection['g_1p'][:,0] - noshear_selection['g_1m'][:,0] ) / 0.02
        r12=( noshear_selection['g_2p'][:,0] - noshear_selection['g_2m'][:,0] ) / 0.02
        r21=( noshear_selection['g_1p'][:,1] - noshear_selection['g_1m'][:,1] ) / 0.02
        r22=( noshear_selection['g_2p'][:,1] - noshear_selection['g_2m'][:,1] ) / 0.02

        try:
            self.selCat.add_columns([r11,r12,r21,r22],names=['r11','r12','r21','r22'])

            R = [ [r11, r12], [r21, r22] ]
            R = np.array(R)
            g1_MC=np.zeros_like(r11)
            g2_MC=np.zeros_like(r22)

            for k in range(len(r11)):
                Rinv = np.linalg.inv(R[:,:,k])
                gMC = np.dot(Rinv,noshear_selection[k]['g_noshear'])
                g1_MC[k]=gMC[0];g2_MC[k]=gMC[1]

                self.selCat.add_columns([g1_MC,g2_MC], names = ['g1_MC','g2_MC'])
        except:
            print('WARNING: response value-adds not added!')

        self.selCat['g1_Rinv'] = self.selCat['g_noshear'][:,0]/(r11_gamma + r11_S)
        self.selCat['g2_Rinv'] = self.selCat['g_noshear'][:,1]/(r22_gamma + r22_S)

        self.selCat.add_column(r11_S,name='R11_S')
        self.selCat.add_column(r22_S,name='R22_S')
        self.selCat['weight'] = weight

        return qualcuts


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

        # match master metacal catalog to source extractor cat
        self.mcCat = self._get_catalogs(overwrite=True)

        # source selection; return metacalibrated shears
        self._make_table()

        # make shear profiles
        self.compute_shear_profiles()

        return


def main(args):


    if (len(args)<4):
        print("arguments missing; call is:\n")
        print("     python make_annular_catalog.py sexcat outcatname.fits mcalcat1 [mcalcat2 mcalcat3...]\n\n")

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
        #pdb.post_mortem(tb)
