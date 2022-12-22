import numpy as np
import meds
import os
import psfex
import piff
from astropy.io import fits
import string
from pathlib import Path
import ipdb
from astropy import wcs
import fitsio
import esutil as eu
from astropy.table import Table
import astropy.units as u
import astropy.coordinates
from astroquery.gaia import Gaia
import superbit_lensing.utils as utils
import glob

'''
Goals:
  - Take as input calibrated images
  - Make masks/weight maps
    -- based on flats/darks: locate bad pixels there
    -- also choose a minimum distance from the edge to mask.
  - Make a catalog (SExtractor)
  - Build a psf model (PSFEx)
  - run the meds maker (use meds.Maker)
  - run ngmix (run library)

TO DO:
    - Make weight files
    - Do a flux calibration for photometric scaling in coaddition/divide by exposure time
    - Actually do darks better: match exposure times, rather than collapsing them all down
    - Run medsmaker
'''

def piff_extender(piff_file, stamp_size=20):
    """
    Utility function to add the get_rec function expected
    by the MEDS package

    """
    psf = piff.read(piff_file)

    type_name = type(psf)

    class PiffExtender(type_name):
        '''
        A helper class that adds functions expected by MEDS
        '''

        def __init__(self, type_name=None):

            self.psf = None
            self.single_psf = type_name

            return

        def get_rec(self,row,col):

            fake_pex = self.psf.draw(x=col, y=row, stamp_size=stamp_size).array

            return fake_pex

        def get_center(self,row,col):

            psf_shape = self.psf.draw(x=col,y=row,stamp_size=stamp_size).array.shape
            cenpix_row = (psf_shape[0]-1)/2
            cenpix_col = (psf_shape[1]-1)/2
            cen = np.array([cenpix_row,cenpix_col])

            return cen

    psf_extended = PiffExtender(type_name)
    psf_extended.psf = psf

    return psf_extended

class BITMeasurement():
    def __init__(self, image_files=None, data_dir=None, outdir=None,
                    run_name=None, log=None, overwrite=False, vb=False):
        '''
        :image_files: Python List of image filenames; must be complete relative or absolute path.
        :flat_files: Python List of image filenames; must be complete relative or absolute path.
        :master_dark: Python List of image filenames; must be complete relative or absolute path.
        :catalog: Object that stores FITS array of catalog
        :coadd: Set to true if the first image file is a coadd image (must be first)
        :weight_files: default weight maps based on bkg noise/shot noise
        :combined_mask_file: combination of master dark and BPM
        :combined_weight_file: combination of combined_mask_file and individual exposure weight files

        '''

        self.image_files = image_files
        self.run_name = run_name
        self.overwrite = overwrite
        self.outdir = outdir
        self.data_dir = data_dir
        self.vb = vb
        self.bpm = None
        self.master_flat = None
        self.master_dark = None
        self.reduced_images = None
        self.coadd_file = None
        self.catalog = None
        self.psf_path = None
        self.weight_files = None
        self.combined_mask = None
        self.combined_weight_files = None
        self.pix_scale = None

        self._setup_dirs()

        if log is None:
            logfile = 'medsmaker.log'
            log = utils.setup_logger(logfile)

        self.logprint = utils.LogPrint(log, vb)

        filepath = Path(os.path.realpath(__file__))
        self.base_dir = filepath.parents[1]

        return

    def _setup_dirs(self):
        for directory in [self.outdir, self.data_dir]:
            if directory is None:
                directory = os.getcwd()
            if not os.path.exists(directory):
                os.mkdir(directory)

    def set_path_to_psf(self,path=None):
        self.psf_path = path
        if self.psf_path is None:
            self.psf_path = os.cwd()+'psf-outputs'
        if not os.path.exists(self.psf_path):
            os.mkdir(self.psf_path)

    def _setup_mask(self, mask_name, mask_dir=None, ext=0):
        if mask_name is None:
            return None
        elif os.path.exists(mask_name):
            return fits.getdata(mask_name, ext=0)
        elif not os.path.exists(mask_name) and os.path.isdir(mask_dir):
            mask_file = os.path.join(mask_dir, mask_name)
            return fits.getdata(mask_name)

    def setup_calib_data(self, bpm=None, master_dark=None, master_flat=None):
        if bpm is not None:
            #self.bpm = fits.getdata(bpm)
            self.bpm = self._setup_mask(mask_name=bpm)
        if master_flat is not None:
            self.master_flat = fits.getdata(master_flat)
        if master_dark is not None:
            self.master_dark = fits.getdata(master_dark)

    def get_weight(self, weight_name, weight_dir=None):
        if weight_dir is None:
            weight_dir = os.path.join(self.outdir,'weight_files')
        self.weight_file = os.path.join(weight_dir, weight_name)

    def set_path_to_science_data(self,path=None):
        if path is None:
            self.science_path = '../Data/timmins2019/raw'
            self.reduced_science_path = '../Data/timmins2019/reduced'
        else:
            self.science_path = path
            self.reduced_science_path = path

    def set_path_to_wcs_data(self,path=None):
        # Possibly deprecated
        if path is None:
            self.wcs_path = '../Data/timmins2019/raw'
        else:
            self.wcs_path = path

    def _get_wcs_info(self,image_filename):
        '''
        Return a new image header with WCS (SIP) information,
        or nothing if the WCS file doesn't exist
        '''
        try:
            # os.path.basename gets the filename if a full path gets supplied
            basename = os.path.basename(image_filename)
            splitted=basename.split('_')
            wcsName=os.path.join(self.wcs_path,str('wcs_'+splitted[2]+'_'+splitted[3]+'.fits'))
            inhead=fits.getheader(wcsName)
            w=wcs.WCS(inhead)
            wcs_sip_header=w.to_header(relax=True)
        except:
            self.logprint('cluster %s has no WCS, skipping...' % wcsName )
            wcs_sip_header=None

        return wcs_sip_header

    def _make_new_fits(self,image_filename):
        '''
        Returns new cluster fits file with the
        updated WCS and some important keywords
        List of KW can probably be changed as needed
        '''
        if os.path.exists(image_filename):
            ClusterFITSFile=fits.open(image_filename)
            ClusterHeader=ClusterFITSFile[0].header
            WCSheader=self._get_wcs_info(image_filename)
            if WCSheader is not None:
                for key in WCSheader.keys():
                    ClusterHeader[key]=WCSheader[key]
                outFITS=fits.PrimaryHDU(ClusterFITSFile[0].data,header=ClusterHeader)
                new_image_filename = os.path.join(self.science_path,image_filename.replace(".fits","WCS.fits"))
                outFITS.writeto(new_image_filename)
                return new_image_filename
        else:
            self.logprint("Could not process %s" % image_filename)
            return None

    def add_wcs_to_science_frames(self):
        '''
        wrapper for _make_new_fits() which returns astrometry-corrected images
        '''
        fixed_image_files = []

        for image_file in self.image_files:
            fixed_image_file = self._make_new_fits(image_file)
            if fixed_image_file is not None:
                fixed_image_files.append(fixed_image_file)

        self.image_files = fixed_image_files

    def reduce(self,overwrite=False,skip_sci_reduce=False):
        # Read in and average together the bias, dark, and flat frames.

        bname = os.path.join(self.mask_path,'master_bias_mean.fits')
        """
        if (not os.path.exists(bname) or (overwrite==True)):
            # Taking median biases and darks instead of mean to eliminate odd noise features
            bias_array=[]
            self.logprint("I get to bias_array")
            for ibias_file in self.bias_files:
                bias_frame = fitsio.read(ibias_file)
                bias_array.append(bias_frame)
                master_bias = np.median(bias_array,axis=0)
                fitsio.write(os.path.join(self.outdir,'master_bias_median.fits'),master_bias,overwrite=True)
        else:
        """
        master_bias = fitsio.read(bname)

        dname = os.path.join(self.mask_path,'master_dark_median.fits')
        if (not os.path.exists(dname) or (overwrite==True)):
            dark_array=[]
            for imaster_dark in self.master_darks:
                hdr = fitsio.read_header(imaster_dark)
                time = hdr['EXPTIME'] / 1000. # exopsure time, seconds
                dark_frame = ((fitsio.read(imaster_dark)) - master_bias) * 1./time
                dark_array.append(dark_frame)
                master_dark = np.median(dark_array,axis=0)
                fitsio.write(os.path.join(self.mask_path,'master_dark_median.fits'),master_dark,overwrite=True)
        else:
            master_dark=fitsio.read(dname)

        fname = os.path.join(self.mask_path,'master_flat_median.fits')
        if (not os.path.exists(fname) or (overwrite==True)):
            flat_array=[]
            # Ideally, all the flats should have the SAME exposure time, or rather, each filter
            # gets its own band with its own flat exptime
            for iflat_file in self.flat_files:
                hdr = fitsio.read_header(iflat_file)
                time = hdr['EXPTIME'] /  1000.
                flat_frame = (fitsio.read(iflat_file) - master_bias - master_dark * time ) * 1./time
                flat_array.append(flat_frame)
                master_flat1 = np.median(flat_array,axis=0)
                master_flat = master_flat1/np.median(master_flat1)
                fitsio.write(os.path.join(self.mask_path,'master_flat_median.fits'),master_flat,overwrite=True)
        else:
            master_flat=fitsio.read(fname)
        if not skip_sci_reduce:
            reduced_image_files=[]

            for this_image_file in self.image_files:
                # WARNING: as written, function assumes science data is in 0th extension
                this_image_fits=fits.open(this_image_file)
                time=this_image_fits[0].header['EXPTIME']/1000.
                this_reduced_image = (this_image_fits[0].data - master_bias)-(master_dark*time)
                this_reduced_image = this_reduced_image/master_flat
                updated_header = this_image_fits[0].header
                updated_header['HISTORY']='File has been bias & dark subtracted and FF corrected'
                this_image_outname=(os.path.basename(this_image_file)).replace(".fits","_cal.fits")
                this_image_outname = os.path.join(self.outdir,this_image_outname)
                reduced_image_files.append(this_image_outname)
                this_outfits=fits.PrimaryHDU(this_reduced_image,header=updated_header)
                this_outfits.writeto(this_image_outname,overwrite=True)

            self.image_files = reduced_image_files
        else:
            pass

    def quick_reduce(self, darkname=None):
        '''
        Just subtract the darks for now, could be extended with others
        '''

        bpm = self.bpm
        master_dark = self.master_dark
        master_flat = self.master_flat

        reduced_images = []

        if master_dark is not None:

            self.logprint(f'Subtracting master_dark...')

            for image in self.image_files:
                imfits = fits.open(image)
                raw = imfits[0].data; hdr = imfits[0].header
                dark_sub = raw - master_dark
                dsub_card = ('DARK_SUB', 'master_dark_clipped.fits', 'dark image subtracted')
                hdr.append(dsub_card)
                outname = image.replace('.fits', '_cal.fits')
                fits.writeto(outname, data=dark_sub.astype(np.float32),
                                header=hdr, overwrite=True)
                reduced_images.append(outname)

        if master_flat is not None:
            print('Not set up to do quick-reduce with flats')

        self.reduced_images = reduced_images

        return

    def get_combined_mask(self, combined_mask_file, dark_sigma_thresh=2.0):
        '''
        Make a combined weight-mask using a master dark, a master flat,
        and any existing weight maps. Right now, this just takes the 0th extension
        of the simulation weights and masks, which is fine.

        When ready, we should put this in the "quick reduce" function as that
        naturally loops over images, weights, and masks.
        '''

        bpm = self.bpm
        master_dark = self.master_dark
        master_flat = self.master_flat
        overwrite = self.overwrite

        try:
            combined_mask_file
        except NameError:
            print('No combined_mask filename specified')

        if os.path.exists(combined_mask_file) and overwrite is False:
            self.logprint(f"\n Loading {combined_mask_file}\n")
            self.combined_mask = fits.getdata(combined_mask_file)

        else:
            master_dark_mask = np.ones_like(master_dark)*master_dark
            clip = master_dark > (dark_sigma_thresh*np.median(master_dark))
            master_dark_mask[clip] = 0 # setting it to 1e-6 might reduce interpolation/dark pixel issues
            combined_mask = np.ones_like(master_dark_mask)*master_dark_mask

            bpm_inverse = 1 - bpm # The default forecast mask is all zeros!
            combined_mask *= bpm_inverse

            fits.writeto(combined_mask_file, data=combined_mask, overwrite=overwrite)

            self.combined_mask = combined_mask
            pdb.set_trace()
            '''
            # Start with dark
            med_dark_array=[]
            flattened=np.ravel(mdark)
            outrav=np.zeros(mflat.size)
            outrav[flattened>=global_dark_thresh]=1
            med_dark_array.append(outrav)
            sum_dark = np.sum(med_dark_array,axis=0)
            # This transforms our bpm=1 array to a bpm=0 array
            darkmask=np.ones(sum_dark.size)
            #darkmask[sum_dark==(len(master_darks))]=0
            darkmask[sum_dark==1]=0
            outfile = fits.PrimaryHDU(darkmask.reshape(np.shape(mdark)))
            outfile.writeto(os.path.join(self.mask_path,'darkmask.fits'),overwrite=True)


            # repeat for flat
            mflat = fits.getdata(mflat_fname)
            med_flat_array=[]
            flattened=np.ravel(mflat)
            outrav=np.zeros(mflat.size)
            outrav[flattened<=global_flat_thresh]=1
            med_flat_array.append(outrav)
            sum_flat = np.sum(med_flat_array,axis=0)
            # This transforms our bpm=1 array to a bpm=0 array
            flatmask=np.ones(sum_flat.size)
            #darkmask[sum_dark==(len(master_darks))]=0
            flatmask[sum_flat==1]=0
            outfile = fits.PrimaryHDU(flatmask.reshape(np.shape(mflat)))
            outfile.writeto(os.path.join(self.mask_path,'flatmask.fits'),overwrite=True)

            # Now generate actual mask
            supermask = (darkmask + flatmask)/2.
            outfile = fits.PrimaryHDU(flatmask.reshape(np.shape(mflat)))
            outfile.writeto(os.path.join(self.mask_path,'supermask.fits'),overwrite=True)
            '''

        return

    def make_combined_weight(self, name, outdir=None, bpm=None,
                                master_dark=None, dark_sigma_thresh=2.0,
                                master_flat=None, flat_thresh=0.85):

        combined_weight_name = os.path.join(outdir, name)

        if (not os.path.exists(combined_weight_name)) or (self.overwrite is True):

            # This will fail for the new MEF images/non-trivial weights and bpm
            weight = fits.getdata(self.weight_file, ext=0)

        return

    def _make_detection_image(self, outfile_name='detection.fits', weightout_name='weight.fits'):
        '''
        :output: output file where detection image is written.

        Runs SWarp on provided (reduced!) image files to make a coadd image
        for SEX and PSFEx detection.
        '''
        ### Code to run SWARP
        if self.reduced_images is not None:
            self.logprint('Running on dark-subtracted images')
            image_files = self.reduced_images
        else:
            image_files = self.image_files

        image_args = ' '.join(image_files)
        detection_file = os.path.join(self.outdir, outfile_name) # This is coadd
        weight_file = os.path.join(self.outdir, weightout_name) # This is coadd weight
        config_arg = '-c ' + os.path.join(self.base_dir, 'superbit/astro_config/swarp.config')
        weight_arg = f'-WEIGHT_IMAGE {self.combined_weight} -WEIGHT_TYPE MAP_WEIGHT'
        resamp_arg = f'-RESAMPLE_DIR {self.outdir}'
        outfile_arg = f'-IMAGEOUT_NAME {detection_file} -WEIGHTOUT_NAME {weight_file}'

        cmd = ' '.join(['swarp ', image_args, weight_arg, resamp_arg, outfile_arg, config_arg])

        self.logprint('swarp cmd is ' + cmd)
        os.system(cmd)
        self.logprint('\n')

        if os.path.curdir != self.outdir:
            cmd = f'mv *.xml *.fits {self.outdir}'
            self.logprint(cmd)
            os.system(cmd)
            self.logprint('\n')

        return detection_file, weight_file


    def select_sources_from_gaia():
        # Use some set of criteria to choose sources for measurement.

        coord = astropy.coordinates.SkyCoord(hdr['CRVAL1'],hdr['CRVAL2'],unit='deg')
        result = Gaia.cone_search_async(coord,radius=10*u.arcminute)
        catalog = result.get_data()
        pass

    def _run_sextractor(self, detection_file, weight_file, sextractor_config_path):
        '''
        Utility method to invoke Source Extractor on supplied detection file
        Returns: file path of catalog
        '''

        if sextractor_config_path is None:
            sextractor_config_path = os.path.join(
                self.base_dir, 'superbit/astro_config/'
                )

        cat_name=detection_file.replace('.fits','_cat.ldac')
        name_arg='-CATALOG_NAME ' + cat_name
        config_arg = sextractor_config_path+'sextractor.mock.config'
        param_arg = '-PARAMETERS_NAME '+sextractor_config_path+'sextractor.param'
        nnw_arg = '-STARNNW_NAME '+sextractor_config_path+'default.nnw'
        filter_arg = '-FILTER_NAME '+sextractor_config_path+'default.conv'

        seg_name = detection_file.replace('.fits','.sgm.fits')
        bg_sub_name = detection_file.replace('.fits','.sub.fits')
        bg_rms_name = detection_file.replace('.fits','.bkg_rms.fits')
        checkim_arg = f'-CHECKIMAGE_TYPE -BACKGROUND,SEGMENTATION,BACKGROUND_RMS'
        checkname_arg = f'-CHECKIMAGE_NAME {bg_sub_name},{seg_name},{bg_rms_name}'

        cmd = ' '.join(['sex', detection_file, name_arg, checkim_arg,
                        checkname_arg, param_arg, nnw_arg, filter_arg,
                        '-c', config_arg
                        ])

        if weight_file is not None:
            weight_arg = \
                    f'-WEIGHT_IMAGE {weight_file} ' + \
                    '-WEIGHT_TYPE MAP_WEIGHT'
            cmd = ' '.join([cmd, weight_arg])

        self.logprint("sex cmd is " + cmd)

        # utils.run_command(cmd, logprint=self.logprint)
        os.system(cmd)

        print("cat_name_is {}".format(cat_name))
        return cat_name

    def make_coadd_catalog(self, sextractor_config_path=None):
        '''
        Wrapper for astromatic tools to make coadd detection image
        from provided exposures and return a coadd catalog
        '''

        if sextractor_config_path is None:
            sextractor_config_path = os.path.join(self.base_dir, 'superbit/astro_config/')

        if self.run_name is not None:
            p = f'{self.run_name}_'
        else:
            p = ''
        outfile_name = f'{p}mock_coadd.fits'
        weightout_name = outfile_name.replace('.fits', '.weight.fits')

        detection_file,weight_file = \
                                self._make_detection_image(outfile_name=outfile_name,
                                                            weightout_name=weightout_name
                                                            )
        self.coadd_file = detection_file

        # Set pixel scale
        self.pix_scale = utils.get_pixel_scale(self.coadd_file)

        # Run SExtractor on coadd
        cat_name = self._run_sextractor(detection_filepath,
                                        weight_filepath,
                                        sextractor_config_path)

        try:
            self.catalog = Table.read(cat_name)

            if source_selection is True:
                self.logprint("selecting sources")
                self._select_sources_from_catalog(fullcat=self.catalog, catname=cat_name)

        except Exception as e:
            self.logprint("coadd catalog could not be loaded; check name?")
            raise(e)

    def make_exposure_catalogs(self, weight_file=None, sextractor_config_path=None):

        sexcat_names = []

        if self.reduced_images is not None:
            self.logprint('Running on dark-subtracted images')
            image_files = self.reduced_images
        else:
            image_files = self.image_files

        for image_file in image_files:
            sexcat = self._run_sextractor(image_file,
                                            weight_file,
                                            sextractor_config_path
                                            )
            sexcat_names.append(sexcat)

        return sexcat_names

    def make_psf_models(self, select_truth_stars=False, im_cats=None,
                        use_coadd=False, psf_mode='piff', psf_seed=None,
                        star_params=None):

        if star_params is None:
            star_keys = {'size_key':'FLUX_RAD','mag_key':'MAG_AUTO'}
            star_params = {'CLASS_STAR':0.92,
                            'MIN_MAG':22,
                            'MAX_MAG':17,
                            'MIN_SIZE':1.1,
                            'MAX_SIZE':3.0,
                            'MIN_SNR': 20
                            }

        if psf_seed is None:
            psf_seed = utils.generate_seeds(1)

        self.psf_models = []

        if self.reduced_images is not None:
            image_files = self.reduced_images
        else:
            image_files = self.image_files

        Nim = len(image_files)

        assert(len(im_cats)==Nim)

        # Will be placed first
        if use_coadd is True:
            Nim += 1

        k = 0
        for i in range(Nim):
            if (i == 0) and (use_coadd is True):
                # TODO: temporary coadd PSF solution!
                # see issue #83
                self.psf_models.append(None)
                continue
            else:
                if use_coadd is True:
                    imagefile = image_files[i-1]
                else:
                    imagefile = image_files[i]

            # TODO: update as necessary
            weightfile = self.mask_file.replace('mask', 'weight')

            if psf_mode == 'piff':
                piff_model = self._make_piff_model(
                    imagefile,
                    select_truth_stars=select_truth_stars,
                    star_params=star_params,
                    psf_seed=psf_seed
                    )
                self.psf_models.append(piff_model)

            elif psf_mode == 'psfex':
                psfex_model_file = self._make_psfex_model(
                    im_cats[i],
                    weightfile=weightfile,
                    select_truth_stars=select_truth_stars,
                    star_params=star_params,
                    psf_seed=psf_seed
                    )

                # create & move checkimages to psfex_output
                psfex_outdir = os.path.join(self.data_dir,'psfex-output')

                if not os.path.exists(psfex_outdir):
                    os.mkdir(psfex_outdir)

                cleanup_cmd = ' '.join(
                    ['mv chi* resi* samp* snap* proto* *.xml', psfex_plotdir]
                    )
                cleanup_cmd2 = ' '.join(
                    ['mv count*pdf ellipticity*pdf fwhm*pdf', psfex_plotdir]
                    )
                os.system(cleanup_cmd)
                os.system(cleanup_cmd2)

                self.psf_models.append(psfex.PSFEx(psfex_model_file))

        # TODO: temporary coadd PSF solution!
        # see issue #83
        self.psf_models[0] = self.psf_models[1]

        return

    def _make_psfex_model(self, im_cat, weightfile='weight.fits',
                          config_path=None,
                          psfex_outdir='./tmp/', psf_seed=None,
                          select_truth_stars=False, star_params=None):
        '''
        Gets called by make_psf_models for every image in self.image_files
        Wrapper for PSFEx. Requires a FITS format catalog with vignettes

        TODO: Implement psf_seed for PSFEx!
        '''

        if config_path is None:
            config_path = os.path.join(
                self.base_dir, 'superbit/astro_config/'
                )

        # If flagged, get a "clean" star catalog for PSFEx input
        if select_truth_stars==True:
            # This will break for any truth file nomenclature that
            # isn't pipeline default
            truthdir = self.data_dir
            truthcat = glob.glob(''.join([truthdir,'*truth*.fits']))[0]
            truthfilen = os.path.join(truthdir,truthcat)
            self.logprint("using truth catalog %s" % truthfilen)
            psfcat_name = self._select_stars_for_psf(
                sscat=im_cat,truthfile=truthfilen
                )

        else:
            psfcat_name = im_cat

        # Now run PSFEx on that image and accompanying catalog

        psfex_config_arg = '-c '+config_path+'psfex.mock.config'
        outcat_name = im_cat.replace('.fits','.psfex.star')
        cmd = ' '.join(
            ['psfex', psfcat_name,psfex_config_arg,'-OUTCAT_NAME', outcat_name]
            )
        self.logprint("psfex cmd is " + cmd)
        os.system(cmd)
        # utils.run_command(cmd, logprint=self.logprint)

        psfex_model_file = imcat_ldac_name.replace('.ldac','.psf')

        # Just return name, the make_psf_models method reads it in
        # as a PSFEx object
        return psfex_model_file

    def _make_piff_model(self, imagefile, weightfile='weight.fits',
                         config_path=None, psfex_outdir='./tmp/',
                         select_truth_stars=False, star_params=None,
                         psf_seed=None):
        '''
        Method to invoke PIFF for PSF modeling
        Returns a "PiffExtender" object with the get_rec() and get_cen()
        functions expected by meds.maker

        First, let's get it to run on one, then we can focus on running list
        '''

        if config_path is None:
            config_path = os.path.join(
                self.base_dir, 'superbit/astro_config/'
            )

        output_dir = os.path.join(self.data_dir, 'piff-output')
        utils.make_dir(output_dir)

        base_piff_config = os.path.join(config_path, 'piff.config')
        run_piff_config = os.path.join(output_dir, 'piff.config')

        # update piff config w/ psf_seed
        config = utils.read_yaml(base_piff_config)
        if psf_seed is None:
            psf_seed = utils.generate_seeds(1)
        config['select']['seed'] = psf_seed
        utils.write_yaml(config, run_piff_config)

        imcat_ldac_name = imagefile.replace('.fits', '_cat.ldac')

        if select_truth_stars is True:
            # This will break for any truth file nomenclature that
            # isn't pipeline default
            truthdir = self.outdir
            try:
                truthcat = glob.glob(os.path.join(truthdir,'*truth*.fits'))[0]
            except OSError:
                # old way
                truthcat = glob.glob(os.path.join(truthdir,'*truth*.dat'))[0]

            truthfilen = os.path.join(truthdir,truthcat)
            self.logprint('using truth catalog %s' % truthfilen)

        psfcat_name = self._select_stars_for_psf(
            sscat=imcat_ldac_name,
            star_params=star_params
            )

        # Now run PIFF on that image and accompanying catalog
        im_name = imagefile.replace('.fits', '.sub.fits')
        image_arg = f'input.image_file_name={im_name}'
        psfcat_arg = f'input.cat_file_name={psfcat_name}'
        output_name = imagefile.split('/')[-1].replace('.fits', '.piff')
        full_output_name = os.path.join(output_dir, output_name)
        output_arg = f'output.file_name={output_name} output.dir={output_dir}'
        cmd = f'piffify {run_piff_config} {image_arg} {psfcat_arg} {output_arg}'

        self.logprint('piff cmd is ' + cmd)
        os.system(cmd)

        piff_extended = piff_extender(full_output_name)

        return piff_extended

    def _select_stars_for_psf(self, sscat, truthfile=None, starkeys=None,
                              star_params=None):
        '''
        Method to obtain stars from SExtractor catalog using the truth catalog from GalSim
            sscat : input ldac-format catalog from which to select stars
            truthcat : the simulation truth catalog written out by GalSim
        '''

        try:
            ss = Table.read(sscat,hdu=2)
        except:
            ss = Table.read(sscat,hdu=1)

        if truthfile is not None:
            # Read in truthfile, obtain stars with redshift cut
            try:
                truthcat = Table.read(truthfile,format='fits')
            except:
                truthcat = Table.read(truthfile,format='ascii')
            stars=truthcat[truthcat['redshift']==0]
            outname = sscat.replace('.ldac','truthstars.ldac')

            # match sscat against truth star catalog -- 0.72" = 5 SuperBIT pixels
            self.logprint("selecting on truth catalog, %d stars found"%len(stars))
            star_matcher = eu.htm.Matcher(16,ra=stars['ra'],dec=stars['dec'])
            matches,starmatches,dist = star_matcher.match(ra=ss['ALPHAWIN_J2000'],
                                                    dec=ss['DELTAWIN_J2000'],radius=0.72/3600.,maxmatch=1)

            # Save result to file, return filename
            ss=ss[matches]
            wg_stars = (ss['SNR_WIN']>star_params['MIN_SNR']) & (ss['CLASS_STAR']>star_params['CLASS_STAR'])
            ss[wg_stars].write(outname, format='fits', overwrite=True)

        else:
            # Do more standard stellar locus matching
            # Would be great to have stellar_locus_params be more customizable...
            outname = sscat.replace('.ldac','stars.ldac')
            self.logprint("Selecting stars on CLASS_STAR...")
            wg_stars = (ss['CLASS_STAR']>star_params['CLASS_STAR']) & \
            (ss['SNR_WIN']>star_params['MIN_SNR'])
            ss[wg_stars].write(outname, format='fits', overwrite=True)

        return outname

    def make_image_info_struct(self, max_len_of_filepath=200, use_coadd=False):
        # max_len_of_filepath may cause issues down the line if the file path
        # is particularly long

        if self.reduced_images is not None:
            image_files = self.reduced_images
        else:
            image_files = self.image_files

        # If coadd used, will be put first
        Nim = len(image_files)
        if use_coadd is True:
            Nim += 1

        image_info = meds.util.get_image_info_struct(Nim, max_len_of_filepath)

        i=0
        for image_file in range(Nim):
            if (i == 0) and (use_coadd is True):
                image_file = self.coadd_file
            else:
                if use_coadd is True:
                    image_file = image_files[i-1]
                else:
                    image_file = image_files[i]

            #bkgsub_name = reduced_file.replace('.fits','.sub.fits')
            segmap_name = image_file.replace('.fits','.sgm.fits')

            image_info[i]['image_path']  =  image_file
            image_info[i]['image_ext']   =  0
            image_info[i]['weight_path'] =  self.combined_weight
            image_info[i]['weight_ext']  =  0
            image_info[i]['bmask_path']  =  self.mask_file
            image_info[i]['bmask_ext']   =  0
            image_info[i]['seg_path']    =  segmap_name
            image_info[i]['seg_ext']     =  0

            # The default is for 0 offset between the internal numpy arrays
            # and the images, but we use the FITS standard of a (1,1) origin.
            # In principle we could probably set this automatically by checking
            # the images
            image_info[i]['position_offset'] = 1

            i+=1

        return image_info

    def make_meds_config(self, extra_parameters=None, use_coadd=False):
        '''
        :extra_parameters: dictionary of keys to be used to update the base MEDS configuration dict

        '''
        # sensible default config.
        config = {'first_image_is_coadd': use_coadd,
                  'cutout_types':['weight','seg','bmask'],
                  'psf_type':'psfex'}

        if extra_parameters is not None:
            config.update(extra_parameters)

        return config

    def _meds_metadata(self, magzp, use_coadd):
        '''
        magzp: float
            The reference magnitude zeropoint
        use_coadd: bool
            Set to True if the first MEDS cutout is from the coadd
        '''

        meta = np.empty(1, [
            ('magzp_ref', np.float),
            ('has_coadd', np.bool)
            ])

        meta['magzp_ref'] = magzp
        meta['has_coadd'] = use_coadd

        return meta

    def _calculate_box_size(self,angular_size,size_multiplier = 2.5, min_size = 16, max_size= 64):
        '''
        Calculate the cutout size for this survey.

        :angular_size: angular size of a source, with some kind of angular units.
        :size_multiplier: Amount to multiply angular size by to choose boxsize.
        :deconvolved:
        :min_size:
        :max_size:

        '''

        pixel_scale = self.pix_scale

        box_size_float = np.ceil( angular_size/pixel_scale)

        # Available box sizes to choose from -> 16 to 256 in increments of 2
        available_sizes = min_size * 2**(np.arange(np.ceil(np.log2(max_size)-np.log2(min_size)+1)).astype(int))

        # If a single angular_size was proffered:
        if isinstance(box_size_float, np.ndarray):
            available_sizes_matrix = available_sizes.reshape(1,available_sizes.size).repeat(angular_size.size,axis=0)
            box_size_float_matrix=box_size_float.reshape(box_size_float.size,1)
            #available_sizes_matrix[box_size_float.reshape(box_size_float.size,1) > available_sizes.reshape(1,available_sizes.size)] = np.max(available_sizes)+1
            available_sizes_matrix[box_size_float.reshape(box_size_float.size,1) > available_sizes_matrix] = np.max(available_sizes)+1
            box_size = np.min(available_sizes_matrix,axis=1)
        else:
            box_size = np.min( available_sizes[ available_sizes > box_size_float ] )
        return box_size

    def make_object_info_struct(self, catalog=None):

        if catalog is None:
            catalog = self.catalog

        obj_str = meds.util.get_meds_input_struct(len(catalog),\
                                                  extra_fields = [('KRON_RADIUS',np.float),('number',np.int),\
                                                                  ('X_IMAGE',np.float),('Y_IMAGE',np.float)])
        obj_str['id'] = catalog['NUMBER']
        obj_str['number'] = np.arange(len(catalog))+1
        obj_str['box_size'] = self._calculate_box_size(catalog['KRON_RADIUS'])
        obj_str['ra'] = catalog['ALPHAWIN_J2000']
        obj_str['dec'] = catalog['DELTAWIN_J2000']
        obj_str['X_IMAGE'] = catalog['X_IMAGE']
        obj_str['Y_IMAGE'] = catalog['Y_IMAGE']
        obj_str['KRON_RADIUS'] = catalog['KRON_RADIUS']

        return obj_str

    def run(self,outfile='mock_superbit.meds', overwrite=False,
            source_selection=False, master_dark=None, select_truth_stars=False,
            psf_mode='piff', use_coadd=True):
        # Make a MEDS, overwriteing if needed

        #### ONLY FOR DEBUG
        #### Set up the paths to the science and calibration data
        #self.set_working_dir()
        #self.set_path_to_psf()
        #self.set_path_to_science_data()
        # Add a WCS to the science
        #self.add_wcs_to_science_frames()
        ####################

        # Reduce the data.
        # self.reduce(overwrite=overwrite,skip_sci_reduce=True)
        # Make a mask.
        # NB: can also read in a pre-existing mask by setting self.mask_file
        #self.make_mask(mask_name='mask.fits',overwrite=overwrite)

        # Combine images, make a catalog.
        config_path = os.path.join(self.base_dir, 'superbit/astro_config/')
        self.quick_reduce(master_dark=master_dark)
        self.make_coadd_catalog(sextractor_config_path=config_path,
                          source_selection=source_selection)
        # Make catalogs for individual exposures
        self.make_exposure_catalogs(sextractor_config_path=config_path)
        # Build a PSF model for each image.
        self.make_psf_models(select_truth_stars=select_truth_stars, im_cats=im_cats, use_coadd=False, psf_mode=psf_mode)
        # Make the image_info struct.
        image_info = self.make_image_info_struct()
        # Make the object_info struct.
        obj_info = self.make_object_info_struct()
        # Make the MEDS config file.
        meds_config = self.make_meds_config()
        # Create metadata for MEDS
        magzp = 30.
        meta = self._meds_metadata(magzp, use_coadd)
        # Finally, make and write the MEDS file.
        medsObj = meds.maker.MEDSMaker(obj_info, image_info, config=meds_config,
                                       psf_data=self.psf_models,meta_data=meta)
        medsObj.write(outfile)
