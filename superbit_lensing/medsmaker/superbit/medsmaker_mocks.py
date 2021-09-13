import ngmix
import numpy as np
import meds
import os
import psfex
from astropy.io import fits
import string
from pathlib import Path
import pdb, pudb
from astropy import wcs
import fitsio
import esutil as eu
from astropy.table import Table
import astropy.units as u
import astropy.coordinates
from astroquery.gaia import Gaia
import superbit_lensing.utils as utils

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


class BITMeasurement():
    def __init__(self, image_files=None, flat_files=None, dark_files=None,
                 bias_files=None, data_dir=None, log=None, vb=False):
        '''
        :image_files: Python List of image filenames; must be complete relative or absolute path.
        :flat_files: Python List of image filenames; must be complete relative or absolute path.
        :dark_files: Python List of image filenames; must be complete relative or absolute path.
        :catalog: Object that stores FITS array of catalog
        :coadd: Set to true if the first image file is a coadd image (must be first)
        '''

        self.image_files = image_files
        self.flat_files = flat_files
        self.dark_files = dark_files
        self.bias_files = bias_files
        self.vb = vb
        self.coadd_file = None
        self.catalog = None
        self.psf_path = None
        self.work_path = None

        if data_dir is None:
            self.data_dir = os.getcwd()
            self.mask_path = './mask_files'
        else:
            self.mask_path = os.path.join(data_dir, 'mask_files')

        if log is None:
            logfile = 'medsmaker.log'
            log = utils.setup_logger(logfile)

        self.logprint = utils.LogPrint(log, vb)

        filepath = Path(os.path.realpath(__file__))
        self.base_dir = filepath.parents[1]

        return

    def set_working_dir(self,path=None):
        if path is None:
            self.work_path = './tmp'
            if not os.path.exists(self.work_path):
                os.mkdir(self.work_path)
        else:
            self.work_path = path
            if not os.path.exists(self.work_path):
                os.mkdir(self.work_path)

    def set_path_to_psf(self,path=None):
        if path is not None:
            self.psf_path = path
            if not os.path.exists(self.psf_path):
                os.mkdir(self.psf_path)
        else:
            self.psf_path = './tmp/psfex_output'
            if not os.path.exists(self.psf_path):
                os.mkdir(self.psf_path)

    def set_path_to_calib_data(self,path=None):
        if path is None:
            self.calib_path = '../Data/calib'
        else:
            self.calib_path = path

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
                fitsio.write(os.path.join(self.work_path,'master_bias_median.fits'),master_bias,clobber=True)
        else:
        """
        master_bias = fitsio.read(bname)

        dname = os.path.join(self.mask_path,'master_dark_median.fits')
        if (not os.path.exists(dname) or (overwrite==True)):
            dark_array=[]
            for idark_file in self.dark_files:
                hdr = fitsio.read_header(idark_file)
                time = hdr['EXPTIME'] / 1000. # exopsure time, seconds
                dark_frame = ((fitsio.read(idark_file)) - master_bias) * 1./time
                dark_array.append(dark_frame)
                master_dark = np.median(dark_array,axis=0)
                fitsio.write(os.path.join(self.mask_path,'master_dark_median.fits'),master_dark,clobber=True)
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
                fitsio.write(os.path.join(self.mask_path,'master_flat_median.fits'),master_flat,clobber=True)
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
                this_image_outname=(os.path.basename(this_image_file)).replace(".fits","_reduced.fits")
                this_image_outname = os.path.join(self.work_path,this_image_outname)
                reduced_image_files.append(this_image_outname)
                this_outfits=fits.PrimaryHDU(this_reduced_image,header=updated_header)
                this_outfits.writeto(this_image_outname,overwrite=True)

            self.image_files = reduced_image_files
        else:
            pass


    def make_mask(self, mask_name='mask.fits', global_dark_thresh=10,
                  global_flat_thresh=0.85, overwrite=False):
        '''
        Use master flat and dark to generate a bad pixel mask.
        Default values for thresholds may be superseded in function call
        '''
        self.mask_file = os.path.join(self.mask_path, mask_name)
        self.logprint("\nUsing mask %s\n" % str(self.mask_file))

        if (not os.path.exists(self.mask_file)) or (overwrite==True):
            # It's bad practice to hard-code filenames in
            mdark_fname = os.path.join(self.mask_path,'master_dark_median.fits')
            mflat_fname = os.path.join(self.mask_path,'master_flat_median.fits')
            mdark = fits.getdata(mdark_fname)
            mflat = fits.getdata(mflat_fname)

            # Start with dark
            med_dark_array=[]
            flattened=np.ravel(mdark)
            outrav=np.zeros(mflat.size)
            outrav[flattened>=global_dark_thresh]=1
            med_dark_array.append(outrav)
            sum_dark = np.sum(med_dark_array,axis=0)
            # This transforms our bpm=1 array to a bpm=0 array
            darkmask=np.ones(sum_dark.size)
            #darkmask[sum_dark==(len(dark_files))]=0
            darkmask[sum_dark==1]=0
            outfile = fits.PrimaryHDU(darkmask.reshape(np.shape(mdark)))
            outfile.writeto(os.path.join(self.mask_path,'darkmask.fits'),overwrite=True)

            # repeat for flat
            med_flat_array=[]
            flattened=np.ravel(mflat)
            outrav=np.zeros(mflat.size)
            outrav[flattened<=global_flat_thresh]=1
            med_flat_array.append(outrav)
            sum_flat = np.sum(med_flat_array,axis=0)
            # This transforms our bpm=1 array to a bpm=0 array
            flatmask=np.ones(sum_flat.size)
            #darkmask[sum_dark==(len(dark_files))]=0
            flatmask[sum_flat==1]=0
            outfile = fits.PrimaryHDU(flatmask.reshape(np.shape(mflat)))
            outfile.writeto(os.path.join(self.mask_path,'flatmask.fits'),overwrite=True)

            # Now generate actual mask
            supermask = (darkmask + flatmask)/2.
            outfile = fits.PrimaryHDU(flatmask.reshape(np.shape(mflat)))
            outfile.writeto(os.path.join(self.mask_path,'supermask.fits'),overwrite=True)

        return

    def _make_detection_image(self, outfile_name='detection.fits', weightout_name='weight.fits'):
        '''
        :output: output file where detection image is written.

        Runs SWarp on provided (reduced!) image files to make a coadd image
        for SEX and PSFEx detection.
        '''
        ### Code to run SWARP

        image_args = ' '.join(self.image_files)
        detection_file = os.path.join(self.work_path, outfile_name) # This is coadd
        weight_file = os.path.join(self.work_path, weightout_name) # This is coadd weight
        config_arg = '-c ' + os.path.join(self.base_dir, 'superbit/astro_config/swarp.config')
        #weight_arg = '-WEIGHT_IMAGE '+self.mask_file
        outfile_arg = '-IMAGEOUT_NAME '+ detection_file + ' -WEIGHTOUT_NAME ' + weight_file
        #cmd = ' '.join(['swarp ',image_args,weight_arg,outfile_arg,config_arg])
        cmd = ' '.join(['swarp ',image_args,outfile_arg,config_arg])
        self.logprint('swarp cmd is ' + cmd)
        # rc = utils.run_command(cmd, logprint=self.logprint)
        os.system(cmd)
        self.logprint('\n')

        cmd = f'mv *.xml *.fits {self.work_path}'
        self.logprint(cmd)
        # rc = utils.run_command(cmd, logprint=self.logprint)
        os.system(cmd)
        self.logprint('\n')

        # pudb.set_trace()

        return detection_file, weight_file

    def _select_sources_from_catalog(self, fullcat, catname='catalog.ldac', min_size =2, max_size=24.0, size_key='KRON_RADIUS'):
        # Choose sources based on quality cuts on this catalog.
        keep = (self.catalog[size_key] > min_size) & (self.catalog[size_key] < max_size)
        self.catalog = self.catalog[keep.nonzero()[0]]


        self.logprint("Selecting analysis objects on CLASS_STAR...") # Adapt based on needs of data; FWHM~8 for empirical!
        keep2 = self.catalog['CLASS_STAR']<=0.9
        self.catalog = self.catalog[keep2.nonzero()[0]]

        # Write trimmed catalog to file
        fullcat_name=catname.replace('.ldac','_full.ldac')
        cmd =  ' '.join(['mv',catname,fullcat_name])
        os.system(cmd)
        # utils.run_command(cmd, logprint=self.logprint)

        # "fullcat" is now actually the filtered-out analysis catalog
        fullcat[2].data = self.catalog
        #fullcat[2].data = gals

        fullcat.writeto(catname,overwrite=True)

    def select_sources_from_gaia():
        # Use some set of criteria to choose sources for measurement.

        coord = astropy.coordinates.SkyCoord(hdr['CRVAL1'],hdr['CRVAL2'],unit='deg')
        result = Gaia.cone_search_async(coord,radius=10*u.arcminute)
        catalog = result.get_data()
        pass

    def make_catalog(self, sextractor_config_path=None, source_selection=False):
        '''
        Wrapper for astromatic tools to make catalog from provided images.
        This returns catalog for (stacked) detection image
        '''

        if sextractor_config_path is None:
            sextractor_config_path = os.path.join(self.base_dir, 'superbit/astro_config/')

        #outfile_name='mock_empirical_psf_coadd.fits'; weightout_name='mock_empirical_psf_coadd.weight.fits'
        outfile_name='mock_coadd.fits'; weightout_name='mock_coadd.weight.fits'
        detection_file, weight_file= self._make_detection_image(outfile_name=outfile_name,weightout_name=weightout_name)
        self.coadd_file = detection_file

        cat_name=detection_file.replace('.fits','_cat.ldac')
        name_arg='-CATALOG_NAME ' + cat_name
        #weight_arg = '-WEIGHT_IMAGE '+weight_file
        config_arg = sextractor_config_path+'sextractor.mock.config'
        param_arg = '-PARAMETERS_NAME '+sextractor_config_path+'sextractor.param'
        nnw_arg = '-STARNNW_NAME '+sextractor_config_path+'default.nnw'
        filter_arg = '-FILTER_NAME '+sextractor_config_path+'default.conv'
        bkgname=os.path.join(self.work_path,outfile_name.replace('.fits','.sub.fits'))
        bkg_arg = '-CHECKIMAGE_NAME ' + bkgname
        #cmd = ' '.join(['sex',detection_file,weight_arg,name_arg, bkg_arg, param_arg,nnw_arg,filter_arg,'-c',config_arg])
        cmd = ' '.join(['sex',detection_file,name_arg, bkg_arg, param_arg,nnw_arg,filter_arg,'-c',config_arg]) 
        self.logprint("sex cmd is " + cmd)

        # pudb.set_trace()
        # utils.run_command(cmd, logprint=self.logprint)
        os.system(cmd)

        try:
            le_cat = fits.open(cat_name)
            self.catalog = le_cat[2].data
            if source_selection is True:
                self.logprint("selecting sources")
                self._select_sources_from_catalog(fullcat=le_cat,catname=cat_name)
        except Exception as e:
            self.logprint("coadd catalog could not be loaded; check name?")
            raise(e)

    def make_psf_models(self, select_stars=False, use_coadd=False):

        self.psfEx_models = []

        Nim = len(self.image_files)

        # Will be placed first
        if use_coadd is True:
            Nim += 1

        k = 0
        for i in range(Nim):
            if (i == 0) and (use_coadd is True):
                imagefile = self.coadd_file
            else:
                if use_coadd is True:
                    imagefile = self.image_files[i-1]
                else:
                    imagefile = self.image_files[i]

            # TODO: update as necessary
            weightfile = self.mask_file.replace('mask', 'weight')
            psfex_model_file = self._make_psf_model(imagefile, weightfile=weightfile, select_stars=select_stars)
            # move checkimages to psfex_output
            cleanup_cmd = ' '.join(['mv chi* resi* samp* snap* proto* *.xml', self.psf_path])
	    cleanup_cmd2 = ' '.join(['mv count*pdf ellipticity*pdf fwhm*pdf', self.psf_path])
            os.system(cleanup_cmd)
	    os.system(cleanup_cmd2)
            # utils.run_command(cmd, logprint=self.logprint)

            self.psfEx_models.append(psfex.PSFEx(psfex_model_file))

    def _make_psf_model(self, imagefile, weightfile='weight.fits', sextractor_config_path=None,
                        psfex_out_dir='./tmp/', select_stars=False):
        '''
        Gets called by make_psf_models for every image in self.image_files
        Wrapper for PSFEx. Requires a FITS-LDAC format catalog with vignettes
        '''
        # First, run SExtractor.
        # Hopefully imagefile is an absolute path!

        if sextractor_config_path is None:
            sextractor_config_path = os.path.join(self.base_dir, 'superbit/astro_config/')

        sextractor_config_file = sextractor_config_path+'sextractor.mock.config'
        sextractor_param_arg = '-PARAMETERS_NAME '+sextractor_config_path+'sextractor.param'
        sextractor_nnw_arg = '-STARNNW_NAME '+sextractor_config_path+'default.nnw'
        sextractor_filter_arg = '-FILTER_NAME '+sextractor_config_path+'default.conv'
        imcat_ldac_name=imagefile.replace('.fits','_cat.ldac')

        bkgname=imagefile.replace('.fits','.sub.fits')
        bkg_arg = '-CHECKIMAGE_NAME ' + bkgname

        cmd = ' '.join(['sex',imagefile,'-WEIGHT_IMAGE',weightfile,'-c',sextractor_config_file,'-CATALOG_NAME ',
                            imcat_ldac_name, bkg_arg, sextractor_param_arg,sextractor_nnw_arg,
                            sextractor_filter_arg])
        self.logprint("sex4psf cmd is " + cmd)
        os.system(cmd)
        # utils.run_command(cmd, logprint=self.logprint)

        # Get a "clean" star catalog for PSFEx input
        # At some point, make truthfilen a command line argument
        if select_stars==True:
            # TODO: Needs to be generalized to new pipeline!
            truthdir='/users/jmcclear/data/superbit/superbit-metacal/GalSim/forecasting/b/round1'
            truthcat = 'truth_gaussJitter_001.dat'
            truthfilen=os.path.join(truthdir,truthcat)
            self.logprint("using truth catalog %s" % truthfilen)
            psfcat_name = self._select_stars_for_psf(sscat=imcat_ldac_name,truthfile=truthfilen)

        else:
            psfcat_name=imcat_ldac_name

        # Now run PSFEx on that image and accompanying catalog
        psfex_config_arg = '-c '+sextractor_config_path+'psfex.mock.config'
        # Will need to make that tmp/psfex_output generalizable
        outcat_name = imagefile.replace('.fits','.psfex.star')
        cmd = ' '.join(['psfex', psfcat_name,psfex_config_arg,'-OUTCAT_NAME',
                            outcat_name, '-PSFVAR_DEGREES','2','-PSF_DIR', self.psf_path])
        self.logprint("psfex cmd is " + cmd)
        os.system(cmd)
        # utils.run_command(cmd, logprint=self.logprint)

        psfex_name_tmp1=(imcat_ldac_name.replace('.ldac','.psf'))
        psfex_name_tmp2= psfex_name_tmp1.split('/')[-1]
        psfex_model_file='/'.join([self.psf_path,psfex_name_tmp2])

        # Just return name, the make_psf_models method reads it in as a PSFEx object
        return psfex_model_file

    def _select_stars_for_psf(self,sscat,truthfile):
        '''
        Method to obtain stars from SExtractor catalog using the truth catalog from GalSim 
            sscat : input ldac-format catalog from which to select stars
            truthcat : the simulation truth catalog written out by GalSim
        '''

        # Read in truthfile, obtain stars with redshift cut
        truthcat = Table.read(truthfile,format='ascii')
        stars=truthcat[truthcat['redshift']==0] 

        self.logprint("selecting on truth catalog, %d stars found"%len(stars))
        # match sscat against truth star catalog
        ss = fits.open(sscat)
        star_matcher = eu.htm.Matcher(16,ra=stars['ra'],dec=stars['dec'])
        ssmatches,starmatches,dist = star_matcher.match(ra=ss[2].data['ALPHAWIN_J2000'],
                                                    dec=ss[2].data['DELTAWIN_J2000'],radius=5E-4,maxmatch=1)

        # Save result to file, return filename
        outname = sscat.replace('.ldac','.star')
        ss[2].data=ss[2].data[ssmatches]
        ss.writeto(outname,overwrite=True)

        return outname

    def make_image_info_struct(self, max_len_of_filepath=200, use_coadd=False):
        # max_len_of_filepath may cause issues down the line if the file path
        # is particularly long

        Nim = len(self.image_files)

        # If used, will be put first
        if use_coadd is True:
            Nim += 1

        image_info = meds.util.get_image_info_struct(Nim, max_len_of_filepath)

        i=0
        for image_file in range(Nim):
            if (i == 0) and (use_coadd is True):
                image_file = self.coadd_file
            else:
                if use_coadd is True:
                    image_file = self.image_files[i-1]
                else:
                    image_file = self.image_files[i]

            bkgsub_name = image_file.replace('.fits','.sub.fits')

            image_info[i]['image_path'] = bkgsub_name
            image_info[i]['image_ext'] = 0
            #image_info[i]['weight_path'] = self.weight_file
            # FOR NOW:
            image_info[i]['weight_path'] = self.mask_file.replace('mask', 'weight')
            image_info[i]['weight_ext'] = 0
            image_info[i]['bmask_path'] = self.mask_file
            image_info[i]['bmask_ext'] = 0

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

    def _meds_metadata(self,magzp=0.0):
        meta = np.empty(1,[('magzp_ref',np.float)])
        meta['magzp_ref'] = magzp
        return meta

    def _calculate_box_size(self,angular_size,size_multiplier = 2.5, min_size = 16, max_size= 64, pixel_scale = 0.206):
        '''
        Calculate the cutout size for this survey.

        :angular_size: angular size of a source, with some kind of angular units.
        :size_multiplier: Amount to multiply angular size by to choose boxsize.
        :deconvolved:
        :min_size:
        :max_size:

        '''
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

        ###
        ### Need to figure out a way to add redshifts here...
        ###

        obj_str = meds.util.get_meds_input_struct(catalog.size,extra_fields = [('KRON_RADIUS',np.float),('number',np.int)])
        obj_str['id'] = catalog['NUMBER']
        obj_str['number'] = np.arange(catalog.size)+1
        obj_str['box_size'] = self._calculate_box_size(catalog['KRON_RADIUS'])
        obj_str['ra'] = catalog['ALPHAWIN_J2000']
        obj_str['dec'] = catalog['DELTAWIN_J2000']
        obj_str['KRON_RADIUS'] = catalog['KRON_RADIUS']

        return obj_str


    def run(self,outfile='mock_superbit.meds', clobber=True, source_selection=False,
            select_stars=False):
        # Make a MEDS, clobbering if needed

        #### ONLY FOR DEBUG
        #### Set up the paths to the science and calibration data
        #self.set_working_dir()
        #self.set_path_to_psf()
        #self.set_path_to_science_data()
        # Add a WCS to the science
        #self.add_wcs_to_science_frames()
        ####################

        # Reduce the data.
        # self.reduce(overwrite=clobber,skip_sci_reduce=True)
        # Make a mask.
        # NB: can also read in a pre-existing mask by setting self.mask_file
        #self.make_mask(mask_name='mask.fits',overwrite=clobber)
        # Combine images, make a catalog.
        config_path = os.path.join(self.base_dir, 'superbit/astro_config/')
        self.make_catalog(sextractor_config_path=config_path,
                          source_selection=source_selection)
        # Build a PSF model for each image.
        self.make_psf_models(select_stars=select_stars)
        # Make the image_info struct.
        image_info = self.make_image_info_struct()
        # Make the object_info struct.
        obj_info = self.make_object_info_struct()
        # Make the MEDS config file.
        meds_config = self.make_meds_config()
        # Create metadata for MEDS
        meta = self._meds_metadata(magzp=30.0)
        # Finally, make and write the MEDS file.
        medsObj = meds.maker.MEDSMaker(obj_info, image_info, config=meds_config,
                                       psf_data=self.psfEx_models,meta_data=meta)
        medsObj.write(outfile)

