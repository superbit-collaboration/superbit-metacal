import ngmix
import numpy as np
import meds
import sys
import os
import psfex
import logging
from astropy.io import fits
import string
import pdb
from astropy import wcs
import fitsio
import glob
import yaml
import esutil as eu
from astropy.table import Table
import astropy.units as u
import astropy.coordinates
from astroquery.gaia import Gaia


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
    def __init__(self, argv=None, config_file=None):
        '''
        '''
        self.catalog = None

        # Define some default default parameters below.
        # These are used in the absence of a .yaml config_file or command line args.
        self._load_config_file("medsconfig_mock.yaml")

        # Check for config_file params to overwrite defaults
        if config_file is not None:
            logger.info('Loading parameters from %s' % (config_file))
            self._load_config_file(config_file)

        # Check for command line args to overwrite config_file and / or defaults
        if argv is not None:
            self._load_command_line(argv)


    def _load_config_file(self, config_file):
        """
        Load parameters from configuration file. Only parameters that exist in the config_file
        will be overwritten.
        """
        with open(config_file) as fsettings:
            config = yaml.load(fsettings, Loader=yaml.FullLoader)
        self._load_dict(config)

    def _args_to_dict(self, argv):
        """
        Converts a command line argument array to a dictionary.
        """
        d = {}
        for arg in argv[1:]:
            optval = arg.split("=", 1)
            option = optval[0]
            value = optval[1] if len(optval) > 1 else None
            d[option] = value
        return d

    def _load_command_line(self, argv):
        """
        Load parameters from the command line argumentts. Only parameters that are provided in
        the command line will be overwritten.
        """
        logger.info('Processing command line args')
        # Parse arguments here
        self._load_dict(self._args_to_dict(argv))

    def _load_dict(self, d):
        """
        Load parameters from a dictionary.
        """
        for (option, value) in d.items():
            if option == "science_files":
                self.science_files = glob.glob(str(value))
            elif option == "bias_files":
                self.bias_files = glob.glob(str(value))
            elif option == "dark_files":
                self.dark_files = glob.glob(str(value))
            elif option == "flat_files":
                self.flat_files = glob.glob(str(value))
            elif option == "working_dir":
                self.working_dir = str(value)
            elif option == "psfex_dir":
                self.psfex_dir = str(value)
            elif option == "out_file":
                self.out_file = str(value)
            elif option == "mask_file":
                self.mask_file = str(value)
            elif option == "mask_dark_file":
                self.mask_dark_file = str(value)
            elif option == "mask_flat_file":
                self.mask_flat_file = str(value)
            elif option == "mask_bias_file":
                self.mask_bias_file = str(value)
            elif option == "mask_src":
                self.mask_src = str(value)
            elif option == "mask_dark_src":
                self.mask_dark_src = str(value)
            elif option == "mask_flat_src":
                self.mask_flat_src = str(value)
            elif option == "mask_bias_src":
                self.mask_bias_src = str(value)
            elif option == "truth_file":
                self.truth_file = str(value)
            elif option == "config_file":
                logger.info('Loading parameters from %s' % (value))
                self._load_config_file(str(value))
            elif option == "swarp_config":
                logger.info('Using SWARP config %s' % (value))
                self.swarp_config = str(value)
            elif option == "psfex_config":
                logger.info('Using PSFEx config %s' % (value))
                self.psfex_config = str(value)
            elif option == "sextractor_dir":
                logger.info('Using sextractor config directory %s' % (value))
                self.sextractor_dir = str(value)
            else:
                raise ValueError("Invalid parameter \"%s\" with value \"%s\"" % (option, value))

        # Download configuration data
        if self.mask_file is not None:
            self._download_missing_data(self.mask_src, self.mask_file)
        else:
            self._download_missing_data(self.mask_dark_src, self.mask_dark_file)
            self._download_missing_data(self.mask_flat_src, self.mask_flat_file)

        # Load truth table
        if self.truth_file is not None:
            self.truthcat = Table.read(self.truth_file, format='ascii')
            logger.info("Using truth catalog %s" % self.truth_file)
        else:
            self.truthcat = None

        # Setup directories
        if not os.path.exists(os.path.dirname(self.out_file)):
            os.makedirs(os.path.dirname(self.out_file))
        if not os.path.exists(self.psfex_dir):
            os.makedirs(self.psfex_dir)
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)
        
    def _download_missing_data(self, src, output):
        if not os.path.exists(output):
            if not os.path.exists(os.path.dirname(output)):
                os.makedirs(os.path.dirname(output))
            cmd = "wget %s -O %s" % (src, output)
            os.system(cmd)

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
            logger.waring('cluster %s has no WCS, skipping...' % wcsName )
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
            logger.error("Could not process %s" % image_filename)
            return None

    def add_wcs_to_science_frames(self):
        '''
        wrapper for _make_new_fits() which returns astrometry-corrected images
        '''
        fixed_science_files = []
        for image_file in self.science_files:
            fixed_image_file = self._make_new_fits(image_file)
            if fixed_image_file is not None:
                fixed_science_files.append(fixed_image_file)
        self.science_files = fixed_science_files

    def reduce(self,overwrite=False,skip_sci_reduce=False):
        # Read in and average together the bias, dark, and flat frames.

        if (not os.path.exists(self.mask_bias_file) or (overwrite==True)):
            # Taking median biases and darks instead of mean to eliminate odd noise features
            bias_array=[]
            logger.debug("I get to bias_array")
            for ibias_file in self.bias_files:
                bias_frame = fitsio.read(ibias_file)
                bias_array.append(bias_frame)
                master_bias = np.median(bias_array,axis=0)
                fitsio.write(self.mask_bias_file, master_bias, clobber=True)
        else:
            master_bias = fitsio.read(self.mask_bias_file)

        if (not os.path.exists(self.mask_dark_file) or (overwrite==True)):
            dark_array=[]
            for idark_file in self.dark_files:
                hdr = fitsio.read_header(idark_file)
                time = hdr['EXPTIME'] / 1000. # exopsure time, seconds
                dark_frame = ((fitsio.read(idark_file)) - master_bias) * 1./time
                dark_array.append(dark_frame)
                master_dark = np.median(dark_array,axis=0)
                fitsio.write(self.mask_dark_file, master_dark, clobber=True)
        else:
            master_dark=fitsio.read(self.mask_dark_file)

        if (not os.path.exists(self.mask_flat_file) or (overwrite==True)):
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
                fitsio.write(self.mask_flat_file, master_flat, clobber=True)
        else:
            master_flat = fitsio.read(self.mask_flat_file)

        if not skip_sci_reduce:
            reduced_science_files=[]
            for this_image_file in self.science_files:
                # WARNING: as written, function assumes science data is in 0th extension
                this_image_fits=fits.open(this_image_file)
                time=this_image_fits[0].header['EXPTIME']/1000.
                this_reduced_image = (this_image_fits[0].data - master_bias)-(master_dark*time)
                this_reduced_image = this_reduced_image/master_flat
                updated_header = this_image_fits[0].header
                updated_header['HISTORY']='File has been bias & dark subtracted and FF corrected'
                this_image_outname=(os.path.basename(this_image_file)).replace(".fits","_reduced.fits")
                this_image_outname = os.path.join(self.working_dir,this_image_outname)
                reduced_science_files.append(this_image_outname)
                this_outfits=fits.PrimaryHDU(this_reduced_image,header=updated_header)
                this_outfits.writeto(this_image_outname,overwrite=True)
            self.science_files=reduced_science_files
        else:
            pass


    def make_mask(self, global_dark_thresh=10, global_flat_thresh=0.85, overwrite=False):
        '''
        Use master flat and dark to generate a bad pixel mask.
        Default values for thresholds may be superseded in function call
        '''

        if (self.mask_file is not None and not os.path.exists(self.mask_file)) or (overwrite==True):
            # It's bad practice to hard-code filenames in
            mdark = fits.getdata(self.mask_dark_file)
            mflat = fits.getdata(self.mask_flat_file)

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
            #out_file = fits.PrimaryHDU(darkmask.reshape(np.shape(mdark)))
            #out_file.writeto(os.path.join(self.mask_path,'darkmask.fits'),overwrite=True)

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
            #outfile = fits.PrimaryHDU(flatmask.reshape(np.shape(mflat)))
            #outfile.writeto(os.path.join(self.mask_path,'flatmask.fits'),overwrite=True)

            # Now generate actual mask
            supermask = (darkmask + flatmask)/2.
            outfile = fits.PrimaryHDU(flatmask.reshape(np.shape(mflat)))
            outfile.writeto(os.path.join(self.mask_file),overwrite=True)

    def _make_detection_image(self, outfile_name='detection.fits', weightout_name='weight.fits'):
        '''
        :output: output file where detection image is written.

        Runs SWarp on provided (reduced!) image files to make a coadd image
        for SEX and PSFEx detection.
        '''
        ### Code to run SWARP
        image_args = ' '.join(self.science_files)
        detection_file = os.path.join(self.working_dir, outfile_name) # This is coadd
        weight_file = os.path.join(self.working_dir, weightout_name) # This is coadd weight
        config_arg = '-c ' + self.swarp_config
        weight_arg = '-WEIGHT_IMAGE '+self.mask_file
        outfile_arg = '-IMAGEOUT_NAME '+ detection_file + ' -WEIGHTOUT_NAME ' + weight_file
        cmd = ' '.join(['swarp ',image_args,weight_arg,outfile_arg,config_arg])
        logger.debug("swarp cmd is " + cmd)
        os.system(cmd)
        return detection_file,weight_file

    def _select_sources_from_catalog(self,fullcat,catname='catalog.ldac',min_size =2,max_size=16.0,size_key='KRON_RADIUS'):
        # Choose sources based on quality cuts on this catalog.
        keep = (self.catalog[size_key] > min_size) & (self.catalog[size_key] < max_size) 
        self.catalog = self.catalog[keep.nonzero()[0]]
        
        logger.info("Selecting analysis objects on FWHM and CLASS_STAR...") # Adapt based on needs of data; FWHM~8 for empirical!
        keep2 = self.catalog['CLASS_STAR']<=0.8
        self.catalog = self.catalog[keep2.nonzero()[0]]

        # Write trimmed catalog to file
        fullcat_name=catname.replace('.ldac','_full.ldac')
        cmd =  ' '.join(['mv',catname,fullcat_name])
        os.system(cmd)
       
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

    def make_catalog(self, source_selection=False):
        '''
        Wrapper for astromatic tools to make catalog from provided images.
        This returns catalog for (stacked) detection image
        '''
        outfile_name='mock_coadd.fits'; weightout_name='mock_coadd.weight.fits'
        detection_file, weight_file= self._make_detection_image(outfile_name=outfile_name, 
                                                    weightout_name=weightout_name)
        
        cat_name=detection_file.replace('.fits','_cat.ldac')
        name_arg='-CATALOG_NAME ' + cat_name
        weight_arg = '-WEIGHT_IMAGE '+weight_file
        config_arg = self.sextractor_dir + 'sextractor.config'
        param_arg = '-PARAMETERS_NAME ' + self.sextractor_dir + 'sextractor.param'
        nnw_arg = '-STARNNW_NAME ' + self.sextractor_dir + 'default.nnw'
        filter_arg = '-FILTER_NAME ' + self.sextractor_dir + 'default.conv'
        bkgname=detection_file.replace('.fits','.sub.fits')
        bkg_arg = '-CHECKIMAGE_NAME ' + bkgname
        cmd = ' '.join(['sex', detection_file, weight_arg, name_arg, bkg_arg, 
                        param_arg, nnw_arg, filter_arg, '-c', config_arg])
        logger.debug("sex cmd is " + cmd)
        os.system(cmd)
        try:
            #le_cat = fits.open('A2218_coadd_catalog.fits')
            le_cat = fits.open(cat_name)
            self.catalog = le_cat[2].data
            if source_selection is True:
                logger.debug("I get here")
                self._select_sources_from_catalog(fullcat=le_cat,catname=cat_name)
        except:
            logger.error("coadd catalog could not be loaded; check name?")
            pdb.set_trace()

    def make_psf_models(self):

        self.psfEx_models = []
        for imagefile in self.science_files:
            #update as necessary
            weightfile=self.mask_file
            psfex_model_file = self._make_psf_model(imagefile,weightfile=weightfile)
            # move checkimages to psfex_output
            cmd = ' '.join(['mv chi* resi* samp* snap* proto*',self.psfex_dir])
            os.system(cmd)

            try:
                self.psfEx_models.append(psfex.PSFEx(psfex_model_file))
            except:
                pdb.set_trace()

    def _make_psf_model(self, imagefile, weightfile='weight.fits', psfex_out_dir='./tmp/'):
        '''
        Gets called by make_psf_models for every image in self.science_files
        Wrapper for PSFEx. Requires a FITS-LDAC format catalog with vignettes
        '''
        # First, run SExtractor.
        # Hopefully imagefile is an absolute path!

        sextractor_config_file = self.sextractor_dir + 'sextractor.config'
        sextractor_param_arg = '-PARAMETERS_NAME ' + self.sextractor_dir + 'sextractor.param'
        sextractor_nnw_arg = '-STARNNW_NAME ' + self.sextractor_dir + 'default.nnw'
        sextractor_filter_arg = '-FILTER_NAME ' + self.sextractor_dir + 'default.conv'
        imcat_ldac_name=imagefile.replace('.fits','_cat.ldac')

        bkgname=imagefile.replace('.fits','.sub.fits')
        bkg_arg = '-CHECKIMAGE_NAME ' + bkgname

        cmd = ' '.join(['sex', imagefile, '-WEIGHT_IMAGE', weightfile,'-c', 
                            sextractor_config_file,'-CATALOG_NAME ', imcat_ldac_name, 
                            bkg_arg, sextractor_param_arg,sextractor_nnw_arg,
                            sextractor_filter_arg])
        logger.debug("sex4psf cmd is " + cmd)
        os.system(cmd)

        # Get a "clean" star catalog for PSFEx input
        if self.truthcat is not None:
            psfcat_name = self._select_stars_for_psf(sscat=imcat_ldac_name)
            
        else:
            psfcat_name=imcat_ldac_name
            
        # Now run PSFEx on that image and accompanying catalog
        psfex_config_arg = '-c ' + self.psfex_config
        # Will need to make that tmp/psfex_output generalizable
        outcat_name = imagefile.replace('.fits','.psfex.star')
        cmd = ' '.join(['psfex', psfcat_name,psfex_config_arg,'-OUTCAT_NAME',
                            outcat_name, '-PSFVAR_DEGREES','3','-PSF_DIR', self.psfex_dir])
        logger.debug("psfex cmd is " + cmd)
        os.system(cmd)
        psfex_name_tmp1=(imcat_ldac_name.replace('.ldac','.psf'))
        psfex_name_tmp2= psfex_name_tmp1.split('/')[-1]
        psfex_model_file='/'.join([self.psfex_dir, psfex_name_tmp2])

        # Just return name, the make_psf_models method reads it in as a PSFEx object
        return psfex_model_file
    

    def _select_stars_for_psf(self,sscat):
        '''
        Method to obtain stars from SExtractor catalog using the truth catalog from GalSim 
            sscat : input ldac-format catalog from which to select stars
                      
        '''
        
        # Read in truth_file, obtain stars with redshift cut
        stars=self.truthcat[self.truthcat['redshift']==0] 

        # match sscat against truth star catalog
        ss = fits.open(sscat)
        star_matcher = eu.htm.Matcher(16,ra=stars['ra'],dec=stars['dec'])
        ssmatches,starmatches,dist = star_matcher.match(ra=ss[2].data['ALPHAWIN_J2000'],
                                                            dec=ss[2].data['DELTAWIN_J2000'],radius=3E-4,maxmatch=1)
        
        # Save result to file, return filename
        outname = sscat.replace('.ldac','.star')
        ss[2].data=ss[2].data[ssmatches]
        ss.writeto(outname,overwrite=True)

        return outname

    def make_image_info_struct(self,max_len_of_filepath = 200):
        # max_len_of_filepath may cause issues down the line if the file path
        # is particularly long

        image_info = meds.util.get_image_info_struct(len(self.science_files),max_len_of_filepath)

        i=0
        for image_file in self.science_files:
            bkgsub_name=image_file.replace('.fits','.sub.fits')

            image_info[i]['image_path'] = bkgsub_name
            image_info[i]['image_ext'] = 0
            #image_info[i]['weight_path'] = self.weight_file
            # FOR NOW:
            image_info[i]['weight_path'] = self.mask_file
            image_info[i]['weight_ext'] = 0
            image_info[i]['bmask_path'] = self.mask_file
            image_info[i]['bmask_ext'] = 0
            i+=1
        return image_info

    def make_meds_config(self,extra_parameters = None):
        '''
        :extra_parameters: dictionary of keys to be used to update the base MEDS configuration dict

        '''
        # sensible default config.
        config = {'first_image_is_coadd': False,'cutout_types':['weight','seg','bmask'],'psf_type':'psfex'}
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

    def make_object_info_struct(self,catalog=None):
        if catalog is None:
            catalog = self.catalog

        
        obj_str = meds.util.get_meds_input_struct(catalog.size,extra_fields = [('KRON_RADIUS',np.float),('number',np.int)])
        obj_str['id'] = catalog['NUMBER']
        obj_str['number'] = np.arange(catalog.size)+1
        obj_str['box_size'] = self._calculate_box_size(catalog['KRON_RADIUS'])
        obj_str['ra'] = catalog['ALPHAWIN_J2000']
        obj_str['dec'] = catalog['DELTAWIN_J2000']
        obj_str['KRON_RADIUS'] = catalog['KRON_RADIUS']
       
        
        return obj_str


    def run(self, clobber=True, source_selection=False):
        # Make a MEDS, clobbering if needed

        # Reduce the data.
        if self.mask_file is None:
            self.reduce(overwrite=clobber,skip_sci_reduce=True)
        # Make a mask.
        self.make_mask(overwrite=clobber)
        # Combine images, make a catalog.
        self.make_catalog(source_selection=source_selection)
        # Build a PSF model for each image.
        self.make_psf_models()
        # Make the image_info struct.
        image_info = self.make_image_info_struct()
        # Make the object_info struct.
        obj_info = self.make_object_info_struct()
        # Make the MEDS config file.
        meds_config = self.make_meds_config()
        # Create metadata for MEDS
        meta = self._meds_metadata(magzp=30.0)
        # Finally, make and write the MEDS file.
        medsObj = meds.maker.MEDSMaker(obj_info,image_info,config=meds_config,psf_data = self.psfEx_models,meta_data=meta)
        medsObj.write(self.out_file)

logging.basicConfig(format="%(message)s", level=logging.DEBUG, stream=sys.stdout)
logger = logging.getLogger("mock_medsmaker")

def main(argv):
    """
    Runs the MEDSMaker from the command line.
    """

    # Do MedsMaker
    bm = BITMeasurement(argv=argv)
    bm.run(clobber=False, source_selection=True)

if __name__ == "__main__":
    main(sys.argv)


