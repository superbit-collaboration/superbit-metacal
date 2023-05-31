import numpy as np
import meds
import os
import psfex
from astropy.io import fits
import string
from pathlib import Path
import pickle
from astropy import wcs
import fitsio
import esutil as eu
from astropy.table import Table
import astropy.units as u
import superbit_lensing.utils as utils
from superbit_lensing.medsmaker.superbit.psf_extender import psf_extender
import glob

import ipdb

'''
Goals:
  - Take as input calibrated images
  - Build a psf model (PSFEx)
  - run the meds maker (use meds.Maker)

'''

class BITMeasurement():
    def __init__(self, image_files, data_dir, target_name, band,
                    outdir, work_dir=None, log=None, vb=False):
        '''
        :data_path: path to the image data not including target name
        :coadd: Set to true if the first image file is a coadd image (must be first)
        '''

        self.image_files = image_files
        self.data_dir = data_dir
        self.target_name = target_name
        self.outdir = outdir
        self.vb = vb
        # Adding this for nomenclature of truth file if config is not supplied
        self.band = band

        self.image_cats = []
        self.detect_img_path = None
        self.detect_cat_path = None
        self.detection_cat = None
        self.psf_models = None

        # TODO: generalize
        self.pix_scale = 0.141 # arcsec per pixel

        # Set up logger
        if log is None:
            logfile = 'medsmaker.log'
            log = utils.setup_logger(logfile)

        self.logprint = utils.LogPrint(log, vb)

        # Set up base (code) directory
        filepath = Path(os.path.realpath(__file__))
        self.base_dir = filepath.parents[1]

        # If desired, set a tmp output directory
        self._set_work_dir(work_dir)

        # Populate list of single-epoch image catalogs
        self._get_image_cats()

        return


    def _set_work_dir(self, work_dir):
        '''
        In case one wants to have psf outputs or (when they were still being
        made) SExtractor and SWarp products saved elsewhere than outdir.
        '''
        if work_dir is None:
            self.work_dir = self.outdir
        else:
            self.work_dir = work_dir

        if not os.path.exists(self.work_dir):
            os.mkdir(self.work_dir)

        return

    def _get_image_cats(self):
        '''
        Get list of single-epoch exposure catalogs using filenames of
        single-epoch exposures. It bugs me to be defining science images in process_2023.py
        but catalogs here, but w/e.
        Note that this assumes OBA convention for data organization:
        [target_name]/[band]/[cal, cat, coadd, etc.]
        '''
        catdir = os.path.join(self.data_dir, self.target_name, self.band, 'cat')
        #imcats = glob.glob(os.path.join(top_dir, 'cat/*cal_cat.fits'))
        ims = self.image_files
        cnames = map(lambda x:os.path.basename(x).replace('cal.fits',\
                            'cal_cat.fits'), ims)
        imcats = list(map(lambda x:os.path.join(catdir, x), cnames))

        if os.path.exists(imcats[0]) == False:
            print(f'No cat files found at location {catdir}')
        else:
            self.image_cats = imcats

    def make_exposure_catalogs(self, sextractor_config_path):

        if os.path.isdir(sextractor_config_path) is False:
            raise f'{sextractor_config_path} does not exist, exiting'

        for imagefile in self.image_files:
            sexcat = self._run_sextractor(imagefile, sextractor_config_path)
            self.image_cats.append(sexcat)

        return

    def _run_sextractor(self, image_file, sextractor_config_path):
        '''
        Utility method to invoke Source Extractor on supplied detection file
        Returns: file path of catalog
        '''
        cpath = sextractor_config_path
        cat_dir = os.path.join(self.data_dir, self.target_name, self.band, 'cat')
        cat_name = os.path.basename(image_file).replace('cal.fits','cal_cat.fits')
        cat_file = os.path.join(cat_dir, cat_name)

        image_arg = f'"{image_file}[0]"'
        weight_arg = f'-WEIGHT_IMAGE "{image_file}[1]" -WEIGHT_TYPE MAP_WEIGHT'
        name_arg='-CATALOG_NAME ' + cat_file
        config_arg = os.path.join(cpath, "sextractor.real.config")
        param_arg = f'-PARAMETERS_NAME {os.path.join(cpath, "sextractor.param")}'
        nnw_arg = f'-STARNNW_NAME {os.path.join(cpath, "default.nnw")}'
        filter_arg = f'-FILTER_NAME {os.path.join(cpath, "default.conv")}'

        bkg_name = image_file.replace('.fits','.sub.fits')
        seg_name = image_file.replace('.fits','.sgm.fits')
        checkname_arg = f'-CHECKIMAGE_NAME  {bkg_name},{seg_name}'

        cmd = ' '.join([
                    'sex', image_arg, weight_arg, name_arg,  checkname_arg,
                    param_arg, nnw_arg, filter_arg, '-c', config_arg
                    ])

        self.logprint("sex cmd is " + cmd)

        os.system(cmd)

        print(f'cat_name_is {cat_file}')
        return cat_file


    def get_detection_files(self):
        '''
        Get detection source file & catalog, assuming OBA convention for data organization:
        [target_name]/[band]/[cal, cat, coadd, etc.]
        '''

        det_dir = os.path.join(self.data_dir, self.target_name, 'det')
        coadd_name = f'coadd/{self.target_name}_coadd_det.fits'
        coadd_cat_name = f'cat/{self.target_name}_coadd_det_cat.fits'

        detection_img_path = os.path.join(det_dir, coadd_name)
        detection_cat_path = os.path.join(det_dir, coadd_cat_name)

        if os.path.exists(detection_img_path) == False:
            raise(f'No detection coadd found at {detection_img_path}')
        else:
            self.detect_img_path = detection_img_path

        if os.path.exists(detection_cat_path) == False:
            raise('No detection coadd found at {detection_img_path}; check name?')
        else:
            self.detect_cat_path = detection_cat_path
            dcat = fits.open(detection_cat_path)
            # If an LDAC
            try:
                self.detection_cat = dcat[2].data
            # If a FITS_1.0
            except:
                self.detection_cat  = dcat[1].data

        return


    def make_psf_models(self, config_path=None, select_truth_stars=False,
                        use_coadd=True, psf_mode='piff', psf_seed=None,
                        star_params=None):
        '''
        Make PSF models. If select_truth_stars is enabled, cross-references an
        externally-supplied star catalog before PSF fitting.
        '''
        self.psf_models = []
        image_files = self.image_files
        image_cats  = self.image_cats

        if star_params is None:
            star_keys = {'size_key': 'FLUX_RADIUS', 'mag_key': 'FLUX_APER'}
            star_params = {'MIN_MAG': 22.6,
                           'MAX_MAG': 16,
                           'MIN_SIZE': 1.1,
                           'MAX_SIZE': 3.0,
                           'MIN_SNR': 20,
                           'CLASS_STAR': 0.95,
                           'truthfilename': f'{self.target_name}_{self.band}_truth.fits'
                           }
            self.logprint(f"Using default star params: {star_params}")


        if config_path is None:
            config_path = os.path.join(self.base_dir, 'superbit/astro_config/')
            self.logprint(f'Using PSF config path {config_path}')

        if psf_seed is None:
            psf_seed = utils.generate_seeds(1)

        Nim = len(image_files)
        self.logprint(f'Nim = {Nim}')
        self.logprint(f'len(image_cats)={len(image_cats)}')
        self.logprint(f'image_cats = {image_cats}')
        assert(len(image_cats)==Nim)

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
                    image_file = image_files[i-1]
                    image_cat = image_cats[i-1]
                else:
                    image_file = image_files[i]
                    image_cat = image_cats[i]

            if psf_mode == 'piff':
                piff_model = self._make_piff_model(
                    image_file, image_cat, config_path=config_path,
                    select_truth_stars=select_truth_stars,
                    star_params=star_params,
                    psf_seed=psf_seed
                    )
                self.psf_models.append(piff_model)

            elif psf_mode == 'psfex':
                psfex_model = self._make_psfex_model(
                    image_cat, config_path=config_path,
                    select_truth_stars=select_truth_stars,
                    star_params=star_params,
                    psf_seed=psf_seed
                    )
                self.psf_models.append(psfex_model)

                # create & move checkimages to psfex_output
                psfex_plotdir = os.path.join(self.data_dir, 'psfex-output')

                if not os.path.exists(psfex_plotdir):
                    os.mkdir(psfex_plotdir)
                cleanup_cmd = ' '.join(
                    ['mv chi* resi* samp* snap* proto* *.xml', psfex_plotdir]
                    )
                cleanup_cmd2 = ' '.join(
                    ['mv count*pdf ellipticity*pdf fwhm*pdf', psfex_plotdir]
                    )
                os.system(cleanup_cmd); os.system(cleanup_cmd2)

            elif psf_mode == 'true':
                true_model = self._make_true_psf_model()
                self.psf_models.append(true_model)

        # TODO: temporary coadd PSF solution!
        # see issue #83
        self.psf_models[0] = self.psf_models[1]

        return


    def _make_psfex_model(self, im_cat, config_path=None,
                          psfex_out_dir='./tmp/', psf_seed=None,
                          select_truth_stars=False, star_params=None):
        '''
        Gets called by make_psf_models for every image in self.image_files
        Wrapper for PSFEx. Requires a FITS format catalog with vignettes

        TODO: Implement psf_seed for PSFEx!
        '''

        # If flagged, get a "clean" star catalog x-refed against another one
        if select_truth_stars==True:
            # This will break for any truth file nomenclature that
            # isn't pipeline default
            truthdir = self.work_dir
            truthcat = glob.glob(''.join([truthdir,'*truth*.fits']))[0]
            truthfilen = os.path.join(truthdir, truthcat)
            if os.path.exists(truthfilen) == False:
                raise(f'Star truth file {truthfilen} not found')
            self.logprint("using truth catalog %s" % truthfilen)
            psfcat_name = self._select_stars_for_psf(
                sscat=im_cat,truthfile=truthfilen
                )
        else:
            psfcat_name = im_cat

        # Now run PSFEx on that image and accompanying catalog
        psfex_config_arg = '-c '+config_path+'psfex.config'
        outcat_name = im_cat.replace('.ldac','.psfex.star')
        cmd = ' '.join(
            ['psfex', psfcat_name,psfex_config_arg,'-OUTCAT_NAME', outcat_name]
            )
        self.logprint("psfex cmd is " + cmd)
        os.system(cmd)

        psfex_model_file = im_cat.replace('.ldac','.psf')

        return psfex.PSFEx(psfex_model_file)


    def _make_piff_model(self, im_file, im_cat, config_path=None, psf_seed=None,
                         select_truth_stars=False, star_params=None):
        '''
        Method to invoke PIFF for PSF modeling
        Returns a "PiffExtender" object with the get_rec() and get_cen()
        functions expected by meds.maker

        First, let's get it to run on one, then we can focus on running list
        '''

        output_dir = os.path.join(self.outdir, 'piff-output',
                        os.path.basename(im_file).split('.fits')[0])
        utils.make_dir(output_dir)

        output_name = os.path.basename(im_file).replace('.fits', '.piff')
        output_path = os.path.join(output_dir, output_name)

        # update piff config w/ psf_seed
        base_piff_config = os.path.join(config_path, 'piff.config')
        run_piff_config = os.path.join(output_dir, 'piff.config')

        config = utils.read_yaml(base_piff_config)
        config['select']['seed'] = psf_seed
        utils.write_yaml(config, run_piff_config)

        # PIFF wants RA in hours, not degrees
        ra  = fits.getval(im_file, 'CRVAL1') / 15.0
        dec = fits.getval(im_file, 'CRVAL2')

        if select_truth_stars is True:
            # This will break for any truth file nomenclature that
            # isn't pipeline default
            truthdir = self.work_dir

            if 'truthfilepath' in star_params:
                truthfile = star_params['truthfilepath']
            else:
                truthfilename = star_params.get('truthfilename', None)
                if truthfilename:
                    truthfile = os.path.join(truthdir, truthfilename)
                else:
                    print('No truth star catalog file specified')
                    truthfile = None

            if truthfile is not None:
                self.logprint('using truth catalog %s' % truthfile)

        else:
            truthfile = None

        psfcat_name = self._select_stars_for_psf(
            sscat=im_cat,
            star_params=star_params,
            truthfile=truthfile
            )

        # Now run PIFF on that image and accompanying catalog
        image_arg  = f'input.image_file_name={im_file}'
        psfcat_arg = f'input.cat_file_name={psfcat_name}'
        coord_arg  = f'input.ra={ra} input.dec={dec}'
        output_arg = f'output.file_name={output_name} output.dir={output_dir}'

        cmd = f'piffify {run_piff_config} {image_arg} {psfcat_arg} ' +\
              f'{output_arg} {coord_arg}'

        self.logprint('piff cmd is ' + cmd)
        os.system(cmd)

        # use stamp size defined in config
        psf_stamp_size = config['psf']['model']['size']

        # Extend PIFF PSF to have needed PSFEx methods for MEDS
        kwargs = {
            'piff_file': output_path
        }
        piff_extended = psf_extender('piff', psf_stamp_size, **kwargs)

        return piff_extended


    def _make_true_psf_model(self, stamp_size=25, psf_pix_scale=None):
        '''
        Construct a PSF image to populate a MEDS file using the actual
        PSF used in the creation of single-epoch images

        NOTE: For now, this function assumes a constant PSF for all images
        NOTE: Should only be used for validation simulations!
        '''

        if psf_pix_scale is None:
            # make it higher res
            # psf_pix_scale = self.pix_scale / 4
            psf_pix_scale = self.pix_scale

        # there should only be one of these
        true_psf_file = glob.glob(
            os.path.join(self.data_dir, '*true_psf.pkl')
            )[0]

        with open(true_psf_file, 'rb') as fname:
            true_psf = pickle.load(fname)

        # Extend True GalSim PSF to have needed PSFEx methods for MEDS
        kwargs = {
            'psf': true_psf,
            'psf_pix_scale': psf_pix_scale
        }
        true_extended = psf_extender('true', stamp_size, **kwargs)

        return true_extended


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
            stars = truthcat
            outname = sscat.replace('.ldac','truthstars.ldac')

            # match sscat against truth star catalog; 0.72" = 5 SuperBIT pixels
            self.logprint(f"selecting on truth catalog, {len(stars)} stars found")
            star_matcher = eu.htm.Matcher(16, ra=stars['ALPHAWIN_J2000'],
                                            dec=stars['DELTAWIN_J2000']
                                            )
            matches, starmatches, dist =\
                    star_matcher.match(ra=ss['ALPHAWIN_J2000'],
                        dec=ss['DELTAWIN_J2000'],
                        radius=1/3600., maxmatch=1
                        )

            # Save result to file, return filename
            self.logprint(f'{len(dist)}/{len(ss)} objects matched to truth star catalog')
            ss=ss[matches]
            wg_stars = (ss['SNR_WIN']>star_params['MIN_SNR'])
            ss[wg_stars].write(outname,format='fits',overwrite=True)

        else:
            # Do more standard stellar locus matching
            # Would be great to have stellar_locus_params be more customizable...
            # NOTE: this fails if all kw are not included. Should be a loop through
            # keywords that are supplied, or there should be checking for mandatory
            # keywords.
            outname = sscat.replace('.ldac','stars.ldac')
            self.logprint("Selecting stars on CLASS_STAR, SIZE and MAG...")
            wg_stars = (ss['CLASS_STAR']>star_params['CLASS_STAR']) & \
            (ss['FLUX_RADIUS']>star_params['MIN_SIZE']) & \
            (ss['MAG_AUTO']<star_params['MIN_MAG']) & \
            (ss['MAG_AUTO']>star_params['MAX_MAG'])
            ss[wg_stars].write(outname, format='fits', overwrite=True)

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
                image_file = self.detect_img_path
            else:
                if use_coadd is True:
                    image_file = self.image_files[i-1]
                else:
                    image_file = self.image_files[i]

            image_info[i]['image_path']  =  image_file
            image_info[i]['image_ext']   =  0
            image_info[i]['weight_path'] =  image_file
            image_info[i]['weight_ext']  =  1
            image_info[i]['bmask_path']  =  image_file
            image_info[i]['bmask_ext']   =  2
            image_info[i]['seg_path']    =  self.detect_img_path
            image_info[i]['seg_ext']     =  2

            # The default is for 0 offset between the internal numpy arrays
            # and the images, but we use the FITS standard of a (1,1) origin.
            # In principle we could probably set this automatically by checking
            # the images
            image_info[i]['position_offset'] = 1

            i+=1

        return image_info


    def make_meds_config(self, use_coadd, psf_mode, extra_parameters=None,
                         use_joblib=False):
        '''
        :extra_parameters: dictionary of keys to be used to update the base MEDS configuration dict

        '''
        # sensible default config.
        config = {
            'first_image_is_coadd': use_coadd,
            'cutout_types':['weight','seg','bmask'],
            'psf_type': psf_mode,
            'use_joblib': use_joblib
            }

        if extra_parameters is not None:
            config.update(extra_parameters)

        return config

    def meds_metadata(self, magzp, use_coadd):
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
        box_size_float = np.ceil(angular_size/pixel_scale)

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
            catalog = self.detection_cat

        ###
        ### Need to figure out a way to add redshifts here...
        ###

        obj_str = meds.util.get_meds_input_struct(catalog.size,\
                                                  extra_fields = [('KRON_RADIUS',np.float),('number',np.int),\
                                                                  ('XWIN_IMAGE',np.float),('YWIN_IMAGE',np.float)])
        obj_str['id'] = catalog['NUMBER']
        obj_str['number'] = np.arange(catalog.size)+1
        obj_str['box_size'] = self._calculate_box_size(catalog['KRON_RADIUS'])
        obj_str['ra'] = catalog['ALPHAWIN_J2000']
        obj_str['dec'] = catalog['DELTAWIN_J2000']
        obj_str['XWIN_IMAGE'] = catalog['XWIN_IMAGE']
        obj_str['YWIN_IMAGE'] = catalog['YWIN_IMAGE']
        obj_str['KRON_RADIUS'] = catalog['KRON_RADIUS']

        return obj_str

    def run(self,outfile='superbit_ims.meds', overwrite=False,
            source_selection=False, select_truth_stars=False, psf_mode='piff',
            use_coadd=True):
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
        self.make_coadd_catalog(sextractor_config_path=config_path,
                          source_selection=source_selection)
        # Make catalogs for individual exposures
        self.make_exposure_catalogs(sextractor_config_path=config_path)
        # Build a PSF model for each image.
        self.make_psf_models(select_truth_stars=select_truth_stars, use_coadd=False, psf_mode=psf_mode)
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
        medsObj = meds.maker.MEDSMaker(
            obj_info, image_info, config=meds_config, psf_data=self.psf_models,
            meta_data=meta
            )
        medsObj.write(outfile)
