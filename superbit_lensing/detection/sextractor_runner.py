import os
import subprocess
from glob import glob

from superbit_lensing import utils

import ipdb

class SExtractorRunner(object):
    '''
    A minimal wrapper class around SExtractor to organize i/o and some
    common value-adds needed for downstream modules.

    go() method handles all needed SExtractor runs, which come in
    three catagories:

    1) Run SE on detection image for fiducial detections & positions
    2) Run SE in dual-image mode for every single-band coadd using (1)
    3) Run SE on all single-epoch-exposures for PSF estimation
    '''

    # required fields for the master config
    _req_fields = [
        'bands',
        'parameters', # SExtractor parameters to compute
        'filter', # convolutional filter
        'nnw', # neural network for stellar classification
        ]

    # NOTE: With some additional work, we could make the pars, filter,
    # and nnw config files be optional with the correct default. However,
    # I have only figured out how to generate the parameters config...
    _opt_fields = {
        'config_dir': None, # directory of *single-band* configs for run
        'dual_mode': True, # run SE in dual-mode using det image if True
        'cat_types': ['det', 'coadd', 'exp']
        }

    def __init__(self, config_file, run_name, basedir, bands=None,
                 config_dir=None, fname_base=None, coadd_fname_base=None,
                 outdir=None, det_image=None, sci_ext=0, wgt_ext=1,
                 logprint=None):
        '''
        config_file: dict
            Filename of a yaml config containing the filenames of the base
            SExtractor config for each band, as well as a few extras
        run_name: str
            The name of the current processing run
        basedir: str
            The base directory path for the set of single-epoch & coadd images
        bands: list of str's
            A list of band names to make catalogs for. Defaults to bands set in
            config_file
        config_dir: str
            A directory where the single-band SExtractor config files are
            located. Defaults to the current working directory
        fname_base: str
            Base name of the input single-epoch exposure files
        coadd_fname_base: str
            Base name of the input coadd files
        outdir: str
            The output directory for produced catalog files. Defaults to
            basedir
        sci_ext: int
            The science frame fits extension
        wgt_ext: int
            The weight frame fits extension
        logprint: utils.LogPrint
            A LogPrint instance. Will default to standard print
        '''

        if logprint is None:
            self.logprint = print
        else:
            if not isinstance(logprint, utils.LogPrint):
                raise TypeError('logprint must be a LogPrint instance!')
            self.logprint = logprint

        if outdir is None:
            outdir = basedir

        if fname_base is None:
            fname_base = self.get_default_fname_base(run_name)
        if coadd_fname_base is None:
            coadd_fname_base = self.get_default_coadd_fname_base(run_name)

        # process str args
        arg_dict = {
            'run_name': run_name,
            'basedir': basedir,
            'fname_base': fname_base,
            'coadd_fname_base': coadd_fname_base,
            'outdir': outdir,
            }
        for name, arg in arg_dict.items():
            if not isinstance(arg, str):
                raise TypeError(f'{name} must be a str!')
            setattr(self, name, arg)

        for name, ext in {'sci_ext':sci_ext, 'wgt_ext':wgt_ext}.items():
            if not isinstance(ext, int):
                raise TypeError(f'{name} must be an int!')

        self.sci_ext = sci_ext
        self.wgt_ext = wgt_ext

        self.parse_config(config_file, config_dir, bands=bands)

        self.get_sci_images(det_image=det_image)
        self.get_wgt_images(det_image=det_image)

        # will be filled during the call to go()
        self.catalogs = {}

        return

    def parse_config(self, config_file, config_dir, bands=None):
        '''
        Parse all class inputs related to the master SExtractor config
        and single-band config files
        '''

        if not isinstance(config_file, str):
            raise TypeError('config_file must be a str!')
        if not os.path.exists(config_file):
            raise OSError(f'Passed master SExtractor config {config_file}' +
                          'does not exist!')

        if config_dir is not None:
            if not isinstance(config_dir, str):
                raise TypeError('config_dir must be a str!')
            self.config_file = os.path.join(config_dir, config_file)
        else:
            self.config_file = config_file
        self.config_dir = config_dir

        config = utils.read_yaml(config_file)

        # do the usual config parsing
        self.config = utils.parse_config(
            config,
            self._req_fields,
            self._opt_fields,
            allow_unregistered=True,
            name='master SExtractror'
            )

        if bands is not None:
            if not isinstance(bands, list):
                raise TypeError(f'{name} must be a list of strings!')
            else:
                # each band must have a passed config file
                for band in bands:
                    if not isinstance(band, str):
                        raise TypeError('Band names must be a str!')
                    if band not in self.config:
                        raise KeyError(f'band {band} not in config!')
        else:
            # must be specified either in constructor or config!
            bands = self.config['bands']

        self.bands = bands

        # NOTE: if the master config sets a config_dir for each single-band
        # config (*not* necessarily the config_dir for the master config!),
        # parse it now. Otherwise it defaults to CWD
        if 'config_dir' in self.config:
            cdir = self.config['config_dir']
            if cdir is not None:
                # update single-band configs
                bands = self.bands.copy()
                if 'det' in self.config:
                    bands.append('det')
                for band in bands:
                    self.config[band] = os.path.join(
                        cdir, self.config[band]
                        )

                # update auxillary files
                fields = ['parameters', 'filter', 'nnw']
                for field in fields:
                    if self.config[field] is not None:
                        self.config[field] = os.path.join(
                            cdir, self.config[field]
                            )

        # for convenience
        self.cat_types = self.config['cat_types']
        self.dual_mode = self.config['dual_mode']

        return

    def get_default_fname_base(self, run_name):
        '''
        If no fname_base is provided, generate a default
        '''

        return run_name

    def get_default_coadd_fname_base(self, run_name):
        '''
        If no coadd_fname_base is provided, generate a default
        '''

        return f'{run_name}_coadd'

    def get_sci_images(self, det_image=None):
        '''
        Grab all available science images for requested bands given
        the basedir

        NOTE: For now, we follow the multi-extention fits convention

        det_image: str
            Can pass a detection image filename if it does not follow the
            same filepath convention as coadd_fname_base
        '''

        ext = self.sci_ext

        # we'll save the images for each band separately
        self.sci_images = {}
        self.coadds = {}

        for band in self.bands:
            self.logprint(f'Gathering sci files for band {band}')
            self.coadds[band] = {}

            # single coadd file per band
            coadd_sci = os.path.join(
                self.basedir, self.coadd_fname_base + f'_{band}.fits'
                )

            if not os.path.exists(coadd_sci):
                raise OSError(f'Coadd file {coadd_sci} not found!')

            # now add sci_fdext
            coadd_sci += f'[{ext}]'

            # (potentially) many single-epoch exposures
            # NOTE: extra ignore fields are in case you are re-processing
            # and have extra files lying around
            sci_names = os.path.join(
                self.basedir, self.fname_base
                ) + f'*[!truth,meds,mcal,sub,wgt,mask,sgm,coadd]_{band}.fits'

            sci_list = glob(sci_names)

            sci_list = [
                # sci.replace('.fits', f'.fits[{ext}]') for sci in sci_list
                sci + f'[{ext}]' for sci in sci_list
                ]

            self.sci_images[band] = sci_list
            self.coadds[band]['sci'] = coadd_sci
            self.logprint(f'Science frames for band {band}: {sci_list}')
            self.logprint(f'Coadd frame for band {band}: {coadd_sci}')

        # if a detection coadd is present, try to grab it
        if det_image is None:
            det_coadd_fname = os.path.join(self.basedir, f'*coadd_det.fits')
            det_coadd = glob(det_coadd_fname)
            det_coadd = os.path.join(
                self.basedir, self.coadd_fname_base + f'_det.fits'
                )
        else:
            det_coadd = det_image

        if os.path.exists(det_coadd):
            self.coadds['det'] = {}
            det_coadd += f'[{ext}]'
            self.coadds['det']['sci'] = det_coadd
            self.logprint(f'Coadd frame for band det: {det_coadd}')
        else:
            self.logprint('No detection coadd found using ' +
                          f'{det_coadd_fname}; skipping')

        return

    def get_wgt_images(self, det_image=None):
        '''
        Can implement more options in the future. For now, we're using
        the provided ext of the sci images

        det_image: str
            Can pass a detection image filename if it does not follow the
            same filepath convention as coadd_fname_base
        '''

        if len(self.sci_images) == 0:
            raise ValueError('There are no science images registered yet!')

        sci_ext = self.sci_ext
        wgt_ext = self.wgt_ext

        # we'll save the images for each band separately
        self.wgt_images = {}

        for band in self.bands:
            sci_list = self.sci_images[band]
            wgt_list = [
                sci.replace(f'[{sci_ext}]', f'[{wgt_ext}]') for sci in sci_list
                ]

            coadd_wgt = self.coadds[band]['sci'].replace(
                f'[{sci_ext}]', f'[{wgt_ext}]'
                )

            self.wgt_images[band] = wgt_list
            self.coadds[band]['wgt'] = coadd_wgt
            self.logprint(f'Weight frames for band {band}: {wgt_list}')
            self.logprint(f'Coadd weight frame for band {band}: {coadd_wgt}')

        # if detection coadd is registered, get weight frame
        band = 'det'
        if band in self.coadds:
            det_sci = self.coadds[band]['sci']
            det_wgt = det_sci.replace(f'[{sci_ext}]', f'[{wgt_ext}]')
            self.coadds['det']['wgt'] = det_wgt
            self.logprint(f'Coadd weight frame for band {band}: {det_wgt}')

        return

    def go(self, outfile_base=None, outdir=None):
        '''
        Handle all needed SExtractor runs, which come in 3 possible catagories:

        1) 'det': Run SE on detection image for fiducial detections & positions
        2) 'coadd': Run SE in dual-image mode for every single-band coadd using (1)
        3) 'exp': Run SE on all single-epoch-exposures for PSF & bkg estimation

        Can choose a subset of these in the master SE config using `cat_types`

        If `dual_mode` in the config is False, then will only run (2) & (3)
        in single image mode

        outfile_base: str
            The base of the output coadd filenames. Defaults to
            {run_name}_coadd_{band}.fits
        outdir: str
            The output directory for the coadd images. Defaults to basedir
        '''

        self.logprint('Setting outfile_base...')
        self.set_outfile_base(outfile_base, outdir=outdir)
        self.logprint(f'outfile_base={self.outfile_base}')

        cat_types = self.cat_types
        dual_mode = self.dual_mode

        if dual_mode is True:
            self.logprint(f'Running in dual-mode')
            if 'det' not in cat_types:
                raise AttributeError('You have requested dual-mode but ' +
                                     '`det` image is not in catalog_types!')
        else:
            self.logprint('Running in single-image mode')

        if 'det' in cat_types:
            self.logprint('Running on detection image...')
            self.run_on_det_image()

        else:
            self.logprint('Skipping detection image as `dual_mode` ' +
                          'is False')

        if 'coadd' in cat_types:
            self.logprint('Running on single-band coadds...')
            self.run_on_coadds(dual_mode=dual_mode)

        if 'exp' in cat_types:
            self.logprint('Running on single-epoch exposures...')
            self.run_on_exposures(dual_mode=dual_mode)

        # TODO: Refactor point!!
        self.logprint('Collating files...')
        self.collate_files()

        return

    def set_outfile_base(self, outfile_base, outdir=None):

        if outdir is None:
            outdir = self.basedir

        if outfile_base is None:
            self.outfile_base = os.path.join(
                outdir, f'{self.run_name}_cat.fits'
            )
        else:
            self.outfile_base = os.path.join(
                outdir, outfile_base
                )

        return

    def run_on_det_image(self):
        '''
        Run SExtractor on the detection image for a catalog of
        fiducial object positions

        NOTE: If this is called, then dual_mode is implicitly True
        '''

        band = 'det'

        if band not in self.coadds:
            self.logprint('WARNING: there is no registered detection ' +
                          'image; skipping')
            return

        det_sci = self.coadds[band]['sci']
        det_wgt = self.coadds[band]['wgt']

        outfile = self.outfile_base.replace('.fits', f'_{band}.fits')

        # NOTE: There is some subtlety in calling the detection image
        # measurement in dual-mode as well to ensure identical object
        # ordering in the single-band dual image catalogs. Unclear if
        # it is strictly necessary, but playing it safe
        self._run_sextractor(
            band,
            outfile,
            det_sci,
            det_wgt,
            dual_mode=True
            )

        self.catalogs[band] = {
            'coadd': outfile,
            'exposures': [] # det has no single-band exposures
            }

        return

    def run_on_coadds(self, dual_mode=False):
        '''
        Run SExtractor on the single-band coadds. These are used for
        the fiducial SE photometry measurements (though you may want
        to use something more sophisticated in a later module)

        dual_mode: bool
            Set to True to run SExtractor in dual mode; extract sources
            in image1 using detections in image2
        '''

        if dual_mode is True:
            if 'det' not in self.catalogs:
                raise ValueError('Must run SExtractor on the detection ' +
                                 'image before you can run in dual-mode!')

        for band in self.bands:
            self.logprint(f'Starting band {band}')

            outfile = self.outfile_base.replace('.fits', f'_{band}.fits')
            self.logprint(f'Saving output catalog to {outfile}')

            sci = self.coadds[band]['sci']
            wgt = self.coadds[band]['wgt']

            # NOTE: if running in dual mode, it will automatically
            # use the detection image for image2
            self._run_sextractor(
                band,
                outfile,
                sci,
                wgt=wgt,
                dual_mode=dual_mode
                )

            if band not in self.catalogs:
                self.catalogs[band] = {
                    'coadd': outfile,
                    'exposures': None
                    }
            else:
                self.catalogs[band]['coadd'] = outfile

        return

    def run_on_exposures(self, dual_mode=False):
        '''
        Run SExtractor on the single-epoch exposures. This is mostly
        needed for the PSF estimation

        dual_mode: bool
            Set to True to run SExtractor in dual mode; extract sources
            in image1 using detections in image2
        '''

        if dual_mode is True:
            if 'det' not in self.catalogs:
                raise ValueError('Must run SExtractor on the detection ' +
                                 'image before you can run in dual-mode!')

        for band in self.bands:
            self.logprint(f'Starting band {band}')

            sci_list = self.sci_images[band]
            wgt_list = self.wgt_images[band]
            Nfiles = len(sci_list)
            assert Nfiles == len(wgt_list)

            ext = self.sci_ext

            for i, files in enumerate(zip(sci_list, wgt_list)):
                sci, wgt = files
                self.logprint(f'Starting frame {i+1} of {Nfiles}: {sci}')

                outfile = sci.replace(f'.fits[{ext}]', f'_cat.fits')
                self.logprint(f'Saving output catalog to {outfile}')

                # NOTE: if running in dual mode, it will automatically
                # use the detection image for image2
                self._run_sextractor(
                    band,
                    outfile,
                    sci,
                    wgt=wgt,
                    dual_mode=dual_mode
                    )

                # in case the single-band coadd cat hasn't been created yet
                if band not in self.catalogs:
                    self.catalogs[band] = {
                        'coadd': None,
                        'exposures': [outfile]
                        }
                elif self.catalogs[band]['exposures'] is None:
                    self.catalogs[band]['exposures'] = [outfile]
                else:
                    self.catalogs[band]['exposures'].append(outfile)

        return

    def _run_sextractor(self, band, outfile, sci, wgt=None, dual_mode=False):
        '''
        band: str
            The band to make a coadd of
        outfile: str
            The name of the output catalog file
        sci: str
            The sci filename for image1
        wgt: str
            The wgt filename for image1
        dual_mode: bool
            Set to True to run SExtractor in dual mode; extract sources
            in image1 using detections in image2 (det image)
        '''

        if dual_mode is True:
            if 'det' not in self.coadds:
                raise ValueError('Cannot run in dual-mode if a ' +
                                 'detection image is not registered!')

            det_sci = self.coadds['det']['sci']
            det_wgt = self.coadds['det']['wgt']
        else:
            det_sci=None,
            det_wgt=None

        sextractor_cmd = self._setup_sextractor_cmd(
            band,
            outfile,
            sci,
            im_wgt=wgt,
            im2_sci=det_sci,
            im2_wgt=det_wgt
            )

        self.logprint()
        self.logprint(f'Sextractor cmd: {sextractor_cmd}')

        try:
            rc = utils.run_command(sextractor_cmd)

            self.logprint(f'SExtractor completed successfully for band {band}')
            self.logprint()

        except Exception as e:
            self.logprint()
            self.logprint('WARNING: SExtractor failed with the following ' +
                          f'error:')
            raise e

        # TODO: Once the default config generation is working,
        # will want to clean them up here!
        # # move any extra created files if needed
        # if os.getcwd() != self.outdir:
        #     self.logprint('Cleaning up local directory...')
        #     cmd = f'mv *.xml *.fits {self.outdir}'
        #     self.logprint()
        #     self.logprint(cmd)
        #     os.system(cmd)
        #     self.logprint()

        return

    def _setup_sextractor_cmd(self, band, outfile, im_sci, im_wgt=None,
                              im2_sci=None, im2_wgt=None):
        '''
        outfile: str
            The name of the output catalog file
        im_sci: str
            The sci filename for main image
        im_wgt: str
            The wgt filename for main image
        im2_sci: str
            The sci filename for the detection image (dual mode)
        im2_wgt: str
            The wgt filename for the detection image (dual mode)
        '''

        config_file = self.config[band]
        config_arg = f'-c {config_file}'

        cat_arg = f'-CATALOG_NAME {outfile}'

        image_args = f'{im_sci}'
        if im2_sci is not None:
            image_args += f', {im2_sci}'

        if im_wgt is not None:
            wgt_args = f'-WEIGHT_IMAGE {im_wgt}'
            if im2_wgt is not None:
                wgt_args = f'{wgt_args},{im2_wgt}'

            # TODO: Should we put this one in the config file instead?
            wgt_args = f'{wgt_args} -WEIGHT_TYPE MAP_WEIGHT'
        else:
            wgt_args = None

        # simultaneously handle sci files w/ & w/o extensions
        base = im_sci.split('.fits')[0]
        bkg_name = f'{base}.sub.fits'
        seg_name = f'{base}.sgm.fits'
        check_arg = f'-CHECKIMAGE_NAME  {bkg_name},{seg_name}'

        cmd = ' '.join([
            'sex', image_args, config_arg, cat_arg, wgt_args, check_arg
            ])

        # now for the optional config fields
        opt_configs = {
            'parameters': 'PARAMETERS_NAME',
            'filter': 'FILTER_NAME',
            'nnw': 'STARNNW_NAME',
        }
        for name, cmd_name in opt_configs.items():
            cmd_name
            val = self.config[name]
            if val is not None:
                cmd += f' -{cmd_name} {val}'
            else:
                # TODO: Figure out how to automatically generate the
                # other configs!
                default_configs = {
                    'parameters': ('default.params', 'sex -dp'),
                    # 'filter': ??
                    # 'nnw': ??
                }
                # Query SExtractor to get the default configs in CWD
                # for name, default_pair in default_configs.items():
                #     default, default_cmd = default_pair

                #     self.logprint(f'WARNING: {name} config not passed')
                #     self.logprint(f'Generating default config {default}:')
                #     self.logprint(f'{default_cmd}')
                #     os.system(default_cmd)

        return cmd

    def collate_files(self):
        # TODO: Refactor point!!
        pass
