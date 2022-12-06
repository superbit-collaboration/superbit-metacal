import os
from glob import glob

from superbit_lensing import utils

import ipdb

class SWarpRunner(object):
    '''
    A minimal wrapper class around SWarp to organize i/o and some
    common value-adds needed for downstream modules.
    '''

    def __init__(self, config_file, run_name, basedir, bands, det_bands,
                 config_dir=None, fname_base=None, outdir=None,
                 sci_ext=0, wgt_ext=1, logprint=None):
        '''
        config_file: str
            The filename of the base SWarp config
        run_name: str
            The name of the current processing run
        basedir: str
            The base directory path for the set of single-exposure images
        bands: list of str's
            A list of band names to make coadds for
        det_bands: list of str's
            A list of band names to use for the detection coadd
        config_dir: str
            A directory where the SWarp config file is located. Defaults to
            current working directory
        fname_base: str
            Base name of the input sci files to coadd
        outdir: str
            The output directory for produced coadd files. Defaults to basedir
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
            fname_base = run_name

        # process str args
        arg_dict = {
            'config_file': config_file,
            'run_name': run_name,
            'basedir': basedir,
            'fname_base': fname_base,
            'outdir': outdir,
            }
        for name, arg in arg_dict.items():
            if not isinstance(arg, str):
                raise TypeError(f'{name} must be a str!')
            setattr(self, name, arg)

        if config_dir is not None:
            if isinstance(config_dir, str):
                self.config_dir = config_dir
                self.config_file = os.path.join(config_dir, config_file)
            else:
                raise TypeError('config_dir must be a str!')

        for name, bnds in {'bands':bands, 'det_bands':det_bands}.items():
            if not isinstance(bnds, list):
                raise TypeError(f'{name} must be a list of strings!')
            else:
                for band in bnds:
                    if not isinstance(band, str):
                        raise TypeError('Band names must be a str!')
        self.bands = bands
        self.det_bands = bands

        for name, ext in {'sci_ext':sci_ext, 'wgt_ext':wgt_ext}.items():
            if not isinstance(ext, int):
                raise TypeError(f'{name} must be an int!')

        self.sci_ext = sci_ext
        self.wgt_ext = wgt_ext

        self.get_sci_images()
        self.get_wgt_images()

        # will be set during go() method
        self.coadds = {}

        return

    def get_sci_images(self):
        '''
        Grab all available science & weight images for requested bands given
        the basedir

        NOTE: For now, we follow the multi-extention fits convention
        '''

        ext = self.sci_ext

        # we'll save the images for each band separately
        self.sci_images = {}

        for band in self.bands:
            # get single exposures, but ignore any pre-existing truth, mcal, etc.
            sci_names = os.path.join(
                self.basedir, self.fname_base
                ) + f'*[!truth,meds,mcal,sub,wgt,mask,sgm,coadd]_{band}.fits'

            sci_list = glob(sci_names)

            sci_list = [
                sci.replace('.fits', f'.fits[{ext}]') for sci in sci_list
                ]

            self.sci_images[band] = sci_list
            self.logprint(f'Science frames for band {band}: {sci_list}')

        return

    def get_wgt_images(self):
        '''
        Can implement more options in the future. For now, we're using
        ext=1 of the sci images
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

            self.wgt_images[band] = wgt_list

        return

    def go(self, outfile_base=None, outdir=None, make_det_image=True):
        '''
        outfile_base: str
            The base of the output coadd filenames. Defaults to
            {run_name}_coadd_{band}.fits
        outdir: str
            The output directory for the coadd images. Defaults to basedir
        make_det_image: bool
            Set to create a composite detection image in addition to the
            single-band coadds. Default is True
        '''

        self.logprint('Setting outfile_base...')
        self.set_outfile_base(outfile_base, outdir=outdir)
        self.logprint(f'outfile_base={self.outfile_base}')

        self.logprint('Making coadds...')
        self.make_coadds()

        if make_det_image is True:
            self.logprint('Making detection image...')
            self.make_detection_image()
        else:
            self.logprint('Skipping detection image as `make_det_image` ' +
                          'is False')

        return

    def set_outfile_base(self, outfile_base, outdir=None):

        if outdir is None:
            outdir = self.basedir

        if outfile_base is None:
            self.outfile_base = os.path.join(
                outdir, f'{self.run_name}_coadd.fits'
            )
        else:
            self.outfile_base = os.path.join(
                outdir, outfile_base
                )

        return

    def make_coadds(self):
        '''
        Make a coadd image using SWarp for each band
        '''

        for b in self.bands:
            self.logprint(f'Starting band {b}')
            self.coadds[b] = {}

            outfile = self.outfile_base.replace('.fits', f'_{b}.fits')
            self._run_swarp(b, outfile)

            self.coadds[b]['sci'] = outfile
            self.coadds[b]['wgt'] = outfile.replace('.fits', '.wgt.fits')

        if len(self.coadds) != len(self.bands):
            self.logprint('WARNING: The number of produced coadds does not ' +
                          'equal the number of passed bands; something ' +
                          'likely has failed!')

        return

    def make_detection_image(self):
        '''
        Once all single-band coadds are made, create the
        detection image
        '''


        for band in self.det_bands:
            if band not in self.coadds:
                raise ValueError('Cannot make detection image until all '
                                 'following single-band coadds are done: '
                                 f'{self.det_bands}')

        band = 'det'
        self.coadds[band] = {}

        outfile = os.path.join(
            self.basedir, f'{self.run_name}_coadd_{band}.fits'
            )
        self._run_swarp(band, outfile, detection=True)

        self.coadds[band]['sci'] = outfile
        self.coadds[band]['wgt'] = outfile.replace('.fits', '.wgt.fits')

        return


    def _run_swarp(self, band, outfile, detection=False):
        '''
        band: str
            The band to make a coadd of
        outfile: str
            The name of the output coadd file
        detection: bool
            Set to True to indicate you are making a detection image
        '''

        if band not in self.bands:
            # make sure band is in self.bands, except for the detection image
            if detection is False:
                raise ValueError(f'{band} is not a registered band!')

        swarp_cmd = self._setup_swarp_cmd(
            band, outfile, detection=detection
            )

        self.logprint()
        self.logprint(f'SWarp cmd: {swarp_cmd}')
        os.system(swarp_cmd)
        self.logprint(f'SWarp completed for band {band}')
        self.logprint()

        # move any extra created files if needed
        if os.getcwd() != self.outdir:
            self.logprint('Cleaning up local directory...')
            cmd = f'mv *.xml *.fits {self.outdir}'
            self.logprint()
            self.logprint(cmd)
            os.system(cmd)
            self.logprint()

        return

    def _setup_swarp_cmd(self, band, outfile, detection=False):
        '''
        band: str
            The band to make a coadd of
        outfile: str
            The name of the output coadd file
        detection: bool
            Set to True to indicate you are making a detection image
        '''

        # a few common cmd args
        config_arg = '-c ' + self.config_file
        weight_outfile = outfile.replace('.fits', '.wgt.fits')
        resamp_arg = '-RESAMPLE_DIR ' + self.outdir
        outfile_arg = '-IMAGEOUT_NAME '+ outfile + ' ' +\
                      '-WEIGHTOUT_NAME ' + weight_outfile

        if detection is False:
            # normal coadds are made from resampling from all single-epoch
            # exposures (& weights) for a given band & target
            sci_im_args = ' '.join(self.sci_images[band])
            wgt_im_args = ','.join(self.wgt_images[band])

            image_args = f'{sci_im_args} -WEIGHT_IMAGE {wgt_im_args}'

            # use config value for single-band
            ctype_arg = ''

        else:
            # detection coadds resample from the single-band coadds
            sci_im_list = [self.coadds[b]['sci'] for b in self.det_bands]
            wgt_im_list = [self.coadds[b]['wgt'] for b in self.det_bands]

            sci_im_args = ' '.join(sci_im_list)
            wgt_im_args = ','.join(wgt_im_list)

            image_args = f'{sci_im_args} -WEIGHT_IMAGE {wgt_im_args}'

            # DES suggests using AVERAGE instead of CHI2 or WEIGHTED
            ctype_arg = '-COMBINE_TYPE AVERAGE'

        cmd = ' '.join([
            'swarp ', image_args, resamp_arg, outfile_arg, config_arg, ctype_arg
            ])

        return cmd
