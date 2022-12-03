import os
from glob import glob

from superbit_lensing import utils

import ipdb

class SWarpRunner(object):
    '''
    A minimal wrapper class around SWarp to organize i/o and some
    common value-adds needed for downstream modules.
    '''

    def __init__(self, config_file, run_name, basedir, bands, config_dir=None,
                 fname_base=None, outdir=None, logprint=None):
        '''
        config_file: str
            The filename of the SWarp config
        run_name: str
            The name of the current processing run
        basedir: str
            The base directory path for the set of single-exposure images
        bands: list of str's
            A list of band names to make coadds for
        config_dir: str
            A directory where the SWarp config file is located. Defaults to
            current working directory
        fname_base: str
            Base name of the input sci files to coadd
        outdir: str
            The output directory for produced coadd files. Defaults to basedir
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

        if not isinstance(bands, list):
            raise TypeError('bands must be a list of strings!')
        else:
            for band in bands:
                if not isinstance(band, str):
                    raise TypeError('Band names must be a str!')
        self.bands = bands

        self.get_sci_images()

        # will be set during go() method
        self.det_image = None
        self.coadds = None

        return

    def get_sci_images(self):
        '''
        Grab all available science images for requested bands given the basedir
        '''

        # we'll save the images for each band separately
        self.sci_images = {}

        for band in self.bands:

            # get single exposures, but ignore any pre-existing truth, mcal, etc.
            sci_names = os.path.join(
                self.basedir, self.fname_base
                ) + f'*[!truth,meds,mcal,sub,wgt,mask,sgm,coadd]_{band}.fits'

            sci = glob(sci_names)

            # TODO: loop through and check the files?

            self.sci_images[band] = sci
            self.logprint(f'Science frames for band {band}: {sci}')

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

        self.logprint('Writing coadds to disk...')
        self.write_coadds()

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

        self.coadds = {}

        for b in self.bands:
            self.logprint(f'Starting band {b}')

            outfile = self.outfile_base.replace('.fits', f'_{b}.fits')
            self._run_swarp(b, outfile)

            self.coadds[b] = outfile

        if len(self.coadds) != len(self.bands):
            self.logprint('WARNING: The number of produced coadds does not ' +
                          'equal the number of passed bands; something ' +
                          'likely has failed!')

        return

    def make_detection_image(self):
        band = 'det'

        outfile = os.path.join(
            self.basedir, f'{self.run_name}_coadd_{band}.fits'
            )
        self._run_swarp(band, outfile, detection=True)

        self.coadds[band] = outfile

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

        swarp_cmd = self._setup_swarp_cmd(band, outfile)
        os.sys(cmd)
        self.logprint()

        # move any extra created files if needed
        if os.getcwd() != self.outdir:
            cmd = f'mv *.xml *.fits {self.outdir}'
            self.logprint(cmd)
            # rc = utils.run_command(cmd, logprint=self.logprint)
            os.system(cmd)
            self.logprint()

        return

    def _setup_swarp_cmd(self, band, outfile):
        '''
        band: str
            The band to make a coadd of
        outfile: str
            The name of the output coadd file
        '''

        ipdb.set_trace()
        image_args = ' '.join(self.sci_images[band])

        weight_outfile = outfile.replace('.fits', '.weight.fits')
        config_arg = '-c ' + self.config_file
        resamp_arg = '-RESAMPLE_DIR ' + self.outdir
        outfile_arg = '-IMAGEOUT_NAME '+ outfile + ' ' +\
                      '-WEIGHTOUT_NAME ' + weight_outfile

        cmd = ' '.join([
            'swarp ', image_args, resamp_arg, outfile_arg, config_arg
            ])

        self.logprint('SWarp cmd: {cmd}')

        return cmd

    def write_coadds(self):
        pass

