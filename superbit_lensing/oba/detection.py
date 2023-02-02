import os
import subprocess
from pathlib import Path
from glob import glob

from superbit_lensing import utils

import ipdb

class DetectionRunner(object):
    '''
    Runner class for detecting sources on the calibrated, bkg-subtracted
    detection image for the SuperBIT onboard analysis (OBA).

    NOTE: At this stage, input coadd images should have a
    WCS solution in the header and the following structure:

    ext0: SCI (calibrated & background-subtracted)
    ext1: WGT (weight; 0 if masked, 1/sky_var otherwise)
    '''

    def __init__(self, config_file, run_dir, target_name=None,
                 sci_ext=0, wgt_ext=1):
        '''
        config_file: pathlib.Path
            The filepath of the base SExtractor config
        run_dir: pathlib.Path
            The OBA run directory for the given target
        target_name: str
            The name of the target. Default is to use the end of
            run_dir
        sci_ext: int
            The science frame fits extension
        wgt_ext: int
            The weight frame fits extension
        '''

        args = {
            'config_file': (config_file, Path),
            'run_dir': (run_dir, Path),
            'sci_ext': (sci_ext, int),
            'wgt_ext': (wgt_ext, int)
        }

        for name, tup in args.items():
            val, allowed_types = tup
            utils.check_type(name, val, allowed_types)
            setattr(self, name, val)

        if target_name is None:
            target_name = run_dir.name

        utils.check_type('target_name', target_name, str)
        self.target_name = target_name

        # the filepath of the detection coadd
        self.det_coadd = None

        # the filepath of the output detection catalog
        self.det_cat = None

        return

    def go(self, logprint, overwrite=False):
        '''
        Make a detection catalog by running SExtractor on the
        coadded detection image

        Steps:

        (1) Grab the detection image
        (2) Run SExtractor on the detection coadd
            NOTE: We ignore the single-band coadds for OBA
        (3) Collate any outputs needed for the final output datastructure

        logprint: utils.LogPrint
            A LogPrint instance for simultaneous logging & printing
        overwrite: bool
            Set to overwrite existing files
        '''

        logprint('Grabbing the detection coadd...')
        self.get_det_coadd(logprint)

        logprint('Running source detection...')
        self.detect_sources(logprint, overwrite=overwrite)

        # TODO: Do we need this?
        logprint('Collating...')
        self.collate(logprint)

        return

    def get_det_coadd(self, logprint):
        '''
        logprint: utils.LogPrint
            A LogPrint instance for simultaneous logging & printing
        '''

        det_dir = (self.run_dir / 'det/coadd/').resolve()
        det_fname = det_dir / f'{self.target_name}_coadd_det.fits'

        if not det_fname.is_file():
            raise OSError(f'{det_fname} does not exist!')

        self.det_coadd = det_fname

        return

    def detect_sources(self, logprint, overwrite=False):
        '''
        Detect sources on the detection coadd using SExtractor

        logprint: utils.LogPrint
            A LogPrint instance for simultaneous logging & printing
        overwrite: bool
            Set to overwrite existing files
        '''

        sci_ext = self.sci_ext
        wgt_ext = self.wgt_ext

        outfile = Path(
            str(self.det_coadd).replace('.fits', '_cat.fits')
            )
        det_sci = str(self.det_coadd) + f'[{sci_ext}]'
        det_wgt = str(self.det_coadd) + f'[{wgt_ext}]'

        if outfile.is_file():
            if overwrite is False:
                raise OSError(f'{outfile} already exists and '
                              'overwrite is False!')
            else:
                logprint(f'{outfile} exists; deleting as ' +
                         'overwrite is True')
                outfile.unlink()

        self._run_sextractor(
            logprint,
            str(outfile),
            det_sci,
            det_wgt,
            dual_mode=True,
            )

        return

    def _run_sextractor(self, logprint, outfile, sci, wgt=None,
                        dual_mode=True):
        '''
        logprint: utils.LogPrint
            A LogPrint instance for simultaneous logging & printing
        outfile: str
            The name of the output catalog file
        sci: str
            The sci filename of image1
        wgt: str
            The wgt filename of image1
        dual_mode: bool
            Set to True to run SExtractor in dual mode; extract sources
            in image1 using detections in image2 (det image)
        '''

        if dual_mode is True:
            # NOTE: this is a bit silly for the OBA, since we are running
            # on just the det image. However running in dual-mode with
            # det being both image1 and image2 guarantees us that the
            # source id order will be the same if we run on the single-
            # band coadds
            det_sci = sci
            det_wgt = wgt
        else:
            det_sci=None,
            det_wgt=None

        sextractor_cmd = self._setup_sextractor_cmd(
            outfile,
            sci,
            im_wgt=wgt,
            im2_sci=det_sci,
            im2_wgt=det_wgt
            )

        logprint()
        logprint(f'Sextractor cmd: {sextractor_cmd}')

        try:
            rc = utils.run_command(sextractor_cmd)

            logprint(f'SExtractor completed successfully')
            logprint()

        except Exception as e:
            logprint()
            logprint('WARNING: SExtractor failed with the following ' +
                          f'error:')
            raise e

        # TODO: Once the default config generation is working,
        # will want to clean them up here!
        # # move any extra created files if needed
        outdir = Path(outfile).parent
        if os.getcwd() != outdir:
            logprint('Cleaning up local directory...')
            cmd = f'mv *.xml *.fits {outdir}'
            logprint()
            logprint(cmd)
            os.system(cmd)
            logprint()

        return

    def _setup_sextractor_cmd(self, outfile, im_sci, im_wgt=None,
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

        config_file = self.config_file
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
        seg_name = f'{base}.sgm.fits'
        ctype = 'SEGMENTATION'
        check_arg = f'-CHECKIMAGE_TYPE {ctype} -CHECKIMAGE_NAME {seg_name}'

        cmd = ' '.join([
            'sex', image_args, config_arg, cat_arg, wgt_args, check_arg
            ])

        # now setup a few additional default configuration files
        config_dir = Path(utils.MODULE_DIR) / 'oba/configs/sextractor/'

        # this sets the photometric parameters that SExtractor computes
        param_file = str(config_dir / 'sb_sextractor.param')
        cmd += f' -PARAMETERS_NAME {param_file}'

        # this sets the detection filter
        filter_file = str(config_dir / 'default.conv')
        cmd += f' -FILTER_NAME {filter_file}'

        # this sets the neural network for the star classifier
        nnw_file = str(config_dir / 'default.nnw')
        cmd += f' -STARNNW_NAME {nnw_file}'

        return cmd

    def collate(self, logprint):
        '''
        Collate outputs to match requirements for final datastructure
        beamed down by OBA

        TODO: At least add the coadd seg map to the next FITS ext!

        logprint: utils.LogPrint
            A LogPrint instance for simultaneous logging & printing
        '''
        pass
