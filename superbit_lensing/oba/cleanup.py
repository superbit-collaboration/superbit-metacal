import shutil
from pathlib import Path

import ipdb

from superbit_lensing import utils

class CleanupRunner(object):
    '''
    Runner class for the final cleanup of the SuperBIT onboard analysis
    pipeline. At this stage all final dataproducts have been created,
    and all that is left is to copy them from their temporary OBA_DIR
    to their final destination in OBA_RESULTS which is automatically
    saved to the DRS's that will be dropped over land when possible.

    Currently, the final outputs to beam (or parachute) down are the
    following:

    (1) CookieCutter output FITS file containing source stamps, masking
        information, and metadata
    (2) TODO: Coadd cluster center cutouts
    (3) TODO: RAW_SCI or CAL_SCI image headers
    (4) Log files & any intermediate and/or created config files
    (5) TODO: ...
    '''

    def __init__(self, run_dir, out_dir, bands, target_name=None,
                 clean_oba_dir=False):
        '''
        run_dir: pathlib.Path
            The OBA run directory for the given target
        out_dir: pathlib.Path
            The permanent output directory for the given target
        bands: list of str's
            A list of band names
        target_name: str
            The name of the target. Default is to check the end of the raw
            & run dirs
        clean_oba_dir: bool
            Set to delete the temporary OBA dir after output writing.
            A bit dangerous!
        '''

        dir_args = {
            'run_dir': run_dir,
            'out_dir': out_dir,
        }
        for name, val in dir_args.items():
            utils.check_type(name, val, Path)
            setattr(self, name, val)

        utils.check_type('bands', bands, list)
        for b in bands:
            utils.check_type('band', b, str)
        self.bands = bands
        self.Nbands = len(bands)

        if target_name is None:
            # NOTE: same default structure as preprocessing.py
            # try looking at the paths, but ensure they are consistent!
            raw_name = raw_dir.name
            run_name = run_dir.name
            out_name = out_dir.name

            if (raw_name != run_name) or (raw_name != out_name):
                raise ValueError('If target_name is not provided, then ' +
                                 'the final dir name of raw_dir and run_dir ' +
                                 'must match!\n' +
                                 'raw_dir={raw_dir}\nrun_dir={run_dir}')
            else:
                target_name = run_name
        else:
            utils.check_type('target_name', target_name, str)
        self.target_name = target_name

        utils.check_type('clean_oba_dir', clean_oba_dir, bool)
        self.clean_oba_dir = clean_oba_dir

        # this dict will store the output files that will be written to disk,
        # indexed by band
        self.outputs = {}

        return

    def go(self, logprint, overwrite=False):
        '''
        Run the OBA cleanup step. Mostly moving final OBA output datatypes
        into permanent storage on disks that automatically update the DRS's,
        as well as compression

        Steps:

        (1) Register all files that are to be saved to permanent storage
            (images, logs, configs, etc.)
        (2) Compress output files
        (3) Write output files to permanent storage
        (4) Cleanup the OBA run directory, if desired

        logprint: utils.LogPrint
            A LogPrint instance for simultaneous logging & printing
        overwrite: bool
            Set to overwrite existing files
        '''

        logprint('Gathering output files...')
        self.gather_outputs(logprint)

        logprint('Compressing output files...')
        self.compress_outputs(logprint)

        logprint('Writing output files to disk...')
        self.write_outputs(logprint)

        logprint('Cleaning up OBA dir...')
        self.cleanup(logprint)

        return

    def gather_outputs(self, logprint):
        '''
        Collect all OBA files that are to be permanently stored to disk.
        These are mostly FITS (CookieCutter stamps), logs, and config files

        logprint: utils.LogPrint
            A LogPrint instance for simultaneous logging & printing
        overwrite: bool
            Set to overwrite existing files
        '''

        target_name = self.target_name
        out_dir = self.out_dir

        for band in self.bands:
            logprint(f'Starting band {band}')
            outputs = []

            band_dir = self.run_dir / band
            out_dir = band_dir / 'out/'

            # CookieCutter cutout FITS file
            cutouts = out_dir / f'{target_name}_{band}_cutouts.fits'
            outputs.append(str(cutouts))

            # generated CookeCutter config file
            cutouts_config = out_dir / f'{target_name}_{band}_cutouts.yaml'
            outputs.append(str(cutouts_config))

            self.outputs[band] = outputs

        # TODO: Add gathering for any additional desired output files here!

        return

    def compress_outputs(self, logprint):
        '''
        Copy final output files to a temporary directory & compress
        '''

        # copy all of the output files
        # tmp_dir = self.run_dir / 'tmp/'
        # logprint(f'Compressing files in temporary dir {str(rmp_dir)}')

        # for bands in self.bands:
        #     logprint('Starting band {band}')

            # for output in self.outputs[band]:
                # logprint(f'Moving {output.name} to tmp')
                # self._copy_file(output, tmp_dir)


        # TODO: finish!
        logprint('WARNING: compress_outputs() not yet implemented!')

        return

    def write_outputs(self, logprint):
        # TODO: finish!
        logprint('WARNING: write_outputs() not yet implemented!')

        return

    def cleanup(self, logprint):
        # TODO: finish!
        logprint('WARNING: cleanup() not yet implemented!')

        return

    def _copy_file(self, filename, dest):
        '''
        Copy a file to the destination

        filename: pathlib.Path
            The filepath of the file to copy
        dest: pathlib.Path
            The destination directory
        '''

        shutil.copy(str(filename), str(dest))

        return
