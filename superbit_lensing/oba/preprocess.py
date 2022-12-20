from pathlib import Path
from glob import glob

class PreprocessRunner(object):
    '''
    Class to manage the running of the SuperBIT onboard analysis
    (OBA) preprocessing
    '''

    def __init__(self, raw_dir, run_dir, out_dir, bands, target_name=None):
        '''
        raw_dir: pathlib.Path
            The directory containing the raw sci frames for the given target
        run_dir: pathlib.Path
            The OBA run directory for the given target
        out_dir: pathlib.Path
            The permanent output directory for the given target
        bands: list of str's
            A list of band names
        target_name: str
            The name of the target. Default is to check the end of the raw
            & run dirs
        '''

        dir_args = {
            'raw_dir': raw_dir,
            'run_dir': run_dir,
            'out_dir': out_dir
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

        # will get populated during call to go()
        self.images = {}

        return

    def go(self, logprint):
        '''
        Run the OBA preprocessing step. This entails the following:

        1) Setup the run_dir
        2) Copy raw sci frames from raw_dir to run_dir
        3) Decompress raw files
        '''

        logprint('Setting up temporary run directories...')
        self.setup_run_dirs(logprint)

        logprint('Copying raw sci files...')
        self.copy_raw_files(logprint)

        logprint('Decompressing raw files...')
        self.decompress_raw_files(logprint)

        logprint('Preprocessing completed!')

        return

    def setup_run_dirs(self, logprint):
        '''
        Setup the OBA run directory for the target
        '''

        dirs = [
            self.run_dir,
            # "derived" detection band
            self.run_dir / 'det' / 'coadd'
            ]

        for b in self.bands:
            dirs.append(self.run_dir / b)
            dirs.append(self.run_dir / b / 'coadd/')
            dirs.append(self.run_dir / b / 'out/')

        for d in dirs:
            logprint(f'Creating directory {d}')
            utils.make_dir(d)

        # now create permanent output dir
        odir = self.out_dir
        logprint(f'Creating permanent output directory {str(odir)}')
        utils.make_dir(odir)

        return

    def copy_raw_files(self, logprint):
        '''
        Copy raw sci frames to the temp OBA run directory for
        a given target
        '''

        bands = self.bands

        for band in bands:
            logprint(f'Starting band {band}')
            self.images[band] = []

            orig = (self.raw_dir / band).resolve()
            dest = (self.run_dir / band).resolve()

            # TODO: Make this more specific once filename
            # conventions are fixed!
            raw_files = glob(
                os.path.join(str(orig), '*.fits')
                )

            logprint(f'Found the following raw files for band {band}:')
            logprint(raw_files)

            for raw_file in raw_files:
                logprint(f'Copying {raw_file} to {str(dest)}:')
                self._copy_raw_files(
                    raw_file, str(dest), logprint=logprint
                    )

                # add to internal image dict
                out_name = Path(raw_file).name
                out_file = dest / out_name
                self.images[band].append(out_file)

        return

    def decompress_raw_files(self, logprint):
        '''
        We've already done the work of registering each copied
        raw sci frame, so now decompress each
        '''

        for band, images in self.images.items():
            logprint(f'Starting band {band}')

            for image in images:
                image = str(image)
                logprint(f'Decompressing file {image}')
                self._decompress_file(image, logprint)

        return

    @staticmethod
    def _copy_file(orig_file, dest, logprint=None):
        '''
        Input paths must be str's, not pathlib Paths at this point
        '''

        cmd = f'cp {orig_file} {dest}'

        if logprint is not None:
            logprint(f'cmd = {cmd}')

        utils.run_cmd(cmd)

        return

    @staticmethod
    def _decompress_file(filename, logprint=None):
        '''
        Inputs paths must be str's, not pathlib Paths at this point
        '''

        # TODO: Check if bzip2 is the correct command!
        cmd = f'bzip2 -dk {filename}'

        if logprint is not None:
            logprint(f'cmd = {cmd}')

        utils.run_cmd(cmd)

        return