from pathlib import Path
from glob import glob
import os
import fitsio

from superbit_lensing import utils
from superbit_lensing.oba.oba_io import band2index

import ipdb

class PreprocessRunner(object):
    '''
    Class to manage the running of the SuperBIT onboard analysis
    (OBA) preprocessing
    '''

    _name = 'preprocessing'

    _compression_method = 'bzip2'
    _compression_args = '-dk' # forces decompression, keep orig file
    _compression_ext = 'bz2'

    # some useful detector meta data to be added to headers; should be static
    _header_info = {
        'GAIN': 0.343, # e- / ADU
        'SATURATE': 65535, # 2^16-1 as we saturate before NL effects (gain < 1)

        # NOTE: Already the SExtractor default, and causes Hierarch warnings
        # 'SATUR_KEY': 'SATURATE' # sets the saturation key SExtractor looks for
    }

    # some key names have been updated over time; check, but this should only
    # be relevant for testing sims
    _updated_keys = {
        'TARGET_RA': 'TRG_RA',
        'TARGET_DEC': 'TRG_DEC'
    }

    # image header key that is used to determine image quality
    # is one of ['GOOD', 'BAD', 'UNVERIFIED'] for SuperBIT
    _img_qual_key = 'IMG_QUAL'
    _img_qual_allowed = ['BAD', 'GOOD', 'UNVERIFIED']
    _img_qual_order = {
        'BAD': 0,
        'UNVERIFIED': 1,
        'GOOD': 2
    }

    def __init__(self, raw_dir, run_dir, out_dir, bands, target_name=None,
                 check_img_qual=True, min_img_qual='unverified'):
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
        check_img_qual: bool
            Set to check for image quality when selecting images for OBA
        min_img_qual: str
            The minimum image quality value to process on. Must be either
            "good" or "unverified" (case agnostic)
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

        utils.check_type('check_img_qual', check_img_qual, bool)
        self.check_img_qual = check_img_qual

        utils.check_type('min_img_qual', min_img_qual, str)
        min_img_qual = min_img_qual.upper()
        _allowed = self._img_qual_allowed
        if min_img_qual.upper() not in _allowed:
            raise ValueError(f'min_img_qual must be one of {_allowed}!')

        # this maps the image quality state to an int we can compare
        self.min_img_qual = min_img_qual
        self.min_img_qual_val = self._img_qual_order[min_img_qual]

        # will get populated during call to go()
        self.images = {}
        self.decompressed_images = {}

        return

    def go(self, logprint, overwrite=False, skip_decompress=False):
        '''
        Run the OBA preprocessing step. This entails the following:

        1) Setup the run_dir (and all subdirs)
        2) Copy raw sci frames from raw_dir to run_dir
        3) Decompress raw files
        4) Update fits headers with useful info

        logprint: utils.LogPrint
            A LogPrint instance for simultaneous logging & printing
        overwrite: bool
            Set to overwrite existing files
        skip_decompress: bool
            Set to skip the raw file decompression if file already exists
        '''

        logprint('Setting up temporary run directories...')
        self.setup_run_dirs(logprint)

        logprint('Copying raw sci files...')
        self.copy_raw_files(logprint)

        logprint('Decompressing raw files...')
        self.decompress_raw_files(logprint, overwrite=(not skip_decompress))

        logprint('Updating fits headers...')
        self.update_headers(logprint)

        logprint('Preprocessing completed!')

        return

    def setup_run_dirs(self, logprint):
        '''
        Setup the OBA run directory for the target

        logprint: utils.LogPrint
            A LogPrint instance for simultaneous logging & printing
        '''

        dirs = [
            self.run_dir,
            # "derived" detection band
            self.run_dir / 'det' / 'coadd'
            ]

        # TODO: decide which of these we don't need!
        for b in self.bands:
            dirs.append(self.run_dir / b)
            dirs.append(self.run_dir / b / 'cal/')
            dirs.append(self.run_dir / b / 'coadd/')
            dirs.append(self.run_dir / b / 'out/')

            # for images that fail during astrometry module
            dirs.append(self.run_dir / b / 'failed/')

            # for any temporary file writing; useful for minimizing
            # memory footprint
            dirs.append(self.run_dir / b / 'tmp/')

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

        logprint: utils.LogPrint
            A LogPrint instance for simultaneous logging & printing
        '''

        bands = self.bands
        cext = self._compression_ext

        Nimages = 0
        for band in bands:
            logprint(f'Starting band {band}')
            self.images[band] = []

            # orig = (self.raw_dir / band).resolve()
            # NOTE: For now, the plan is for all raw files to be in
            # the same dir, regardless of band
            orig = self.raw_dir
            dest = (self.run_dir / band).resolve()

            bindx = band2index(band)

            # NOTE: This glob is safe as OBA files have a fixed convention
            search = str(orig / f'{self.target_name}*_{bindx}_*.fits.{cext}')
            raw_files = glob(search)

            Nraw = len(raw_files)
            if Nraw == 0:
                logprint(f'WARNING: found zero raw files for band {band}')
                logprint('Skipping')
                continue
            else:
                Nimages += Nraw

            logprint(f'Found the following {Nraw} raw files for band {band}:')
            remove_indices = []
            for i, raw in enumerate(raw_files):
                logprint(f'{i}: {raw}')

                if self.check_img_qual is True:
                    # check if the image quality is acceptable, if desired
                    min_qual = self.min_img_qual_val
                    min_qual_val = self.min_img_qual_val
                    raw_hdr = fitsio.read_header(raw)

                    try:
                        img_qual = raw_hdr[self._img_qual_key]
                        img_qual_val = self._img_qual_order[img_qual]

                        logprint(f'Quality: {img_qual}')

                        if img_qual_val < min_qual_val:
                            logprint(
                                f'WARNING: Image quality does not meet the ' +\
                                f'required standard of {min_qual}; skipping'
                                )
                            remove_indices.append(i)

                    except KeyError as e:
                        raise KeyError(f'IMG_QUAL must be in image header ' +\
                                       'if check_img_qual is set to True!')

            # wait to remove any images that do not satisfy requirements until
            # *after* loop (and in reverse) to avoid indexing issues
            for index in sorted(remove_indices, reverse=True):
                del raw_files[index]

            for raw_file in raw_files:
                logprint(f'Copying {raw_file} to {str(dest)}:')
                self._copy_file(
                    raw_file, str(dest), logprint=logprint
                    )

                # add to internal image dict
                out_name = Path(raw_file).name
                out_file = dest / out_name
                self.images[band].append(out_file)

        # make sure that at least *one* image was found!
        if Nimages == 0:
            raise OSError('No images found for the OBA to run on!')

        return

    def decompress_raw_files(self, logprint, overwrite=False):
        '''
        We've already done the work of registering each copied
        raw sci frame, so now decompress each

        logprint: utils.LogPrint
            A LogPrint instance for simultaneous logging & printing
        overwrite: bool
            Set to overwrite existing raw files
        '''

        for band, images in self.images.items():
            logprint(f'Starting band {band}')
            self.decompressed_images[band] = []

            for image in images:
                image = str(image)
                logprint(f'Decompressing file {image}')

                decompressed_image = image.replace(
                    f'.{self._compression_ext}', ''
                    )
                self.decompressed_images[band].append(
                    Path(decompressed_image)
                    )

                # NOTE: check if decompressed file already exists; can
                # cause issues otherwise
                if os.path.exists(decompressed_image):
                    # will cause a bzip2 error if not handled
                    logprint(f'{decompressed_image} already exists')
                    if overwrite is True:
                        logprint('Deleting file as overwrite is True')
                        os.remove(decompressed_image)
                    else:
                        logprint('Keeping file as overwrite is False')
                        continue

                self._decompress_file(image, logprint)

        return

    @staticmethod
    def _copy_file(orig_file, dest, logprint=None):
        '''
        Input paths must be str's, not pathlib Paths at this point

        orig_file: str
            The filepath of the original file
        dest: str
            The filepath of the destination
        logprint: utils.LogPrint
            A LogPrint instance for simultaneous logging & printing
        '''

        cmd = f'cp {orig_file} {dest}'

        if logprint is not None:
            logprint(f'cmd = {cmd}')

        utils.run_command(cmd, logprint=logprint)

        return

    @staticmethod
    def _decompress_file(filename, logprint=None):
        '''
        Inputs paths must be str's, not pathlib Paths at this point

        filename: str
            Name of the file to decomporess
        logprint: utils.LogPrint
            A LogPrint instance for simultaneous logging & printing
        '''

        # TODO: Check if bzip2 is the correct command!
        cmd = f'bzip2 -dk {filename}'

        if logprint is not None:
            logprint(f'cmd = {cmd}')

        rc = utils.run_command(cmd, logprint=logprint)

        return

    def update_headers(self, logprint, ext=0):
        '''
        Add useful and / or required metadata for downstream processes
        in the raw image headers

        logprint: utils.LogPrint
            A LogPrint instance for simultaneous logging & printing
        ext: int
            The fits extension to update the header of
        '''

        logprint('Setting the following values in each raw image header:')
        for key, val in self._header_info.items():
            logprint(f'{key}: {val}')

        for band, images in self.decompressed_images.items():
            logprint(f'Starting band {band}')

            Nimages = len(images)
            for i, image in enumerate(images):
                image_name = image.name
                logprint(f'Updating {image_name}; {i+1} of {Nimages}')

                with fitsio.FITS(str(image), 'rw') as f:
                    hdr = f[0].read_header()

                    # remap any old keys to updated ones
                    # NOTE: In principle, this should only matter for OBA
                    # testing on older sims
                    for old_key, new_key in self._updated_keys.items():
                        if old_key in hdr:
                            val = hdr[old_key]
                            comment = hdr.get_comment(old_key)
                            f[0].write_key(new_key, val, comment=comment)
                            logprint(f'Replaced {old_key} with {new_key}')

                    for key, val in self._header_info.items():
                        # things have changed over time, so only add if the key
                        # doesn't already exist
                        if key not in hdr:
                            # fits.setval(str(image), key, value=val, ext=ext)
                            f[0].write_key(key, val)

        return
