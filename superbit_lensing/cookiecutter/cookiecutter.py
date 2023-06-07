'''
The CookieCutter class is essentially a "lite" version of the Multi-Object
Data Structure (MEDS) format used by the SuperBIT onboard analysis (OBA).
It stores the image cutouts of detected sources for a variety of image types
such as science, weight, mask, segmentation, etc. It is particularly useful in
that it will allow for non-float dtypes, which significantly decreases the
disk (and thus bandwidth) requirement. Metadata for each cutout source is also
saved in a separate FITS extension

The CookieCutter format was designed by Eric Huff & Spencer Everett of JPL
'''

import numpy as np
from numpy.lib import recfunctions as rf
from pathlib import Path
from glob import glob
import fitsio
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import galsim as gs
import astropy.units as u
from astropy.table import Table, vstack
import glob
from time import time
import copy

from superbit_lensing import utils
from .config import CookieCutterConfig

import ipdb

class CookieCutter(object):
    '''
    The CookieCutter takes a catalog of detected sources and the set of
    images they come from and saves only the cutouts of each source
    realization along with some minimal metadata needed to re-construct
    the original images where there is overlap with the stamps. It can
    also store an additional stamp that holds per-pixel bitmap and
    segmentation information. While it can be used in general, it was
    designed specifically as an output format for the SuperBIT onboard
    analysis (OBA)

    It is effectively a lightweight replacement for the Multi-Epoch Data
    Structure format: https://github.com/esheldon/meds

    The CookieCutter format can be initialized from either a config file
    specifying the needed catalog and image filepath information, or from
    a previously-constructed CookieCutter output file (which is FITS format
    with additional structure)

    The current structure is as follows, for each image index:

    EXT_{INDEX}: IMAGE{EXT}
    EXT_{INDEX+1}: MASK{EXT}
    ...
    EXT_{-1}: META

    The final extension is the metadata table, which has a row for each
    obj stamp instance (not just each obj!)
    '''

    def __init__(self, cookiecutter_file=None, config=None, logprint=None):
        '''
        cookiecutter_file: str, Path
            Initialize a cookiecutter object from an already-existing
            CookieCutter file
        config: str, Path, dict
            A filepath to a config file (or its corresponding dictionary) that
            specifies which files and catalogs to use to build the CookieCutter
            object

        Only one of these two arguments above can be specified, of course.

        logprint: utils.LogPrint
            A LogPrint instance for simultaneous logging & printing

        ------

        If the catalog provided does not have a box size field, this will
        calculate one on the fly. You can change how this is done by re-
        assigning the compute_boxsize() method, an attribute of
        this class, which expects a single argument (catalog, a numpy structured
        array) and returns an integer scalar that will be used as the cutout
        box size.

        NOTE: In the constructor we just parse the inputs. We don't load anything
        until we're told to
        '''

        if config is not None:
            utils.check_type('config', config, (str, Path, dict))

            if not isinstance(config, dict):
                config = utils.read_yaml(str(config))

            # also sets defaults for optional params
            self.config = CookieCutterConfig(config)
        else:
            self.config = None

        if (cookiecutter_file is None) and (config is not None):
            # Will initialize from config instead during go()
            self._cookiecutter_file = None

            # Decide where to put the output.
            outdir = config['output']['dir']
            outfile = config['output']['filename']
            if outdir is not None:
                self.outfile = Path(outdir) / outfile
            else:
                self.outfile = Path(outfile)

        elif (cookiecutter_file is not None) and (config is None):
            # Initialize from file
            utils.check_type(
                'cookiecutter_file', cookiecutter_file, (str, Path)
                )
            self._cookiecutter_file = Path(cookiecutter_file)
            self.outfile = self._cookiecutter_file

        else:
            raise ValueError('You should pass only a cookie cutout file or a ' +
                             'config, not both.')

        # metadata for the cutouts; will get populated later
        self._meta = None

        self.wcs_origin = 0

        # can keep a single image array cached in memory at once. Useful for
        # speeding up the image reconstruction process
        self._cached_image = None
        self._cached_extname = None

        if logprint is None:
            logprint = print
        else:
            utils.check_type('logprint', logprint, utils.LogPrint)
        self.logprint = logprint

        return

    def initialize(self):
        '''
        Initalize CookieCutter from either a config or file. Actually
        loads the catalog or stamp data into memory
        '''

        config = self.config
        cc_file = self._cookiecutter_file

        # Parse the config file.
        # Decide if we're initializing a cookie cutter object from an existing
        # cookie cutout file or if we're building one from scratch.
        if (cc_file is None) and (config is not None):
            # Initialize from config
            input_dir = config['input']['dir']

            catalog_file = Path(config['input']['catalog'])

            if input_dir is not None:
                catalog_file = input_dir / catalog_file

            self.input_dir = input_dir

            ext = config['input']['catalog_ext']
            catalog = fitsio.read(str(catalog_file), ext=ext)

            self.catalog_file = catalog_file
            self.catalog = catalog

            self.wcs_type = config['input']['wcs_type']

            # if we are making a center stamp, stick it at the end of the cat
            self.make_center_stamp = config['output']['make_center_stamp']
            self.center_stamp_index = None
            self.center_stamp_id = None
            # sometimes more useful to have this (last entry)
            self.center_stamp_rel_index = -1

            self._fits = None

        else:
            self._fits = fitsio.FITS(cc_file, 'r')
            return

        return

    def go(self):
        '''
        (1) Initialize the CookieCutter & load catalog or stamps into memory
        (2) Determine the boxsize for each stamp, if needed
        (3) Construct the CookieCutter format & populate with obj stamps
        '''

        self.initialize()

        # don't need to construct CookieCutter format if loading from a file
        if self._fits is not None:
            return

        self.setup_boxsizes()

        # can't be done until now as we want to allow for the center boxsize
        # to be of arbitrary size
        if self.make_center_stamp is True:
            self.add_center_stamp_to_cat()

        # The catalog read, now let's get the image information.
        self.register_images()

        return

    def setup_boxsizes(self):
        '''
        Determine if the input catalog needs a `boxsize` column, and if so
        build from defaults
        '''

        config = self.config
        catalog = self.catalog

        # Use a method attached to the class, so that the user can redefine it.
        # Stick here to the boxsize name provided.
        if config['input']['boxsize_tag'] not in catalog.dtype.names:
            self.add_boxsize_to_cat(catalog)

        return

    def add_boxsize_to_cat(self, catalog):
        # Use the numpy recfunctions library, because I'm a geezer.
        boxsizes = self._compute_boxsize(catalog)
        self.catalog = rf.append_fields(
            catalog, 'boxsize', boxsizes, dtypes=['i4'], usemask=False
            )
        self.config['input']['boxsize'] = 'boxsize'

        return

    def _compute_boxsize(self, catalog, min_size=8, max_size=64):
        '''
        This is a crude rule of thumb -- smallest power of two
        that encloses a quadrature sum of 8 pixels and 4x the flux radius.
        You probably have a better idea. Can override by providing your own
        `boxsize` field

        catalog: np.recarray
            The catalog to estimate the box size from
        min_size: int
            The minimum box size (i.e. edge length)
        max_size: int
            The maximum box size (i.e. edge length)
        '''

        # Assume we have a FLUX_RADIUS in the catalog
        # NOTE: have to use int32 as the square of the boxsize
        # is often computed; can cause overflow errors if users
        # are not careful
        radius = 2**(
            np.ceil(np.log2(np.sqrt( 8**2 + (4*catalog['FLUX_RADIUS'])**2)))
            ).astype('i4')

        radius[radius < min_size] = min_size
        radius[radius > max_size] = max_size

        return radius

    def add_center_stamp_to_cat(self):
        '''
        If desired, add an artificial detection at the target center of
        arbitrary size
        '''

        center_size = self.config['output']['center_stamp_size']
        center_ra, center_dec = self.config['output']['center_stamp_pos']

        utils.check_type('center_size', center_size, int)

        for pos in [center_ra, center_dec]:
            utils.check_type('center_pos', pos, (int, float))
            if pos is None:
                raise ValueError('Must set the center_pos if make_center_stamp' +
                                 ' is True')

        ra_tag = self.config['input']['ra_tag']
        dec_tag = self.config['input']['dec_tag']
        id_tag = self.config['input']['id_tag']
        center_cat = Table()
        center_cat[ra_tag] = [center_ra]
        center_cat[dec_tag] = [center_dec]
        center_cat[id_tag] = int(np.max(self.catalog[id_tag])) + 1
        center_cat['boxsize'] = center_size

        # NOTE: most cols will be empty, but that's fine
        self.catalog = vstack([
            Table(self.catalog), center_cat
            ]).as_array()

        self.center_stamp_index = len(self.catalog) - 1
        self.center_stamp_id = center_cat[id_tag][0]

        return

    def register_images(self, progress=1000, print_skips=False):
        '''
        Use the config to initialize the CookieCutter structure and save obj
        stamps from each input file to a corrsponding set of FITS extensions

        For current format structure, see class docstring

        Basic procedure:

        We get a list of base (SCI image) & supporting (weight, mask, etc.)
        images from the config file, along with a catalog (including RA/DEC)
        that defines source positions where we define the stamp cutouts to
        extract. If a `boxsize` field is not present, then we make a guess
        based off of the defaults in _compute_boxsize() (expects FLUX_RADIUS
        for now)

        For each image, find all the sources from the catalog contained in
        the image:
          -- read in the WCS
          -- compute the pixel coordinantes in this image for this source.
          -- if the source is inside the image, make a cutout.
          -- check for the presence of mask or weight information;
             if they exist, make cutouts of these as well.

        To maximize disk efficiency, we determine the required memory
        allocation by pre-looping through all sources and exclude any
        that have no overlap in the image (which also excludes an entry in
        the metadata table for that image)

        progress: int
            The number of objects stamps to write to the CookieCutter before
            printing out the progress update, if desired
        print_skips: bool
            Set to print out each time an obj is skipped due to no overlap
        '''

        utils.check_type('progress', progress, int)

        config = self.config

        images = [
            config['images'][image] for image in config['images'].keys()
            ]

        # need Table format for the loop over objs to work correctly below
        # catalog = Table(self.catalog)
        catalog = self.catalog
        Nsources = len(catalog)

        id_tag = config['input']['id_tag']
        ra_tag = config['input']['ra_tag']
        dec_tag = config['input']['dec_tag']
        boxsize_tag = config['input']['boxsize_tag']

        # it is *much* faster to access a ndarray than the row of a recarray
        ra = catalog[ra_tag]
        dec = catalog[dec_tag]
        boxsizes = catalog[boxsize_tag]
        obj_ids = catalog[id_tag]

        ra_unit = u.Unit(config['input']['ra_unit'])
        dec_unit = u.Unit(config['input']['dec_unit'])

        sci_dtype = config['output']['sci_dtype']
        msk_dtype = config['output']['msk_dtype']

        seg_type = config['segmentation']['type']

        # NOTE: We consider the center stamp here as we will subtract any
        # overlapping stamps from it, making the final large stamp more
        # compressable
        make_center_stamp = config['output']['make_center_stamp']
        center_stamp_size = config['output']['center_stamp_size']
        center_stamp_indx = self.center_stamp_index
        center_stamp_id = self.center_stamp_id

        overwrite = config['output']['overwrite']

        outfile = self.outfile
        if outfile.is_file():
            if overwrite is True:
                print(f'{str(outfile)} exist')
                print('Deleting as overwrite is True...')
                outfile.unlink()
            else:
                raise OSError(f'{outfile} already exists and overwrite is False!')

        meta_dtype = [
            ('object_id', 'u4'),
            ('xcen', float),
            ('ycen', float),
            ('start_index', 'u4'),
            ('end_index', 'u4'),
            ('sky_bkg', float),
            ('sky_var', float),
            ('img_ext', 'u2')
            ]

        # this will hold the object metadata for each cutout
        # NOTE: this means that there is Nexp rows for each object, where
        # Nexp is the number of exposures that the object stamp has at least one
        # interesecting pixel in the image
        meta = np.zeros(
            Nsources * len(images),
            dtype=meta_dtype
            )

        # this keeps track of the *used* meta rows, as the actual number
        # will be smaller due to objects not showing up in every image
        meta_size = 0

        total_start = time()

        info_index = 0
        with fitsio.FITS(outfile, 'rw') as fits:

            Nimages = len(images)
            for image_index, image in enumerate(images):
                im_file = image['image_file']
                im_name = Path(im_file).name # not always the same
                self.logprint(f'Starting image {im_name}; {image_index+1} of ' +
                              f'{Nimages}')

                start = time()

                # handles all input image file parsing
                # NOTE: sensible defaults are set in config.py for fields
                # that are not set explicitly
                imageObj = ImageLocator(
                    image_file=image['image_file'],
                    image_ext=image['image_ext'],
                    weight_file=image['weight_file'],
                    weight_ext=image['weight_ext'],
                    mask_file=image['mask_file'],
                    mask_ext=image['mask_ext'],
                    background_file=image['background_file'],
                    background_ext=image['background_ext'],
                    skyvar_file=image['skyvar_file'],
                    skyvar_ext=image['skyvar_ext'],
                    segmentation_file=image['segmentation_file'],
                    segmentation_ext=image['segmentation_ext'],
                    input_dir=self.input_dir
                    )

                # TODO: Something equivalent for mask would be nice,
                # but more complicated as we are combining multiple ext's
                # into the mask
                if sci_dtype is None:
                    # use the native dtype
                    sci_dtype = imageObj.image[0,0].dtype
                # else:
                #     if sci_dtype != this_dtype:
                #         im_file = imageObj._image_file
                #         raise ValueError(f'Image {im_file} dtype of {this_dtype}' +
                #                          f' is not consistent with {sci_dtype}!')

                image_wcs = imageObj.get_wcs(
                    'image', wcs_type=self.wcs_type
                    )

                image_hdr = remove_essential_FITS_keys(
                    imageObj.image.read_header()
                    )

                # if making the segmask, we'll need this later
                if imageObj.segmentation is not None:
                    # NOTE: We use ext0 even though the segmentation map is in a
                    # different coadd ext as the WCS typically lives in ext0
                    seg_wcs = imageObj.get_wcs(
                        'segmentation', ext=0, wcs_type=self.wcs_type
                        )

                    # pre-compute the image positions of the sources
                    # NOTE: This is *much* faster than individual WCS evals of
                    # SkyCoords in a loop!
                    seg_x, seg_y = self._compute_image_pos(
                        ra, dec, seg_wcs, wcs_type=self.wcs_type
                        )

                    seg_shape = imageObj.segmentation.get_info()['dims']

                # place the file path info here
                image_hdr.add_record({
                    'name': 'IMG_FILE',
                    'value': str(Path(image['image_file']).name),
                    'comment': 'The image that has been cookie-cut'
                })
                image_hdr.add_record({
                    'name': 'IMG_EXT',
                    'value': image['image_ext'],
                    'comment': 'The extension of the cut image'
                })

                # NOTE: In principle, a "center" (but really "target") stamp
                # could fall off of an image with a large enough dither, but
                # let's ignore that for now (the slicing takes care of this)
                image_hdr.add_record({
                    'name': 'CEN_STMP',
                    'value': make_center_stamp,
                    'comment': 'Is there a center stamp'
                })

                if make_center_stamp is True:
                    image_hdr.add_record({
                        'name': 'CEN_ID',
                        'value': center_stamp_id,
                        'comment': 'Object ID of the center stamp'
                        })

                # if some checkimages are passed, save some statistics
                # to the header
                if imageObj.weight is not None:
                    wgt = imageObj.weight
                    image_hdr['wgt_mu'] = np.mean(wgt[:,:])
                    image_hdr['wgt_std'] = np.std(wgt[:,:])
                    image_hdr['wgt_med'] = np.median(wgt[:,:])
                if imageObj.background is not None:
                    bkg = imageObj.background
                    image_hdr['bkg_mu'] = np.mean(bkg[:,:])
                    image_hdr['bkg_std'] = np.std(bkg[:,:])
                    image_hdr['bkg_med'] = np.median(bkg[:,:])
                if imageObj.skyvar is not None:
                    skyvar = imageObj.skyvar
                    image_hdr['skyvar_mu'] = np.mean(skyvar[:,:])
                    image_hdr['skyvar_std'] = np.std(skyvar[:,:])
                    image_hdr['skyvar_med'] = np.median(skyvar[:,:])

                image_shape = imageObj.image.get_info()['dims']

                # We know in advance how many cutout pixels we'll need to store
                self.logprint(f'Determining memory requirement')
                Npix, slice_info, skip_list = self._compute_slice_info(
                    imageObj.image,
                    catalog,
                    wcs=image_wcs,
                    progress=progress,
                    )

                # NOTE: This was the old way, which unnecessarily allocates
                # memory for stamps w/ no overlap in the given image
                pix_percent = 100. * Npix / np.sum(boxsizes**2)
                im_percent = 100 * Npix / (image_shape[0] * image_shape[1])
                self.logprint(f'image {im_name} needs {Npix} pixels; ' +
                              f'{pix_percent:.2f}% of max stamp pixels; ' +
                              f'{im_percent:.2f}% of full image')

                # one dimension for data, one dimension for composite MASK+SEG
                science_image_dimensions = (1, Npix)
                mask_image_dimensions = (1, Npix)

                # Store each image's cutouts in a new extension.
                # NOTE: Because the image and background have different datatypes
                # to the mask, we need two different extensions.
                fits.create_image_hdu(
                    img=None,
                    dtype=sci_dtype,
                    dims=science_image_dimensions,
                    extname=f'IMAGE{image_index}',
                    header=image_hdr
                    )

                fits.create_image_hdu(
                    img=None,
                    dtype=msk_dtype,
                    dims=mask_image_dimensions,
                    extname=f'MASK{image_index}',
                    )

                # the arrays we will store the cutouts in
                sci_array = np.zeros(
                    science_image_dimensions, dtype=sci_dtype
                    )
                mask_array = np.zeros(
                    mask_image_dimensions, dtype=msk_dtype
                    )

                if make_center_stamp is True:
                    # load center stamp into memory so that we can subtract
                    # overlapping stamps from it (more compressable)
                    icen, cen_pos, cen_im_slice, cen_cutout_slice, cen_size = \
                        slice_info[self.center_stamp_rel_index]

                    center_cutout = imageObj.image[cen_im_slice].astype(sci_dtype)
                    center_shape = center_cutout.shape

                    # this is the "origin" of the center cutout, needed for
                    # determining relative positions in the transformed frame
                    cen_origin = [
                        cen_im_slice[0].start,
                        cen_im_slice[1].start
                    ]

                pixels_written = 0
                hdr_written = False
                for indx in range(Nsources):
                    if indx % progress == 0:
                        self.logprint(f'{indx} of {Nsources}')

                    if indx in skip_list:
                        # if there is no overlap in the image, skip this obj
                        if print_skips is True:
                            self.logprint(f'Object {indx} has no overlap in ' +
                                          f'image {im_name}; skipping')
                        continue

                    boxsize = boxsizes[indx]
                    obj_id = obj_ids[indx]

                    iobj, obj_pos, image_slice, cutout_slice, cutout_size =\
                        slice_info.pop(0)
                    assert iobj == indx

                    cutout_shape = (boxsize, boxsize)

                    image_cutout = np.zeros(cutout_shape, dtype=sci_dtype)

                    if indx == center_stamp_indx:
                        # in this case, *replace* the full cutout with the
                        # stamp-subtracted one (more compressable)
                        sci_cutout = center_cutout
                    else:
                        sci_cutout = imageObj.image_array[image_slice].astype(sci_dtype)

                    image_cutout[cutout_slice] = sci_cutout
                    science_output = image_cutout.flatten()

                    if imageObj.background is not None:
                        sky_cutout = np.zeros(cutout_shape)
                        sky_cutout[cutout_slice] = imageObj.background_array[image_slice]
                        # NOTE: While the median is most robust here, this is already the
                        # the slowly-varying (course) sky bkg, & medians are very
                        # expensive
                        sky_bkg = np.mean(sky_cutout)
                    else:
                        sky_bkg = None

                    if imageObj.weight is not None:
                        weight_cutout = np.zeros(cutout_shape)
                        weight_cutout[cutout_slice] = imageObj.weight_array[image_slice]
                        weight = np.mean(weight_cutout)
                    else:
                        weight = None

                    if imageObj.skyvar is not None:
                        skyvar_cutout = np.zeros(cutout_shape)
                        skyvar_cutout[cutout_slice] = imageObj.skyvar_array[image_slice]
                        sky_var = np.mean(sky_cutout)
                    elif weight is not None:
                        # for OBA at least, this is true
                        safe_weight = weight_cutout.copy()
                        safe_weight[safe_weight == 0] = np.inf
                        sky_var = np.mean(1. / safe_weight)
                    else:
                        sky_var = None

                    # TODO: can generalize this post SuperBIT OBA
                    # we treat the mask differently, as it is required
                    mask_cutout = np.zeros(cutout_shape)
                    if imageObj.mask is not None:
                        mask_cutout[cutout_slice] = imageObj.mask_array[image_slice]

                    if imageObj.segmentation is not None:
                        # TODO: It would be nice to generalize this, but it is a
                        # common case that the segmentation map comes from a
                        # different image than the rest (coadd vs. single-epoch)
                        # NOTE: the segmentation WCS lives in a different ext as
                        # it is coming from the detection coadd

                        seg_obj_pos = [seg_x[indx], seg_y[indx]]
                        seg_out = self._compute_obj_slices(
                            seg_shape, seg_obj_pos, boxsize
                            )

                        seg_slice, seg_cutout_slice, seg_cutout_size = seg_out
                        seg_cutout = np.zeros(cutout_shape)
                        seg_cutout[seg_cutout_slice] = imageObj.segmentation_array[seg_slice]
                    else:
                        seg_cutout = None

                    if seg_cutout is not None:
                        mask_output = self._combine_mask_and_seg(
                            obj_id, mask_cutout, seg_cutout, seg_type=seg_type
                            )
                    else:
                        mask_output = mask_cutout

                    # shouldn't happen, but just in case:
                    msk_max_val = int(2**(8 * np.dtype(msk_dtype).itemsize))
                    if (mask_output > msk_max_val).any():
                        raise ValueError('The combined segmask has values '
                                         f'above {msk_max_val}, which is the '
                                         f'maximum value for a set msk_dtype '
                                         f'of {msk_dtype}!')
                    mask_output = mask_output.astype(msk_dtype).flatten()

                    # This is how we look up object positions to read later.
                    meta[info_index]['object_id'] = obj_id
                    meta[info_index]['xcen'] = obj_pos[0]
                    meta[info_index]['ycen'] = obj_pos[1]
                    meta[info_index]['img_ext'] = image_index

                    if sky_bkg is not None:
                        meta[info_index]['sky_bkg'] = sky_bkg
                    else:
                        meta[info_index]['sky_bkg'] = -1
                    if sky_var is not None:
                        meta[info_index]['sky_var'] = sky_var
                    else:
                        meta[info_index]['sky_var'] = -1

                    indx_start = pixels_written
                    indx_end = pixels_written + cutout_size

                    sci_array[0, indx_start:indx_end] = science_output
                    mask_array[0, indx_start:indx_end] = mask_output

                    meta[info_index]['start_index'] = indx_start
                    meta[info_index]['end_index'] = indx_end

                    pixels_written += cutout_size

                    # NOTE: Due to some strange fitsio design choices, the image
                    # headers have actually *not* been written yet! So do it
                    # explicitly once
                    if hdr_written is False:
                        fits[f'IMAGE{image_index}'].write_keys(image_hdr)
                        hdr_written = True

                    # Subtract from the center stamp, if possible
                    if make_center_stamp is True:
                        try:
                            relative_pos = list(
                                np.array(obj_pos) - np.array(cen_origin)
                                )
                            overlap_slice, cutout_slice = intersecting_slices(
                                center_shape, cutout_shape, relative_pos
                            )
                            center_cutout[overlap_slice] = 0
                        except NoOverlapError:
                            pass

                    meta_size += 1
                    info_index += 1

                fits[f'IMAGE{image_index}'].write(sci_array)
                fits[f'MASK{image_index}'].write(mask_array)

                end = time()
                dT = end - start
                self.logprint(
                    f'Image {image_index+1} stamp writing time: {dT:.1f} s'
                    )

            total_end = time()
            total_dT = total_end - total_start
            self.logprint(f'Total stamp writing time: {total_dT:.1f}')
            self.logprint(f'Writing time per image: {total_dT/Nimages:.1f} s')

            # strip unused rows of meta
            meta = meta[0:meta_size]

            start = time()

            # NOTE: while we wanted this to be ext1, there are issues that
            # make it easier for it to be -1
            fits.create_table_hdu(data=meta, extname='META')
            fits['META'].write(meta)

            end = time()
            dT = end - start
            self.logprint(f'Writing time for metadata: {dT:.1f} s')

        # Finally, populate the object info table if we need it for later.
        self._meta = meta

        return

    def _compute_slice_info(self, image, catalog, wcs=None, progress=1000):
        '''
        Compute all of the needed slice information per object for this image,
        including:

        - The number of cutout pixels needed to be allocated; Npix
        - The slice information for objects *included* in the image:
          - Object ID
          - Object position (X,Y)
          - Image slice
          - Cutout slice
          - Cutout size

        image: fitsio.hdu.image.ImageHDU
            A FITS Image HDU object
        catalog: np.recarray
            The catalog of detected sources and their properties,
            namely `boxsize`
        image_pos: 2-tuple of floats
            A tuple of (x, y)
        wcs: astropy.WCS
            The WCS of the image
        progress: int
            The number of objects stamps to write to the CookieCutter before
            printing out the progress update, if desired

        returns:

        Npix: int
            The number of cutout pixels that need to be allocated for
            the given image extension
        slice_info: tuple
            A tuple of slice info for both image and cutout (see above)
        skip_list: list of int's
            A list of obj indices that should be skipped due to no overlap
        '''

        Nsources = len(catalog)

        image_shape = image.get_info()['dims']
        boxsize_tag = self.config['input']['boxsize_tag']

        if wcs is None:
            # best guess
            wcs = self.get_wcs(
                'image', ext=0, wcs_type=self.wcs_type
                )

        # pre-compute the image positions of the sources
        x, y = self._compute_image_pos(
            catalog['ALPHAWIN_J2000'],
            catalog['DELTAWIN_J2000'],
            wcs,
            wcs_type=self.wcs_type
            )

        boxsizes = catalog[boxsize_tag]

        Npix = 0

        skip_list = []
        slice_info = []
        for indx in range(len(catalog)):
            try:
                if indx % progress == 0:
                    self.logprint(f'{indx} of {Nsources}')

                boxsize = boxsizes[indx]
                obj_pos = [x[indx], y[indx]]

                slices = self._compute_obj_slices(
                    image_shape, obj_pos, boxsize
                    )
                image_slice, cutout_slice, cutout_size = slices

                # NOTE: This fails if exactly 1 slice is empty unless we load
                # the full array into memory first. This is an issue we have now
                # flagged w/ fitiso:
                # https://github.com/esheldon/fitsio/issues/359
                # handle the slice(0,0) edge case
                skip = False
                for sl in image_slice:
                    if sl.stop == sl.start:
                        skip_list.append(indx)
                        skip = True
                for sl in cutout_slice:
                    if sl.stop == sl.start:
                        # only add once
                        if skip is False:
                            skip_list.append(indx)
                            skip = True

                if skip is True:
                    continue

                Npix += cutout_size

                slice_info.append(
                    (indx,
                     obj_pos,
                     image_slice,
                     cutout_slice,
                     cutout_size,
                     )
                )

            except NoOverlapError:
                # if there is no overlap in the image, do not add the stamp
                # to the allocated memory for this extension
                skip_list.append(indx)
                continue

        # for speedups
        skip_list = np.array(skip_list)

        return Npix, slice_info, skip_list

    def _compute_image_pos(self, ra, dec, wcs, wcs_type='astropy'):
        '''
        Compute the position of an object in pixel coordinates
        given an image with a registered WCS

        NOTE: This method is much faster than the public method, as it
        is vectorized on low-level datatypes

        ra: np.ndarray
            A vector of right ascension positions
        dec: np.ndarray
            A vector of declination positions
        wcs: astropy.WCS OR galsim.FitsWCS
            A WCS object, from either astropy or GalSim
        wcs_type: str
            The name of the WCS type. Must be one of 'astropy'
            or 'galsim'

        returns:
        x: np.ndarray
            The x position in image coords
        y: np.ndarray
            The y position in image coords
        '''

        _allowed_wcs = ['astropy', 'galsim']

        if wcs_type == 'astropy':
            # NOTE: flipped due to FITS vs numpy conventions!
            y, x = wcs.all_world2pix(
                ra, dec, self.wcs_origin, ra_dec_order=True
                )
        elif wcs_type == 'galsim':
            # NOTE: have to do the slow way...
            x = np.zeros_like(ra)
            y = np.zeros_like(dec)
            i = 0
            for r, d in zip(ra, dec):
                sc = gs.CelestialCoord(
                    ra=r*gs.degrees, dec=d*gs.degrees
                    )
                im_pos = wcs.toImage(sc)

                # NOTE: flipped due to FITS vs numpy conventions!
                x[i] = im_pos.y
                y[i] = im_pos.x
                i += 1
        else:
            raise ValueError('wcs_type must be one of {_allowed_wcs}!')

        return x, y

    def compute_image_pos(self, image, obj, wcs=None):
        '''
        Compute the position of an object in pixel coordinates
        given an image with a registered WCS

        NOTE: This method is quite slow, but uses the typical high-level
        astropy WCS interface

        image: fitsio.hdu.image.ImageHDU
            A FITS Image HDU object
        obj: astropy.table.Row
            A row of an astropy table
        wcs: astropy.WCS
            An astropy WCS instance, if you want to use one other tahn
            the SCI image (such as the coadd segmap)
        '''

        id_tag = self.config['input']['id_tag']
        ra_tag = self.config['input']['ra_tag']
        dec_tag = self.config['input']['dec_tag']

        ra_unit = u.Unit(self.config['input']['ra_unit'])
        dec_unit = u.Unit(self.config['input']['dec_unit'])

        coord = SkyCoord(
            ra=obj[ra_tag]*ra_unit, dec=obj[dec_tag]*dec_unit
            )

        if wcs is None:
            # default behavior is to grab it from the passed image header
            wcs = WCS(image.read_header())

        image_shape = image.get_info()['dims']

        # x, y = wcs.world_to_pixel(coord)
        x, y = wcs.all_world2pix(
            coord.ra.value, coord.dec.value, 0
            )
        object_pos_in_image = [x.item(), y.item()]

        # NOTE: reversed as numpy arrays have opposite convention!
        object_pos_in_image_array = object_pos_in_image[-1::-1]

        return object_pos_in_image_array

    def _compute_obj_slices(self, image_shape, object_pos_in_image_array,
                            boxsize):
        '''
        Compute the image & cutout slices corrresponding to the given obj

        image_shape: tuple, list
            The shape of the image we are grabbing cutouts from
        obj_pos_in_image_array: np.recarray row, astropy.Table row
            The object being considered
        boxsize:

        returns: (image_slice, cutout_slice, cutout_size)
            A tuple of the slices needed to correctly grab the object
            cutout in the original image and the cutout box respectively,
            as well as the total cutout size in pixels
        '''

        cutout_shape = (boxsize, boxsize)

        image_slice, cutout_slice = intersecting_slices(
            image_shape, cutout_shape, object_pos_in_image_array
        )

        cutout_size = boxsize**2

        return image_slice, cutout_slice, cutout_size

    def _get_center_slice(self, wcs):
        '''
        Compute the slice of the center stamp, given an image's WCS

        wcs: astropy.WCS
            A WCS instance for a given image
        '''

        x0, y0 = wcs.wcs.crpix

        xsize, ysize = self.config['output']['center_stamp_size']

        dx = xsize / 2.
        dy = ysize / 2.

        # round conservatively, if needed
        xmin = int(np.floor(x0-dx))
        xmax = int(np.ceil(x0+dx))
        ymin = int(np.floor(y0-dy))
        ymax = int(np.ceil(y0+dy))
        xslice = slice(xmin, xmax)
        yslice = slice(ymin, ymax)

        # NOTE: flipped due to FITS vs numpy conventions
        center_slice = (yslice, xslice)

        return center_slice

    def _combine_mask_and_seg(self, obj_id, mask, seg, seg_type='minimal'):
        '''
        Combine the mask and segmentation maps into an efficient single-map
        representation

        obj_id: int
            The cutout object's ID (should be consistent w/ segmentation val)
        mask: np.ndarray (shape = cutout_shape)
            The mask cutout for a given source
        seg: np.ndarray (shape = cutout_shape)
            The segmentation cutout for a given source
        seg_type: str
            The type of segmentation map to use. Currently only one registered
            type:
            - minimal: Convert normal segmentation values (pix_val=object_id)
              into the minimal set that can be reconstructed later on:
              - 0: unassigned sky
              - 1: this object (of the current stamp)
              - 2: neighbor (contained in other cutouts)
        '''

        _allowed = ['minimal']

        # these are the input SExtractor SEGMENTATION values
        _sky = 0
        _obj = obj_id

        if seg_type not in _allowed:
            raise ValueError(f'{seg_type} is not a registered segmentation '
                             f'type! Must be one of the following: {_allowed}')

        seg_obj = self.config['segmentation']['obj']
        seg_neighbor = self.config['segmentation']['neighbor']

        for name, val in dict(
                zip(
                    ['obj', 'neighbor'],
                    [seg_obj, seg_neighbor]
                    )
                ).items():
            if val in mask:
                raise ValueError(f'{name} of {val} is already present in '
                                 'the mask!')

        combined_mask = mask.copy()

        if seg_type == 'minimal':
            # NOTE: used to keep track of sky here as well, but not needed
            combined_mask[seg == _obj] += seg_obj
            combined_mask[(seg != _obj) & (seg != _sky)] = seg_neighbor

        return combined_mask

    @property
    def meta(self):
        if self._meta is None:
            # should only happen if was instantiated with a file
            self._meta = self._fits['META'].read()

        return self._meta

    def clip_meta(self, start, end):
        '''
        Clip the metadata table to a given start & end row

        start: int
            The starting index for the new metadata table
        end: int
            The ending index for the new metadata table
        '''

        utils.check_type('start', start, int)
        utils.check_type('end', end, int)

        if end < start:
            raise ValueError(f'Cannot clip the metadata table given the ' +
                             f'bounds ({start}, {end})!')

        self._meta = self._meta[start:end]

        return

    # NOTE: this is for backwards compatibility. Base def is now meta
    @property
    def objectInfoTable(self):
        return self.meta

    def get_cutout(self, objectid, extnumber=None, filename=None,
                   cutout_type='IMAGE', cache_image=False):
        '''
        Request a single cutout of one object from one image.
        Must always provide:
        objectid:  unique object id, as listed in the metadata table
        --------------
        Must always provide ONLY one of either:
          filename:  enough of the name of the file where your
                     requested cutout originates that we can uniquely glob it;
                     a UTC string, for example.
          extnumber: an extension number (corresponding to the order in which
                     this file was stored in the CookieCutter. Note that this
                     does NOT correspond to the actual order of the fits
                     extensions in the raw file.
        ---------------

        Optional:
        cutout_type: str
            Name of the cutout type. Defaults to IMAGE; may also provide MASK
        cache_image: bool
            Set to use and / or cache the current image in memory
        '''

        # Construct the file extension from the requested cutout:
        if (filename is None) and (extnumber is not None):
            extnumber = int(extnumber) # Avoiding needless silliness
            extname = f'{cutout_type}{extnumber}'

        # Look up the row in the table.
        if (filename is not None) and (extnumber is None):
            filematch = [
                filename in thing for thing in self.meta['imagefile']
                ]
            if len(filematch) == 1:
                extnumber = filematch.index(True)
                extname = f'{cutout_type}{extnumber}'
            else:
                raise ValueError(f'the provided file string, {filename}, is ' +
                                 'not unique in the list of provided image files.')

        if (filename is not None) and (extnumber is not None):
            raise ValueError('should provide only the extension number or a ' +
                             'unique fragment of the input filename, but not both')
        if (filename is None) and (extnumber is None):
            raise ValueError('should provide at least one of the extension ' +
                             'number or a unique fragment of the input filename')

        # we should have arrived here with an extension name.
        # Now get the row from the table with the object info.
        # At this point, the extension name should exist.
        row = (self.meta['object_id'] == objectid) &\
            (self.meta['img_ext'] == extnumber)
        if np.sum(row) != 1:
            raise ValueError(f'somehow, the objectindex {objectid} and image ' +
                             f'extension {extnumber} pairing does not exist, ' +
                             'or is not unique in your metadata')
        entry = self.meta[row]

        # are read as arrays
        start = int(entry['start_index'])
        end = int(entry['end_index'])

        if cache_image is True:
            if (self._cached_image is None) or (self._cached_extname != extname):
                self._cached_image = self._fits[extname].read()
                self._cached_extname = extname

            cutout1d = self._cached_image[0,start:end]
        else:
            cutout1d = self._fits[extname][0,start:end]

        # NOTE: cutouts are always square.
        boxsize = np.sqrt(cutout1d.size)
        cutout2d = cutout1d.reshape(
            int(boxsize),
            int(boxsize)
            )

        return cutout2d

    def _get_cutout_by_index(self, obj_indx, cutout_type='IMAGE'):
        '''
        A private, faster version of get_cutout that does no checking and
        access by internal meta index instead of obj_id

        obj_indx: int
            The index of the meta table corresponding to the object cutout
        cutout_type: str
            Name of the cutout type. Defaults to IMAGE; may also provide MASK
        '''

        obj = self.meta[obj_indx]
        start = obj['start_index']
        end = obj['end_index']

        ext_number = obj['img_ext']
        ext_name = f'{cutout_type}{ext_number}'

        cutout1d = self._fits[ext_name][0,start:end]

        # NOTE: cutouts are always square.
        boxsize = int(np.sqrt(cutout1d.size))
        cutout2d = cutout1d.reshape(boxsize, boxsize)

        return cutout2d

    def get_cutouts(self, objectids=None, extnumbers=None, filenames=None,
                     cutoutTypes=['IMAGE', 'MASK']):

        # Can specify all three, will try to find everything that matches at least one.
        raise NotImplementedError('get_cutouts() not yet implemented!')

        return

    def reconstruct_image(self, cutout_type='IMAGE', extnumber=None,
                          filename=None, print_skips=False):
        '''
        cutout_type can by any of 'IMAGE' or 'MASK' (add new types if
        we implement them...)

        Look up the correct image extension.
        Get the old header
        build the correctly-sized and typed 2d array.
        Get the wcs.
        Read in the catalog
        for each catalog entry:
          compute image coordinates
          build cutout and image slices.
          add background level + cutout flux to 2d image array
        '''

        if (filename is None) and (extnumber is not None):
            extnumber = int(extnumber) # Avoiding needless silliness
            extname = f'{cutout_type}{extnumber}'

        if (filename is not None) and (extnumber is None):
            filematch = [
                filename in thing for thing in self._meta['imagefile']
                ]

            if len(filematch) == 1:
                extnumber = filematch.index(True)
                extname = f'{cutout_type}{extnumber}'
            else:
                raise ValueError(f'the provided file string, {filename}, is ' +
                                 'not unique in the list of provided image files.')

        # At this stage we have the image header name.
        # Reconstruct the original header.
        cc_header = fitsio.read_header(self._cookiecutter_file, ext=extname)
        orig_header = reconstruct_original_FITS_header(cc_header)

        try:
            has_center_stamp = cc_header['CEN_STMP']
        except KeyError:
            # can happen for MASK images
            has_center_stamp = False

        if has_center_stamp is True:
            center_id = cc_header['CEN_ID']

        # NOTE: reversed due to FITS vs numpy conventions
        image_shape = (orig_header['NAXIS2'], orig_header['NAXIS1'])

        if orig_header['BITPIX'] < 0:
            dtype = f'i{orig_header["BITPIX"]//8}'
        else:
            dtype = f'u{orig_header["BITPIX"]//8}'

        new_image = np.zeros(image_shape, dtype=dtype)

        meta_indices = np.where(self.meta['img_ext'] == extnumber)
        meta_in_image = self.meta[meta_indices]
        Nmeta = len(meta_in_image)

        # Now loop over objects in this image.
        for indx, obj in zip(meta_indices[0], meta_in_image):

            # passing the index explicitly saves *lots* of time
            # over many iterations
            cutout = self._get_cutout_by_index(
                indx,
                cutout_type
                )

            # guaranteed to be perfect square, for now
            assert cutout.shape[0] == cutout.shape[1]
            boxsize = cutout.shape[0]

            # NOTE: much faster w/ vectorized operations
            obj_pos = (obj['xcen'], obj['ycen'])

            try:
                slice_info = self._compute_obj_slices(
                    image_shape, obj_pos, boxsize
                    )
                image_slice, cutout_slice, cutout_size = slice_info

                if (has_center_stamp is True) and \
                   (obj['object_id'] == center_id):
                    # the areas with source stamps have been zero'd out
                    new_image[image_slice] += cutout[cutout_slice]
                else:
                    new_image[image_slice] = cutout[cutout_slice]

            except NoOverlapError:
                if print_skips is True:
                    print(f'No overlap for obj {obj["object_id"]}')

        return new_image, orig_header

class ImageLocator(object):
    '''
    Lightweight container & access protocol for all input images associated
    with a CookieCutter ingested image
    '''

    # the list of registered image types
    _allowed_types = [
        'image',
        'weight',
        'mask',
        'background',
        'skyvar',
        'segmentation',
    ]

    def __init__(self, image_file, image_ext=0, weight_file=None,
                 weight_ext=0, mask_file=None, mask_ext=0,
                 background_file=None, background_ext=0,
                 skyvar_file=None, skyvar_ext=0, segmentation_file=None,
                 segmentation_ext=0, input_dir=None):

        # TODO: clean this up when there is time

        if input_dir is not None:
            image_file = Path(input_dir) / Path(image_file)

            if weight_file is not None:
                weight_file = Path(input_dir) / Path(weight_file)
            if mask_file is not None:
                mask_file = Path(input_dir) / Path(mask_file)
            if background_file is not None:
                background_file = Path(input_dir) / Path(background_file)
            if skyvar_file is not None:
                skyvar_file = Path(input_dir) / Path(skyvar_file)
            if segmentation_file is not None:
                segmentation_file = Path(input_dir) / Path(segmentation_file)

        else:
            image_file = Path(image_file)

            if weight_file is not None:
                weight_file = Path(weight_file)
            if mask_file is not None:
                mask_file = Path(mask_file)
            if background_file is not None:
                background_file = Path(background_file)
            if skyvar_file is not None:
                skyvar_file = Path(skyvar_file)
            if segmentation_file is not None:
                segmentation_file = Path(segmentation_file)

        self.input_dir = input_dir

        self._image_file = image_file
        self._image_ext = image_ext
        self._image = None
        self._image_array = None

        self._weight_file = weight_file
        self._weight_ext = weight_ext
        self._weight = None
        self._weight_array = None

        self._mask_file = mask_file
        self._mask_ext = mask_ext
        self._mask = None
        self._mask_array = None

        self._background_file = background_file
        self._background_ext = background_ext
        self._background = None
        self._background_array = None

        self._skyvar_file = skyvar_file
        self._skyvar_ext = skyvar_ext
        self._skyvar = None
        self._skyvar_array = None

        self._segmentation_file = segmentation_file
        self._segmentation_ext = segmentation_ext
        self._segmentation = None
        self._segmentation_array = None

        return

    @property
    def image(self):
        if self._image is None:
            self._image = fitsio.FITS(
                self._image_file, 'r'
                )[self._image_ext]

        return self._image

    @property
    def image_array(self):
        '''
        Same as image(), but loads the *full* image into memory once
        '''
        if self._image_array is None:
            self._image_array = fitsio.FITS(
                self._image_file, 'r'
                )[self._image_ext].read()

        return self._image_array

    @property
    def weight(self):
        if self._weight_file is None:
            return None
        else:
            if self._weight is None:
                self._weight = fitsio.FITS(
                    self._weight_file, 'r'
                    )[self._weight_ext]

            return self._weight

    @property
    def weight_array(self):
        '''
        Same as weight(), but loads the *full* weight into memory once
        '''
        if self._weight_array is None:
            self._weight_array = fitsio.FITS(
                self._weight_file, 'r'
                )[self._weight_ext].read()

        return self._weight_array

    @property
    def mask(self):
        if self._mask_file is None:
            return None
        else:
            if self._mask is None:
                self._mask = fitsio.FITS(
                    self._mask_file, 'r'
                    )[self._mask_ext]

            return self._mask

    @property
    def mask_array(self):
        '''
        Same as mask(), but loads the *full* mask into memory once
        '''
        if self._mask_array is None:
            self._mask_array = fitsio.FITS(
                self._mask_file, 'r'
                )[self._mask_ext].read()

        return self._mask_array

    @property
    def background(self):
        if self._background_file is None:
            return None
        else:
            if self._background is None:
                self._background = fitsio.FITS(
                    self._background_file, 'r'
                    )[self._background_ext]

            return self._background

    @property
    def background_array(self):
        '''
        Same as background(), but loads the *full* background into memory once
        '''
        if self._background_array is None:
            self._background_array = fitsio.FITS(
                self._background_file, 'r'
                )[self._background_ext].read()

        return self._background_array

    @property
    def skyvar(self):
        if self._skyvar_file is None:
            return None
        else:
            if self._skyvar is None:
                self._skyvar = fitsio.FITS(
                    self._background_file, 'r'
                    )[self._skyvar_ext]

            return self._skyvar

    @property
    def skyvar_array(self):
        '''
        Same as skyvar(), but loads the *full* skyvar into memory once
        '''
        if self._skyvar_array is None:
            self._skyvar_array = fitsio.FITS(
                self._skyvar_file, 'r'
                )[self._skyvar_ext].read()

        return self._skyvar_array

    @property
    def segmentation(self):
        if self._segmentation_file is None:
            return None
        else:
            if self._segmentation is None:
                self._segmentation = fitsio.FITS(
                    self._segmentation_file, 'r'
                    )[self._segmentation_ext]

            return self._segmentation

    @property
    def segmentation_array(self):
        '''
        Same as segmentation(), but loads the *full* segmentation into memory once
        '''
        if self._segmentation_array is None:
            self._segmentation_array = fitsio.FITS(
                self._segmentation_file, 'r'
                )[self._segmentation_ext].read()

        return self._segmentation_array

    def get_wcs(self, image_type, ext=0, wcs_type='astropy'):
        '''
        In some cases, the WCS of an image will not live in the same extension as
        the passed image data. In that case, you can request to get the WCS
        from an arbitrary extension for a given image type

        image_type: str
            The name of the image type whose WCS you want to grab
        ext: int
            The image extension where the WCS lives
        wcs_type: str
            The WCS object type to make. Can be one of 'astropy' or 'galsim'
        '''

        utils.check_type('image_type', image_type, str)
        utils.check_type('ext', ext, int)
        utils.check_type('wcs_type', wcs_type, str)

        if image_type not in self._allowed_types:
            raise ValueError(f'{image_type} is not one of the allowed image ' +
                             f'types; must be one of {self._allowed_types}')

        image_file = getattr(self, f'_{image_type}_file')


        _allowed_wcs = ['astropy', 'galsim']
        if wcs_type == 'astropy':
            hdr = fitsio.read_header(image_file, ext=ext)
            wcs = WCS(hdr)
        elif wcs_type == 'galsim':
            wcs = gs.FitsWCS(str(image_file))
        else:
            raise ValueError(f'wcs_type must be one of {_allowed_wcs}!')

        return wcs

#------------------------------------------------------------------------------
# Some helper funcs relevant to CookieCutter

def write_2d_cookiecutter(cc_file, out_file=None, logprint=None,
                          keep_1d_segmask=True):
    '''
    Given a standard cookiecutter file, we can easily make an alternative
    version of the cookiecutter which stores the 2D reconstructed
    images of cutouts instead of the 1D array list of cutouts. While a larger
    file, it *should* compress well with most of the image being 0's, and it has
    the advantage of storing no overlapping pixel information

    cc_file: str, pathlib.Path
        The filepath to the already-written CookieCutter file
    out_file: str, pathlib.Path
        The filepath of the output 2D CookieCutter file (defaults w/ "_2d"
        suffix)
    logprint: utils.LogPrint
        A LogPrint instance for simultaneous logging & printing
    keep_1d_segmask: bool
        Set to keep the seg-mask as the 1d cutout arrays (as the segmask is
        only properly defined on cutouts, not full image)
    '''

    if isinstance(cc_file, str):
        cc_file = Path(cc_file)
    if isinstance(out_file, str):
        out_file = Path(out_file)

    if out_file is None:
        out_file = Path(
            str(cc_file).replace('.fits', '_2d.fits')
            )

    if logprint is not None:
        logprint(f'Writing 2D CookieCutter file to {out_file.name}, ' +
                 f'using {cc_file.name} as input')

    cc = CookieCutter(cookiecutter_file=cc_file, logprint=logprint)
    cc.initialize()

    with fitsio.FITS(out_file, 'rw') as fits:
        try:
            for ext in range(len(cc._fits)):
                image_hdr = cc._fits[ext].read_header()
                ext_name = cc._fits[ext].get_extname()

                # we save 2 CC ext's for each image ext
                img_ext = ext // 2

                if ext_name == f'IMAGE{img_ext}':
                    cutout_type = 'IMAGE'
                elif ext_name == f'MASK{img_ext}':
                    cutout_type = 'MASK'
                elif ext_name == 'META':
                    cutout_type = 'META'
                else:
                    raise ValueError(
                        f'Extension name {ext_name} not recognized!'
                        )

                if (logprint is not None) and (cutout_type == 'IMAGE'):
                    img_file = Path(image_hdr['IMG_FILE']).name
                    logprint(f'Reconstructing image {img_file}')

                if cutout_type == 'IMAGE':
                    # first, reconstruct the original image at the cutouts
                    new_image, orig_hdr = cc.reconstruct_image(
                        cutout_type=cutout_type, extnumber=img_ext
                        )
                elif cutout_type == 'MASK':
                    # NOTE: The CookieCutter segmask is *not* unique on a full
                    # 2D image, only on the stamps! so keep as-is
                    new_image = cc._fits[ext].read()
                    orig_hdr = image_hdr

                if cutout_type != 'META':
                    image_dtype = new_image.dtype
                    image_shape = new_image.shape

                    # first, create the new HDU
                    fits.create_image_hdu(
                        img=new_image,
                        dtype=image_dtype,
                        dims=image_shape,
                        extname=ext_name,
                        header=orig_hdr
                        )

                    # next, write the header
                    fits[ext_name].write_keys(orig_hdr)

                    # finally, write the full 2D image
                    fits[ext_name].write(new_image)

                else:
                    fits.create_table_hdu(
                        data=cc.meta,
                        extname='META',
                        header=image_hdr
                        )

                    fits[ext_name].write_keys(image_hdr)
                    fits[ext_name].write(cc.meta)

        except Exception as e:
            # we do this to cleanup the failed FITS writing
            out_file.unlink()
            raise e

    if logprint is not None:
        logprint('Done!')

    return

def intersecting_slices(big_array_shape, small_array_shape, position):
    '''
    Stripped-down version of astropy ndimage utilities.
    from here:
    https://docs.astropy.org/en/stable/_modules/astropy/nddata/utils.html#Cutout2D

    Parameters:
    -----------

    big_array_shape : tuple of int or int; shape of large array
    small_array_shape: tuple of int or int; shape of small array to be
        cut out
    position: position of the small array center wrt to the big array.
        NOTE: Must be in the same order as big/small_array_shape, but
        this is reverse (i.e. (y,x)) for FITS images!!

    Returns:
    --------
    big_slices : slices such that big_array[big_slices] extracts the desired cutout
      region.
    small_slices : slices such that cutout[small_slices] extracts the non-empty
        pixels in the desired size cutout.

    TODO: Fail gracefully when there's no overlap.
    '''

    min_indices = [
        int(np.ceil(pos - (small_shape / 2.0))) for
        (pos, small_shape) in zip(position, small_array_shape)
        ]

    max_indices = [
        int(np.ceil(pos + (small_shape / 2.0))) for
        (pos, small_shape) in zip(position, small_array_shape)
        ]

    for e_max in max_indices:
        if e_max < 0:
            raise NoOverlapError('Arrays do not overlap')
        for e_min, large_shape in zip(min_indices, big_array_shape):
            if e_min >= large_shape:
                raise NoOverlapError('Arrays do not overlap')

    big_slices = tuple(
        slice(max(0, min_indices), min(large_shape, max_indices)) for
        (min_indices, max_indices, large_shape) in zip(
            min_indices, max_indices, big_array_shape
            )
        )

    small_slices = tuple(
        slice(0, slc.stop - slc.start) for slc in big_slices
        )

    return big_slices, small_slices

def is_slice1_contained_in_slice2(slice1, slice2):
    '''
    Determines if slice1 is *fully* containd in slice2. Mostly useful
    for determining if an object should be considered part of the central
    stamp

    slice1: slice
        The slice object we are testing
    slice2: slice
        The slice object we are comparing to
    '''

    start1, end1 = slice1.start, slice1.stop
    start2, end2 = slice2.start, slice2.stop

    if (start1 < start2) or (stop1 > stop2):
        return False

    return True

class NoOverlapError(ValueError):
    '''
    Raised when determining the overlap of non-overlapping arrays
    in intersecting_slices()
    '''
    pass

def reconstruct_original_FITS_header(inheader):
    '''
    Old keys were stored with "ORIG_" prepended.
    Find all the keys that start with "ORIG_". Strip this off, and
    replace any record in the current header.
    '''

    header = copy.deepcopy(inheader)
    keys = header.keys()

    for ikey in keys:
        if "ORIG_" in ikey:
            new_keyname = ikey.split("ORIG_")[-1]
            # Check to see if the new keyname is in the header.
            if new_keyname in keys:
                header[new_keyname] = header[ikey]
            else:
                new_record = fitsio.FITSRecord(
                    {
                        'name': new_keyname,
                        'value': header[ikey],
                        'comment':''
                    })
                header.add_record(new_record)

    return header

def remove_essential_FITS_keys(header, exclude_keys=None, backup_keys=True):
    '''
    FITS format requries certain essential header keys that we want to remove
    when transfering header information, as they will no longer be true for the
    FITS extensions that we save
    '''

    if exclude_keys is None:
        exclude_keys = ESSENTIAL_FITS_KEYS

    # NOTE: This doesn't fail if one of the keys is not in the header.
    # This is desirable for this application.
    for key in exclude_keys:
        if backup_keys:
            new_keyname = f'ORIG_{key}'
            new_record = fitsio.FITSRecord(
                {
                    'name': new_keyname,
                    'value': header[key],
                    'comment': f'original image value of {key}'
                })
            header.add_record(new_record)

        header.delete(key)

    return header

# Used by remove_essential_fits_keys()
# TODO: determine what to do with EXTEND
ESSENTIAL_FITS_KEYS = ['SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2']#, 'EXTEND']
