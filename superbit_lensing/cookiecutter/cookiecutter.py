import numpy as np
from numpy.lib import recfunctions as rf
from pathlib import Path
from glob import glob
import fitsio
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Table
import glob
from time import time

from superbit_lensing import utils
from .config import CookieCutterConfig

import ipdb

'''
The CookieCutter class is essentially a "lite" version of the Multi-Object
Data Structure (MEDS) format used by the SuperBIT onboard analysis (OBA).
It stores the image cutouts of detected sources for a variety of image types
such as science, weight, mask, segmentation, etc. It is particularly useful in
that it will allow for non-float dtypes, which significantly decreases the
disk (and thus bandwidth) requirement. Metadata for each cutout source is also
saved in a separate FITS extension

The CookieCutter format was designed by Eric Huff of JPL
'''

# TODO:
# - add obj positions to metadata!
# - make inv function to recover original images!

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

class NoOverlapError(ValueError):
    '''Raised when determining the overlap of non-overlapping arrays.'''
    pass

class ImageLocator(object):
    '''
    TODO: Decide whether to set defaults to...
    '''

    def __init__(self, image_file=None, image_ext=None, weight_file=None,
                 weight_ext=None, mask_file=None, mask_ext=None,
                 background_file=None, background_ext=None,
                 skyvar_file=None, skyvar_ext=None, input_dir=None):

        if input_dir is not None:
            image_file = Path(input_dir) / Path(image_file)

            if weight_file is not None:
                weight_file = Path(input_dir) / Path(weight_file)
            if mask_file is not None:
                mask_file = Path(input_dir) / Path(mask_file)
            if background_file is not None:
                background_file = Path(input_dir) / Path(mask_file)

        else:
            image_file = Path(image_file)

            if weight_file is not None:
                weight_file = Path(weight_file)
            if mask_file is not None:
                mask_file = Path(mask_file)
            if background_file is not None:
                background_file = Path(mask_file)

        self._image_file = image_file
        if image_ext is None:
            self._image_ext = 0
        else:
            self._image_ext = image_ext

        self._weight_file = weight_file
        if weight_ext is None:
            self._weight_ext = 0
        else:
            self._weight_ext = weight_ext

        self._mask_file = mask_file
        if mask_ext is None:
            self._mask_ext = 0
        else:
            self._mask_ext = mask_ext

        self._background_file = background_file
        if background_ext is None:
            self._background_ext = 0
        else:
            self._background_ext = background_ext

        self._skyvar_file = skyvar_file
        if skyvar_ext is None:
            self._skyvar_ext = 0
        else:
            self._skyvar_ext = skyvar_ext

    @property
    def image(self):
        return fitsio.FITS(self._image_file, 'r')[self._image_ext]

    @property
    def weight(self):
        if self._weight_file is None:
            return None
        else:
            return fitsio.FITS(self._weight_file, 'r')[self._weight_ext]

    @property
    def mask(self):
        if self._mask_file is None:
            return None
        else:
            return fitsio.FITS(self._mask_file, 'r')[self._mask_ext]

    @property
    def skybkg(self):
        if self._background_file is None:
            return None
        else:
            return fitsio.FITS(self._background_file, 'r')[self._background_ext]

    @property
    def skyvar(self):
        if self._skyvar_file is None:
            return None
        else:
            return fitsio.FITS(self._background_file, 'r')[self._skyvar_ext]

def removeEssentialFITSkeys(header, exclude_keys=None):
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
        header.delete(key)

    return header

class CookieCutter(object):
    '''
    For meting out cutouts to processing functions.
    Lightweight replacement for MEDS.

    What this should do:
     - be initialized from _either_ an existing file, or a config file
       specifying a catalog, a list of images(+ optionally weights/masks)
       and metadata specifying cutout size, etc.
     - TODO:...
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
        assigning the calculateBoxsizeFromCatalog() method, an attribute of
        this class, which expects a single argument (catalog, a numpy structured
        array) and returns an integer scalar that will be used as the cutout
        box size.

        NOTE: In the constructor we just parsethe inputs. We don't load anything
        until we're told to.
        '''

        if config is not None:
            utils.check_type('config', config, (str, Path, dict))

            if not isinstance(config, dict):
                config = utils.read_yaml(str(config))

        # also sets defaults for optional params
        self.config = CookieCutterConfig(config)

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

        if logprint is None:
            logprint = print
        else:
            utils.check_type('logprint', logprint, utils.LogPrint)
            self.logprint = logprint

        return

    # @classmethod
    # def from_file(cls, cookiecutter_file):

        # cookiecutter = CookieCutter()
        # cookiecutter._fits = ...
    #     # return cookiecutter

    def go(self):
        '''
        TODO:...
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

            ext = config['input']['catalog_ext']
            catalog = fitsio.read(str(catalog_file), ext=ext)

            self.catalog_file = catalog_file
            self.catalog = catalog

        else:
            # self._fits = fitsio.FITS(cc_file, 'r')
            # return
            # TODO: check that the above works!
            raise NotImplementedError('Creating from a cookiecutter file ' +
                                      'is not yet implemented!')

        # It's unlikely that the catalog actually has a 'boxsize' field,
        # in which case we need to create one.
        # Use a method attached to the class, so that the user can redefine it.
        # Stick here to the boxsize name provided.
        if config['input']['boxsize_tag'] not in catalog.dtype.names:
            self._updateCatalogWithBoxsizes(catalog)

        # The catalog read, now let's get the image information.
        images = [
            config['images'][image] for image in config['images'].keys()
            ]

        self._createFromImages(
            images,
            input_dir=input_dir
            )

        return

    def _updateCatalogWithBoxsizes(self, catalog):
        # Use the numpy recfunctions library, because I'm a geezer.
        boxsizes = self.calculateBoxsizeFromCatalog(catalog)
        self.catalog = rf.append_fields(
            catalog, 'boxsize', boxsizes, dtypes=['i4'], usemask=False
            )

        return

    def calculateBoxsizeFromCatalog(self, catalog, min_size=8, max_size=128):
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

    def _createFromImages(self, images, input_dir=None):
        '''
        images: list of dict's
            Each entry in the list is a dict containing the image, weight, mask,
            and background file+extension info.
        input_dir: Path
            The path to append to all image filenames, if desired

        TODO: CONFIRM THAT THE OBJECT DATA TABLE IS INITIALIZED WITH SUFFICIENT
        LENGTH TO HOLD THE IMAGE FILENAME.
        '''

        # need Table format for the loop over objs to work correctly below
        catalog = Table(self.catalog)
        Nsources = len(catalog)

        id_tag = self.config['input']['id_tag']
        ra_tag = self.config['input']['ra_tag']
        dec_tag = self.config['input']['dec_tag']
        boxsize_tag = self.config['input']['boxsize_tag']

        ra_unit = u.Unit(self.config['input']['ra_unit'])
        dec_unit = u.Unit(self.config['input']['dec_unit'])

        sci_dtype = self.config['output']['sci_dtype']
        msk_dtype = self.config['output']['msk_dtype']

        overwrite = self.config['output']['overwrite']

        # We get a list of images from the config file.
        # We get a catalog (ra/dec) as an input (either from a catalog file, or as an argument)

        # For each image, find all the sources in that image.
        #   -- read in the WCS
        #   -- compute the pixel coordinantes in this image for this source.
        #   -- if the source is inside the image, make a cutout.
        #   -- check for the presence of mask or weight information;
        #      if they exist, make cutouts of these as well.

        outfile = self.outfile
        if outfile.is_file():
            if overwrite is True:
                print(f'{str(outfile)} exist')
                print('Deleting as overwrite is True...')
                outfile.unlink()
            else:
                raise OSError(f'{outfile} already exists and overwrite is False!')

        object_info_table = np.empty(
            Nsources * len(images),
            dtype=[('object_id', int),
                   ('image_file', 'S64'),
                   ('start_pos', int),
                   ('end_pos', int),
                   ('sky_bkg', float),
                   ('sky_var', float),
                   ('extension', 'u1')])

        info_index = 0

        with fitsio.FITS(outfile, 'rw') as fits:

            Nimages = len(images)
            for image_index, image in enumerate(images):
                im_file = image['image_file']
                self.logprint(f'Starting image {im_file}; {image_index+1} of ' +
                              f'{Nimages}')

                imageObj = ImageLocator(
                    image_file=image['image_file'],
                    image_ext=image['image_ext'],
                    weight_file=image['weight_file'],
                    weight_ext=image['weight_ext'],
                    mask_file=image['mask_file'],
                    mask_ext=image['mask_ext'],
                    input_dir=input_dir
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

                image_wcs = WCS(imageObj.image.read_header())
                imageHDR = removeEssentialFITSkeys(imageObj.image.read_header())

                # place the file path info here
                imageHDR['image_path'] = Path(image['image_file']).parent

                image_shape = imageObj.image.get_info()['dims']

                # We know in advance how many cutout pixels we'll need to store
                # NOTE: Need to cast to int32 or overflow can happen
                npix = np.sum(catalog[boxsize_tag][:]**2)

                # one dimension for data, one dimension for sky.
                science_image_dimensions = (1, npix)
                mask_image_dimensions = (1, npix)

                # Store each image's cutouts in a new extension.
                # Because the image and background have different datatypes to
                # the mask, we need two different extensions.
                fits.create_image_hdu(
                    img=None,
                    dtype=sci_dtype,
                    dims=science_image_dimensions,
                    extname=f'IMAGE{image_index}',
                    header=imageHDR
                    )

                fits.create_image_hdu(
                    img=None,
                    # TODO: make this more flexible!
                    dtype='i1',
                    dims=mask_image_dimensions,
                    extname=f'MASK{image_index}'
                    )

                start = time()

                pixels_written = 0
                for indx, iobj in enumerate(catalog):
                    if indx % 1000 == 0:
                        self.logprint(f'{indx} of {Nsources}')

                    coord = SkyCoord(
                        ra=iobj[ra_tag]*ra_unit, dec=iobj[dec_tag]*dec_unit
                        )

                    # TODO / NOTE: refactor this bit into a function & inv
                    # function, which will be used to re-construct the images
                    #
                    # x and y are, for some reason, needlessly returned as
                    # numpy arrays.
                    x, y = image_wcs.world_to_pixel(coord)
                    object_pos_in_image = [x.item(), y.item()]

                    # NOTE: reversed as numpy arrays have opposite convention!
                    object_pos_in_image_array = object_pos_in_image[-1::-1]

                    boxsize = iobj[boxsize_tag]
                    cutout_shape = [boxsize, boxsize]
                    cutout_pixels = boxsize**2

                    try:
                        image_slice, cutout_slice = intersecting_slices(
                            image_shape, cutout_shape, object_pos_in_image_array
                            )
                    except NoOverlapError as e:
                        # TODO: Figure out how to handle this correctly...
                        pass
                        # self.logprint(f'Stamp {indx} has no overlap; skipping')
                        # continue

                    image_cutout = np.zeros(cutout_shape, dtype=sci_dtype)
                    sci_cutout = imageObj.image[image_slice].astype(sci_dtype)
                    image_cutout[cutout_slice] = sci_cutout
                    science_output = image_cutout.flatten()

                    if indx == 0:
                        fits[f'IMAGE{image_index}'].write(
                            science_output,
                            start=[0, pixels_written],
                            header=imageHDR
                            )
                    else:
                        fits[f'IMAGE{image_index}'].write(
                            science_output,
                            start=[0, pixels_written]
                            )

                    # TODO / QUESTION: should we use mean or median for the following?
                    if imageObj.skybkg is not None:
                        sky_cutout = np.zeros_like(image_cutout)
                        sky_cutout[cutout_slice] = imageObj.skybkg[image_slice]
                        sky_bkg = np.median(sky_cutout)
                    else:
                        sky_bkg = None

                    if imageObj.skyvar is not None:
                        skyvar_cutout = np.zeros_like(image_cutout)
                        skyvar_cutout[cutout_slice] = imageObj.skyvar[image_slice]
                        sky_var = np.mean(sky_cutout)
                    else:
                        sky_var = None

                    if imageObj.weight is not None:
                        # print('Weight extension currently not implemented!')
                        weight = None
                        # weight_cutout = np.zeros_like(image_cutout)
                        # weight_cutout[cutout_slice] = imageObj.weight[image_slice]
                    else:
                        weight = None

                    # TODO: This currently won't work if a mask is not provided
                    if imageObj.mask is not None:
                        mask_cutout = np.zeros_like(image_cutout)
                        mask_cutout[cutout_slice] = imageObj.mask[image_slice]
                    else:
                        weight = None

                    # combine these into one bitplane.
                    # TODO: Update this line w/ extra info, such as coadd seg!
                    # maskbits_output = mask_cutout + 2*mask_cutout
                    maskbits_output = mask_cutout.astype(msk_dtype)
                    fits[f'MASK{image_index}'].write(
                        maskbits_output, start=[0, pixels_written]
                        )

                    object_info_table[info_index]['object_id'] = iobj[id_tag]
                    object_info_table[info_index]['image_file'] = image['image_file']

                    # This is how we look up object positions to read later.
                    object_info_table[info_index]['extension'] = image_index
                    object_info_table[info_index]['start_pos'] = pixels_written
                    object_info_table[info_index]['end_pos'] = pixels_written +\
                        cutout_pixels

                    if sky_bkg is not None:
                        object_info_table[info_index]['sky_bkg'] = sky_bkg
                    else:
                        object_info_table[info_index]['sky_bkg'] = -1
                    if sky_var is not None:
                        object_info_table[info_index]['sky_var'] = sky_var
                    else:
                        object_info_table[info_index]['sky_var'] = -1

                    pixels_written += cutout_pixels
                    info_index = info_index+1

            end = time()
            dT = end - start
            self.logprint(f'Total stamp writing time: {dT:.1f}')
            self.logprint(f'Writing time per image: {dT/Nimages:.1f} s')


            # NOTE: while we wanted this to be ext1, there are issues that
            # make it easier for it to be -1
            start = time()
            fits.create_table_hdu(data=object_info_table, extname='META')
            end = time()
            dT = end - start
            self.logprint(f'Writing time for metadata: {dT:.1f} s')

            # self._fits = fits

        # Finally, populate the object info table if we need it for later.
        self._object_info_table = object_info_table

        return

    @property
    def objectInfoTable(self):
        if self._object_info_table is None:
            self._object_info_table = fitsio.read(
                self._cookiecutter_file, ext='META'
                )

        return self._object_info_table

    def _getOneCutout(self, objectid, extnumber=None, filename=None,
                      cutoutType='IMAGE'):
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
        cutoutType: defaults to IMAGE; may also provide MASK.
        '''

        # Construct the file extension from the requested cutout:
        if (filename is None) and (extnumber is not None):
            extnumber = int(extnumber) # Avoiding needless silliness
            extname = f'{cutoutType}{extnumber}'

        # Look up the row in the table.
        if (filename is not None) and (extnumber is None):
            filematch = [
                filename in thing for thing in self._object_info_table['imagefile']
                ]
            if len(filematch) == 1:
                extnumber = filematch.index(True)
                extname = f'{cutoutType}{extnumber}'
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
        row = (self._object_info_table['object index'] == objectid) &\
            (self._object_info_table['extnumber'] == extnumber)
        if np.sum(row) != 1:
            raise ValueError(f'somehow, the object index {objectid} and image ' +
                             'extension {extnumber} pairing does not exist, ' +
                             'or is not unique in your metadata')
        entry = self._object_info_table[row]

        with fitsio.FITS(self._cookiecutter_file, 'r') as fits:
            cutout1d = fits[extname][row[startpos]:row[endpos]]
            # NOTE: cutouts are always square.
            cutout2d = cutout1d.reshape(int(np.sqrtcutout1d.size),int(np.sqrtcutout1d.size))

        return cutout2d

    def getCutouts(self, objectids=None, extnumbers=None, filenames=None,
                   cutoutTypes=['IMAGE', 'MASK']):
        # Can specify all three, will try to find everything that matches at least one.
        raise NotImplementedError('getCutouts() not yet implemented!')

        return

# Used by removeEssentialFitsKeys()
ESSENTIAL_FITS_KEYS = ['SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2', 'EXTEND']
