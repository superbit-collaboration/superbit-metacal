import numpy as np
from numpy.lib import recfunctions as rf
from pathlib import Path
from glob import glob
import fitsio
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u

from superbit_lensing import utils



def intersecting_slices(big_array_shape,small_array_shape,position):
    '''

    Stripped-down version of astropy ndimage utilities.
    from here: https://docs.astropy.org/en/stable/_modules/astropy/nddata/utils.html#Cutout2D

    
    Parameters:
    -----------
    
    big_array_shape : tuple of int or int; shape of large array
    small_array_shape: tuple of int or int; shape of small array to be 
      cut out
    position: position of the small array center wrt to the big array.
    

    Returns:
    --------
    big_slices : slices such that big_array[big_slices] extracts the desired cutout 
      region.
    small_slices : slices such that cutout[small_slices] extracts the non-empty pixels in the desired size cutout.

    TODO: Fail gracefully when there's no overlap.

    '''

    min_indices = [int(np.ceil(pos - (small_shape / 2.0))) for (pos, small_shape) in zip(position, small_array_shape)]
    max_indices = [int(np.ceil(pos + (small_shape / 2.0))) for (pos, small_shape) in zip(position, small_array_shape)]

    for e_max in max_indices:
        if e_max < 0:
            raise NoOverlapError("Arrays do not overlap.")
        for e_min, large_shape in zip(min_indices, big_array_shape):
            if e_min >= large_shape:
                raise NoOverlapError("Arrays do not overlap.")

    big_slices = tuple(
        slice(max(0, min_indices), min(large_shape, max_indices))
        for (min_indices, max_indices, large_shape) in zip(min_indices, max_indices, big_array_shape)
        )

    small_slices = tuple(slice(0, slc.stop - slc.start) for slc in big_slices)
    return big_slices, small_slices


class NoOverlapError(ValueError):
    """Raised when determining the overlap of non-overlapping arrays."""
    pass


class ImageLocator(object):
    def __init__(self,imagefile = None, imageext = None, weightfile = None, weightext = None,
                     maskfile = None, maskext = None,backgroundfile = None, backgroundext = None,
                     globalpath = None):
        if globalpath is not None:
            imagefile = Path(globalpath) / Path(imagefile)
            if weightfile is not None:
                weightfile = Path(globalpath) / Path(weightfile)
            if maskfile is not None:
                maskfile = Path(globalpath) / Path(maskfile)
            if backgroundfile is not None:
                backgroundfile = Path(globalPath) / Path(maskfile)
        else:
            imagefile = Path(imagefile)
            if weightfile is not None:
                weightfile = Path(weightfile)
            if maskfile is not None:
                maskfile = Path(maskfile)
            if backgroundfile is not None:
                backgroundfile = Path(maskfile)
            
        self._imagefile = imagefile
        if imageext is None:
            self._imageext = 0
        else:
            self._imageext = imageext
        self._weightfile = weightfile
        if weightext is None:
            self._weightext = 0
        else:
            self._weightext = weightext
        self._maskfile = maskfile
        if maskext is None:
            self._maskext = 0
        else:
            self._maskext = maskext
        self._backgroundfile = backgroundfile
        if backgroundext is None:
            self._backgroundext = 0
        else:
            self._backgroundext = backgroundext

    @property
    def image(self):
        return fitsio.FITS(self._imagefile,'r')[self._imageext]
    
    @property
    def weight(self):
        if self._weightfile is None:
            return None
        else:
            return fitsio.FITS(self._weightfile,'r')[self._weightext]
    
    @property
    def mask(self):
        if self._maskfile is None:
            return None
        else:
            return fitsio.FITS(self._maskfile,'r')[self._maskext]

    @property
    def background(self):
        if self._backgroundfile is None:
            return None
        else:
            return fitsio.FITS(self._backgroundfile,'r')[self._backgroundext]



def removeEssentialFITSkeys(header,excludekeys = ['SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2', 'EXTEND']):
    for ikey in excludekeys:
        header.delete(ikey)
    # Note that this doesn't fail if one of the keys is not in the header.
    # This is desirable for this application.
    return header



class CookieCutter(object):
    '''
    For meting out cutouts to processing functions.
    Lightweight replacement for meds.
    
    What this should do:
     - be initialized from _either_ an existing file, or a config file
       specifying a catalog, a list of images(+ optionally weights/masks) 
       and metadata specifying cutout size, etc.
     - 
    '''
    def __init__(self,cookiecutterfile = None, config=None):
        '''
        
        config: A dictionary (see example yaml file) specifying which files and catalogs to use to build the 
                 cookie cutter object.
        cookiecutterfile: initialize a cookiecutter object from an already-existing cookie cutter file.
        
        Only one of these two arguments above can be specified, of course.

        ------

        If the catalog provided does not have a box size field, this will calculate one on the fly.
        You can change how this is done by re-assigning the "calculateBoxsizeFromCatalog" method,
          an attribute of this class, which expects a single argument (catalog, a numpy structured array) 
          and returns an integer scalar that will be used as the cutout box size.
        
        '''

        # This pattern is just here to test the inputs. We don't load anything until we're told to.
        self.config = config
        if (cookiecutterfile is None) and (config is not None):
            # Initialize from config
            pass
        elif (cookiecutterfile is not None) and (config is None):
            # Initialize from file
            pass
        else:
            raise ValueError("You should pass only a cookie cutout file or a config, not both.")

    def go(self):
        # Parse the config file.
        # Decide if we're initializing a cookie cutter object from an existing cookie cutout file,
        #  or if we're building one from scratch.
        if (cookiecutterfile is None) and (config is not None):
            # Initialize from config
            if config['global input path'] is not None:
                catalogfile = Path(config['global input path']) / Path(config['input catalog']['filename'])
            else:
                catalogfile =  Path(config['input catalog filename'])

            catalog = fitsio.read(catalogfile,ext=config['input catalog']['file extension'])

        # It's unlikely that the catalog actually has a 'boxsize' field, in which case we need to create one.
        # Use a method attached to the class, so that the user can redefine it.
        # Stick here to the boxsize name provided.
        if config['input catalog']['boxsize tag'] not in catalog.dtype.names:
            self._updateCatalogWithBoxsizes(catalog)

        #The catalog read, now let's get the image information.
        images = [config['images'][imagename] for imagename in config['images'].keys()]
            
        # Decide where to put the output.
        if 'global input path' in config.keys():
            outputpath = Path(config['global output path']) / config['output filename']
        else:
            outputpath = Path(config['output filename'])
        if 'global input path' in config.keys():
            global_input_path = config['global input path']
        else:
            global_input_path = None
        
        self._createFromImages(images=images,catalog=catalog,
                    ratag = config['input catalog']['ra tag'],dectag = config['input catalog']['dec tag'],
                    boxsizetag = config['input catalog']['boxsize tag'],
                    globalpath = global_input_path)
                    
    def _updateCatalogWithBoxsizes(self,catalog):
        # Use the numpy recfunctions library, because I'm a geezer.
        boxsizes = self.calculateBoxsizeFromCatalog(catalog)
        new_catalog = rf.append_fields(catalog,'boxsize',boxsizes,dtypes=['i2'],usemask=False)
        return new_catalog
        
    def calculateBoxsizeFromCatalog(self,catalog):
        # Assume we have a FLUX_RADIUS in the catalog.
        radius = 2**(np.ceil(np.log2(np.sqrt( 8**2 + (4*catalog['FLUX_RADIUS'])**2)))).astype('i2')
        # This is a crude rule of thumb -- smallest power of two
        #  that encloses a quadrature sum of 8 pixels and 4x the flux radius.
        # You probably have a better idea.
        return radius
        
    def _createFromImages(self,images = None,catalog=None,ratag=  'RA', dectag = 'DEC',boxsizetag='boxsize', outfile = None):
        '''
        catalog: catalog[i][ratag], catalog[i][dectag], catalog[i]['boxsize']
        ratag: string such that catalog call above works
        dectag: string such that catalog call above works
        boxsize: sring such that catalog call above works
        images: list of dictionaries, each containing image/weight/mask/background file+extension info.
        outfile: where to put the output.



        TODO: CONFIRM THAT THE OBJECT DATA TABLE IS INITIALIZED WITH SUFFICIENT LENGTH TO HOLD THE IMAGE FILENAME.
        '''
        
        # We get a list of images from the config file.
        # We get a catalog (ra/dec) as an input (either from a catalog file, or as an argument)

        # For each image, find all the sources in that image.
        #   -- read in the WCS
        #   -- compute the pixel coordinantes in this image for this source.
        #   -- if the source is inside the image, make a cutout.
        #   -- check for the presence of mask or weight information;
        #      if they exist, make cutouts of these as well.


        # Create a place to put the results.
        
        with fitsio.FITS(filename,"rw",clobber=True) as fits:
            self.fits = fits

        object_info_table = np.empty(catalog.size * len(images), dtype = [('object id',int),
                                                                          ('imagefile','S64'),
                                                                          ('startpos',int),
                                                                          ('skylevel',float]))
        info_index = 0
                                                        
        for image_index,image in enumerate(images):
            
            imageObj = ImageLocator(imagefile  = image['imagefile'], imageext = image['image ext'],\
                                    weightfile = image['weightfile'],weightext = image['weight ext'],\
                                    maskfile   = image['maskfile'], maskext = image['mask ext'])

            imageHDR = removeEssentialFITSkeys(imageObj.image.read_header())
            
            
            image_shape = imageObj.image.get_info()['dims']
            
            npix = catalog.size * np.sum(catalog[boxsizetag][:]**2) # We know in advance how many cutout pixels we'll need to store.


            
            science_image_dimensions = (2,npix) # one dimension for data, one dimension for sky.
            mask_image_dimensions = (1,npix)

            # Store each image's cutouts in a new extension.
            # Because the image and background have different datatypes to the mask,
            # we need two different extensions.
            fits.create_image_hdu(img=None,dtype='i2',dims=dims,extname=f'SCI{image_index}',header=imageHDR)

            
            fits.create_image_hdu(img=None,dtype='i1',dims=dims,extname=f'MASK{image_index}')

            pixels_written = 0
            
            for obindx,iobj in enumerate(catalog):
                coord = SkyCoord(ra=iobj[ratag],dec=iobj[dectag])
                x,y = image_wcs.world_to_pixel(coord)
                object_pos_in_image = [x.item(),y.item()] # x and y are, for some reason, needlessly returned as numpy arrays.
                image_slice, cutout_slice = intersecting_slices(image_shape,iobj[boxsizetag],object_pos_in_image)

                cutout_shape = (iobj[boxsizetag],iobj[boxsizetag])
                image_cutout = np.zeros(cutout_shape)
                image_cutout[cutout_slice] = imageObj.image[image_slice]
                science_output = image_cutout.flatten() #np.vstack([image_cutout.flatten(order='C'),sky_cutout.flatten(order='C')])

                if objindx == 0:
                    fits[f'SCI{image_index}'].write(science_output,start=[0,pixels_written],header=imageHDR)
                else:
                    fits[f'SCI{image_index}'].write(science_output,start=[0,pixels_written])
                

                if imageObj.sky is not None:
                    sky_cutout = np.zeros_like(image_cutout)
                    sky_cutout[cutout_slice] = imageObj.sky[image_slice]
                    sky_level = np.median(sky_cutout)

                if imageObj.weight is not None:
                    weight_cutout = np.zeros_like(image_cutout)
                    weight_cutout[cutout_slice] = imageObj.weight[image_slice]
                if imageObj.mask is not None:
                    mask_cutout = np.zeros_like(image_cutout)
                    mask_cutout[cutout_slice] = imageObj.mask[image_slice]

                maskbits_output = weight_cutout + 2*mask_cutout # combine these into one bitplane.
                fits[f'MASK{image_index}'].write(maskbits_output,start=[0,pixels_written])

                meta = {'object id':iobj['id'],'startpos':pixels_written, 'background':sky_level}

                object_info_table[info_index]['object id'] = iobj['id']
                object_info_table[info_index]['imagefile'] = image['imagefile']
                object_info_table[info_index]['startpos'] = npix_sci_written # This is how we look up objects later.
                object_info_table[info_index]['background'] = sky_level
                
                npix_sci_written = npix_sci_written + sky_cutout.size
                info_index = info_index+1
        fits.close()

        # Finally, write the ancillary table we'll need to interpret the data to the last extension.
        fitsio.write(self.config['objectinfo output'],object_info_table)
        
