import os
import numpy as np
import astropy.io.fits as fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
from shapely.geometry.polygon import Polygon
from shapely.geometry import MultiPolygon, box
from rtree import index

class HotColdSExtractor:

    def __init__(self, image_files, hc_config, target_name):
        # Load in config arguments
        self.coadd_file = hc_config['coadd_file']
        self.data_dir = hc_config['data_dir']
        self.config_dir = hc_config['config_dir']
        self.outdir = hc_config['outdir']
        self.modes = hc_config['modes']
        self.buffer_radius = hc_config['buffer_radius']
        self.n_neighbors = hc_config['n_neighbors']

        self.target_name = target_name
        self.image_files = image_files

        # Initialize merged catalog and exclusion zone
        self.merged_data = []
        self.exclusion_zones = []

    def run(self, imagefile):
        '''
        Function that calls the SExtractor command building function based on the selected mode.
        '''
        cold_cat, hot_cat, default_cat = None, None, None

        # Run 'cold' (bright-mode) Source Extractor
        if 'cold' in self.modes:
            cold_cat = self._run_sextractor(self.config_dir, imagefile, self.outdir, "cold", self.data_dir)
            print(f"Cold mode catalog complete: {cold_cat}")

        # Run 'hot' (faint-mode) Source Extractor
        if 'hot' in self.modes:
            hot_cat = self._run_sextractor(self.config_dir, imagefile, self.outdir, "hot", self.data_dir)
            print(f"Hot mode catalog complete: {hot_cat}")

        # Run 'default' (one-mode) Source Extractor
        if 'default' in self.modes:
            default_cat = self._run_sextractor(self.config_dir, imagefile, self.outdir, "default", self.data_dir)
            print(f"Default mode catalog complete: {default_cat}")

        # Check if the catalogs exist before merging
        if cold_cat and hot_cat and os.path.exists(cold_cat) and os.path.exists(hot_cat):
            # Define output catalog name
            base_name = os.path.basename(imagefile)
            outname = base_name.replace('.fits', '_cat' + '_merged'+'.fits')  
            # Merge the catalogs
            print(f"Merging catalogs {cold_cat} and {hot_cat}")
            self._merge_catalogs(cold_cat, hot_cat, self.buffer_radius, self.n_neighbors, outname)
            print(f"Merged catalog complete: {outname}")
        else:
            raise FileNotFoundError(f"One or both of the catalogs {cold_cat}, {hot_cat} do not exist, merged catalog NOT created.")


    def _run_sextractor(self, sextractor_config_path, image_file, outdir, mode, datadir):
        '''
        Runs source extractor using os.system(cmd) on the given image file in the given mode
        '''
        print("Processing " + f'{image_file}' + " in mode " + f'{mode}')

        # Define config file path
        cpath = sextractor_config_path

        # Define output catalog name/path
        cat_dir = os.path.join(outdir)
        base_name = os.path.basename(image_file)

        # Different output catalog nomenclature for coadd versus single exposure
        if "det/cat" in outdir:
            cat_name = base_name.replace('.fits', '_det_cat.fits')
        else:
            cat_name = base_name.replace('.fits', '_cat' + f'_{mode}'+'.fits')

        cat_file = os.path.join(cat_dir, cat_name)

        # Define image file, check if it exists
        image = os.path.join(datadir, image_file)
        if not os.path.exists(image):
            raise FileNotFoundError(f"The file '{image}' does not exist.")

        # Construct the SExtractor command
        cmd = self._construct_sextractor_cmd(image, cat_file, cpath, mode)
        print("sex cmd is " + cmd)

        # Print command to terminal and run
        os.system(cmd)

        print(f'cat_name_is {cat_file}')
        return cat_file

    def make_exposure_catalogs(self):
        '''
        Wrapper script that runs through the list of single exposures,
        runs SExtractor on each, and returns a list of the catalogs.
        '''

        exposure_catalogs = []

        for imagefile in self.image_files:
            sexcat = self.run(imagefile)
            exposure_catalogs.append(sexcat)

        return exposure_catalogs
    
    def make_coadd_catalog(self):
        '''
        Wrapper script that runs SExtractor on the coadd image,
        and returns the catalog.
        '''
        self.coadd_file = os.path.join(self.data_dir, self.target_name, "det", "coadd.fits")
        self.outdir = os.path.join(self.data_dir, self.target_name, "det", "cat")
        self.run(self.coadd_file)


    def _construct_sextractor_cmd(self, image, cat_file, cpath, mode):
        '''
        Construct the sextractor command based on the given mode.
        '''
        # Extract the base name of the image (without path and .fits extension)
        image_basename = os.path.basename(image).replace('.fits', '')

        # Define Arguments for SExtractor command
        image_arg =             f'"{image}[0]"'
        weight_arg =            f'-WEIGHT_IMAGE "{image}[1]" -WEIGHT_TYPE MAP_WEIGHT'
        weight_type_arg =       '-WEIGHT_TYPE MAP_WEIGHT'
        name_arg =              f'-CATALOG_NAME {cat_file}'
        param_arg =             f'-PARAMETERS_NAME {os.path.join(cpath, "sextractor.param")}'
        nnw_arg =               f'-STARNNW_NAME {os.path.join(cpath, "default.nnw")}'
        checktype_arg =         f'-CHECKIMAGE_TYPE -BACKGROUND,SEGMENTATION,APERTURES'

        bkg_name =              image.replace('.fits','.sub.fits')
        seg_name =              image.replace('.fits','.sgm.fits')
        aper_name_base =        f"{image_basename}_apertures"

        if mode == "hot":
            config_arg =        f'-c {os.path.join(cpath, "sextractor.hot.config")}'
            filter_arg =        f'-FILTER_NAME {os.path.join(cpath, "default.conv")}'
            aper_name =         os.path.join(self.outdir, f"{aper_name_base}.hot.fits")
        elif mode == "cold":
            config_arg =        f'-c {os.path.join(cpath, "sextractor.cold.config")}'
            filter_arg =        f'-FILTER_NAME {os.path.join(cpath, "gauss_5.0_9x9.conv")}'
            aper_name =         os.path.join(self.outdir, f"{aper_name_base}.cold.fits")
        elif mode == "default":
            config_arg =        f'-c {os.path.join(cpath, "sextractor.config")}'
            filter_arg =        f'-FILTER_NAME {os.path.join(cpath, "default.conv")}'
            aper_name =         os.path.join(self.outdir, f"{aper_name_base}.default.fits")

        checkname_arg =         f'-CHECKIMAGE_NAME {bkg_name},{seg_name},{aper_name}'

        # Make the SExtractor command
        cmd = ' '.join([
            'sex', image_arg, weight_arg, weight_type_arg, name_arg,  checktype_arg, checkname_arg,
            param_arg, nnw_arg, filter_arg, config_arg
                        ])

        return cmd

    def create_ellipse(self, ra, dec, a, b, theta_deg):
        # Check for NaN or Infinity values in the inputs
        if np.isnan(ra) or np.isnan(dec) or np.isnan(a) or np.isnan(b) or np.isnan(theta_deg):
            print("Skipped due to NaN values")
            return None
        if np.isinf(ra) or np.isinf(dec) or np.isinf(a) or np.isinf(b) or np.isinf(theta_deg):
            print("Skipped due to Infinity values")
            return None

        theta_rad = np.radians(180 - theta_deg) # Convert theta from degrees to radians (flip angle due to Polygon using different convention)
        t = np.linspace(0, 2*np.pi, 50)
        ellipse_x = ra + a * np.cos(t) * np.cos(theta_rad) - b * np.sin(t) * np.sin(theta_rad)
        ellipse_y = dec + a * np.cos(t) * np.sin(theta_rad) + b * np.sin(t) * np.cos(theta_rad)
        return Polygon(np.column_stack((ellipse_x, ellipse_y)))


    def get_kron_min_radius(self, fits_file):
        with fits.open(fits_file) as hdul:
            header = hdul[0].header  # the header is in the Primary HDU
            return header['SEXAPEK3'] # Kron minimum radius attribute in header

    def _merge_catalogs(self, cold_cat, hot_cat, buffer_radius, n_neighbors, outname):

        # Load in hot and cold catalogs
        with fits.open(cold_cat) as hdul1:
            cold_data = hdul1[1].data
        with fits.open(hot_cat) as hdul2:
            hot_data = hdul2[1].data

        cold_idx = index.Index()
        buffer_radius = self.buffer_radius
        n_neighbors = self.n_neighbors

        # Get the Kron minimum radius from the cold catalog
        kron_min_radius_cold = self.get_kron_min_radius(cold_cat)
        kron_min_radius_hot = self.get_kron_min_radius(hot_cat)

        print(f'Creating ellipses for {len(cold_data)} cold sources')
        # Loop over each source in the cold catalog
        for i, cold_source in enumerate(cold_data):
            # Convert the cold source parameters to a SkyCoord object
            cold_coord = SkyCoord(cold_source['ALPHAWIN_J2000'], cold_source['DELTAWIN_J2000'], unit='deg')
            
            # Create an ellipse for the cold source
            try:
                ellipse = self.create_ellipse(cold_coord.ra.deg, cold_coord.dec.deg, cold_source['A_WORLD']*kron_min_radius_cold, 
                                cold_source['B_WORLD']*kron_min_radius_cold, cold_source['THETA_IMAGE'])
                # Add a check here to see if the ellipse was created successfully
                if ellipse is None:
                    continue
            except Exception as e:
                print(f"Exception occurred: {e}")
                print(f"Cold source parameters: {cold_source}")
                continue

            # Add the ellipse to the exclusion zones and the r-tree index
            self.exclusion_zones.append(ellipse)
            cold_idx.insert(i, ellipse.bounds)

            # Add the cold source to the merged catalog
            self.merged_data.append(cold_source)
        print(f'Created {len(self.exclusion_zones)} exclusion zones')

        print(f'Creating ellipses for {len(hot_data)} hot sources')
        # Loop over each source in the hot catalog
        for hot_source in hot_data:
            # Convert the hot source parameters to a SkyCoord object
            hot_coord = SkyCoord(hot_source['ALPHAWIN_J2000'], hot_source['DELTAWIN_J2000'], unit='deg')

            # Create an ellipse for the hot source
            try:
                hot_object = self.create_ellipse(hot_coord.ra.deg, hot_coord.dec.deg, hot_source['A_WORLD']*kron_min_radius_hot,
                                            hot_source['B_WORLD']*kron_min_radius_hot, hot_source['THETA_IMAGE'])
                # Add a check here to see if the hot object was created successfully
                if hot_object is None:
                    continue
            except Exception as e:
                print(f"Exception occurred: {e}")
                print(f"Hot source parameters: {hot_source}")
                continue

            # Create a buffer around the hot source
            hot_buffer = hot_object.buffer(buffer_radius)
            hot_buffer_bounds = box(*hot_buffer.bounds)

            # Get the list of 5 nearest cold sources
            nearest_sources_ids = list(cold_idx.nearest(hot_buffer_bounds.bounds, n_neighbors))
            
            # Get the actual nearest cold sources
            nearest_sources = [self.exclusion_zones[i] for i in nearest_sources_ids if 0 <= i < len(self.exclusion_zones)]
            
            if not any(source.contains(hot_object) or source.intersects(hot_object) for source in nearest_sources):
                self.merged_data.append(hot_source)
        print('Hot sources referenced to exclusion zone, merging catalogs')
        # Save the merged catalog:
        merged_table = Table(rows=self.merged_data)
        hdu = fits.BinTableHDU(merged_table)
        hdu.writeto(os.path.join(self.outdir, outname), overwrite=True)


