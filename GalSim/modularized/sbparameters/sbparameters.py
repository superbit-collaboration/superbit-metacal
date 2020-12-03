import logging
import sys
import yaml
import galsim

global logger

logging.basicConfig(format="%(message)s", 
                    level=logging.INFO, 
                    stream=sys.stdout)
logger = logging.getLogger("mock_superbit_data")

__all__ = ["SuperBITParameters"]

class SuperBITParameters:
        def __init__(self, config_file=None, argv=None):
            """
            Initialize default params and overwirte with 
            config_file params and / or commmand line
            parameters.
            """
            # Check for config_file params to overwrite defaults
            if config_file is not None:
                logger.info('Loading parameters from %s' % (config_file))
                self._load_config_file(config_file)

            # Check for command line args to overwrite 
            # config_file and / or defaults
            if argv is not None:
                self._load_command_line(argv)

        def _load_config_file(self, config_file):
            """
            Load parameters from config file. Only parameters that exist in 
            the config_file will be overwritten.
            """
            with open(config_file) as fsettings:
                config = yaml.load(fsettings, Loader=yaml.FullLoader)
            self._load_dict(config)
        
        def _args_to_dict(self, argv):
            """
            Converts a command line argument array to a dictionary.
            """
            d = {}
            for arg in argv[1:]:
                optval = arg.split("=", 1)
                option = optval[0]
                value = optval[1] if len(optval) > 1 else None
                d[option] = value
            return d

        def _load_command_line(self, argv):
            """
            Load parameters from the command line argumentts. 
            Only parameters that are provided in
            the command line will be overwritten.
            """
            # Parse arguments here
            self._load_dict(self._args_to_dict(argv))

        def _load_dict(self, d):
            """
            Load parameters from a dictionary.
            """
            for (option, value) in d.items():
                if option == "pixel_scale":     
                    self.pixel_scale = float(value)
                elif option == "sky_bkg":        
                    self.sky_bkg = float(value) 
                elif option == "sky_sigma":     
                    self.sky_sigma = float(value)
                elif option == "gain":          
                    self.gain = float(value)   
                elif option == "read_noise":
                    self.read_noise = float(value)
                elif option == "dark_current":
                    self.dark_current = float(value)
                elif option == "dark_current_std":
                    self.dark_current_std = float(value)
                elif option == "image_xsize":   
                    self.image_xsize = int(value)    
                elif option == "image_ysize":   
                    self.image_ysize = int(value)    
                elif option == "center_ra":     
                    self.center_ra = float(value) * galsim.hours
                elif option == "center_dec": 
                    self.center_dec = float(value) * galsim.degrees
                elif option == "n_exp":      
                    self.n_exp = int(value)          
                elif option == "exp_time":   
                    self.exp_time = float(value) 
                elif option == "n_bkg_gal":     
                    self.n_bkg_gal = int(value)     
                elif option == "n_cluster_gal":     
                    self.n_cluster_gal = int(value)     
                elif option == "n_stars": 
                    self.n_stars = int(value)    
                elif option == "tel_diam": 
                    self.tel_diam = float(value)
                elif option == "lam":     
                    self.lam = float(value)      
                elif option == "jitter_psf_path": 
                    self.jitter_psf_path = str(value) 
                elif option == "mass": 
                    self.mass = float(value)         
                elif option == "nfw_conc":   
                    self.nfw_conc = float(value) 
                elif option == "nfw_z_cluster": 
                    self.nfw_z_cluster = float(value)
                elif option == "omega_m":   
                    self.omega_m = float(value)  
                elif option == "omega_lam":
                    self.omega_lam = float(value)
                elif option == "config_file":
                    logger.info('Loading parameters from %s' % (value))
                    self._load_config_file(str(value))
                elif option == "cosmosdir":
                    self.cosmosdir = str(value)
                elif option == "cosmosdir":
                    self.cosmosdir = str(value)
                elif option == "cat_file_name":
                    self.cat_file_name = str(value)
                elif option == "fit_file_name":
                    self.fit_file_name = str(value)
                elif option == "cluster_cat_name":
                    self.cluster_cat_name = str(value)
                elif option == "bp_file":
                    self.bp_file = str(value)
                elif option == "outdir":
                    self.outdir = str(value)
                elif option == "noise_seed":     
                    self.noise_seed = int(value)     
                elif option == "galobj_seed":     
                    self.galobj_seed = int(value)     
                elif option == "cluster_seed":     
                    self.cluster_seed = int(value)     
                elif option == "stars_seed":     
                    self.stars_seed = int(value)     
                elif option == "nstruts":     
                    self.nstruts = int(value)      
                elif option == "strut_thick":     
                    self.strut_thick = float(value)     
                elif option == "strut_theta":  
                    self.strut_theta = float(value)        
                elif option == "obscuration":  
                    self.obscuration = float(value)     
                else:
                    raise ValueError("Invalid parameter \"%s\" "
                                     "with value \"%s\"" % (option, value))

            # Derive the parameters from the base parameters
            self.image_xsize_arcsec = self.image_xsize * self.pixel_scale 
            self.image_ysize_arcsec = self.image_ysize * self.pixel_scale 
            self.center_coords = galsim.CelestialCoord(self.center_ra,
                                                       self.center_dec)
            self.strut_angle = self.strut_theta * galsim.degrees
            
            # The catalog returns objects that are appropriate for HST in 
            # 1 second exposures.  So for our telescope we scale up by the 
            # relative area, exposure time, pixel scale & detector gain 
            # NEED TO VERIFY IF THIS IS CORRECT              
            hst_eff_area = 2.4**2 * (1.-0.33**2)
            sbit_eff_area = self.tel_diam**2 * (1.-0.380**2) 
            self.flux_scaling = ((sbit_eff_area / hst_eff_area) 
                                * self.exp_time 
                                * self.gain 
                                * (self.pixel_scale/.05)**2) 
