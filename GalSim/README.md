This directory contains the GalSim scripts used to generate the simulated superbit images used for medsmaking & ngmix/metacalibration.

Excellent documentation for GalSim is available here: https://galsim-developers.github.io/GalSim/_build/html/index.html

The actual scripts you'll want to run are `mock_superBIT_data.py` to generate mock observations, and `mock_superbit_empirical.cluster.py` 
to generate mock Abell 2218 observations using the 2019 flight PSF. Outputs are saved in a location set by the user; you can set the location
by modifying the `outdir` variable.

Currently, user can set exposure time, number of expected galaxies in observation, etc. by accessing the relevant keywords in `main()`. 
Note that code expects the existence of a `data/` directory, containing the PSFs files to be used (i.e. PSFEx PSF models, jitter kernels, etc.). 
At present, those locations are hard-coded (sorry) in the `psf_path`, 

These scripts also rely on the COSMOS_23.5 and COSMOS_25.2 catalogs to be stored in a `data` directory, as well as their supporting files. 
COSMOS catalogs can be accessed here: https://galsim-developers.github.io/GalSim/_build/html/real_gal.html#downloading-the-cosmos-catalog. 

`debug3.py` is the script that actually invokes GalSim; user has to set the exposure time (`exp_time`) within  `main()` (sorry). Five files per exposure time will be created, as well as the accompanying "truth" catalogs. `debug3.py` expects to find the COSMOS_23.5_training_sample/ catalogs
in a data/ directory at the same level as ``debug3.py``

output-debug is the result of running debug3.py; it presently contains 5 simulated observations as well as the GalSim truth
table associated with those simulated images

