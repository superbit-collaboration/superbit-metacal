This directory contains the GalSim scripts used to generate the simulated superbit images used for medsmaking & ngmix/metacalibration.

These scripts require the `galsim` python module to be installed. Excellent documentation for GalSim, including download instructions, is available here: 
https://galsim-developers.github.io/GalSim/_build/html/index.html

The actual scripts you'll want to run are `mock_superBIT_data.py` to generate mock observations, and `mock_superBIT_empirical.cluster.py` 
to generate mock Abell 2218 observations using the 2019 flight PSF. Outputs are saved in a location set by the user; you can set the location
by modifying the `outdir` variable. In particular, `mock_superBIT_empirical.cluster.py` will create observations mimicking each exposure in the 2019 flight
by stepping through each of the PSFs (see below) and guessing at the exposure time. 

There are a few debug scripts as well: `mock_superBIT_data.debug.py` and ` mock_superBIT_data.gaussian.py`. These are mostly included
for archival reasons; it's unlikely the user will have to invoke them unless they too are debugging some new module. Five files per exposure
time will be created, as well as the accompanying "truth" catalogs.

All scripts rely on the COSMOS_23.5 and COSMOS_25.2 catalogs to be stored in a `data` directory, as well as their supporting files. 
COSMOS catalogs can be accessed here: https://galsim-developers.github.io/GalSim/_build/html/real_gal.html#downloading-the-cosmos-catalog.

Both `mock_superBIT_data.py` and `mock_superBIT_empirical_cluster.py` will look for a folder containing their PSF models in the `data/` directory.
The `data/` folder in this repository contains both the flight jitter model PSFs (in `data/flight_jitter_only_oversampled_1x/` with three different exposure times)
and the PSFEx fits to the 2019 flight A2218 data (in `data/empirical_2019Flight_PSF/`). 

Currently, user can set exposure time, number of expected galaxies in observation, etc. by accessing the relevant keywords in `main()`. 
Note that code expects the existence of a `data/` directory, containing the PSFs files to be used (i.e. PSFEx PSF models, jitter kernels, etc.). 
At present, those locations are hard-coded (sorry) in the `psf_path`. 

To generate `Chromatic` galaxies convolved with the SuperBIT throughputs in different filters, the mock data scripts expect a throughput
file in the `data/` directory. User can modify locations and names of files being used by setting the `bp_dir` and `bp_file` variables
in the `make_a_galaxy()` and `make_cluster_galaxy` methods. 
