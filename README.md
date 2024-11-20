# superbit-metacal
This `process-real` branch contains a collection of routines used to perform ngmix fits, including metacalibration, on _real_ SuperBIT images.

This repo has recently been significantly refactored into the new `superbit_lensing` module, which you can include in your desired environment by running `python setup.py install` or `pip install -e .` without the need to add the various submodules to your `PYTHONPATH`. The module includes the following four submodules which can be used independently if desired:

  - `galsim`: Contains scripts that generate simulated SuperBIT observations used for validation and forecasting analyses. This is _not_ called if real data is processed, although it is still accessible in this branch. 
  - `medsmaker`: Contains small modifications to the original superbit-ngmix scripts that make coadd images, runs SExtractor & PSFEx, and creates MEDS files.
  - `metacalibration`: Contains scripts used to run the ngmix/metacalibration algorithms on the MEDS files produced by Medsmaker.
  - `shear-profiles`: Contains scripts to compute the tangential/cross shear profiles and output to a file, as well as plots of the shear profiles.

More detailed descriptions for each stage are contained in their respective directories.

## Running the Pipeline 
To run the full pipeine in sequence (or a particular subset), we have created the `SuperBITPipeline` class in `superbit_lensing/pipe.py` along with a subclass for each of the submodules. This is run by passing a single yaml configuration file that defines the run options for the pipeline run. As of right now, the `SuperBITPipeline` is broken in this branch, so a seperate yaml file is required to run each cluster. If you would like to understand how to run the `SuperBITPipeline` anyways, the intructions are in the `main` branch `README.md`. 

To run the pipeline currently, you will need to create your own job script. A template job script can be accessed in `superbit-metacal/job_scripts/job_Abell3571.sh`. This job script defines input and output paths for your data, and calls the `medsmaker`, `metacalibration`, and `shear profiles` scripts independently. These scripts are `process_2023.py`, `ngmix_fit_superbit3.py` and `make_annular_catalog.py`, respectively. They can be found in the `superbit-lensing` folder. The arguments passed to these scripts, such as the cluster name and band, are defined and described in each script's `parse_args()` function. 



## Preliminary Steps

A proper python environment is needed in order to run this pipeline. The steps to building such environment are as follows: 

Step 1: Clone this github repo:

`git clone https://github.com/superbit-collaboration/superbit-metacal.git`

Step 2: Install conda or miniconda if you don't have it already:

`https://conda.io/projects/conda/en/latest/user-guide/install/index.html`

...confirm that you are in your `base` environment 

Step 3:  Build a specific run environment with a given configuration file (`.yaml` or `.yml`)

The current recommended config file is `sbmcal_py12.yaml`. If there are any problems with package conflicts in this environment, another config option is `env.yaml`. This is a simple environment with few dependencies, so working through the pipeline in this environment will yield errors regarding missing packages. Simply `conda install -c conda-forge *package*` the package in question. This will install the latest version of packages that may be out of date in `sbmcal_py12.yaml`. 

If you are running into issues with conda, you can also look at `Install.md` which takes you through installing the necessary packages with `pip`. This is also a good option. You can create an environment without a configuration file, and follow the steps in `Install.md`. 

Create env from yaml:

`conda env create --name *give your env a name* --file *config file name*`

recommended: `conda env create --name sbmcal_139 --file sbmcal_py12.yaml`

Activate new env:

`conda activate sbmcal_139`

Step 4: Build Superbit Lensing:

`cd /path/to/repos/superbit-metacal`

`python setup.py install`

Step 5: Build Meds:

`Git clone https://github.com/esheldon/meds.git`

`cd /path/to/repos/meds`

`python setup.py install`

Step 6: Pip install repository

Cd to this repo again:`cd /path/to/repos/superbit-metacal`

`pip install -e /path/to/repos/superbit-metacal`

## NGMIX

As explained above, NGMIX is a package used in the `metacalibration` module. To run this pipeline, it is required that ngmix version 1.3.9 is used (hense the environment name ending in 139). Unfortunately, ngmix insists on python 3.6 or 3.7 in order to be installed, but our environment uses python 3.12. It is therefore not possible to simply conda forge this package into your environment. Instead, you can do the following to manually install it:

Step 1: `curl -OL https://github.com/esheldon/ngmix/archive/refs/tags/v1.3.9.tar.gz`

Step 2: `tar -xzf v1.3.9.tar.gz`

Step 3: Navigate to wherever this package is downloaded and do `pip install -e . `



## For the experts

If you want to add a new submodule to the pipeline, simply define a new subclass `MyCustomModule(SuperBITModule)` that implements the abstract `run()` function of the parent class and add it to `pipe.MODULE_TYPES` to register it with the rest of the pipeline. You should also implement the desired required & optional parameters that can be present in the module config with the class variables `_req_fields` and `_opt_fields`, which should be lists.

Contact @sweverett at spencer.w.everett@jpl.nasa.gov or @mccleary at j.mccleary@northeastern.edu you have any questions about running the pipeline - or even better, create an issue!
