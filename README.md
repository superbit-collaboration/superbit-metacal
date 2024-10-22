# superbit-metacal
Contains a collection of routines used to perform ngmix fits, including metacalibration, on simulated SuperBIT images.

This repo has recently been significantly refactored into the new `superbit_lensing` module, which you can include in your desired environment by running `python setup.py install` without the need to add the various submodules to your `PYTHONPATH`. The module includes the following four submodules which can be used independently if desired:

  - `galsim`: Contains scripts that generate the simulated SuperBIT observations used for validation and forecasting analyses.
  - `medsmaker`: Contains small modifications to the original superbit-ngmix scripts that make coadd images, runs SExtractor & PSFEx, and creates MEDS files.
  - `metacalibration`: Contains scripts used to run the ngmix/metacalibration algorithms on the MEDS files produced by Medsmaker.
  - `shear-profiles`: Contains scripts to compute the tangential/cross shear profiles and output to a file, as well as plots of the shear profiles.

More detailed descriptions for each stage are contained in their respective directories.

To run the full pipeine in sequence (or a particular subset), we have created the `SuperBITPipeline` class in `superbit_lensing/pipe.py` along with a subclass for each of the submodules. This is run by passing a single yaml configuration file that defines the run options for the pipeline run. The most important arguments are as follows:

- `run_name`: The name of the run, which is also used to specify the `outdir` if you do not provide one; **Required**
- `order`: A list of submodule names that you want the pipeline to run in the given order; **Required**
- `vb`: Verbose. Only affects terminal output; everything is saved to a pipeline log file as well as a log for each submodule; **Required**
- `ncores`: The number of CPU cores to use. Will default to half of the available cores if not provided. Can overwrite for specific submodules in their respective configs if desired; _Optional_
- `run_diagnostics`: A bool. Set to `True` run the diagnostics, including plots which are saved in `{outdir}/plots/`; _Optional_

These should be set in the `run_options` field of the config file, while options for each submodule should be set in a field with the same name (e.g. `medsmaker: {...}`). Once the configuration is set, run the pipeline by doing the following:
```
import superbit_lensing.utils as utils
from superbit_lensing.pipe import SuperBITPipeline

log = utils.setup_logger({logfile}, logdir={logdir})
pipe = SuperBITPipeline(config_file, log)

rc = pipe.run()

assert(rc == 0)
```
An example of a pipeline run along with a test configuration is given in `pipe.main()`, which can be run with

`python pipe_test.py`.

The example configuration file is shown in `configs/pipe_test.yaml`. An example wrapper script you can use to run the `SuperBITPipeline` is shown in `superbit-lensing/process_all.py`.

The available config options for each submodule are defined in the various module classes in `superbit_lensing.pipe.py`, such as `GalSimModule`. The required & optional fields are given in `_req_fields` and `_opt_fields` respectively. The pipeline runner tells you if you fail to pass a required field or if you pass something that it doesn't understand.

## To Build A Specific Run Environement 

Step 1: Clone this github repo:

`git clone https://github.com/superbit-collaboration/superbit-metacal.git`

Step 2: Install `conda` or `miniconda` if you don't have it already:

https://conda.io/projects/conda/en/latest/user-guide/install/index.html

...confirm that you are in your `base` environment

Step 3: Build a specific run environment with a given configuration file (.yaml or .yml)

The current recommended config file is `sbmcal_py12.yaml`. If there are any problems with package conflicts in this environment, another config option is `env.yaml`. This is a simple environment with few dependencies, so working through the pipeline in this environment will yield errors regarding missing packages. Simply `conda install -c conda-forge *package*` the package in question. This will install the latest version of packages that may be out of date in `sbmcal_py12.yaml`.

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

cd to this repo again:`cd /path/to/repos/superbit-metacal`

`pip install -e /path/to/repos/superbit-metacal`

NGMIX

As explained above, NGMIX is a package used in the metacalibration module. To run this pipeline, it is required that `ngmix` version 1.3.9 is used (hense the environment name ending in 139). Unfortunately, ngmix insists on python 3.6 or 3.7 in order to be installed, but our environment uses python 3.12. It is therefore not possible to simply conda forge this package into your environment. Instead, you can do the following to manually install it:

Step 1: `curl -OL https://github.com/esheldon/ngmix/archive/refs/tags/v1.3.9.tar.gz`

Step 2: `tar -xzf v1.3.9.tar.gz`

Step 3: Navigate to wherever this package is downloaded and do `pip install -e .` 

## For the experts

If you want to add a new submodule to the pipeline, simply define a new subclass `MyCustomModule(SuperBITModule)` that implements the abstract `run()` function of the parent class and add it to `pipe.MODULE_TYPES` to register it with the rest of the pipeline. You should also implement the desired required & optional parameters that can be present in the module config with the class variables `_req_fields` and `_opt_fields`, which should be lists.

Contact @sweverett at spencer.w.everett@jpl.nasa.gov or @mccleary at j.mccleary@northeastern.edu you have any questions about running the pipeline - or even better, create an issue!
