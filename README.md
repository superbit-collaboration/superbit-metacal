# superbit-metacal
Contains a collection of modules and routines used build the SuperBIT measurement pipeline, as well as make realistic image simulations for validation.

This repo has been significantly refactored into the new `superbit_lensing` module, which you can include in your desired environment by following the installation instructions below. The module includes a number of modules which can be used independently if desired:

  - `galsim`: Contains scripts & classes that generate the simulated SuperBIT observations used for validation and forecasting analyses. Uses a flat config
  - `imsim`: A new image simulation module that aims to handle science and validation simulations using the same framework through regristration of various object, psf, shear, etc. types. Uses a hierarchial config and handles multi-band observations.
  - `medsmaker`: Contains small modifications to the original superbit-ngmix scripts that make coadd images with SWARP, runs SExtractor for source detection, PIFF or PSFEx for PSF estimation, and collates all outputs in a MEDS file. Older style that does coaddition, detection, and MEDS-making in one step.
  - `coadd`: A new coaddition module that runs SWarp independently. Handles multi-band observations and can create a detection image.
  - `detection`: A new detection module that runs SExtractor independently. Handles multi-band observations and can run in dual-mode using a detection image.
  - `metacalibration`: Contains scripts & classes used to run the ngmix implementation of metacalibration on the MEDS files produced by `medsmaker`.
  - `shear-profiles`: Contains scripts & classes to compute the tangential/cross shear profiles and output to a file, as well as plots of the shear profiles.
  - `analysis`: Contains scripts to produce standardized analysis plots on stacks of cluster realizations needed for shear calibration validation.

More detailed descriptions for each stage are contained in their respective directories.

## Pipeline running & automated diagnostics

To run the full pipeine in sequence (or a particular subset), we have created the `SuperBITPipeline` class in `superbit_lensing/pipe.py` along with a subclass for each of the submodules. This is run by passing a single yaml configuration file that defines the run options for the pipeline run. The most important arguments are as follows:

- `run_name`: The name of the run, which is also used to specify the `outdir` if you do not provide one; **Required**
- `order`: A list of submodule names that you want the pipeline to run in the given order; **Required**
- `vb`: Verbose. Only affects terminal output; everything is saved to a pipeline log file as well as a log for each submodule; **Required**
- `ncores`: The number of CPU cores to use. Will default to half of the available cores if not provided. Can overwrite for specific submodules in their respective configs if desired; _Optional_
- `run_diagnostics`: A bool. Set to `True` run the diagnostics, including plots which are saved in `{outdir}/plots/`; _Optional_

These should be set in the `run_options` field of the config file, while options for each submodule should be set in a field with the same name (e.g. `medsmaker: {...}`). To see a full example of a pipeline config, run a pipe test first which is explained below, and then look at the corresponding `pipe_test.yaml` file.

The available config options for each submodule are defined in the various module classes in `superbit_lensing.pipe.py`, such as `GalSimModule`. The required & optional fields are given in `_req_fields` and `_opt_fields` respectively. The pipeline runner tells you if you fail to pass a required field or if you pass something that it doesn't understand.

Once the configuration is set, you can run the pipeline in a script by doing the following:
```
import superbit_lensing.utils as utils
from superbit_lensing.pipe import SuperBITPipeline

config_file = ...
logdir = ...
logfile = ...

log = utils.setup_logger(logfile, logdir=outdir)
pipe = SuperBITPipeline(config_file, log)

rc = pipe.run()

assert(rc == 0)
```
or simply pass your favorite config file to `python run_pipe.py {my_config.yaml}`

# Installation

Clone the repo in your desired local directory. The `utils.py` module will automatically sort out it's current location (`MODULE_DIR`) which anchors other useful paths (`BASE_DIR`, `TEST_DIR`, etc.)

## Conda environment

At the moment, we provide two different conda environments to run the pipeline, as we have modules for both `ngmix` metacal APIs (`v1.X` and `v2.X`). To build a specific run environment (e.g. `env_v1.3.9.yaml`):

`conda env create --name sbmcal_139 --file env_v1.3.9.yaml`

Activate new env:

`conda activate sbmcal_139`

These env files are written with `--no-builds` and so hopefully are OS agnostic, but no guarantees.

## Extra dependencies

The [meds](https://github.com/esheldon/meds) and [psfex](https://github.com/esheldon/psfex) packages need to be built from source. Clone each and cd to their respective repos:

`cd /path/to/repos/{meds/psfex}`

Build it (in your desired environment!):

`python setup.py install`

cd back to this repo:

`cd /path/to/repos/superbit-metacal`

To get around adding the repository to your `PATH` or `PYTHONPATH` directly or having to rebuild the respository every time you make a change to the code, we can pip install with the `-e` flag which will overwrite the directory in site-packages with a symbolic link to the repository, meaning any changes to code in there will automatically be reflected when running the pipeline. So do it:

`pip install -e /path/to/repos/superbit-metacal`

## Quickstart & Pipe Testing

To test that your local installation worked or to validate that your code updates haven't broken the pipeline before pushing changes (but you would never do that, right?), you can run what we call a "pipe test". This test of the pipeline is optimized for speed and submodule/feature coverage, not scientifically useful outputs. The easiest way to get started is to first make a copy of `configs/path_config.yaml` with the path templates edited to match your installation of this repo and the required `GalSim` COSMOS catalogs (but don't edit the file directly as it is tracked, while your local version is not). The second thing you need to do is unpack two provided tar files:

- `tar -xvf superbit_lensing/galsim/data/gaia/GAIAstars_2023filter.tar` (so you can sample GAIA stars for actual SuperBIT targets)
- `tar -xvf superbit_lensing/shear_profiles/truth/nfw_truth_files_sample.tar` (so you can compare your measured shear profile to a true NFW)

**NOTE:** Since we are currently not including the source input catalog `cosmos15_superbit2023_phot_shapes.csv` in the repo, we have added a 1000-row subset `sample_cosmos15_superbit2023_phot_shapes.csv` in the `galsim/data/` directory that you can use to try a pipe test before you are sent the file by a current SuperBIT team member.

Once both sets of tasks are done, simply run:

`python superbit_lensing/pipe_test.py -path_config=configs/{my_path_config}.yaml --fresh`

You should mostly care about whether it succeeded or not, but you can look at `configs/pipe_test_gs.yaml` to see what kind of images you are producing. Take a look at the top of `superbit_lensing/pipe_test.py` for more details and alternative ways to run it with more control.

## Meta-configs and HPC jobs

Under construction! Bug @sweverett if you are far enough along to care about this. For now, here is a quick outline for gattaca users:

1. Make a copy of `configs/gauss-uberseg-meta.yaml` and call it whatever you want corresponding to some `run_name`, which you should also update in the config itself
2. Update the paths in that meta config to run from your local gattaca repo
3. Choose any subset of mass & redshift combinations already listed in the meta config (setting different values would require creating new truth files)
4. Change any other optional settings in the meta config that you desire
5. You can use the existing `configs/auss-uberseg-gs.yaml` as your GalSim config, though you'll have to update the path to it as your gattaca repo as outlined in (2). Optionally can change parameters in it or a copy of it
6. Run `python superbit_lensing/prep_jobs.py {meta_config.yaml}`
7. Inspect the subdirectories created by the script, located in the `base_dir` you set in the meta config. You can see the configs it automatically generated for each mass/redshfit/realization
8. Ask a team member for the truth catalogs for each (m,z) pair you want to run and store them wherever the `nfw_dir` is set in your meta config
9. When ready, edit job_scripts/gattaca_run_pipe.sh with your desired `RUN_NAME`, `MASS`, `Z`, etc. and run it

## For the experts

If you want to add a new submodule to the pipeline, simply define a new subclass `MyCustomModule(SuperBITModule)` in `pipe.py` that implements the abstract `run()` function of the parent class and add it to `pipe.MODULE_TYPES` to register it with the rest of the pipeline. You should also implement the desired required & optional parameters that can be present in the module config with the class variables `_req_fields` and `_opt_fields`, which should be lists.

Contact @sweverett at spencer.w.everett@jpl.nasa.gov or @mccleary at j.mccleary@northeastern.edu you have any questions about running the pipeline - or even better, create an issue!
