# OBA Run Instructions

More detailed instructions to follow. In the meantime, here is a quick overview on how to run the SuperBIT onboard analysis (OBA). All scripts are located in the `superbit_lensing/oba` directory.

**NOTE:** The OBA has an internal model of the QCC directory structure, but the root directory of the QCC can be sent to anything (default `/` of course). This allows us to easily run local tests or on the downloaded QCC files on `hen`. This is always set with the `-root_dir` optional arg. 

- **`root_dir` on `hen` will always be `/data/downloads/` unless you are testing on sims!**

### Quick use

Don't like reading? You probably just want to do the following:

1. `ssh bit@hen.astro.utoronto.ca` (usual password)
2. `conda activate oba`
3. `cd /home/bit/git/bit/superbit-metacal/superbit_lensing/oba`
4. `python prep_oba.py {TARGET_NAME} 0 1 0 0 -1 -1 1 -root_dir /data/downloads/`
5. `python qcc_run_oba.py 0 {TARGET_NAME} -root_dir /data/downloads/`
6. Inspect intermediate files in `/data/downloads/home/bit/oba_temp/{TARGET_NAME}/`
7. Profit

## Installation

### Locally

Install the repository in the usual way given the repo README. Make sure you build the `oba` conda env using `env_oba.yaml` and activate it.

### On Hen

The repo & `oba` conda env are already installed for the `bit` user, so just login with the usual password.

### On QCC

The main OBA scripts have all been registered to COW so you can interface in the usual way w/ the GUI. The script input parameters follow the same conventions as the following, but you don't need to specify a `root_dir` of course

## OBA Prep

For a variety of reasons (mostly catered to the case of extremely limited on-float bandwidth), we decided to make running the OBA a two-step process. First, you must run the [`prep_oba.py`](https://github.com/superbit-collaboration/superbit-metacal/blob/oba/superbit_lensing/oba/prep_oba.py) script to check for the number of images per band that meet your quality requirements to see if you are ready to run. Invoke the script by doing the following:
```
python prep_oba.py {TARGET_NAME} {REQ_U_IMAGES} {REQ_B_IMAGES} {REQ_G_IMAGES} {REQ_R_IMAGES} {REQ_NIR_IMAGES} {REQ_LUM_IMAGES} {ALLOW_UNVERIFIED 1/0} [-root_dir {ROOT_DIR}]
```
Here is the script docstring:
```
This script is used for preparing the SuperBIT onboard analysis (OBA) on the
QCC flight computer. It expects a target name and a series of integers for
each SuperBIT band that define the following:

1) How many images are required per-band for this target to run the OBA
   - Can be zero; will still analyze those bands & send down cutouts
2) Whether to ignore a given band in the OBA (by setting to -1)

In addition, the final argument `allow_unverified` is used to allow for some
flexibility in what images can be accepted for the OBA. The fiducial plan is
for all raw SCI images to be examined by a image checker that will set the
header IMG_QUAL to one of ['GOOD', 'BAD', 'UNVERIFIED']. In the case that the
image checker does not work or is not run on some images, you can choose to
allow unverified images in the analysis
```
If the script succeeds, then it will tell you the location of an output OBA configuration file for your chosen target. It uses the [global oba config](https://github.com/superbit-collaboration/superbit-metacal/blob/oba/superbit_lensing/oba/configs/oba_global_config.yaml) as a base and adds a few target-specific additions such as the bands to use and whether you want to allow unverified images.

You should definitely look at this config file to make sure it's going to do what you want it to do!

### Locally or on hen

Just use your favorite text editor!

### QCC

Use the COW version of the `print_oba_config.py` and `update_oba_config.py` scripts (don't remember the COW names offhand, but they are obvious)

## Run OBA

With the OBA config file made, we can actually run the thing! Invoke using:
```
python qcc_run_oba.py {OBA_MODE 0/1} {OBA_ARG TARGET_NAME/CONFIG_FILE} [-root_dir {ROOT_DIR}]
```
You will almost always run in `OBA_MODE = 0` which just means pass the `TARGET_NAME` and it will figure out the rest using the config file we just made with `prep_oba.py`. If instead you want to pass a bespoke OBA config file you made for any reason, simply run in `OBA_MODE = 1` and then pass the config filepath instead of the target name.

## OBA Outputs

The whole point of the OBA is to do all the hard reduction work on the QCC but only save the cutouts for an optimized weak-lensing sort of lossy data compression. This means that the default behaviour is to delete all intermediate data products in `ROOT_DIR/home/bit/oba_temp/{TARGET_NAME}` and save only the following to `ROOT_DIR/data/bit/oba_results/` (all `bzip2`'d):
- OBA config file
- Generated config files, such as for the `CookieCutter`
- `CookieCutter` object cutout FITS files (one for each band)
- - Will describe format later
- - Can save `1d`, `2d`, or both (will describe later)

To save useful intermediate data products for inspection such as the coadds, look at the section on OBA tips below for the correct config setting

## Temporary OBA directory structure

Each OBA run for a given `TARGET_NAME` will create a temporary directory at `ROOT_DIR/home/bit/oba_temp/{TARGET_NAME}`. It will have the following directory structure for each band (including the derived detection band `det`):
- root - (compressed & uncompressed copies of the raw files are put here)
- `cals/` - calibrated images that are dark & flat-field corrected, background-subtracted, masked, etc.
- `cats/` - `SExtractor` catalogs run on the single exposures (for astrometric registration). Will be the detection catalog only for band `det`
- `coadd/` - The `SWarp` coadd image. Will be the detection image for band `det`
- `failed/` - Images that failed any step are placed here, usually due to astrometric failures
- `out/` - The output files actually saved to disk
- `tmp/` - A temporary staging ground for final compressed output files

## Tips for successful OBA running

- Did you look at the config file before you ran? Are you sure it is doing what you want?
- OBA fail for any reason in the middle? Look at the log to figure out which stage the failure happened at, and then remove any stages before it in the config field `modules` once you've resolved the issue
   - Afraid you screwed something up & want to start over? Either run with the `--fresh` flag in `qcc_run_oba.py` or set `run_options['fresh']` to `true`
- If you want to look at any intermediate data products such as the coadds, you'll want to make sure that the config field `cleanup['clean_oba_dir']` is `false` so it doesn't automatically delete the OBA temp dir. Otherwise it only saves the nominal OBA results (cutouts, logs, configs, etc.)
- Coaddition really slow and you aren't on the QCC? Feel free to edit the `superbit_lensing/oba/configs/swarp/swarp.config` file to change `NTHREADS` to something larger (0 defaults to all available). **Don't** do this on QCC of course!
- If you are running on early images or big foreground objects (likely for pretty pictures), you might want to allow `IMG_QUAL = BAD`. In that case, set the OBA config file `run_options['min_image_quality']` field to `bad`
- Want to know more about config options? Take a look at the small class definition in [`config.py`](https://github.com/superbit-collaboration/superbit-metacal/blob/oba/superbit_lensing/oba/config.py)

## Extra OBA scripts

Will write this section as needed! For now, just ask me (@sweverett) if you are running any other script on the QCC
