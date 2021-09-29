This directory contains scripts to perform gaussian mixture modeling to galaxy shapes (`ngmix`) and
shear calibration (`metacalibration`). For a discussion of these algorithms, see 
https://github.com/esheldon/ngmix/wiki


`ngmix_fit_superbit.py`: uses the `MaxMetacalBootstrapper` class in the `ngmix` module to perform metacalibration
`make_annular_catalog.py`: this script turns the output files from the `ngmix_fit_superbit.py` scripts into a single
catalog ready for weak lensing analysis, with X, Y, RA, Dec, and metacalibrated shapes. Note that the *actual shear responsivity correction* of ngmix shapes
is performed here! 

Requirements:

`pip install mof`

`https://github.com/esheldon/meds`

`https://github.com/esheldon/ngmix`


